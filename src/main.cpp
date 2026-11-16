// DIAGNOSTICO FEDERADO TEMPORARIO: requer MLPModel.h corrigido (MLP2).
// Monitor MQTT em fedkd/cmd/push para obter fp_pre, fp_kd, fp_ft e continuity.
/**
 * main.cpp — FedKD-MR ESP32 (clientes heterogêneos: MLP / CNN1D / GRU)
 * ======================================================================
 * Fluxo autônomo:
 *   boot → treino local supervisionado
 *        → KD com teacher_probs
 *        → fine-tuning supervisionado
 *        → impressão de métricas → idle
 *
 * Dependências:
 *   NeuralNetwork.h  (apenas para CLIENT_ID == 1, MLP)
 *   LittleFS, ArduinoJson, Config.h
 *   Model.h, MLPModel.h, CNN1DModel.h, GRUModel.h  (este projeto)
 *
 * Compilar com PlatformIO. Antes de gravar:
 *   1. Ajuste CLIENT_ID em Config.h  (1=MLP, 2=CNN1D, 3=GRU)
 *   2. Copie os arquivos do cliente para data/
 *   3. Upload Filesystem Image (LittleFS)
 *   4. Upload firmware
 */

// ── Activações para NeuralNetwork.h (MLP, cliente 1) ─────────────────────────
// Devem ser definidas antes de qualquer #include para manter índices corretos.
// São ignoradas nos clientes 2 e 3 (CNN1DModel/GRUModel não usam esses defines).
#define ACTIVATION__PER_LAYER
#define Sigmoid
#define Tanh
#define ReLU
#define LeakyReLU
#define ELU
#define SELU
#define Softmax

#include "Config.h"

// ── Inclui apenas a biblioteca MLP para o cliente 1 ─────────────────────────
#if MODEL_ARCH == MODEL_ARCH_MLP
  #include <NeuralNetwork.h>
#endif

#include "Model.h"
#include "MLPModel.h"
#include "CNN1DModel.h"
#include "GRUModel.h"

#include <LittleFS.h>
#include "FedKDCheckpoint.h"
#include "FedKDWire.h"
#include <ArduinoJson.h>
#include <esp_task_wdt.h>

#ifdef ENABLE_MQTT
#include <WiFi.h>
#include <PubSubClient.h>
#endif

// O servidor usa --temperature 2.0. Manter sincronizado com o YAML.
#ifndef KD_TEMPERATURE
#define KD_TEMPERATURE 2.0f
#endif

// ── Tipos ─────────────────────────────────────────────────────────────────────
#ifndef IDFLOAT
  #define IDFLOAT float
#endif

struct ClassMetrics {
    int tp = 0, fp = 0, fn = 0, tn = 0;
    float precision() const { return (tp + fp) > 0 ? (float)tp / (tp + fp) : 0.f; }
    float recall()    const { return (tp + fn) > 0 ? (float)tp / (tp + fn) : 0.f; }
    float f1()        const {
        float p = precision(), r = recall();
        return (p + r) > 0 ? 2.f * p * r / (p + r) : 0.f;
    }
};

struct RunMetrics {
    int n_samples    = 0;
    int n_correct    = 0;
    float mse_last   = 0.f;
    unsigned long ms = 0;
    bool kd_valid = false; // true somente se todas as epocas de KD terminarem
    ClassMetrics cls[N_CLASSES];
    float accuracy() const { return n_samples > 0 ? (float)n_correct / n_samples : 0.f; }
    float macro_f1() const {
        float sum = 0.f;
        for (int i = 0; i < N_CLASSES; i++) sum += cls[i].f1();
        return sum / N_CLASSES;
    }
};

// ── Resultado do pipeline (retornado por run_pipeline()) ─────────────────────
struct PipelineResult {
    float         acc_local  = 0.f;   // avaliação pré-KD  (D_test)
    float         acc_kd     = 0.f;   // avaliação pós-KD  (D_test)
    float         acc_ft     = 0.f;   // avaliação pós-FT  (D_test)
    float         f1_local   = 0.f;
    float         f1_kd      = 0.f;
    float         f1_ft      = 0.f;
    float         loss_local = 0.f;   // MSE último epoch treino local
    float         loss_kd    = 0.f;   // T^2 KL(Q || softmax(z/T)) média última época (0 se pulado)
    float         loss_ft    = 0.f;   // MSE último epoch FT  (0 se pulado)
    unsigned long t_local_ms = 0;     // duração treino local
    unsigned long t_kd_ms    = 0;     // duração KD  (0 se pulado)
    unsigned long t_ft_ms    = 0;     // duração FT  (0 se pulado)
    uint32_t hash_pre = 0;
    uint32_t hash_kd = 0;
    uint32_t hash_ft = 0;
    bool hash_pre_ok = false;
    bool hash_kd_ok = false;
    bool hash_ft_ok = false;
    int continuity = -1; // -1 = sem rodada anterior ou impossivel comparar; 0=falha; 1=OK
    bool completed = false;
    int checkpoint_status = 0; // 0=desativado, 1=salvo, 2=restaurado e salvo
};

// ── Variáveis globais ─────────────────────────────────────────────────────────
Model* model      = nullptr;  // ← agora é Model*, não NeuralNetwork*
bool pipeline_done = false;
// Mantém o estado treinado do modelo entre rodadas enquanto o ESP32 permanecer ligado.
static bool g_local_initialized = false;
static bool g_previous_ft_hash_valid = false;
static uint32_t g_previous_ft_hash = 0;
static bool g_resume_checked = false;
static bool g_restored_this_run = false;
static int g_last_completed_round = 0;

#ifdef ENABLE_MQTT
static WiFiClient    g_wifi_client;
static PubSubClient  g_mqtt_client(g_wifi_client);
static volatile bool g_teacher_received = false;
static volatile bool g_start_cmd        = false;
static volatile int  g_current_round    = 0;
static volatile bool g_pipeline_busy = false;
static volatile bool g_teacher_waiting = false;
#endif

// ── Utilidades ────────────────────────────────────────────────────────────────
static void print_sep() { Serial.println("----------------------------------------------------"); }
static void print_mem() { Serial.printf("  Heap livre: %u bytes\n", (unsigned)esp_get_free_heap_size()); }

static const float MAX_PARAM = 5.0f;

// ── Diagnóstico de continuidade de pesos (somente teste da Fase 1) ───────────
// Serializa o modelo em um arquivo temporário, calcula FNV-1a sobre TODOS os
// bytes serializados e remove o arquivo. Não altera os parâmetros do modelo.
// ATENÇÃO: requer espaço livre no LittleFS (GRU pode ocupar >100 KiB).
static bool log_weight_fingerprint(Model& m, const char* stage, uint32_t* hash_out = nullptr) {
    const char* tmp_path = "/.fedkd_fp_diag.tmp";
    if (LittleFS.exists(tmp_path)) LittleFS.remove(tmp_path);

    File out = LittleFS.open(tmp_path, "w");
    if (!out) {
        Serial.printf("[PESOS] %s ERRO: sem arquivo temporario\n", stage);
        return false;
    }
    bool written = m.saveWeights(out);
    out.flush();
    size_t saved_size = out.size();
    out.close();
    if (!written) {
        LittleFS.remove(tmp_path);
        Serial.printf("[PESOS] %s ERRO: serializacao incompleta (verificar espaco LittleFS)\n", stage);
        return false;
    }

    File in = LittleFS.open(tmp_path, "r");
    if (!in) {
        LittleFS.remove(tmp_path);
        Serial.printf("[PESOS] %s ERRO: leitura de diagnostico indisponivel\n", stage);
        return false;
    }
    uint32_t hash = 2166136261UL;  // FNV-1a 32-bit (diagnostico, nao criptografico)
    size_t nbytes = 0;
    uint8_t buf[256];
    while (in.available()) {
        size_t n = in.read(buf, sizeof(buf));
        if (n == 0) break;
        for (size_t i = 0; i < n; ++i) {
            hash ^= buf[i];
            hash *= 16777619UL;
        }
        nbytes += n;
    }
    in.close();
    LittleFS.remove(tmp_path);
    if (nbytes != saved_size) {
        Serial.printf("[PESOS] %s ERRO: leitura parcial (%u/%u bytes)\n",
                      stage, (unsigned)nbytes, (unsigned)saved_size);
        return false;
    }
    if (hash_out) *hash_out = hash;
    Serial.printf("[PESOS] %s hash=%08lX bytes=%u\n",
                  stage, (unsigned long)hash, (unsigned)nbytes);
    return true;
}

// ── Leitura do metadata.json ──────────────────────────────────────────────────
struct DatasetSchema {
    int     n_features    = 0;
    int     n_classes     = N_CLASSES;
    int     bytes_per_row = 0;
    int     label_offset  = 0;
    bool    encoded_labels = false;
    int     feat_offsets[N_INPUT];
    uint8_t label_type;  // 0=uint8, 1=int8, 2=int32
};

static bool load_schema(const char* meta_path, DatasetSchema& s) {
    File f = LittleFS.open(meta_path, "r");
    if (!f) { Serial.printf("[ERR] Não encontrou %s\n", meta_path); return false; }
    JsonDocument doc;
    if (deserializeJson(doc, f) != DeserializationError::Ok) {
        Serial.println("[ERR] Falha ao parsear metadata.json");
        f.close(); return false;
    }
    f.close();
    s.bytes_per_row  = doc["bytes_per_row"] | 0;
    s.encoded_labels = doc["label_map"].is<JsonObject>();
    const char* label_col = doc["label_column"] | "label";
    JsonArray schema = doc["schema"];
    int feat_idx = 0;
    for (JsonObject col : schema) {
        const char* name = col["name"] | "";
        const char* type = col["type"] | "float32";
        int offset       = col["offset"] | 0;
        if (strcmp(name, label_col) == 0) {
            s.label_offset = offset;
            if      (strcmp(type, "uint8") == 0) s.label_type = 0;
            else if (strcmp(type, "int8")  == 0) s.label_type = 1;
            else                                  s.label_type = 2;
        } else if (strcmp(name, "timestamp") != 0 && feat_idx < N_INPUT) {
            s.feat_offsets[feat_idx++] = offset;
        }
    }
    s.n_features = feat_idx;
    if (s.n_features != N_INPUT) {
        Serial.printf("[ERR] Schema: %d features; esperado %d\n", s.n_features, N_INPUT);
        return false;
    }
    return true;
}

static long read_label(uint8_t* row, const DatasetSchema& s) {
    if      (s.label_type == 0) return (long)(*(uint8_t*)(row + s.label_offset));
    else if (s.label_type == 1) return (long)(*(int8_t*) (row + s.label_offset));
    else { int32_t v; memcpy(&v, row + s.label_offset, 4); return (long)v; }
}
static int label_to_idx(long lv) { return (int)(lv - 1); }

// ── Funções MQTT ──────────────────────────────────────────────────────────────
#ifdef ENABLE_MQTT

static const char* model_arch_name() {
#if MODEL_ARCH == MODEL_ARCH_MLP
    return "MLP";
#elif MODEL_ARCH == MODEL_ARCH_CNN1D
    return "CNN1D";
#else
    return "GRU";
#endif
}
static void publish_identity() {
    if (!g_mqtt_client.connected()) return;
    char mac[24];
    snprintf(mac, sizeof(mac), "%012llX", (unsigned long long)ESP.getEfuseMac());
    char json[256];
    int n = snprintf(json, sizeof(json),
        "{\"client_id\":\"%s\",\"model_arch\":\"%s\",\"device_id\":\"%s\",\"firmware_version\":\"bloco6-v1\",\"experiment_id\":\"%s\",\"round\":%d}",
        CLIENT_NAME, model_arch_name(), mac, FedKDCheckpoint::experimentId(), (int)g_current_round);
    if (n>0 && (size_t)n<sizeof(json))
        g_mqtt_client.publish(TOPIC_IDENTITY_PUSH, (const uint8_t*)json, (unsigned)n, false);
    Serial.printf("[IDENTIDADE] client=%s arquitetura=%s device=%s firmware=bloco6-v1\n",
                  CLIENT_NAME, model_arch_name(), mac);
}

static void mqtt_on_message(char* topic, byte* payload, unsigned int length) {
    if (strcmp(topic, TOPIC_TEACHER_PULL) == 0) {
        if (!g_teacher_waiting || !g_pipeline_busy || g_current_round<=0) {
            Serial.println("[WIRE] teacher fora da janela ativa rejeitado"); return;
        }
        if (!FedKDWire::valid((const uint8_t*)payload, length,
                              (uint32_t)g_current_round, FedKDCheckpoint::experimentId())) {
            Serial.println("[WIRE] teacher de rodada/experimento incorreto rejeitado"); return;
        }
        const size_t data_len = (size_t)length - FedKDWire::HEADER_LEN;
        if (data_len == 0 || data_len % (N_CLASSES*sizeof(float)) != 0) {
            Serial.println("[WIRE] teacher com comprimento invalido"); return;
        }
        const int n_pub_recv = (int)(data_len / (N_CLASSES*sizeof(float)));
        // Conferencia basica de probabilidades e finitude antes de substituir arquivo.
        const uint8_t* raw = ((const uint8_t*)payload) + FedKDWire::HEADER_LEN;
        for (int i=0;i<n_pub_recv;++i) {
            float sum=0.f;
            for(int j=0;j<N_CLASSES;++j) {
                float v; memcpy(&v,raw+(i*N_CLASSES+j)*sizeof(float),sizeof(float));
                if(!isfinite(v) || v<0.f || v>1.f) {
                    Serial.println("[WIRE] teacher invalido (valor fora de [0,1])"); return;
                }
                sum+=v;
            }
            if(fabsf(sum-1.f)>0.01f) {
                Serial.println("[WIRE] teacher invalido (soma de probabilidades)"); return;
            }
        }
        File f = LittleFS.open(TEACHER_PROBS_PATH, "w");
        if (!f) { Serial.println("[WIRE] falha ao gravar teacher"); return; }
        size_t written=f.write(raw,data_len); f.close();
        if (written != data_len) { Serial.println("[WIRE] teacher parcialmente gravado"); return; }
        File fm = LittleFS.open(TEACHER_PROBS_META_PATH, "w");
        if(!fm) { Serial.println("[WIRE] sem metadados teacher"); return; }
        fm.printf("{\"n_pub\":%d}",n_pub_recv); fm.close();
        g_teacher_received=true;
        g_teacher_waiting=false;
        Serial.printf("[WIRE] teacher aceito experimento=%s round=%d amostras=%d\n",
                      FedKDCheckpoint::experimentId(),(int)g_current_round,n_pub_recv);
    } else if (strcmp(topic, TOPIC_CMD_PULL) == 0) {
        JsonDocument doc;
        if (deserializeJson(doc, (char*)payload, length) == DeserializationError::Ok) {
            const char* cmd = doc["cmd"] | "";
            if (strcmp(cmd, "start") == 0) {
                const char* experiment_id = doc["experiment_id"] | "";
                const int incoming_round = doc["round"] | 0;
                if (incoming_round <= 0 || g_pipeline_busy || g_start_cmd ||
                    (g_last_completed_round>0 && incoming_round<=g_last_completed_round) ||
                    !FedKDCheckpoint::setExperimentId(experiment_id)) {
                    Serial.println("[CKPT] START rejeitado: experiment_id ausente, invalido ou diferente do experimento ativo.");
                    return;
                }
                g_current_round = incoming_round;
                g_start_cmd = true;
                publish_identity();
                Serial.printf("[MQTT] 'start' recebido (experimento=%s, round=%d)\n",
                    FedKDCheckpoint::experimentId(), g_current_round);
            }
        }
    }
}

static bool mqtt_connect() {
    Serial.printf("[MQTT] Conectando Wi-Fi: %s", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < CONNECTION_TIMEOUT) {
        delay(500); Serial.print("."); esp_task_wdt_reset();
    }
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\n[MQTT] ERRO: Wi-Fi não conectou.");
        return false;
    }
    Serial.printf("\n[MQTT] Wi-Fi OK — IP: %s\n", WiFi.localIP().toString().c_str());
    g_mqtt_client.setServer(MQTT_BROKER, MQTT_PORT);
    g_mqtt_client.setCallback(mqtt_on_message);
    if (!g_mqtt_client.setBufferSize(2048))
        Serial.println("[MQTT] AVISO: buffer 2048 não alocado");
    if (!g_mqtt_client.connect(CLIENT_NAME)) {
        Serial.printf("[MQTT] ERRO: broker %s:%d (state=%d)\n",
                      MQTT_BROKER, MQTT_PORT, g_mqtt_client.state());
        return false;
    }
    g_mqtt_client.subscribe(TOPIC_TEACHER_PULL);
    g_mqtt_client.subscribe(TOPIC_CMD_PULL);
    publish_identity();
    Serial.printf("[MQTT] Conectado. Subscrito: %s | %s\n",
                  TOPIC_TEACHER_PULL, TOPIC_CMD_PULL);
    return true;
}

/**
 * FeedForward em todas as amostras de XY_PUB_PATH e publica logits
 * como float32 row-major (n_pub × N_CLASSES × 4 bytes).
 * Usa Model::feedForward() — funciona para MLP, CNN1D e GRU.
 */
static int publish_logits_mqtt(Model& m) {
    DatasetSchema s;
    if (!load_schema(METADATA_JSON_PATH, s)) return -1;

    File fc = LittleFS.open(XY_PUB_PATH, "r");
    if (!fc) { Serial.println("[MQTT] Não abriu dataset público"); return -1; }
    int n_pub = (int)(fc.size() / s.bytes_per_row);
    fc.close();
    if (n_pub <= 0) { Serial.println("[MQTT] Dataset público vazio"); return -1; }

    size_t buf_size = (size_t)n_pub * N_CLASSES * sizeof(float);
    float* lb = (float*)malloc(buf_size);
    if (!lb) { Serial.println("[MQTT] malloc logits falhou"); return -1; }

    uint8_t* rowbuf = (uint8_t*)malloc(s.bytes_per_row);
    if (!rowbuf) { free(lb); return -1; }
    float x[N_INPUT];

    File f = LittleFS.open(XY_PUB_PATH, "r");
    int idx = 0;
    while (f.available() >= s.bytes_per_row && idx < n_pub) {
        if (f.read(rowbuf, s.bytes_per_row) != (size_t)s.bytes_per_row) break;
        for (int i = 0; i < N_INPUT; i++) {
            float v; memcpy(&v, rowbuf + s.feat_offsets[i], 4); x[i] = v;
        }
        float* pred = m.feedForward(x);
        float z[N_CLASSES];
        if (!m.getLogits(z)) {
            Serial.println("[LOGITS] Modelo nao disponibilizou saidas pre-softmax");
            f.close(); free(rowbuf); free(lb); return -1;
        }
        // Verifica coerencia numerica entre logits reconstruidos e a saida
        // de inferencia. Falhar impede comunicar logits semanticamente incorretos.
        float mx = z[0];
        for (int k=1; k<N_CLASSES; ++k) if (z[k]>mx) mx=z[k];
        float denom=0.f, reconstructed[N_CLASSES];
        for (int k=0; k<N_CLASSES; ++k) {
            if (!isfinite(z[k])) { f.close(); free(rowbuf); free(lb); return -1; }
            reconstructed[k]=expf(z[k]-mx); denom+=reconstructed[k];
        }
        if (!(denom>0.f) || !isfinite(denom)) {
            f.close(); free(rowbuf); free(lb); return -1;
        }
        for (int k=0; k<N_CLASSES; ++k) {
            reconstructed[k]/=denom;
            if (!isfinite(pred[k]) || fabsf(reconstructed[k]-pred[k])>0.02f) {
                Serial.printf("[LOGITS] Falha de consistencia softmax (classe %d, estimado %.5f, modelo %.5f)\n",
                              k,reconstructed[k],pred[k]);
                f.close(); free(rowbuf); free(lb); return -1;
            }
            lb[idx * N_CLASSES + k] = z[k];
        }
        idx++;
        esp_task_wdt_reset();
    }
    f.close();
    if (idx != n_pub) { free(rowbuf); free(lb); return -1; }

    // Garante conexão antes de publicar (treino pode ter demorado > keepalive)
    if (!g_mqtt_client.connected()) {
        Serial.println("[MQTT] Reconectando antes de publicar logits...");
        if (g_mqtt_client.connect(CLIENT_NAME)) {
            g_mqtt_client.subscribe(TOPIC_TEACHER_PULL);
            g_mqtt_client.subscribe(TOPIC_CMD_PULL);
            publish_identity();
        } else {
            Serial.printf("[MQTT] Falha ao reconectar (state=%d)\n", g_mqtt_client.state());
            free(rowbuf); free(lb);
            return -1;
        }
    }

    const size_t msg_size = FedKDWire::HEADER_LEN + buf_size;
    uint8_t* wire=(uint8_t*)malloc(msg_size);
    if(!wire) { free(rowbuf);free(lb);return -1; }
    FedKDWire::build(wire,(uint32_t)g_current_round,FedKDCheckpoint::experimentId());
    memcpy(wire+FedKDWire::HEADER_LEN,lb,buf_size);
    publish_identity();
    g_teacher_waiting=true;
    bool ok = g_mqtt_client.publish(TOPIC_LOGITS_PUSH,wire,(unsigned int)msg_size,false);
    free(wire);
    if(!ok) g_teacher_waiting=false;
    Serial.printf("[MQTT] Logits: %d amostras → %s (%u bytes) [%s]\n",
                  idx, TOPIC_LOGITS_PUSH, (unsigned)msg_size, ok ? "OK" : "FALHA");
    free(rowbuf); free(lb);
    return ok ? idx : -1;
}

static bool wait_for_teacher_probs(uint32_t timeout_ms) {
    uint32_t deadline = millis() + timeout_ms;
    Serial.printf("[MQTT] Aguardando teacher_probs (timeout=%lus)...\n",
                  (unsigned long)(timeout_ms / 1000));
    while (millis() < deadline) {
        g_mqtt_client.loop();
        if (g_teacher_received) {
            Serial.println("[MQTT] teacher_probs recebido.");
            return true;
        }
        static uint32_t last_print = 0;
        if (millis() - last_print >= 10000) {
            last_print = millis();
            Serial.printf("[MQTT] ... %lus restantes\n",
                          (unsigned long)((deadline - millis()) / 1000));
        }
        esp_task_wdt_reset();
        delay(200);
    }
    g_teacher_waiting=false;
    Serial.println("[MQTT] TIMEOUT aguardando teacher_probs — KD será pulado.");
    return false;
}

#endif  // ENABLE_MQTT

// ── Leitura do teacher_probs_meta ────────────────────────────────────────────
static int read_n_pub() {
    File f = LittleFS.open(TEACHER_PROBS_META_PATH, "r");
    if (!f) { Serial.println("[AVISO] teacher_probs_meta.json não encontrado"); return 0; }
    JsonDocument doc;
    deserializeJson(doc, f);
    f.close();
    return doc["n_pub"] | 0;
}

// ── Treino supervisionado local ───────────────────────────────────────────────
// Usa Model& — funciona com MLP, CNN1D e GRU sem alterar nada aqui.
static RunMetrics train_supervised(Model& m,
                                   const char* bin_path,
                                   const char* meta_path,
                                   int epochs,
                                   float lr_w, float lr_b,
                                   const char* label) {
    RunMetrics result;
    DatasetSchema s;
    if (!load_schema(meta_path, s)) return result;

    m.setLR(lr_w, lr_b);

    uint8_t* rowbuf = (uint8_t*)malloc(s.bytes_per_row);
    if (!rowbuf) { Serial.println("[ERR] malloc rowbuf"); return result; }
    float x[N_INPUT];
    float y[N_CLASSES];

    unsigned long t0 = millis();
    Serial.printf("\n[%s] epochs=%d  lr=%.4f\n", label, epochs, lr_w);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        File f = LittleFS.open(bin_path, "r");
        if (!f) { Serial.printf("[ERR] Não abriu %s\n", bin_path); break; }

        int n_ok = 0, n_total = 0;
        while (f.available() >= s.bytes_per_row) {
            if (f.read(rowbuf, s.bytes_per_row) != (size_t)s.bytes_per_row) break;

            for (int i = 0; i < N_INPUT; i++) {
                float v; memcpy(&v, rowbuf + s.feat_offsets[i], 4); x[i] = v;
            }
            long lv = read_label(rowbuf, s);
            int  idx = label_to_idx(lv);
            if (idx < 0 || idx >= N_CLASSES) { n_total++; continue; }
            for (int k = 0; k < N_CLASSES; k++) y[k] = (k == idx) ? 1.f : 0.f;

            // ── DEBUG cirúrgico: 1ª amostra, 1ª época ──────────────────────
            if (epoch == 1 && n_total == 0) {
                float* pred = m.feedForward(x);
                Serial.print("[BP-DEBUG] Saida pre-BP: ");
                for (int k = 0; k < N_CLASSES; k++) Serial.printf("%.4f ", pred[k]);
                Serial.println();
                m.backProp(y);
                m.clampParams(MAX_PARAM);
                pred = m.feedForward(x);
                Serial.print("[BP-DEBUG] Saida pos-BP: ");
                for (int k = 0; k < N_CLASSES; k++) Serial.printf("%.4f ", pred[k]);
                Serial.println();
            } else {
                m.feedForward(x);
                m.backProp(y);
                m.clampParams(MAX_PARAM);
            }

            // Acurácia
            float* pred = m.feedForward(x);
            int pred_cls = 0;
            for (int k = 1; k < N_CLASSES; k++)
                if (pred[k] > pred[pred_cls]) pred_cls = k;
            if (pred_cls == idx) n_ok++;
            n_total++;

            esp_task_wdt_reset();
        }
        f.close();

        float acc = n_total > 0 ? (float)n_ok / n_total : 0.f;
        result.mse_last = m.getMSE(n_total > 0 ? n_total : 1);
        Serial.printf("  Época %2d/%d — acc=%.3f  mse=%.5f  n=%d\n",
                      epoch, epochs, acc, result.mse_last, n_total);
    }
    result.ms = millis() - t0;
    free(rowbuf);
    return result;
}

// ── Treino KD ────────────────────────────────────────────────────────────────
static RunMetrics train_kd(Model& m,
                            const char* pub_bin_path,
                            const char* pub_meta_path,
                            const char* tp_path,
                            int n_pub,
                            int epochs,
                            float lr_w, float lr_b) {
    RunMetrics result;
    DatasetSchema s;
    if (!load_schema(pub_meta_path, s)) return result;

    size_t tp_size = (size_t)n_pub * N_CLASSES * sizeof(float);
    float* tp = (float*)malloc(tp_size);
    if (!tp) { Serial.println("[ERR] malloc teacher_probs"); return result; }

    File ftf = LittleFS.open(tp_path, "r");
    if (!ftf) { Serial.println("[ERR] teacher_probs.bin"); free(tp); return result; }
    size_t read_bytes = ftf.read((uint8_t*)tp, tp_size);
    ftf.close();
    if (read_bytes != tp_size) {
        Serial.printf("[ERR] teacher_probs: leu %u/%u bytes\n",
                      (unsigned)read_bytes, (unsigned)tp_size);
        free(tp); return result;
    }
    Serial.printf("  teacher_probs: %d × %d = %u bytes\n",
                  n_pub, N_CLASSES, (unsigned)tp_size);

    m.setLR(lr_w, lr_b);

    uint8_t* rowbuf = (uint8_t*)malloc(s.bytes_per_row);
    if (!rowbuf) { free(tp); return result; }
    float x[N_INPUT];
    float y_soft[N_CLASSES];

    unsigned long t0 = millis();
    if (!isfinite(KD_TEMPERATURE) || KD_TEMPERATURE <= 0.f) {
        Serial.println("[KD-FATAL] KD_TEMPERATURE invalida"); free(rowbuf); free(tp); return result;
    }
    Serial.printf("\n[KD] epochs=%d  lr=%.4f  T=%.3f  loss=T^2*KL\n", epochs, lr_w, (float)KD_TEMPERATURE);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        File f = LittleFS.open(pub_bin_path, "r");
        if (!f) break;
        int sidx = 0; float epoch_kl_sum=0.f; bool epoch_ok=true;
        while (f.available() >= s.bytes_per_row && sidx < n_pub) {
            if (f.read(rowbuf, s.bytes_per_row) != (size_t)s.bytes_per_row) break;
            for (int i = 0; i < N_INPUT; i++) {
                float v; memcpy(&v, rowbuf + s.feat_offsets[i], 4); x[i] = v;
            }
            float* row_tp = tp + sidx * N_CLASSES;
            for (int k = 0; k < N_CLASSES; k++) y_soft[k] = row_tp[k];

            if (epoch == 1 && sidx == 0) {
                Serial.println("\n[KD-DEBUG] y_soft amostra 0:");
                float sum_s = 0;
                for (int k = 0; k < N_CLASSES; k++) {
                    Serial.printf("  classe %d: %.4f\n", k, y_soft[k]);
                    sum_s += y_soft[k];
                }
                Serial.printf("  soma=%.4f  %s\n", sum_s,
                    (sum_s > 0.99f && sum_s < 1.01f) ? "OK" : "ERRO");
            }

            m.feedForward(x);
            const float sample_kl = m.backPropKD(y_soft, KD_TEMPERATURE);
            if(!isfinite(sample_kl)) {
                Serial.printf("[KD-FATAL] Teacher/gradiente invalido na amostra %d\n", sidx);
                epoch_ok=false; break;
            }
            epoch_kl_sum += sample_kl;
            m.clampParams(MAX_PARAM);
            sidx++;
            esp_task_wdt_reset();
        }
        f.close();
        if(!epoch_ok || sidx!=n_pub) {
            Serial.printf("[KD-FATAL] Epoca incompleta: %d/%d amostras\n", sidx, n_pub);
            result.ms=0; free(rowbuf); free(tp); return result;
        }
        result.mse_last = epoch_kl_sum / sidx;
        Serial.printf("  Epoca KD %2d/%d — loss_T2KL=%.6f  n=%d\n",
                      epoch, epochs, result.mse_last, sidx);
    }
    result.ms = millis() - t0;
    result.kd_valid = true;
    free(rowbuf); free(tp);
    return result;
}

// ── Avaliação ────────────────────────────────────────────────────────────────
static RunMetrics evaluate(Model& m,
                            const char* bin_path,
                            const char* meta_path,
                            const char* label) {
    RunMetrics result;
    DatasetSchema s;
    if (!load_schema(meta_path, s)) return result;

    uint8_t* rowbuf = (uint8_t*)malloc(s.bytes_per_row);
    if (!rowbuf) return result;
    float x[N_INPUT];

    File f = LittleFS.open(bin_path, "r");
    if (!f) { free(rowbuf); return result; }

    while (f.available() >= s.bytes_per_row) {
        if (f.read(rowbuf, s.bytes_per_row) != (size_t)s.bytes_per_row) break;
        for (int i = 0; i < N_INPUT; i++) {
            float v; memcpy(&v, rowbuf + s.feat_offsets[i], 4); x[i] = v;
        }
        long lv = read_label(rowbuf, s);
        int  true_cls = label_to_idx(lv);
        if (true_cls < 0 || true_cls >= N_CLASSES) continue;

        float* pred = m.feedForward(x);
        int pred_cls = 0;
        for (int k = 1; k < N_CLASSES; k++)
            if (pred[k] > pred[pred_cls]) pred_cls = k;

        result.n_samples++;
        if (pred_cls == true_cls) result.n_correct++;
        for (int k = 0; k < N_CLASSES; k++) {
            if (k == true_cls && k == pred_cls)       result.cls[k].tp++;
            else if (k == pred_cls && k != true_cls)  result.cls[k].fp++;
            else if (k == true_cls && k != pred_cls)  result.cls[k].fn++;
            else                                       result.cls[k].tn++;
        }
        esp_task_wdt_reset();
    }
    f.close(); free(rowbuf);

    const char* cls_names[] = {"EMPTY", "STATIONARY", "APPROACHING", "LEAVING"};
    Serial.printf("\n[%s] Acc=%.3f  MacroF1=%.3f  n=%d\n",
                  label, result.accuracy(), result.macro_f1(), result.n_samples);
    for (int k = 0; k < N_CLASSES; k++) {
        Serial.printf("  %12s: P=%.3f R=%.3f F1=%.3f (tp=%d fp=%d fn=%d)\n",
                      cls_names[k],
                      result.cls[k].precision(),
                      result.cls[k].recall(),
                      result.cls[k].f1(),
                      result.cls[k].tp, result.cls[k].fp, result.cls[k].fn);
    }
    return result;
}

// ── Fábrica de modelos ────────────────────────────────────────────────────────
static Model* create_model() {
#if MODEL_ARCH == MODEL_ARCH_MLP
    Serial.println("Modelo: MLP (NeuralNetwork.h)");
    unsigned int layers_init[] = MODEL_LAYERS;
    byte         actv_init[]   = MODEL_ACTV;
    unsigned int* layers = new unsigned int[MODEL_N_LAYERS];
    byte*         actv   = new byte[MODEL_N_LAYERS - 1];
    memcpy(layers, layers_init, MODEL_N_LAYERS   * sizeof(unsigned int));
    memcpy(actv,   actv_init,   (MODEL_N_LAYERS-1) * sizeof(byte));
    // MLPModel toma posse de layers e actv — não deletar aqui!
    // NeuralNetwork.h armazena o ponteiro layers internamente; deletar agora = dangling ptr.
    return new MLPModel(layers, MODEL_N_LAYERS, actv);

#elif MODEL_ARCH == MODEL_ARCH_CNN1D
    Serial.println("Modelo: CNN1D (Conv32+Conv64+GMP+Dense4)");
    return new CNN1DModel();

#elif MODEL_ARCH == MODEL_ARCH_GRU
    Serial.println("Modelo: GRU (hidden=64, Dense4)");
    return new GRUModel();

#else
    #error "MODEL_ARCH não definido — verifique Config.h"
#endif
}

// ── Pipeline principal ───────────────────────────────────────────────────────
static PipelineResult run_pipeline() {
    PipelineResult pr;
    print_sep();
    Serial.printf("FedKD-MR — Cliente %s\n", CLIENT_NAME);
    print_sep();
    print_mem();

    // BLOCO 5: preflight obrigatório: não executar KD/FT sobre conjuntos antigos.
    DatasetSchema ds5;
    if (!load_schema(METADATA_JSON_PATH, ds5) || ds5.bytes_per_row <= 0) {
        Serial.println("[BLOCO5-FATAL] metadata invalido");
        return pr;
    }
    const char* b5paths[] = {XY_TRAIN_PATH, XY_TEST_PATH, XY_PUB_PATH};
    const char* b5names[] = {"train", "test", "public"};
    for (int bi=0; bi<3; ++bi) {
        File bf=LittleFS.open(b5paths[bi], "r");
        if(!bf) { Serial.printf("[BLOCO5-FATAL] arquivo ausente: %s\n", b5paths[bi]); return pr; }
        size_t nb=bf.size(); bf.close();
        if(nb==0 || nb % (size_t)ds5.bytes_per_row != 0) {
            Serial.printf("[BLOCO5-FATAL] formato invalido: %s (%u bytes)\n", b5paths[bi], (unsigned)nb);
            return pr;
        }
        Serial.printf("[BLOCO5] %s: %u registros\n", b5names[bi], (unsigned)(nb/(size_t)ds5.bytes_per_row));
    }

    // Inicializa o modelo somente na primeira execução. Nas rodadas seguintes,
    // usa os parâmetros resultantes de KD + FT da rodada anterior.
    if (!model) {
        model = create_model();
        if (!model) {
            Serial.println("[FATAL] Falha ao criar modelo");
            return pr;
        }
        print_mem();
    }

#ifdef ENABLE_MQTT
    const int checkpoint_round = (int)g_current_round;
#else
    const int checkpoint_round = 1;
#endif
    // Restauracao opt-in, somente na primeira rodada observada apos boot.
    // Identidade de experimento explicita evita misturar execucoes anteriores.
    if (!FedKDCheckpoint::enabled()) {
        Serial.println("[FATAL] Sem experiment_id no comando de inicio; nao executar rodada.");
        return pr;
    }
    if (!g_resume_checked) {
        uint32_t ck_hash=0;
        int status=FedKDCheckpoint::restoreForRound(*model, checkpoint_round, &ck_hash);
        g_resume_checked=true;
        if(status<0) {
            Serial.println("[FATAL] Estado persistente incompativel; rodada nao executada.");
            return pr;
        }
        if(status==1) {
            g_local_initialized=true;
            g_restored_this_run=true;
            g_last_completed_round=checkpoint_round-1;
            g_previous_ft_hash_valid=false; // nao comparar formato distinto de hash
            Serial.println("[CKPT] Treino inicial dispensado: pesos restaurados.");
        }
    }

    if (g_last_completed_round && checkpoint_round != g_last_completed_round + 1) {
        Serial.printf("[FATAL] Sequencia invalida: ultima=%d comando=%d\n",g_last_completed_round,checkpoint_round);
        return pr;
    }
    // ── Etapa 1: Treino local supervisionado (somente inicialização) ──
    RunMetrics m_local;
    if (!g_local_initialized) {
        print_sep();
        Serial.println("ETAPA 1 — Treino local inicial supervisionado (D_priv)");
        m_local = train_supervised(
            *model, XY_TRAIN_PATH, METADATA_JSON_PATH,
            LOCAL_EPOCHS, LOCAL_LR_WEIGHTS, LOCAL_LR_BIASES, "Local"
        );
        g_local_initialized = true;
        Serial.printf("  Concluído em %lums\n", m_local.ms);
        print_mem();
    } else {
        print_sep();
        Serial.println("ETAPA 1 — Treino inicial já executado; preservando pesos da rodada anterior.");
    }

    // Assinatura depois da inicializacao (rodada 1) ou dos pesos preservados.
    pr.hash_pre_ok = log_weight_fingerprint(*model, "PRE_KD", &pr.hash_pre);
    if (g_local_initialized && g_previous_ft_hash_valid && pr.hash_pre_ok) {
        pr.continuity = (pr.hash_pre == g_previous_ft_hash) ? 1 : 0;
        Serial.printf("[CONTINUIDADE] rodada anterior POST_FT=%08lX / atual PRE_KD=%08lX => %s\n",
            (unsigned long)g_previous_ft_hash, (unsigned long)pr.hash_pre,
            pr.continuity == 1 ? "OK" : "FALHA");
    } else {
        Serial.println("[CONTINUIDADE] Sem assinatura anterior valida para comparar.");
    }

    RunMetrics eval_pre = evaluate(*model, XY_TEST_PATH, METADATA_JSON_PATH,
                                   "Avaliação pré-KD (D_test)");
    pr.acc_local  = eval_pre.accuracy();
    pr.f1_local   = eval_pre.macro_f1();
    pr.loss_local = m_local.mse_last;
    pr.t_local_ms = m_local.ms;

    // ── Etapa 2: Knowledge Distillation ──────────────────────────
    print_sep();
    Serial.println("ETAPA 2 — Knowledge Distillation (D_pub + teacher_probs)");

#ifdef ENABLE_MQTT
    if(publish_logits_mqtt(*model)<=0 || !wait_for_teacher_probs(120000UL)) {
        Serial.println("[FATAL] Sem consenso valido desta rodada; checkpoint nao sera salvo.");
        return pr;
    }
    int n_pub = read_n_pub();
#else
    int n_pub = read_n_pub();
#endif

    RunMetrics m_kd, m_ft;  // timing = 0 se KD/FT forem pulados
    if (n_pub <= 0) {
        Serial.println("[FATAL] n_pub=0: sem KD; checkpoint nao sera salvo.");
        return pr;
    } else {
        m_kd = train_kd(
            *model,
            XY_PUB_PATH, METADATA_JSON_PATH,
            TEACHER_PROBS_PATH, n_pub,
            KD_EPOCHS, KD_LR_WEIGHTS, KD_LR_BIASES
        );
        if(!m_kd.kd_valid) {
            Serial.println("[FATAL] KD incompleta; FT e checkpoint cancelados");
            return pr;
        }
        Serial.printf("  KD concluído em %lums\n", m_kd.ms);
        print_mem();

        RunMetrics eval_kd = evaluate(*model, XY_TEST_PATH, METADATA_JSON_PATH,
                                      "Avaliação pós-KD (D_test)");
        pr.acc_kd  = eval_kd.accuracy();
        pr.f1_kd   = eval_kd.macro_f1();
        pr.loss_kd = m_kd.mse_last;
        // Diagnostico: estado do modelo imediatamente depois da KD, antes do FT.
        pr.hash_kd_ok = log_weight_fingerprint(*model, "POST_KD", &pr.hash_kd);
        Serial.printf("[DIAG] acc_PRE_KD=%.6f acc_POST_KD=%.6f delta_KD=%+.6f\n",
                      pr.acc_local, pr.acc_kd, pr.acc_kd - pr.acc_local);

        // ── Etapa 3: Fine-tuning ──────────────────────────────────
        print_sep();
        Serial.println("ETAPA 3 — Fine-tuning supervisionado (D_priv)");
        m_ft = train_supervised(
            *model, XY_TRAIN_PATH, METADATA_JSON_PATH,
            FT_EPOCHS, FT_LR_WEIGHTS, FT_LR_BIASES, "Fine-tuning"
        );
        Serial.printf("  Fine-tuning concluído em %lums\n", m_ft.ms);
        print_mem();
        pr.loss_ft = m_ft.mse_last;
    }
    pr.t_kd_ms = m_kd.ms;
    pr.t_ft_ms = m_ft.ms;

    // ── Avaliação final ───────────────────────────────────────────
    print_sep();
    Serial.println("AVALIAÇÃO FINAL — D_test");
    RunMetrics eval_post = evaluate(*model, XY_TEST_PATH, METADATA_JSON_PATH,
                                    "Avaliação pós-FT (D_test)");
    pr.acc_ft = eval_post.accuracy();
    pr.f1_ft  = eval_post.macro_f1();

    print_sep();
    Serial.println("RESUMO");
    Serial.printf("  Acc pré-KD : %.3f\n", eval_pre.accuracy());
    Serial.printf("  Acc pós-FT : %.3f   (Δ = %+.3f)\n",
                  eval_post.accuracy(),
                  eval_post.accuracy() - eval_pre.accuracy());
    Serial.printf("  F1  pré-KD : %.3f\n", eval_pre.macro_f1());
    Serial.printf("  F1  pós-FT : %.3f   (Δ = %+.3f)\n",
                  eval_post.macro_f1(),
                  eval_post.macro_f1() - eval_pre.macro_f1());
    print_sep();
    // Assinatura do estado que devera reaparecer no PRE_KD da proxima rodada.
    pr.hash_ft_ok = log_weight_fingerprint(*model, "POST_FT", &pr.hash_ft);
    g_previous_ft_hash_valid = pr.hash_ft_ok;
    if (pr.hash_ft_ok) g_previous_ft_hash = pr.hash_ft;
    Serial.printf("[DIAG] acc_PRE_KD=%.6f acc_POST_KD=%.6f acc_POST_FT=%.6f delta_FT=%+.6f\n",
                  pr.acc_local, pr.acc_kd, pr.acc_ft, pr.acc_ft - pr.acc_kd);
    {
        if(!FedKDCheckpoint::save(*model, checkpoint_round)) {
            Serial.println("[FATAL] Falha ao salvar checkpoint; nao publicar done.");
            return pr;
        }
        pr.checkpoint_status=g_restored_this_run?2:1;
    }
    g_last_completed_round=checkpoint_round;
    pr.completed=true;
    Serial.println("Pipeline concluído. Modelo preservado em memória para a próxima rodada.");
    return pr;
}

// ── setup / loop ─────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(500);

    esp_task_wdt_init(WDT_TIMEOUT_SEC, true);
    esp_task_wdt_add(NULL);

    if (!LittleFS.begin(false)) {
        Serial.println("[FATAL] LittleFS não montou.");
        while (true) delay(1000);
    }
    Serial.printf("LittleFS: total=%u  usado=%u bytes\n",
                  (unsigned)LittleFS.totalBytes(),
                  (unsigned)LittleFS.usedBytes());

    const char* required[] = {
        XY_TRAIN_PATH, XY_TEST_PATH, METADATA_JSON_PATH,
        XY_PUB_PATH, TEACHER_PROBS_PATH, TEACHER_PROBS_META_PATH
    };
    for (const char* p : required) {
        Serial.printf("  %s: %s\n", p, LittleFS.exists(p) ? "OK" : "AUSENTE");
    }

#ifdef ENABLE_MQTT
    if (mqtt_connect()) {
        Serial.println("[SETUP] MQTT pronto. Aguardando 'start'...");
    } else {
        Serial.println("[SETUP] MQTT indisponivel — nao executar protocolo com checkpoint sem ID de experimento.");
    }
#else
    run_pipeline();
    pipeline_done = true;
#endif
}

void loop() {
#ifdef ENABLE_MQTT
    if (!g_mqtt_client.connected()) {
        Serial.println("[MQTT] Reconectando...");
        if (g_mqtt_client.connect(CLIENT_NAME)) {
            g_mqtt_client.subscribe(TOPIC_TEACHER_PULL);
            g_mqtt_client.subscribe(TOPIC_CMD_PULL);
            publish_identity();
            Serial.println("[MQTT] Reconectado.");
        } else {
            esp_task_wdt_reset(); delay(5000); return;
        }
    }
    g_mqtt_client.loop();

    if (g_start_cmd) {
        g_start_cmd        = false;
        g_teacher_received = false;
        g_teacher_waiting=false;
        g_pipeline_busy=true;
        int round = g_current_round;
        Serial.printf("\n[MQTT] === Round %d iniciado ===\n", round);
        PipelineResult pr = run_pipeline();
        g_pipeline_busy=false;
        g_teacher_waiting=false;
        if (!pr.completed) {
            Serial.println("[FATAL] Rodada incompleta; NAO publicar done.");
            return;
        }
        // Campos adicionais de diagnostico nao alteram o contrato basico 'done'.
        // MQTT: monitorar fedkd/cmd/push para obter os hashes sem abrir Serial.
        char done_msg[640];
        int done_len = snprintf(done_msg, sizeof(done_msg),
            "{\"status\":\"done\",\"round\":%d,"
            "\"acc_local\":%.4f,\"acc_kd\":%.4f,\"acc_ft\":%.4f,"
            "\"f1_local\":%.4f,\"f1_kd\":%.4f,\"f1_ft\":%.4f,"
            "\"loss_local\":%.5f,\"loss_kd\":%.5f,\"loss_ft\":%.5f,"
            "\"heap_free\":%u,"
            "\"t_local_ms\":%lu,\"t_kd_ms\":%lu,\"t_ft_ms\":%lu,"
            "\"fp_pre\":%lu,\"fp_kd\":%lu,\"fp_ft\":%lu,"
            "\"fp_pre_ok\":%d,\"fp_kd_ok\":%d,\"fp_ft_ok\":%d,"
            "\"continuity\":%d,\"checkpoint_status\":%d,\"experiment_id\":\"%s\"}",
            round,
            pr.acc_local, pr.acc_kd, pr.acc_ft,
            pr.f1_local,  pr.f1_kd,  pr.f1_ft,
            pr.loss_local, pr.loss_kd, pr.loss_ft,
            (unsigned)esp_get_free_heap_size(),
            pr.t_local_ms, pr.t_kd_ms, pr.t_ft_ms,
            (unsigned long)pr.hash_pre, (unsigned long)pr.hash_kd, (unsigned long)pr.hash_ft,
            (int)pr.hash_pre_ok, (int)pr.hash_kd_ok, (int)pr.hash_ft_ok,
            pr.continuity, pr.checkpoint_status, FedKDCheckpoint::experimentId());
        if (done_len > 0 && (size_t)done_len < sizeof(done_msg)) {
            bool sent = g_mqtt_client.publish(TOPIC_CMD_PUSH,
                           (uint8_t*)done_msg, (unsigned int)done_len, false);
            Serial.printf("[DIAG] Mensagem done publicada=%s bytes=%d\n", sent ? "sim" : "nao", done_len);
        } else {
            Serial.println("[DIAG] ERRO: mensagem done truncada, nao publicada");
        }
        Serial.printf("[MQTT] Round %d: 'done' publicado — acc_local=%.3f acc_ft=%.3f\n",
                      round, pr.acc_local, pr.acc_ft);
    }
    esp_task_wdt_reset();
    delay(100);
#else
    esp_task_wdt_reset();
    delay(5000);
#endif
}
