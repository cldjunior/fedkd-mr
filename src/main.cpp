/**
 * main.cpp — FedKD-MR ESP32 (Etapa 1: treino local autônomo)
 * ===========================================================
 * Fluxo autônomo sem menu:
 *   boot → treino local supervisionado
 *        → KD com teacher_probs pré-computado
 *        → fine-tuning supervisionado
 *        → impressão de métricas
 *        → idle (loop vazio)
 *
 * Dependências: NeuralNetwork.h (biblioteca do projeto Atlântico),
 *               LittleFS, ArduinoJson, Config.h (este projeto).
 *
 * Compilar com PlatformIO. Antes de gravar:
 *   1. Ajuste CLIENT_ID em Config.h
 *   2. Copie os arquivos do cliente para a pasta data/ do projeto
 *   3. Upload Filesystem Image (LittleFS)
 *   4. Upload firmware
 */

// ── Activações disponíveis ───────────────────────────────────────────────────
// TODAS as 7 devem ser definidas para manter os índices corretos:
// Sigmoid=0, Tanh=1, ReLU=2, LeakyReLU=3, ELU=4, SELU=5, Softmax=6
// Omitir qualquer uma desloca os índices → MODEL_ACTV {1,1,6} fica errado
#define ACTIVATION__PER_LAYER
#define Sigmoid
#define Tanh
#define ReLU
#define LeakyReLU
#define ELU
#define SELU
#define Softmax

#include "Config.h"
#include <NeuralNetwork.h>
#include <LittleFS.h>
#include <ArduinoJson.h>
#include <esp_task_wdt.h>

#ifdef ENABLE_MQTT
#include <WiFi.h>
#include <PubSubClient.h>
#endif

// ── Tipos ─────────────────────────────────────────────────────────────────────
#ifndef IDFLOAT
  #define IDFLOAT float
#endif

// Métricas por classe (precisão, recall, F1 — calculadas ao final)
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
    ClassMetrics cls[N_CLASSES];

    float accuracy() const { return n_samples > 0 ? (float)n_correct / n_samples : 0.f; }
    float macro_f1() const {
        float sum = 0.f;
        for (int i = 0; i < N_CLASSES; i++) sum += cls[i].f1();
        return sum / N_CLASSES;
    }
};

// ── Variáveis globais ─────────────────────────────────────────────────────────
NeuralNetwork* model = nullptr;
bool pipeline_done   = false;

// Buffers de arquitetura — permanecem vivos enquanto o NeuralNetwork existir
// (NeuralNetwork guarda ponteiros; não podem ser stack vars nem re-alocados por round)
static unsigned int g_model_layers[MODEL_N_LAYERS];
static byte         g_model_actv  [MODEL_N_LAYERS - 1];

#ifdef ENABLE_MQTT
static WiFiClient      g_wifi_client;
static PubSubClient    g_mqtt_client(g_wifi_client);
static volatile bool   g_teacher_received = false;  // sinaliza chegada de teacher_probs
static volatile bool   g_start_cmd        = false;  // sinaliza comando 'start' do servidor
static volatile int    g_current_round    = 0;      // número do round atual
#endif

// ── Resumo de métricas por round ──────────────────────────────────────────────
// Preenchido por run_pipeline(); lido por loop() para montar o payload "done".
struct RoundSummary {
    float        acc_local  = 0.f;   // acurácia após treino local (eval pré-KD)
    float        acc_kd     = 0.f;   // acurácia após KD           (eval pós-KD)
    float        acc_ft     = 0.f;   // acurácia após fine-tuning  (eval final)
    float        f1_local   = 0.f;
    float        f1_kd      = 0.f;
    float        f1_ft      = 0.f;
    float        loss_local = 0.f;   // MSE final da etapa 1
    float        loss_kd    = 0.f;   // MSE final da etapa 2 (soft targets)
    float        loss_ft    = 0.f;   // MSE final da etapa 3
    unsigned long t_local_ms = 0;
    unsigned long t_kd_ms    = 0;
    unsigned long t_ft_ms    = 0;
    uint32_t     heap_free   = 0;    // heap após pipeline completo
};
static RoundSummary g_last_round;

// ── Utilidades ────────────────────────────────────────────────────────────────
static void print_sep() { Serial.println("----------------------------------------------------"); }

static void print_mem() {
    Serial.printf("  Heap livre: %u bytes\n", (unsigned)esp_get_free_heap_size());
}

// ── Proteção anti-NaN: clipa pesos e biases após BackProp ─────────────────────
// Com datasets pequenos (n < 50 por classe) gradientes explodem ao longo das
// epochs e corrompem todos os parâmetros. clamp_params() limita cada peso e
// bias ao intervalo [-MAX_PARAM, +MAX_PARAM] e zera qualquer NaN/Inf residual.
// Custo: ~3 648 comparações por chamada (96×32 + 32×16 + 16×4) — desprezível.
static const float MAX_PARAM = 5.0f;

static void clamp_params(NeuralNetwork& nn) {
    for (unsigned int l = 0; l < nn.numberOflayers; l++) {
        unsigned int nout = nn.layers[l]._numberOfOutputs;
        unsigned int nin  = nn.layers[l]._numberOfInputs;
        for (unsigned int i = 0; i < nout; i++) {
            // Bias
            float b = (float)nn.layers[l].bias[i];
            if (!isfinite(b))        nn.layers[l].bias[i] = (IDFLOAT)0.0f;
            else if (b >  MAX_PARAM) nn.layers[l].bias[i] = (IDFLOAT) MAX_PARAM;
            else if (b < -MAX_PARAM) nn.layers[l].bias[i] = (IDFLOAT)(-MAX_PARAM);
            // Pesos
            for (unsigned int j = 0; j < nin; j++) {
                float w = (float)nn.layers[l].weights[i][j];
                if (!isfinite(w))        nn.layers[l].weights[i][j] = (IDFLOAT)0.0f;
                else if (w >  MAX_PARAM) nn.layers[l].weights[i][j] = (IDFLOAT) MAX_PARAM;
                else if (w < -MAX_PARAM) nn.layers[l].weights[i][j] = (IDFLOAT)(-MAX_PARAM);
            }
        }
    }
}

// ── Leitura do metadata.json ──────────────────────────────────────────────────
struct DatasetSchema {
    // Apenas o necessário para parsing: índices e offsets das features + label
    int     n_features   = 0;
    int     n_classes    = N_CLASSES;
    int     bytes_per_row = 0;
    int     label_offset = 0;
    bool    encoded_labels = false;
    // offsets das features de entrada (até N_INPUT)
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

    s.bytes_per_row   = doc["bytes_per_row"] | 0;
    s.encoded_labels  = doc.containsKey("label_map");
    const char* label_col = doc["label_column"] | "label";

    JsonArray schema = doc["schema"];
    int feat_idx = 0;
    for (JsonObject col : schema) {
        const char* name = col["name"] | "";
        const char* type = col["type"] | "float32";
        int offset       = col["offset"] | 0;
        int bytes        = col["bytes"]  | 4;

        if (strcmp(name, label_col) == 0) {
            s.label_offset = offset;
            if      (strcmp(type, "uint8") == 0) s.label_type = 0;
            else if (strcmp(type, "int8")  == 0) s.label_type = 1;
            else                                  s.label_type = 2;  // int32
        } else if (strcmp(name, "timestamp") != 0 && feat_idx < N_INPUT) {
            s.feat_offsets[feat_idx++] = offset;
        }
    }
    s.n_features = feat_idx;

    if (s.n_features != N_INPUT) {
        Serial.printf("[ERR] Schema tem %d features; esperado %d\n", s.n_features, N_INPUT);
        return false;
    }
    return true;
}

static long read_label(uint8_t* row, const DatasetSchema& s) {
    if      (s.label_type == 0) return (long)(*(uint8_t*) (row + s.label_offset));
    else if (s.label_type == 1) return (long)(*(int8_t*)  (row + s.label_offset));
    else { int32_t v; memcpy(&v, row + s.label_offset, 4); return (long)v; }
}

// Converte label encoded (1-based) para índice 0-based
static int label_to_idx(long label_enc) {
    return (int)(label_enc - 1);  // EMPTY=1→0, STATIONARY=2→1, ...
}

// ── Treino supervisionado local ───────────────────────────────────────────────
static RunMetrics train_supervised(NeuralNetwork& nn,
                                   const char* bin_path,
                                   const char* meta_path,
                                   int epochs,
                                   float lr_w, float lr_b,
                                   const char* label) {
    RunMetrics result;
    DatasetSchema s;
    if (!load_schema(meta_path, s)) return result;

    nn.LearningRateOfWeights = lr_w;
    nn.LearningRateOfBiases  = lr_b;

    uint8_t* rowbuf = (uint8_t*)malloc(s.bytes_per_row);
    if (!rowbuf) { Serial.println("[ERR] malloc rowbuf falhou"); return result; }

    IDFLOAT* x = new IDFLOAT[N_INPUT];
    IDFLOAT  y[N_CLASSES];

    unsigned long t0 = millis();
    Serial.printf("\n[%s] epochs=%d  lr=%.4f\n", label, epochs, lr_w);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        File f = LittleFS.open(bin_path, "r");
        if (!f) { Serial.printf("[ERR] Não abriu %s\n", bin_path); break; }

        int n_ok = 0, n_total = 0;
        while (f.available() >= s.bytes_per_row) {
            if (f.read(rowbuf, s.bytes_per_row) != (size_t)s.bytes_per_row) break;

            // Parse features
            for (int i = 0; i < N_INPUT; i++) {
                float v; memcpy(&v, rowbuf + s.feat_offsets[i], 4);
                x[i] = (IDFLOAT)v;
            }

            // Parse label → one-hot
            long lv  = read_label(rowbuf, s);
            int  idx = label_to_idx(lv);
            if (idx < 0 || idx >= N_CLASSES) { n_total++; continue; }
            for (int k = 0; k < N_CLASSES; k++) y[k] = (k == idx) ? 1.f : 0.f;

            // Forward + backprop
            IDFLOAT* pred = nn.FeedForward(x);

            // ── TRAIN-DEBUG cirúrgico (amostra 0, época 1) ───────────────────
            if (epoch == 1 && n_total == 0) {
                // 1) Verifica se x[] tem NaN ou Inf
                int n_bad = 0;
                float x_min = 1e30f, x_max = -1e30f;
                for (int i = 0; i < N_INPUT; i++) {
                    float v = (float)x[i];
                    if (isnan(v) || isinf(v)) n_bad++;
                    if (v < x_min) x_min = v;
                    if (v > x_max) x_max = v;
                }
                Serial.printf("\n[BP-DEBUG] Entrada x: %d/%d NaN/Inf  min=%.3f  max=%.3f\n",
                              n_bad, N_INPUT, x_min, x_max);

                // 2) Saída do modelo
                Serial.print("[BP-DEBUG] Saida:  ");
                for (int k = 0; k < N_CLASSES; k++) Serial.printf("%.4f ", (float)pred[k]);
                Serial.println();

                // 3) Pesos nas camadas 0 e 1 ANTES do BackProp
                float w0_pre = (float)nn.layers[0].weights[0][0];
                float w1_pre = (float)nn.layers[1].weights[0][0];
                Serial.printf("[BP-DEBUG] W L0[0][0]=%.6f  L1[0][0]=%.6f  [PRE-BackProp]\n",
                              w0_pre, w1_pre);

                nn.BackProp(y);  // ← BackProp aqui
                clamp_params(nn);

                // 4) Mesmos pesos DEPOIS do BackProp
                float w0_post = (float)nn.layers[0].weights[0][0];
                float w1_post = (float)nn.layers[1].weights[0][0];
                Serial.printf("[BP-DEBUG] W L0[0][0]=%.6f  L1[0][0]=%.6f  [POS-BackProp]\n",
                              w0_post, w1_post);

                bool corrupted = isnan(w0_post) || isnan(w1_post)
                              || isinf(w0_post) || isinf(w1_post);
                Serial.printf("[BP-DEBUG] -> %s\n", corrupted
                    ? "FALHA: BackProp corrompeu pesos!"
                    : "OK: pesos validos apos BackProp.");
            } else {
                nn.BackProp(y);
                clamp_params(nn);
            }
            // ── fim TRAIN-DEBUG ───────────────────────────────────────────────

            // Acurácia
            int pred_cls = 0;
            for (int k = 1; k < N_CLASSES; k++)
                if (pred[k] > pred[pred_cls]) pred_cls = k;
            if (pred_cls == idx) n_ok++;
            n_total++;

            esp_task_wdt_reset();
        }
        f.close();

        float acc = n_total > 0 ? (float)n_ok / n_total : 0.f;
        // getMeanSqrdError(n) divide sumSquaredError / (N_CLASSES * n) e zera o acumulador
        result.mse_last = (float)nn.getMeanSqrdError(n_total > 0 ? n_total : 1);
        Serial.printf("  Época %2d/%d — acc=%.3f  mse=%.5f  n=%d\n",
                      epoch, epochs, acc, result.mse_last, n_total);
    }
    result.ms = millis() - t0;
    delete[] x;
    free(rowbuf);
    return result;
}

// ── Treino KD — soft targets do teacher ──────────────────────────────────────
static RunMetrics train_kd(NeuralNetwork& nn,
                            const char* pub_bin_path,
                            const char* pub_meta_path,
                            const char* tp_path,
                            int n_pub,
                            int epochs,
                            float lr_w, float lr_b) {
    RunMetrics result;
    DatasetSchema s;
    if (!load_schema(pub_meta_path, s)) return result;

    // Carrega teacher_probs inteiro na RAM
    // n_pub × N_CLASSES × 4 bytes — ex: 50 × 4 × 4 = 800 bytes
    size_t tp_size = (size_t)n_pub * N_CLASSES * sizeof(float);
    float* tp = (float*)malloc(tp_size);
    if (!tp) { Serial.println("[ERR] malloc teacher_probs falhou"); return result; }

    File ftf = LittleFS.open(tp_path, "r");
    if (!ftf) { Serial.println("[ERR] Não abriu teacher_probs.bin"); free(tp); return result; }
    size_t read_bytes = ftf.read((uint8_t*)tp, tp_size);
    ftf.close();
    if (read_bytes != tp_size) {
        Serial.printf("[ERR] teacher_probs: leu %u bytes, esperado %u\n",
                      (unsigned)read_bytes, (unsigned)tp_size);
        free(tp); return result;
    }
    Serial.printf("  teacher_probs carregado: %d amostras × %d classes = %u bytes\n",
                  n_pub, N_CLASSES, (unsigned)tp_size);

    nn.LearningRateOfWeights = lr_w;
    nn.LearningRateOfBiases  = lr_b;

    uint8_t* rowbuf = (uint8_t*)malloc(s.bytes_per_row);
    if (!rowbuf) { free(tp); Serial.println("[ERR] malloc rowbuf falhou"); return result; }

    IDFLOAT* x      = new IDFLOAT[N_INPUT];
    IDFLOAT  y_soft[N_CLASSES];

    unsigned long t0 = millis();
    Serial.printf("\n[KD] epochs=%d  lr=%.4f\n", epochs, lr_w);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        File f = LittleFS.open(pub_bin_path, "r");
        if (!f) { Serial.printf("[ERR] Não abriu %s\n", pub_bin_path); break; }

        int sample_idx = 0;
        while (f.available() >= s.bytes_per_row && sample_idx < n_pub) {
            if (f.read(rowbuf, s.bytes_per_row) != (size_t)s.bytes_per_row) break;

            // Parse features
            for (int i = 0; i < N_INPUT; i++) {
                float v; memcpy(&v, rowbuf + s.feat_offsets[i], 4);
                x[i] = (IDFLOAT)v;
            }

            // Soft targets do teacher (linha sample_idx do buffer)
            float* row_tp = tp + sample_idx * N_CLASSES;
            for (int k = 0; k < N_CLASSES; k++)
                y_soft[k] = (IDFLOAT)row_tp[k];

            // ── KD-DEBUG: imprime y_soft da 1ª amostra da 1ª época ──────────
            if (epoch == 1 && sample_idx == 0) {
                Serial.println("\n[KD-DEBUG] y_soft passado ao BackProp (amostra 0, epoca 1):");
                for (int k = 0; k < N_CLASSES; k++)
                    Serial.printf("  classe %d: %.4f\n", k, (float)y_soft[k]);
                float sum_soft = 0;
                for (int k = 0; k < N_CLASSES; k++) sum_soft += (float)y_soft[k];
                Serial.printf("  soma total: %.4f  (deve ser ~1.0)\n", sum_soft);
                bool is_onehot = true;
                for (int k = 0; k < N_CLASSES; k++)
                    if (y_soft[k] > 0.01f && y_soft[k] < 0.99f) { is_onehot = false; break; }
                Serial.printf("  -> %s\n", is_onehot
                    ? "ALERTA: parece one-hot — verificar teacher_probs.bin!"
                    : "OK: distribuicao suave — KD com soft targets confirmado.");
            }
            // ── fim KD-DEBUG ──────────────────────────────────────────────────

            // Forward + backprop com soft targets
            nn.FeedForward(x);
            nn.BackProp(y_soft);
            clamp_params(nn);

            sample_idx++;
            esp_task_wdt_reset();
        }
        f.close();

        result.mse_last = (float)nn.getMeanSqrdError(sample_idx > 0 ? sample_idx : 1);
        Serial.printf("  Época KD %2d/%d — mse=%.5f  n=%d\n",
                      epoch, epochs, result.mse_last, sample_idx);
    }
    result.ms = millis() - t0;

    delete[] x;
    free(rowbuf);
    free(tp);
    return result;
}

// ── Avaliação (sem backprop) ─────────────────────────────────────────────────
static RunMetrics evaluate(NeuralNetwork& nn,
                            const char* bin_path,
                            const char* meta_path,
                            const char* label) {
    RunMetrics result;
    DatasetSchema s;
    if (!load_schema(meta_path, s)) return result;

    uint8_t* rowbuf = (uint8_t*)malloc(s.bytes_per_row);
    if (!rowbuf) return result;
    IDFLOAT* x = new IDFLOAT[N_INPUT];

    File f = LittleFS.open(bin_path, "r");
    if (!f) { free(rowbuf); delete[] x; return result; }

    while (f.available() >= s.bytes_per_row) {
        if (f.read(rowbuf, s.bytes_per_row) != (size_t)s.bytes_per_row) break;

        for (int i = 0; i < N_INPUT; i++) {
            float v; memcpy(&v, rowbuf + s.feat_offsets[i], 4);
            x[i] = (IDFLOAT)v;
        }

        long lv  = read_label(rowbuf, s);
        int  true_cls = label_to_idx(lv);
        if (true_cls < 0 || true_cls >= N_CLASSES) continue;

        IDFLOAT* pred = nn.FeedForward(x);
        int pred_cls  = 0;
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
    f.close();
    delete[] x;
    free(rowbuf);

    const char* cls_names[] = {"EMPTY", "STATIONARY", "APPROACHING", "LEAVING"};
    Serial.printf("\n[%s] Acc=%.3f  MacroF1=%.3f  n=%d\n",
                  label, result.accuracy(), result.macro_f1(), result.n_samples);
    for (int k = 0; k < N_CLASSES; k++) {
        Serial.printf("  %12s: P=%.3f R=%.3f F1=%.3f (tp=%d fp=%d fn=%d)\n",
                      cls_names[k],
                      result.cls[k].precision(),
                      result.cls[k].recall(),
                      result.cls[k].f1(),
                      result.cls[k].tp,
                      result.cls[k].fp,
                      result.cls[k].fn);
    }
    return result;
}

// ── Leitura do teacher_probs_meta ────────────────────────────────────────────
static int read_n_pub() {
    File f = LittleFS.open(TEACHER_PROBS_META_PATH, "r");
    if (!f) {
        Serial.println("[AVISO] teacher_probs_meta.json não encontrado; usando n_pub=0");
        return 0;
    }
    JsonDocument doc;
    deserializeJson(doc, f);
    f.close();
    return doc["n_pub"] | 0;
}

// ── Funções MQTT ─────────────────────────────────────────────────────────────
#ifdef ENABLE_MQTT

static void mqtt_on_message(char* topic, byte* payload, unsigned int length) {
    if (strcmp(topic, TOPIC_TEACHER_PULL) == 0) {
        File f = LittleFS.open(TEACHER_PROBS_PATH, "w");
        if (f) { f.write(payload, length); f.close(); }
        int n_pub_recv = (int)(length / (N_CLASSES * sizeof(float)));
        File fm = LittleFS.open(TEACHER_PROBS_META_PATH, "w");
        if (fm) {
            char buf[32];
            snprintf(buf, sizeof(buf), "{\"n_pub\":%d}", n_pub_recv);
            fm.print(buf); fm.close();
        }
        g_teacher_received = true;
        Serial.printf("\n[MQTT] teacher_probs recebido: %d amostras × %d classes (%u bytes)\n",
                      n_pub_recv, N_CLASSES, length);
    }
    else if (strcmp(topic, TOPIC_CMD_PULL) == 0) {
        JsonDocument doc;
        if (deserializeJson(doc, (char*)payload, length) == DeserializationError::Ok) {
            const char* cmd = doc["cmd"] | "";
            if (strcmp(cmd, "start") == 0) {
                g_current_round = doc["round"] | 0;
                g_start_cmd     = true;
                Serial.printf("[MQTT] Comando 'start' recebido (round %d)\n", g_current_round);
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
        Serial.println("\n[MQTT] ERRO: Wi-Fi não conectou. Verifique WIFI_SSID/PASSWORD.");
        return false;
    }
    Serial.printf("\n[MQTT] Wi-Fi OK — IP: %s\n", WiFi.localIP().toString().c_str());
    g_mqtt_client.setServer(MQTT_BROKER, MQTT_PORT);
    g_mqtt_client.setCallback(mqtt_on_message);
    if (!g_mqtt_client.setBufferSize(2048))
        Serial.println("[MQTT] AVISO: falha ao aumentar buffer — usando 256 bytes padrão");
    if (!g_mqtt_client.connect(CLIENT_NAME)) {
        Serial.printf("[MQTT] ERRO: broker %s:%d não respondeu (state=%d)\n",
                      MQTT_BROKER, MQTT_PORT, g_mqtt_client.state());
        return false;
    }
    g_mqtt_client.subscribe(TOPIC_TEACHER_PULL);
    g_mqtt_client.subscribe(TOPIC_CMD_PULL);
    Serial.printf("[MQTT] Conectado ao broker %s:%d\n", MQTT_BROKER, MQTT_PORT);
    Serial.printf("[MQTT] Subscrito: %s | %s\n", TOPIC_TEACHER_PULL, TOPIC_CMD_PULL);
    return true;
}

static int publish_logits_mqtt(NeuralNetwork& nn) {
    // O treino pode ter durado mais que o keepalive (15 s padrão) sem loop().
    // Reconecta se necessário antes de publicar.
    if (!g_mqtt_client.connected()) {
        Serial.println("[MQTT] Reconectando antes de publicar logits...");
        if (!g_mqtt_client.connect(CLIENT_NAME)) {
            Serial.printf("[MQTT] Falha ao reconectar (state=%d)\n", g_mqtt_client.state());
            return -1;
        }
        g_mqtt_client.subscribe(TOPIC_TEACHER_PULL);
        g_mqtt_client.subscribe(TOPIC_CMD_PULL);
        Serial.println("[MQTT] Reconectado.");
    }
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
    IDFLOAT* x = new IDFLOAT[N_INPUT];
    File f = LittleFS.open(XY_PUB_PATH, "r");
    int idx = 0;
    while (f.available() >= s.bytes_per_row && idx < n_pub) {
        if (f.read(rowbuf, s.bytes_per_row) != (size_t)s.bytes_per_row) break;
        for (int i = 0; i < N_INPUT; i++) {
            float v; memcpy(&v, rowbuf + s.feat_offsets[i], 4); x[i] = (IDFLOAT)v;
        }
        IDFLOAT* pred = nn.FeedForward(x);
        for (int k = 0; k < N_CLASSES; k++)
            lb[idx * N_CLASSES + k] = (float)pred[k];
        idx++; esp_task_wdt_reset();
    }
    f.close();
    bool ok = g_mqtt_client.publish(TOPIC_LOGITS_PUSH,
                                    (uint8_t*)lb, (unsigned int)buf_size, false);
    Serial.printf("[MQTT] Logits publicados: %d amostras → %s  (%u bytes) [%s]\n",
                  idx, TOPIC_LOGITS_PUSH, (unsigned)buf_size, ok ? "OK" : "FALHA");
    delete[] x; free(rowbuf); free(lb);
    return ok ? idx : -1;
}

static bool wait_for_teacher_probs(uint32_t timeout_ms) {
    uint32_t deadline = millis() + timeout_ms;
    Serial.printf("[MQTT] Aguardando teacher_probs (timeout=%lus)...\n",
                  (unsigned long)(timeout_ms / 1000));
    static uint32_t last_print = 0;
    while (millis() < deadline) {
        g_mqtt_client.loop();
        if (g_teacher_received) {
            Serial.println("[MQTT] teacher_probs recebido — prosseguindo com KD.");
            return true;
        }
        uint32_t rem = deadline - millis();
        if (millis() - last_print >= 10000) {
            last_print = millis();
            Serial.printf("[MQTT] ... %lus restantes\n", (unsigned long)(rem / 1000));
        }
        esp_task_wdt_reset(); delay(200);
    }
    Serial.println("[MQTT] TIMEOUT aguardando teacher_probs — KD será pulado.");
    return false;
}

#endif  // ENABLE_MQTT

// ── Pipeline principal ───────────────────────────────────────────────────────
static void run_pipeline() {
    print_sep();
    Serial.println("FedKD-MR — Pipeline de treinamento local");
    Serial.printf("Cliente: %s  |  Modelo: [%d → %d → ... → %d]\n",
                  CLIENT_NAME, N_INPUT, N_CLASSES, N_CLASSES);
    print_sep();
    print_mem();

    // ── Criar modelo ─────────────────────────────────────────────
    // Libera modelo do round anterior antes de alocar novo.
    // g_model_layers / g_model_actv são globais estáticos: NeuralNetwork
    // guarda ponteiros para eles; não podem ser stack vars nem re-alocados.
    if (model != nullptr) {
        delete model;
        model = nullptr;
        Serial.println("[MEM] Modelo anterior liberado.");
    }

    unsigned int layers_init[] = MODEL_LAYERS;
    byte         actv_init[]   = MODEL_ACTV;
    memcpy(g_model_layers, layers_init, MODEL_N_LAYERS     * sizeof(unsigned int));
    memcpy(g_model_actv,   actv_init,   (MODEL_N_LAYERS-1) * sizeof(byte));

    model = new NeuralNetwork(g_model_layers, MODEL_N_LAYERS, g_model_actv);
    model->LearningRateOfWeights = LOCAL_LR_WEIGHTS;
    model->LearningRateOfBiases  = LOCAL_LR_BIASES;

    Serial.printf("\nModelo criado: %u camadas\n", (unsigned)MODEL_N_LAYERS);
    print_mem();

    // ── Etapa 1: Treino local supervisionado ─────────────────────
    print_sep();
    Serial.println("ETAPA 1 — Treino local supervisionado (D_priv)");
    RunMetrics m_local = train_supervised(
        *model,
        XY_TRAIN_PATH, METADATA_JSON_PATH,
        LOCAL_EPOCHS, LOCAL_LR_WEIGHTS, LOCAL_LR_BIASES,
        "Local"
    );
    Serial.printf("  Concluído em %lums\n", m_local.ms);
    print_mem();

    // ── Avaliação pós-treino local ────────────────────────────────
    RunMetrics eval_pre = evaluate(*model, XY_TRAIN_PATH, METADATA_JSON_PATH,
                                   "Avaliação pré-KD (D_priv)");

    // Registra métricas da etapa 1
    g_last_round = RoundSummary{};          // zera tudo antes de preencher
    g_last_round.acc_local  = eval_pre.accuracy();
    g_last_round.f1_local   = eval_pre.macro_f1();
    g_last_round.loss_local = m_local.mse_last;
    g_last_round.t_local_ms = m_local.ms;
    // acc_kd/acc_ft default = acc_local (caso KD não execute por n_pub=0)
    g_last_round.acc_kd = g_last_round.acc_local;
    g_last_round.acc_ft = g_last_round.acc_local;
    g_last_round.f1_kd  = g_last_round.f1_local;
    g_last_round.f1_ft  = g_last_round.f1_local;

    // ── Etapa 2: Knowledge Distillation (KD) ─────────────────────
    print_sep();
    Serial.println("ETAPA 2 — Knowledge Distillation (D_pub + teacher_probs)");

#ifdef ENABLE_MQTT
    // Federação real: publica logits no servidor e aguarda teacher_probs agregado.
    // O callback mqtt_on_message() salva teacher_probs em LittleFS (TEACHER_PROBS_PATH)
    // e atualiza TEACHER_PROBS_META_PATH → read_n_pub() e train_kd() funcionam sem mudanças.
    publish_logits_mqtt(*model);
    wait_for_teacher_probs(120000UL);   // 2 min de timeout
    int n_pub = read_n_pub();           // lê n_pub salvo pelo callback
#else
    // Modo offline: usa teacher_probs pré-computado gravado em LittleFS
    int n_pub = read_n_pub();
#endif
    if (n_pub <= 0) {
        Serial.println("[AVISO] n_pub=0 — pulando KD (teacher_probs não disponível)");
    } else {
        model->LearningRateOfWeights = KD_LR_WEIGHTS;
        model->LearningRateOfBiases  = KD_LR_BIASES;

        // ── KD-DEBUG: peso[camada1][0][0] antes do KD ────────────────────────
        // layers[0] = camada de input (sem pesos); layers[1] = 1ª camada oculta
        float w_antes = model->layers[1].weights[0][0];
        Serial.printf("\n[KD-DEBUG] Peso[1][0][0] ANTES  do KD: %.6f\n", w_antes);
        // ── fim KD-DEBUG ─────────────────────────────────────────────────────

        RunMetrics m_kd = train_kd(
            *model,
            XY_PUB_PATH, METADATA_JSON_PATH,
            TEACHER_PROBS_PATH, n_pub,
            KD_EPOCHS, KD_LR_WEIGHTS, KD_LR_BIASES
        );

        // ── KD-DEBUG: peso[camada0][0][0] depois do KD ───────────────────────
        float w_depois = model->layers[1].weights[0][0];
        Serial.printf("[KD-DEBUG] Peso[1][0][0] DEPOIS do KD: %.6f\n", w_depois);
        Serial.printf("[KD-DEBUG] Variacao: %+.6f  -> %s\n",
                      w_depois - w_antes,
                      (w_depois != w_antes) ? "pesos atualizados — KD executou backprop."
                                            : "ALERTA: peso nao mudou!");
        // ── fim KD-DEBUG ─────────────────────────────────────────────────────

        Serial.printf("  KD concluído em %lums\n", m_kd.ms);
        print_mem();

        // Registra métricas da etapa 2
        g_last_round.loss_kd  = m_kd.mse_last;
        g_last_round.t_kd_ms  = m_kd.ms;

        // Avaliação intermédia pós-KD (antes do fine-tuning)
        RunMetrics eval_kd = evaluate(*model, XY_TRAIN_PATH, METADATA_JSON_PATH,
                                      "Avaliação pós-KD / pré-FT (D_priv)");
        g_last_round.acc_kd = eval_kd.accuracy();
        g_last_round.f1_kd  = eval_kd.macro_f1();

        // ── Etapa 3: Fine-tuning supervisionado pós-KD ───────────
        print_sep();
        Serial.println("ETAPA 3 — Fine-tuning supervisionado (D_priv)");
        RunMetrics m_ft = train_supervised(
            *model,
            XY_TRAIN_PATH, METADATA_JSON_PATH,
            FT_EPOCHS, FT_LR_WEIGHTS, FT_LR_BIASES,
            "Fine-tuning"
        );
        Serial.printf("  Fine-tuning concluído em %lums\n", m_ft.ms);
        // Registra métricas da etapa 3
        g_last_round.loss_ft = m_ft.mse_last;
        g_last_round.t_ft_ms = m_ft.ms;
        print_mem();
    }

    // ── Avaliação final ───────────────────────────────────────────
    print_sep();
    Serial.println("AVALIAÇÃO FINAL — D_priv (comparar com pré-KD)");
    RunMetrics eval_post = evaluate(*model, XY_TRAIN_PATH, METADATA_JSON_PATH,
                                    "Avaliação pós-KD (D_priv)");

    // Registra métricas finais
    g_last_round.acc_ft    = eval_post.accuracy();
    g_last_round.f1_ft     = eval_post.macro_f1();
    g_last_round.heap_free = (uint32_t)ESP.getFreeHeap();

    // ── Resumo comparativo ────────────────────────────────────────
    print_sep();
    Serial.println("RESUMO");
    Serial.printf("  Acc local  : %.3f   F1=%.3f\n",
                  g_last_round.acc_local, g_last_round.f1_local);
    Serial.printf("  Acc pós-KD : %.3f   F1=%.3f   (Δacc=%+.3f)\n",
                  g_last_round.acc_kd, g_last_round.f1_kd,
                  g_last_round.acc_kd - g_last_round.acc_local);
    Serial.printf("  Acc pós-FT : %.3f   F1=%.3f   (Δacc=%+.3f)\n",
                  g_last_round.acc_ft, g_last_round.f1_ft,
                  g_last_round.acc_ft - g_last_round.acc_local);
    Serial.printf("  Loss local/KD/FT : %.5f / %.5f / %.5f\n",
                  g_last_round.loss_local, g_last_round.loss_kd, g_last_round.loss_ft);
    Serial.printf("  Tempos  local=%lums  KD=%lums  FT=%lums\n",
                  g_last_round.t_local_ms, g_last_round.t_kd_ms, g_last_round.t_ft_ms);
    Serial.printf("  Heap livre: %u bytes\n", (unsigned)g_last_round.heap_free);
    print_sep();
    Serial.println("Pipeline concluído. Entrando em modo idle.");

#ifdef SAVE_MODEL
    // Opcional: persistir modelo em flash para próximo boot
    // (requer saveModelToFlash de ModelUtil.cpp)
    // saveModelToFlash(*model, MODEL_PATH);
    Serial.println("  (persistência de modelo desabilitada — defina SAVE_MODEL)");
#endif
}

// ── setup / loop ─────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(500);

    // Watchdog: aumenta timeout para comportar treino
    esp_task_wdt_init(WDT_TIMEOUT_SEC, true);
    esp_task_wdt_add(NULL);

    // Monta LittleFS
    if (!LittleFS.begin(false)) {
        Serial.println("[FATAL] LittleFS não montou. Verifique o filesystem.");
        while (true) delay(1000);
    }
    Serial.printf("LittleFS: total=%u  usado=%u bytes\n",
                  (unsigned)LittleFS.totalBytes(),
                  (unsigned)LittleFS.usedBytes());

    // Verifica arquivos obrigatórios
    const char* required[] = {
        XY_TRAIN_PATH, METADATA_JSON_PATH,
        XY_PUB_PATH, TEACHER_PROBS_PATH, TEACHER_PROBS_META_PATH
    };
    for (const char* p : required) {
        if (!LittleFS.exists(p))
            Serial.printf("[AVISO] Arquivo não encontrado: %s\n", p);
        else
            Serial.printf("  OK: %s\n", p);
    }

    // ── Inicializa MQTT ou executa pipeline offline ──────────────
#ifdef ENABLE_MQTT
    bool mqtt_ok = mqtt_connect();
    if (mqtt_ok) {
        // Modo federado: pipeline será disparado por loop() ao receber 'start'
        Serial.println("[SETUP] MQTT pronto. Aguardando comando 'start' do servidor...");
        Serial.println("[SETUP] (Inicie o servidor: python fedkd_server.py --host <IP>)");
        // pipeline_done permanece false → loop() gerencia os rounds
    } else {
        // Fallback offline: executa pipeline local com teacher_probs do LittleFS
        Serial.println("[SETUP] MQTT indisponível — executando pipeline local (modo offline).");
        run_pipeline();
        pipeline_done = true;
    }
#else
    // Modo compilado sem MQTT: executa pipeline uma única vez
    run_pipeline();
    pipeline_done = true;
#endif
}

void loop() {
#ifdef ENABLE_MQTT
    // ── Mantém conexão MQTT ativa ──────────────────────────────
    if (!g_mqtt_client.connected()) {
        Serial.println("[MQTT] Reconectando ao broker...");
        if (g_mqtt_client.connect(CLIENT_NAME)) {
            g_mqtt_client.subscribe(TOPIC_TEACHER_PULL);
            g_mqtt_client.subscribe(TOPIC_CMD_PULL);
            Serial.println("[MQTT] Reconectado.");
        } else {
            // Tenta novamente após 5 s
            esp_task_wdt_reset();
            delay(5000);
            return;
        }
    }
    g_mqtt_client.loop();

    // ── Processa comando 'start' recebido via MQTT ─────────────
    if (g_start_cmd) {
        g_start_cmd        = false;
        g_teacher_received = false;

        int round = g_current_round;
        Serial.printf("\n[MQTT] === Round %d iniciado ===\n", round);

        // Re-aloca modelo se necessário (rounds subsequentes reutilizam o existente)
        run_pipeline();

        // Publica confirmação de conclusão do round com métricas completas
        char done_msg[384];
        snprintf(done_msg, sizeof(done_msg),
            "{"
            "\"status\":\"done\","
            "\"round\":%d,"
            "\"client\":\"%s\","
            "\"acc_local\":%.4f,\"acc_kd\":%.4f,\"acc_ft\":%.4f,"
            "\"f1_local\":%.4f,\"f1_kd\":%.4f,\"f1_ft\":%.4f,"
            "\"loss_local\":%.5f,\"loss_kd\":%.5f,\"loss_ft\":%.5f,"
            "\"t_local_ms\":%lu,\"t_kd_ms\":%lu,\"t_ft_ms\":%lu,"
            "\"heap_free\":%u"
            "}",
            round, CLIENT_NAME,
            g_last_round.acc_local, g_last_round.acc_kd, g_last_round.acc_ft,
            g_last_round.f1_local,  g_last_round.f1_kd,  g_last_round.f1_ft,
            g_last_round.loss_local, g_last_round.loss_kd, g_last_round.loss_ft,
            g_last_round.t_local_ms, g_last_round.t_kd_ms, g_last_round.t_ft_ms,
            (unsigned)g_last_round.heap_free
        );
        g_mqtt_client.publish(TOPIC_CMD_PUSH,
                              (uint8_t*)done_msg, (unsigned int)strlen(done_msg),
                              /*retain=*/false);
        Serial.printf("[MQTT] Round %d métricas publicadas em %s\n",
                      round, TOPIC_CMD_PUSH);
    }

    esp_task_wdt_reset();
    delay(100);

#else
    // Sem MQTT: pipeline já executou em setup(), loop vazio
    esp_task_wdt_reset();
    delay(5000);
#endif
}
