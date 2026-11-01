/**
 * main_collector_auto.cpp — FedKD-MR: Coleta Automática com Inferência On-Device
 * ================================================================================
 * Requer que o modelo federado já tenha sido treinado e os pesos salvos em
 * MODEL_PATH ("/model_fedkd.nn") pelo main.cpp com a flag SAVE_MODEL ativa.
 *
 * Fluxo:
 *   boot → carrega pesos do modelo do LittleFS
 *        → coleta janelas do sensor LD2410C em Engineering Mode
 *        → feedForward() → argmax → rótulo inferido
 *        → descarta se confiança < MIN_AUTO_CONFIDENCE
 *        → grava amostra rotulada em dataset_priv.bin (append)
 *
 * Comandos Serial:
 *   c / C  — toggle coleta (pausar / retomar)
 *   r / R  — apagar dataset_priv.bin e reiniciar contadores
 *   x / X  — exportar dataset_priv.bin como hex dump (protocolo DUMP)
 *   d / D  — finalizar coleta (imprime resumo)
 *   ?      — status imediato
 *   v / V  — diagnóstico do sensor
 *   b / B  — raw bytes do sensor (80 bytes, 3 s)
 *   UPLOAD /path size\n  — receber arquivo do PC (protocolo UPLOAD)
 *
 * Para compilar: adicione um environment "collector_auto" em platformio.ini:
 *   [env:collector_auto]
 *   build_flags =
 *       ${env.build_flags}       ; herda flags comuns (CLIENT_ID, Wi-Fi, etc.)
 *       -D SAVE_MODEL            ; não usado aqui, mas não prejudica
 *   build_src_filter = +<main_collector_auto.cpp> +<LD2410C.cpp>
 *
 * ATENÇÃO: Grave com  pio run -e collector_auto -t upload  (SEM -t uploadfs)
 *          para preservar o LittleFS e os arquivos já coletados.
 */

// ── Activações para NeuralNetwork.h (MLP, cliente 1) ─────────────────────────
#define ACTIVATION__PER_LAYER
#define Sigmoid
#define Tanh
#define ReLU
#define LeakyReLU
#define ELU
#define SELU
#define Softmax

#include <Arduino.h>
#include <LittleFS.h>
#include <esp_task_wdt.h>

#include "Config.h"

#if MODEL_ARCH == MODEL_ARCH_MLP
  #include <NeuralNetwork.h>
#endif

#include "Model.h"
#include "MLPModel.h"
#include "CNN1DModel.h"
#include "GRUModel.h"
#include "LD2410C.h"

// ── Pinos e baudrates ─────────────────────────────────────────────────────────
static constexpr int      SENSOR_RX_PIN  = 16;
static constexpr int      SENSOR_TX_PIN  = 17;
static constexpr int      OUT_PIN        = 18;
static constexpr uint32_t SENSOR_BAUD    = 256000;
static constexpr uint32_t USB_BAUD       = 115200;

// ── Dataset ───────────────────────────────────────────────────────────────────
static constexpr int   T_WINDOW    = 8;
static constexpr int   N_FEAT      = 12;
static constexpr int   N_INP       = T_WINDOW * N_FEAT;   // 96
static constexpr int   BPR_AUTO    = N_INP * 4 + 1;       // 385 bytes/amostra
static constexpr float MAX_DIST    = 500.0f;
static constexpr float MAX_ENERGY  = 100.0f;

// ── Confiança mínima para aceitar rótulo inferido ─────────────────────────────
// Amostras onde max(probs) < MIN_AUTO_CONFIDENCE são descartadas silenciosamente.
// Aumente para um dataset mais limpo (mais rejeições); diminua para mais volume.
#ifndef MIN_AUTO_CONFIDENCE
  #define MIN_AUTO_CONFIDENCE 0.60f
#endif

static constexpr float  AUTO_THRESHOLD_CM  = 8.0f;  // filtro direcional (mesma lógica do collector)

static const char* PRIV_PATH      = "/dataset_priv.bin";
static const char* CLASS_NAMES[4] = {"EMPTY", "STATIONARY", "APPROACHING", "LEAVING"};

// ── Estado global ─────────────────────────────────────────────────────────────
static Model*    g_model    = nullptr;
static bool      g_model_ok = false;

LD2410C radar(Serial2);

struct AutoState {
    bool  collecting  = true;   // inicia coletando (ao contrário do collector manual)
    bool  done        = false;
    int   counts[4]   = {};
    int   total       = 0;
    int   rejected    = 0;      // amostras descartadas por baixa confiança
    bool  eng_ok      = false;
};
static AutoState g;

static uint32_t g_bytes_rx  = 0;
static uint32_t g_frames_ok = 0;

static float frame_buf[T_WINDOW][N_FEAT];
static int   frame_idx = 0;

// Buffer de linha para comandos multi-palavra (UPLOAD)
static char  s_line_buf[256];
static int   s_line_pos = 0;

// ── Fábrica de modelo (igual a main.cpp) ──────────────────────────────────────
static Model* create_model() {
#if MODEL_ARCH == MODEL_ARCH_MLP
    Serial.println("[AUTO] Modelo: MLP");
    unsigned int layers_init[] = MODEL_LAYERS;
    byte         actv_init[]   = MODEL_ACTV;
    unsigned int* layers = new unsigned int[MODEL_N_LAYERS];
    byte*         actv   = new byte[MODEL_N_LAYERS - 1];
    memcpy(layers, layers_init, MODEL_N_LAYERS   * sizeof(unsigned int));
    memcpy(actv,   actv_init,   (MODEL_N_LAYERS-1) * sizeof(byte));
    return new MLPModel(layers, MODEL_N_LAYERS, actv);

#elif MODEL_ARCH == MODEL_ARCH_CNN1D
    Serial.println("[AUTO] Modelo: CNN1D");
    return new CNN1DModel();

#elif MODEL_ARCH == MODEL_ARCH_GRU
    Serial.println("[AUTO] Modelo: GRU");
    return new GRUModel();

#else
    #error "MODEL_ARCH não definido — verifique Config.h"
#endif
}

// ── Carregamento de pesos ─────────────────────────────────────────────────────
static bool load_model_weights() {
    if (!LittleFS.exists(MODEL_PATH)) {
        Serial.printf("[AUTO] AVISO: %s não existe. Execute ao menos um round de\n"
                      "       federação com SAVE_MODEL ativo antes de usar este firmware.\n",
                      MODEL_PATH);
        return false;
    }
    File f = LittleFS.open(MODEL_PATH, "r");
    if (!f) {
        Serial.printf("[AUTO] Falha ao abrir %s\n", MODEL_PATH);
        return false;
    }
    bool ok = g_model->loadWeights(f);
    Serial.printf("[AUTO] loadWeights: %s (%u bytes)\n",
                  ok ? "OK" : "FALHA", (unsigned)f.size());
    f.close();
    return ok;
}

// ── Verificação de direção (igual ao collector manual) ────────────────────────
static bool check_direction(int label_1based) {
    if (label_1based != 3 && label_1based != 4) return true;
    float first_dist = 0.f, last_dist = 0.f;
    for (int t = 0; t < T_WINDOW; t++) {
        float d = frame_buf[t][0] * MAX_DIST;
        if (d > 5.f) {
            if (first_dist < 1.f) first_dist = d;
            last_dist = d;
        }
    }
    if (first_dist < 1.f || last_dist < 1.f) return false;
    float delta = last_dist - first_dist;
    if (label_1based == 3) return delta < -AUTO_THRESHOLD_CM;
    else                   return delta >  AUTO_THRESHOLD_CM;
}

// ── Inferência: retorna label 1-based (0 = descartado) ───────────────────────
static int infer_label() {
    // Monta vetor flat [T_WINDOW × N_FEAT]
    float x[N_INP];
    for (int t = 0; t < T_WINDOW; t++)
        for (int k = 0; k < N_FEAT; k++)
            x[t * N_FEAT + k] = frame_buf[t][k];

    float* probs = g_model->feedForward(x);

    // Argmax
    int best = 0;
    for (int k = 1; k < N_CLASSES; k++)
        if (probs[k] > probs[best]) best = k;

    // Filtro de confiança
    if (probs[best] < MIN_AUTO_CONFIDENCE) {
        g.rejected++;
        return 0;  // descartado
    }

    return best + 1;  // 1-based
}

// ── Gravação de amostra ───────────────────────────────────────────────────────
static bool write_sample(int label_1based) {
    File f = LittleFS.open(PRIV_PATH, "a");
    if (!f) { Serial.println("[ERR] Nao abriu dataset_priv.bin"); return false; }
    for (int t = 0; t < T_WINDOW; t++)
        for (int k = 0; k < N_FEAT; k++)
            f.write((uint8_t*)&frame_buf[t][k], 4);
    uint8_t lbl = (uint8_t)label_1based;
    f.write(&lbl, 1);
    f.close();
    return true;
}

// ── Impressão de status ───────────────────────────────────────────────────────
static void print_status(int last_label, float last_conf) {
    Serial.printf(
        "\r[%s] E:%3d S:%3d A:%3d L:%3d  Total:%4d  Rej:%3d  Last:%-12s(%.2f)  buf:%d/%d   ",
        g.collecting ? "COLETANDO" : "PAUSADO  ",
        g.counts[0], g.counts[1], g.counts[2], g.counts[3],
        g.total, g.rejected,
        last_label ? CLASS_NAMES[last_label - 1] : "(desc.)",
        last_conf,
        frame_idx, T_WINDOW
    );
}

static void print_done() {
    Serial.println("\n\n" + String('=', 60));
    Serial.println("Coleta automática finalizada!");
    Serial.printf("  Total aceito : %d amostras\n", g.total);
    Serial.printf("  Rejeitados   : %d (confiança < %.2f)\n", g.rejected, (float)MIN_AUTO_CONFIDENCE);
    for (int i = 0; i < N_CLASSES; i++)
        Serial.printf("  %-12s: %d\n", CLASS_NAMES[i], g.counts[i]);
    File f = LittleFS.open(PRIV_PATH, "r");
    if (f) {
        Serial.printf("  Arquivo: %s  (%u bytes)\n", PRIV_PATH, (unsigned)f.size());
        f.close();
    }
    Serial.println(String('=', 60));
}

// ── Protocolo UPLOAD (idêntico ao main_collector.cpp) ────────────────────────
static void handle_upload(const char* cmd) {
    Serial.println();
    char path[128];
    long expected_bytes = 0;
    if (sscanf(cmd, "UPLOAD %127s %ld", path, &expected_bytes) != 2 || expected_bytes <= 0) {
        Serial.println("UPLOAD_ERR bad_command");
        return;
    }
    if (strcmp(path, "/dataset_priv.bin") == 0 || strcmp(path, "dataset_priv.bin") == 0) {
        Serial.println("UPLOAD_ERR protected_file");
        return;
    }
    size_t free_bytes = LittleFS.totalBytes() - LittleFS.usedBytes();
    if ((size_t)expected_bytes > free_bytes) {
        Serial.printf("UPLOAD_ERR no_space free=%u needed=%ld\n", (unsigned)free_bytes, expected_bytes);
        return;
    }
    File f = LittleFS.open(path, "w");
    if (!f) { Serial.printf("UPLOAD_ERR open_failed %s\n", path); return; }

    Serial.println("UPLOAD_READY");

    char     hex_line[300];
    int      hex_pos      = 0;
    long     bytes_written = 0;
    bool     error         = false;
    uint32_t last_byte_ms  = millis();
    const uint32_t TIMEOUT_MS = 30000UL;

    while (millis() - last_byte_ms < TIMEOUT_MS) {
        if (!Serial.available()) { delay(1); continue; }
        char c = (char)Serial.read();
        last_byte_ms = millis();
        if (c == '\r') continue;
        if (c == '\n') {
            hex_line[hex_pos] = '\0';
            if (strcmp(hex_line, "UPLOAD_END") == 0) break;
            int len = hex_pos;
            if (len == 0 || len < 2) { hex_pos = 0; continue; }
            if (len % 2 != 0) { error = true; Serial.printf("UPLOAD_ERR odd_hex_len %d\n", len); break; }
            uint8_t decoded[150];
            int decoded_len = 0;
            for (int i = 0; i < len && !error; i += 2) {
                uint8_t hi = hex_line[i], lo = hex_line[i+1];
                if      (hi >= '0' && hi <= '9') hi -= '0';
                else if (hi >= 'A' && hi <= 'F') hi -= 'A'-10;
                else if (hi >= 'a' && hi <= 'f') hi -= 'a'-10;
                else { error = true; break; }
                if      (lo >= '0' && lo <= '9') lo -= '0';
                else if (lo >= 'A' && lo <= 'F') lo -= 'A'-10;
                else if (lo >= 'a' && lo <= 'f') lo -= 'a'-10;
                else { error = true; break; }
                decoded[decoded_len++] = (hi << 4) | lo;
            }
            if (!error) {
                size_t nw = f.write(decoded, decoded_len);
                if (nw != (size_t)decoded_len) {
                    error = true;
                    Serial.printf("UPLOAD_ERR write_failed wrote=%u expected=%d\n", (unsigned)nw, decoded_len);
                } else {
                    bytes_written += decoded_len;
                    Serial.write('+'); Serial.write('\n');
                }
            }
            if (error) break;
            hex_pos = 0;
        } else {
            if (hex_pos < (int)(sizeof(hex_line)-1)) hex_line[hex_pos++] = c;
            else { error = true; Serial.println("UPLOAD_ERR line_overflow"); break; }
        }
    }
    f.close();
    if (error) { LittleFS.remove(path); return; }
    if (millis() - last_byte_ms >= TIMEOUT_MS) {
        LittleFS.remove(path);
        Serial.printf("UPLOAD_ERR timeout after %ld bytes\n", bytes_written);
        return;
    }
    if (bytes_written != expected_bytes) {
        LittleFS.remove(path);
        Serial.printf("UPLOAD_ERR size_mismatch got=%ld expected=%ld\n", bytes_written, expected_bytes);
        return;
    }
    Serial.printf("UPLOAD_OK %ld\n", bytes_written);
    Serial.printf("[INFO] %s gravado: %ld bytes  LittleFS: %u/%u\n",
                  path, bytes_written,
                  (unsigned)LittleFS.usedBytes(), (unsigned)LittleFS.totalBytes());

    // Se recebemos novos pesos do modelo, recarrega automaticamente
    if (strcmp(path, MODEL_PATH) == 0 && g_model) {
        Serial.println("[AUTO] Novos pesos detectados — recarregando modelo...");
        g_model_ok = load_model_weights();
        Serial.printf("[AUTO] Modelo: %s\n", g_model_ok ? "OK" : "FALHA");
    }
}

// ── Processamento de comandos Serial ─────────────────────────────────────────
static int   s_last_label = 0;
static float s_last_conf  = 0.f;

static void handle_command(char c) {
    switch (c) {
        case 'c': case 'C':
            g.collecting = !g.collecting;
            frame_idx = 0;
            Serial.printf("\n→ Coleta: %s\n", g.collecting ? "ATIVA" : "PAUSADA");
            break;
        case 'd': case 'D':
            g.done = true;
            break;
        case 'r': case 'R':
            LittleFS.remove(PRIV_PATH);
            memset(g.counts, 0, sizeof(g.counts));
            g.total = 0; g.rejected = 0; frame_idx = 0;
            s_last_label = 0; s_last_conf = 0.f;
            Serial.println("\n→ dataset_priv.bin apagado!");
            break;
        case '?':
            Serial.println();
            print_done();
            break;
        case 'x': case 'X': {
            File f = LittleFS.open(PRIV_PATH, "r");
            if (!f) { Serial.println("\n[ERR] Arquivo nao encontrado"); break; }
            long sz = f.size();
            Serial.printf("\nDUMP_START %ld\n", sz);
            int col = 0;
            while (f.available()) {
                uint8_t b = f.read();
                Serial.printf("%02X", b);
                if (++col >= 64) { Serial.println(); col = 0; delay(4); }
            }
            if (col > 0) Serial.println();
            Serial.println("DUMP_END");
            f.close();
            break;
        }
        case 'b': case 'B': {
            Serial.println("\n── Raw bytes do sensor (próximos 80) ──────");
            uint32_t t0 = millis();
            int count = 0;
            while (count < 80 && millis() - t0 < 3000) {
                if (Serial2.available()) {
                    uint8_t b = Serial2.read();
                    Serial.printf("%02X ", b);
                    if (++count % 16 == 0) Serial.println();
                }
            }
            Serial.println("\n───────────────────────────────────────────");
            break;
        }
        case 'v': case 'V': {
            const LD2410C_Frame& fr = radar.getFrame();
            Serial.println("\n── Diagnóstico ──────────────────────────────");
            Serial.printf("  valid       : %s\n", fr.valid ? "SIM" : "NAO");
            Serial.printf("  engineering : %s\n", fr.engineering ? "SIM" : "NAO");
            Serial.printf("  target_state: %d\n", fr.target_state);
            Serial.printf("  moving_dist : %u cm  energy: %u\n", fr.moving_dist_cm, fr.moving_energy);
            Serial.printf("  static_dist : %u cm  energy: %u\n", fr.stationary_dist_cm, fr.stationary_energy);
            Serial.print ("  moving_gates: ");
            for (int i = 0; i < 9; i++) Serial.printf("%3u ", fr.moving_gate[i]);
            Serial.println();
            Serial.print ("  static_gates: ");
            for (int i = 0; i < 9; i++) Serial.printf("%3u ", fr.stationary_gate[i]);
            Serial.println();
            Serial.printf("  modelo_ok   : %s\n", g_model_ok ? "SIM" : "NAO");
            Serial.printf("  bytes_rx    : %lu\n", g_bytes_rx);
            Serial.printf("  frames_ok   : %lu\n", g_frames_ok);
            Serial.println("─────────────────────────────────────────────");
            break;
        }
        default:
            break;
    }
}

// ── setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.setRxBufferSize(1024);
    Serial.begin(USB_BAUD);
    delay(300);

    esp_task_wdt_init(WDT_TIMEOUT_SEC, true);
    esp_task_wdt_add(NULL);

    pinMode(OUT_PIN, INPUT);

    if (!LittleFS.begin(false)) {
        Serial.println("[FATAL] LittleFS nao montou.");
        while (true) delay(1000);
    }

    Serial.println("\n" + String('=', 60));
    Serial.println("FedKD-MR — Coleta Automática com Inferência On-Device");
    Serial.printf("Cliente: %s\n", CLIENT_NAME);
    Serial.printf("LittleFS: total=%u  usado=%u bytes\n",
                  (unsigned)LittleFS.totalBytes(), (unsigned)LittleFS.usedBytes());

    // Dataset existente
    if (LittleFS.exists(PRIV_PATH)) {
        File f = LittleFS.open(PRIV_PATH, "r");
        if (f) {
            int n = (int)(f.size() / BPR_AUTO);
            Serial.printf("[INFO] dataset_priv.bin: %d amostras existentes\n", n);
            Serial.println("  Envie 'r' para apagar ou 'c' para pausar/retomar.");
            f.close();
        }
    }

    // Cria e carrega modelo
    g_model  = create_model();
    g_model_ok = g_model ? load_model_weights() : false;

    if (!g_model_ok) {
        Serial.println("[AVISO] Modelo não carregado — coleta PAUSADA até pesos disponíveis.");
        Serial.printf("  Envie os pesos via: UPLOAD %s <size>\n", MODEL_PATH);
        g.collecting = false;
    }

    // Inicializa sensor
    g.eng_ok = radar.begin(SENSOR_RX_PIN, SENSOR_TX_PIN, SENSOR_BAUD);
    Serial.printf("Engineering Mode: %s\n", g.eng_ok ? "OK" : "NAO (modo basico)");
    Serial.println(String('=', 60));
    Serial.println("Comandos: c=Pausar/Retomar  r=Apagar  x=Exportar  d=Finalizar  ?=Status");
    Serial.println("          v=Diagnóstico  b=Raw bytes  UPLOAD /path size=Receber arquivo");
    Serial.printf ("Confiança mínima: %.2f\n", (float)MIN_AUTO_CONFIDENCE);
    if (g.collecting) Serial.println("Coleta ativa — aguardando janelas do sensor...");
}

// ── loop ──────────────────────────────────────────────────────────────────────
void loop() {
    // ── Comandos Serial ───────────────────────────────────────────────────────
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\r') continue;
        if (c == '\n') {
            s_line_buf[s_line_pos] = '\0';
            if (s_line_pos >= 7 && strncmp(s_line_buf, "UPLOAD ", 7) == 0)
                handle_upload(s_line_buf);
            else if (s_line_pos == 1)
                handle_command(s_line_buf[0]);
            s_line_pos = 0;
        } else {
            if (s_line_pos < (int)(sizeof(s_line_buf)-1))
                s_line_buf[s_line_pos++] = c;
            if (s_line_pos == 1 && c != 'U' && c != 'u') {
                handle_command(c);
                s_line_pos = 0;
            }
        }
    }

    // ── Finalização ───────────────────────────────────────────────────────────
    if (g.done) {
        print_done();
        g.done = false;
        return;
    }

    // ── Leitura do sensor ─────────────────────────────────────────────────────
    g_bytes_rx += Serial2.available();
    if (!radar.update()) { esp_task_wdt_reset(); return; }
    g_frames_ok++;

    const LD2410C_Frame& fr = radar.getFrame();
    if (!fr.valid) return;

    // Retry Engineering Mode se necessário
    if (!fr.engineering) {
        static uint32_t last_eng_retry = 0;
        if (millis() - last_eng_retry > 8000) {
            last_eng_retry = millis();
            Serial.println("[AVISO] Frame Basic — re-habilitando Engineering Mode...");
            if (radar.enableEngineeringMode()) Serial.println("  → enviado");
        }
        return;
    }

    print_status(s_last_label, s_last_conf);

    // Só coleta se modelo carregado e modo ativo
    if (!g.collecting || !g_model_ok) return;

    radar.fillFeatureVector(frame_buf[frame_idx]);
    frame_idx++;

    if (frame_idx < T_WINDOW) return;
    frame_idx = 0;

    // Inferência
    int label = infer_label();
    if (label == 0) {
        // Baixa confiança — descartado (g.rejected já incrementado em infer_label)
        s_last_label = 0;
        s_last_conf  = 0.f;
        return;
    }

    // Filtro direcional (mesma lógica do collector manual)
    if (!check_direction(label)) return;

    // Obtém confiança para exibição
    {
        float x[N_INP];
        for (int t = 0; t < T_WINDOW; t++)
            for (int k = 0; k < N_FEAT; k++)
                x[t * N_FEAT + k] = frame_buf[t][k];
        float* probs = g_model->feedForward(x);
        s_last_conf  = probs[label - 1];
    }
    s_last_label = label;

    if (write_sample(label)) {
        g.counts[label - 1]++;
        g.total++;
    }

    esp_task_wdt_reset();
}
