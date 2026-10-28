/**
 * main_collector.cpp — FedKD-MR: Coleta de Dados Reais no ESP32
 * ==============================================================
 * Firmware de coleta autônoma: lê o HLK-LD2410C via UART, monta
 * janelas de 8 timesteps × 12 features e grava diretamente no
 * LittleFS (dataset_priv.bin) no formato binário do FedKD-MR.
 *
 * Ao final da coleta, o ESP32 já tem o dataset_priv.bin pronto.
 * Basta reflashear o firmware principal (main.cpp) e iniciar a
 * federação normalmente — o servidor envia cmd/start e o ESP32
 * lê os dados reais já gravados na flash.
 *
 * ── Conexão hardware ─────────────────────────────────────────────
 *
 *   LD2410C        ESP32
 *   ────────────── ────────────────────────────────────────────
 *   TX (pino 1) →  GPIO 16  (RX2)   [3.3V — conexão direta OK]
 *   RX (pino 2) →  GPIO 17  (TX2)   [3.3V — conexão direta OK]
 *   GND (pino 4)→  GND
 *   VCC (pino 5)→  5V / VIN         [>200 mA — use cabo USB bom]
 *   OUT (pino 3)→  (opcional) GPIO 18 — nível alto = presença
 *
 *   ATENÇÃO: o VCC do sensor é 5 V. Não conecte TX/RX do sensor
 *   diretamente à alimentação de 3.3 V do ESP32.
 *   Os níveis lógicos do sensor são 3.3 V — conexão direta segura.
 *
 * ── Controle via Serial USB (115200) ─────────────────────────────
 *
 *   e → label: EMPTY        (1)
 *   s → label: STATIONARY   (2)
 *   a → label: APPROACHING  (3)
 *   l → label: LEAVING      (4)
 *   p → Pausar / retomar coleta
 *   d → Finalizar coleta (imprime estatísticas)
 *   r → Reiniciar/apagar dataset_priv.bin (CUIDADO!)
 *   ? → Status atual
 *
 * ── Formato de saída (dataset_priv.bin) ──────────────────────────
 *
 *   Igual ao gerado por prepare_dataset.py:
 *   Cada linha = 96 float32 (little-endian) + 1 uint8 label (1-based)
 *   = 385 bytes por amostra
 *
 *   Features (12 por timestep, 8 timesteps = 96 valores):
 *     t0_moving_distance_cm  … t7_moving_distance_cm
 *     t0_moving_energy       … t7_moving_energy
 *     t0_stationary_dist_cm  … (etc.)
 *     moving_g0..g3, stationary_g0..g3  (por timestep)
 *
 *   Normalização: valores brutos do sensor divididos pelos limites
 *   físicos (dist_max=500 cm, energy_max=100), resultando em [0,1].
 *   Consistente com fillFeatureVector() do LD2410C e com a inferência
 *   do modelo no firmware principal.
 *
 * ── Como usar ─────────────────────────────────────────────────────
 *
 *   1. Compile e grave este firmware no ESP32:
 *        pio run -e collector -t upload
 *
 *   2. Abra o Serial Monitor (115200 baud).
 *
 *   3. Posicione-se / configure a cena e envie o label:
 *        e  → sala vazia
 *        s  → fique parado na frente do sensor
 *        a  → aproxime-se do sensor caminhando
 *        l  → afaste-se do sensor caminhando
 *
 *   4. O firmware acumula janelas automaticamente enquanto
 *      o label está ativo. Pause (p) ao trocar de cena.
 *
 *   5. Quando tiver amostras suficientes, envie d (done).
 *      O firmware imprime as estatísticas e encerra a gravação.
 *
 *   6. Grave o firmware principal (main.cpp) no ESP32 sem apagar
 *      a partição LittleFS:
 *        pio run -e mlp -t upload      (não faça Upload Filesystem Image)
 *
 *   7. Inicie a federação normalmente.
 *
 * Nota: dataset_pub.bin e teacher_probs.bin permanecem sintéticos
 * (gerados por prepare_dataset.py --n_synth 200 --n_pub_per_class 20).
 * Somente dataset_priv.bin é substituído por dados reais.
 */

#include <Arduino.h>
#include <LittleFS.h>
#include "LD2410C.h"

// ── Pinos e baudrates ─────────────────────────────────────────────────────────
static constexpr int    SENSOR_RX_PIN  = 16;
static constexpr int    SENSOR_TX_PIN  = 17;
static constexpr int    OUT_PIN        = 18;    // opcional — OUT do sensor
static constexpr uint32_t SENSOR_BAUD = 256000;
static constexpr uint32_t USB_BAUD    = 115200;

// ── Dataset ───────────────────────────────────────────────────────────────────
static constexpr int    T_WINDOW   = 8;
static constexpr int    N_FEATURES = 12;
static constexpr int    N_INPUT    = T_WINDOW * N_FEATURES;  // 96
static constexpr int    N_CLASSES  = 4;
static constexpr float  MAX_DIST   = 500.0f;    // cm — alcance máximo LD2410C
static constexpr float  MAX_ENERGY = 100.0f;    // escala nativa do sensor
static constexpr int    BPR        = N_INPUT * 4 + 1;  // 385 bytes/linha

// ── Detecção automática de direção (APPROACHING / LEAVING) ────────────────────
// Limiar mínimo de variação de distância em toda a janela para aceitar amostra.
// Aumentar se gravar muitos falsos; diminuir se descartar demais.
static constexpr float  AUTO_THRESHOLD_CM = 8.0f;

static const char* PRIV_PATH      = "/dataset_priv.bin";
static const char* CLASS_NAMES[4] = {"EMPTY","STATIONARY","APPROACHING","LEAVING"};

// ── Estado global ─────────────────────────────────────────────────────────────
LD2410C radar(Serial2);

struct State {
    int     current_label  = 0;       // 0=sem label, 1-4=classes
    bool    paused         = false;
    bool    done           = false;
    int     counts[4]      = {};      // amostras por classe (0-based idx)
    int     total          = 0;
    bool    eng_ok         = false;
};
static State g;

// Contador de bytes brutos recebidos do sensor (diagnóstico)
static uint32_t g_bytes_rx  = 0;
static uint32_t g_frames_ok = 0;

// Buffer de frames para formar uma janela
static float frame_buf[T_WINDOW][N_FEATURES];
static int   frame_idx = 0;

// Normalização delegada ao fillFeatureVector() da própria classe LD2410C

// ── Verificação de direção de movimento ───────────────────────────────────────
// Para APPROACHING: a distância de movimento deve diminuir ao longo da janela.
// Para LEAVING: a distância de movimento deve aumentar.
// EMPTY e STATIONARY não têm restrição de direção.
// Retorna true se a janela deve ser gravada, false para descartar.
static bool check_direction(int label_1based) {
    if (label_1based != 3 && label_1based != 4) return true;  // E e S: sempre grava

    // Coleta distâncias de movimento (feat[0] * MAX_DIST = cm)
    // Ignora frames com distância zero (sem alvo móvel detectado)
    float first_dist = 0.0f, last_dist = 0.0f;
    for (int t = 0; t < T_WINDOW; t++) {
        float d = frame_buf[t][0] * MAX_DIST;  // feat[0] = moving_dist_cm / MAX_DIST
        if (d > 5.0f) {
            if (first_dist < 1.0f) first_dist = d;
            last_dist = d;
        }
    }
    if (first_dist < 1.0f || last_dist < 1.0f) return false;  // sem alvo móvel

    float delta = last_dist - first_dist;  // positivo = afastando, negativo = aproximando

    if (label_1based == 3) return delta < -AUTO_THRESHOLD_CM;   // APPROACHING
    else                   return delta >  AUTO_THRESHOLD_CM;   // LEAVING
}

// ── Gravação de amostra no LittleFS ──────────────────────────────────────────
// Layout: [t0_f0 t0_f1 … t0_f11 | t1_f0 … t7_f11] float32 LE + label uint8
static bool write_sample(int label_1based) {
    File f = LittleFS.open(PRIV_PATH, "a");   // append
    if (!f) {
        Serial.println("[ERR] Nao abriu dataset_priv.bin para append");
        return false;
    }
    // Achata: timestep-major (igual ao prepare_dataset.py)
    for (int t = 0; t < T_WINDOW; t++) {
        for (int k = 0; k < N_FEATURES; k++) {
            f.write((uint8_t*)&frame_buf[t][k], 4);
        }
    }
    uint8_t lbl = (uint8_t)label_1based;
    f.write(&lbl, 1);
    f.close();
    return true;
}

// ── Impressão de status ───────────────────────────────────────────────────────
static void print_status() {
    Serial.printf(
        "\r[%s] Label: %-12s  E:%3d S:%3d A:%3d L:%3d  Total:%4d  buf:%d/%d   ",
        g.paused ? "PAUSADO " : "GRAVANDO",
        g.current_label ? CLASS_NAMES[g.current_label - 1] : "(nenhum)",
        g.counts[0], g.counts[1], g.counts[2], g.counts[3],
        g.total, frame_idx, T_WINDOW
    );
}

static void print_done() {
    Serial.println("\n\n" + String('=', 55));
    Serial.println("Coleta finalizada!");
    Serial.printf("  Total de amostras: %d\n", g.total);
    for (int i = 0; i < N_CLASSES; i++) {
        Serial.printf("  %s: %d\n", CLASS_NAMES[i], g.counts[i]);
    }
    File f = LittleFS.open(PRIV_PATH, "r");
    if (f) {
        Serial.printf("  Arquivo: %s  (%u bytes)\n", PRIV_PATH,
                      (unsigned)f.size());
        f.close();
    }
    Serial.println("\nProximos passos:");
    Serial.println("  1. Grave firmware principal (main.cpp) SEM apagar LittleFS");
    Serial.println("     pio run -e mlp -t upload   (nao faca Upload Filesystem Image)");
    Serial.println("  2. Inicie a federacao normalmente");
    Serial.println(String('=', 55));
}

// ── Processamento de comandos Serial ─────────────────────────────────────────
static void handle_command(char c) {
    switch (c) {
        case 'e': case 'E':
            g.current_label = 1; frame_idx = 0;
            Serial.printf("\n→ Label: EMPTY\n"); break;
        case 's': case 'S':
            g.current_label = 2; frame_idx = 0;
            Serial.printf("\n→ Label: STATIONARY\n"); break;
        case 'a': case 'A':
            g.current_label = 3; frame_idx = 0;
            Serial.printf("\n→ Label: APPROACHING\n"); break;
        case 'l': case 'L':
            g.current_label = 4; frame_idx = 0;
            Serial.printf("\n→ Label: LEAVING\n"); break;
        case 'p': case 'P':
            g.paused = !g.paused; frame_idx = 0;
            Serial.printf("\n→ %s\n", g.paused ? "PAUSADO" : "RETOMADO"); break;
        case 'd': case 'D':
            g.done = true; break;
        case 'r': case 'R':
            LittleFS.remove(PRIV_PATH);
            memset(g.counts, 0, sizeof(g.counts));
            g.total = 0; frame_idx = 0;
            Serial.println("\n→ dataset_priv.bin apagado!"); break;
        case '?':
            Serial.println();
            print_done();
            break;
        case 'x': case 'X': {
            // Exporta dataset_priv.bin como hex para captura no PC
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
            // Dump hex dos próximos bytes do sensor (diagnóstico de baud rate)
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
            // Diagnóstico: imprime o último frame bruto do sensor
            const LD2410C_Frame& fr = radar.getFrame();
            Serial.println("\n── Diagnóstico do sensor ──────────────────");
            Serial.printf("  valid       : %s\n", fr.valid ? "SIM" : "NAO");
            Serial.printf("  engineering : %s\n", fr.engineering ? "SIM" : "NAO");
            Serial.printf("  target_state: %d  (0=nenhum 1=mov 2=estat 3=ambos)\n", fr.target_state);
            Serial.printf("  moving_dist : %u cm   energy: %u\n", fr.moving_dist_cm, fr.moving_energy);
            Serial.printf("  static_dist : %u cm   energy: %u\n", fr.stationary_dist_cm, fr.stationary_energy);
            Serial.printf("  detect_dist : %u cm\n", fr.detection_dist_cm);
            Serial.print ("  moving_gates: ");
            for (int i = 0; i < 9; i++) Serial.printf("%3u ", fr.moving_gate[i]);
            Serial.println();
            Serial.print ("  static_gates: ");
            for (int i = 0; i < 9; i++) Serial.printf("%3u ", fr.stationary_gate[i]);
            Serial.println();
            float feat[12];
            radar.fillFeatureVector(feat);
            Serial.print ("  feat[0..11] : ");
            for (int i = 0; i < 12; i++) Serial.printf("%.3f ", feat[i]);
            Serial.printf("  bytes_rx    : %lu\n", g_bytes_rx);
            Serial.printf("  frames_ok   : %lu\n", g_frames_ok);
            Serial.println("───────────────────────────────────────────");
            break;
        }
        default:
            break;
    }
}

// ── setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(USB_BAUD);
    delay(300);

    pinMode(OUT_PIN, INPUT);

    if (!LittleFS.begin(false)) {
        Serial.println("[FATAL] LittleFS nao montou. Rode Upload Filesystem Image primeiro.");
        while (true) delay(1000);
    }

    // Verifica arquivo existente
    if (LittleFS.exists(PRIV_PATH)) {
        File f = LittleFS.open(PRIV_PATH, "r");
        if (f) {
            long sz = f.size();
            f.close();
            int n = (int)(sz / BPR);
            Serial.printf("[INFO] dataset_priv.bin existente: %d amostras (%ld bytes)\n",
                          n, sz);
            Serial.println("  Envie 'r' para apagar e reiniciar, ou continue coletando.");
        }
    }

    g.eng_ok = radar.begin(SENSOR_RX_PIN, SENSOR_TX_PIN, SENSOR_BAUD);

    Serial.println("\n" + String('=', 55));
    Serial.println("FedKD-MR — Firmware de Coleta de Dados Reais");
    Serial.printf("Engineering Mode: %s\n", g.eng_ok ? "OK" : "NAO (modo basico)");
    Serial.println(String('=', 55));
    Serial.println("Comandos: e=EMPTY  s=STATIONARY  a=APPROACHING  l=LEAVING");
    Serial.println("          p=Pausar/Retomar  d=Finalizar  r=Apagar  ?=Status");
    Serial.println("Aguardando label inicial...\n");
}

// ── loop ──────────────────────────────────────────────────────────────────────
void loop() {
    // Comandos Serial
    while (Serial.available()) {
        char c = (char)Serial.read();
        handle_command(c);
    }

    if (g.done) {
        print_done();
        g.done = false;
        g.current_label = 0;
        return;
    }

    // Leitura do sensor
    g_bytes_rx += Serial2.available();   // conta bytes antes do update consumir
    if (!radar.update()) return;
    g_frames_ok++;

    const LD2410C_Frame& fr = radar.getFrame();
    if (!fr.valid)       return;
    if (!fr.engineering) {
        // Frame Basic Mode — tenta re-habilitar Engineering Mode a cada 8s
        static uint32_t last_eng_retry = 0;
        if (millis() - last_eng_retry > 8000) {
            last_eng_retry = millis();
            Serial.println("[AVISO] Frame Basic Mode — re-habilitando Engineering Mode...");
            if (radar.enableEngineeringMode())
                Serial.println("  → comando enviado");
            else
                Serial.println("  → ACK timeout — verifique conexão TX do ESP32 ao sensor");
        }
        return;
    }

    print_status();

    if (g.paused || g.current_label == 0) return;

    // Acumula frame no buffer da janela
    radar.fillFeatureVector(frame_buf[frame_idx]);
    frame_idx++;

    if (frame_idx < T_WINDOW) return;

    // Janela completa → verifica direção e grava
    frame_idx = 0;
    if (!check_direction(g.current_label)) return;  // direção errada — descarta
    if (write_sample(g.current_label)) {
        g.counts[g.current_label - 1]++;
        g.total++;
    }
}
