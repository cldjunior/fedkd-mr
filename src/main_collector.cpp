/**
 * main_collector.cpp — FedKD-MR: Coleta de Dados Reais no ESP32
 * ==============================================================
 * [MODIFICADO] Adiciona suporte ao protocolo UPLOAD serial para receber
 * arquivos compartilhados do fedkd_orchestrator.py sem apagar dataset_priv.bin.
 *
 * Protocolo UPLOAD (novo — compatível com fedkd_orchestrator.py):
 *   PC → ESP32:  "UPLOAD /dataset_pub.bin 30800\n"
 *   ESP32 → PC:  "UPLOAD_READY\n"
 *   PC → ESP32:  <linha hex MAIÚSCULA, ex: "AABBCC...FF\n">  (repetir)
 *   PC → ESP32:  "UPLOAD_END\n"
 *   ESP32 → PC:  "UPLOAD_OK 30800\n"   (sucesso)
 *              | "UPLOAD_ERR motivo\n" (falha)
 *
 * Comandos existentes mantidos (compatibilidade total):
 *   e/s/a/l/p/d/r/? — controle de coleta (sem alteração)
 *   x               — exporta dataset_priv.bin como hex dump (sem alteração)
 *   b/v             — diagnóstico do sensor (sem alteração)
 *
 * Para gravar apenas o firmware sem apagar o LittleFS:
 *   pio run -e collector -t upload   ← NÃO use -t uploadfs
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
static constexpr float  AUTO_THRESHOLD_CM = 8.0f;

static const char* PRIV_PATH      = "/dataset_priv.bin";
static const char* CLASS_NAMES[4] = {"EMPTY","STATIONARY","APPROACHING","LEAVING"};

// ── Estado global ─────────────────────────────────────────────────────────────
LD2410C radar(Serial2);

struct State {
    int     current_label  = 0;
    bool    paused         = false;
    bool    done           = false;
    int     counts[4]      = {};
    int     total          = 0;
    bool    eng_ok         = false;
};
static State g;

static uint32_t g_bytes_rx  = 0;
static uint32_t g_frames_ok = 0;

static float frame_buf[T_WINDOW][N_FEATURES];
static int   frame_idx = 0;

// ── Buffer de linha (partilhado entre loop() e handle_upload()) ───────────────
// Necessário para detectar o comando "UPLOAD /path size" que é multi-palavra.
static char  s_line_buf[256];
static int   s_line_pos = 0;

// ── Verificação de direção de movimento ───────────────────────────────────────
static bool check_direction(int label_1based) {
    if (label_1based != 3 && label_1based != 4) return true;
    float first_dist = 0.0f, last_dist = 0.0f;
    for (int t = 0; t < T_WINDOW; t++) {
        float d = frame_buf[t][0] * MAX_DIST;
        if (d > 5.0f) {
            if (first_dist < 1.0f) first_dist = d;
            last_dist = d;
        }
    }
    if (first_dist < 1.0f || last_dist < 1.0f) return false;
    float delta = last_dist - first_dist;
    if (label_1based == 3) return delta < -AUTO_THRESHOLD_CM;
    else                   return delta >  AUTO_THRESHOLD_CM;
}

// ── Gravação de amostra no LittleFS ──────────────────────────────────────────
static bool write_sample(int label_1based) {
    File f = LittleFS.open(PRIV_PATH, "a");
    if (!f) {
        Serial.println("[ERR] Nao abriu dataset_priv.bin para append");
        return false;
    }
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
        Serial.printf("  Arquivo: %s  (%u bytes)\n", PRIV_PATH, (unsigned)f.size());
        f.close();
    }
    Serial.println("\nProximos passos:");
    Serial.println("  1. Execute o fedkd_orchestrator.py no PC");
    Serial.println("  2. Ou grave firmware principal manualmente:");
    Serial.println("     pio run -e mlp -t upload   (sem -t uploadfs)");
    Serial.println(String('=', 55));
}

// ── Protocolo UPLOAD serial ───────────────────────────────────────────────────
/**
 * Recebe um arquivo do PC e grava no LittleFS.
 *
 * cmd: string completa da linha, ex: "UPLOAD /dataset_pub.bin 30800"
 *
 * Fluxo:
 *   1. Analisa caminho e tamanho esperado
 *   2. Abre arquivo para escrita (SOBRESCREVE se existir — nunca apaga dataset_priv.bin)
 *   3. Responde "UPLOAD_READY"
 *   4. Lê linhas hex até "UPLOAD_END"
 *   5. Responde "UPLOAD_OK <bytes>" ou "UPLOAD_ERR <motivo>"
 *
 * Timeout: 30 s sem receber byte novo.
 */
static void handle_upload(const char* cmd) {
    // Encerra linha de status pendente (print_status usa \r sem \n).
    // Sem este println, o UPLOAD_READY/ERR ficaria na mesma "linha" serial do status.
    Serial.println();

    // Analisa: "UPLOAD /caminho tamanho"
    char     path[128];
    long     expected_bytes = 0;
    if (sscanf(cmd, "UPLOAD %127s %ld", path, &expected_bytes) != 2
        || expected_bytes <= 0) {
        Serial.println("UPLOAD_ERR bad_command");
        return;
    }

    // Proteção: nunca sobrescreve dataset_priv.bin via UPLOAD
    if (strcmp(path, "/dataset_priv.bin") == 0 ||
        strcmp(path, "dataset_priv.bin")  == 0) {
        Serial.println("UPLOAD_ERR protected_file");
        return;
    }

    // Verifica espaço livre aproximado
    size_t free_bytes = LittleFS.totalBytes() - LittleFS.usedBytes();
    if ((size_t)expected_bytes > free_bytes) {
        Serial.printf("UPLOAD_ERR no_space free=%u needed=%ld\n",
                      (unsigned)free_bytes, expected_bytes);
        return;
    }

    // Abre arquivo para escrita
    File f = LittleFS.open(path, "w");
    if (!f) {
        Serial.printf("UPLOAD_ERR open_failed %s\n", path);
        return;
    }

    Serial.println("UPLOAD_READY");

    // Lê linhas hex decodificando-as diretamente no arquivo
    char     hex_line[300];   // até 128 hex chars (256 bytes) + margem
    int      hex_pos      = 0;
    long     bytes_written = 0;
    bool     error         = false;
    uint32_t last_byte_ms  = millis();
    const uint32_t TIMEOUT_MS = 30000UL;

    while (millis() - last_byte_ms < TIMEOUT_MS) {
        if (!Serial.available()) {
            delay(1);
            continue;
        }

        char c = (char)Serial.read();
        last_byte_ms = millis();

        if (c == '\r') continue;
        if (c == '\n') {
            hex_line[hex_pos] = '\0';

            if (strcmp(hex_line, "UPLOAD_END") == 0) {
                break;  // transferência concluída
            }

            // Decodifica hex — linha deve ter comprimento par
            int len = hex_pos;
            if (len == 0) {
                hex_pos = 0;
                continue;  // linha vazia — ignora
            }
            // Linha muito curta (< 2 chars) não pode ser dado válido.
            // Pode ser um stray byte (\r ou similar) antes do UPLOAD_END.
            // Ignora silenciosamente em vez de abortar.
            if (len < 2) {
                hex_pos = 0;
                continue;
            }
            if (len % 2 != 0) {
                error = true;
                Serial.printf("UPLOAD_ERR odd_hex_len %d\n", len);
                break;
            }

            // Decodifica toda a linha hex para um buffer temporário e grava
            // em um único f.write() para minimizar stalls do LittleFS.
            uint8_t decoded[150];   // 300 chars hex / 2 = 150 bytes máx
            int decoded_len = 0;
            for (int i = 0; i < len && !error; i += 2) {
                uint8_t hi = hex_line[i];
                uint8_t lo = hex_line[i + 1];
                if      (hi >= '0' && hi <= '9') hi -= '0';
                else if (hi >= 'A' && hi <= 'F') hi -= 'A' - 10;
                else if (hi >= 'a' && hi <= 'f') hi -= 'a' - 10;
                else { error = true; break; }
                if      (lo >= '0' && lo <= '9') lo -= '0';
                else if (lo >= 'A' && lo <= 'F') lo -= 'A' - 10;
                else if (lo >= 'a' && lo <= 'f') lo -= 'a' - 10;
                else { error = true; break; }
                decoded[decoded_len++] = (hi << 4) | lo;
            }
            if (!error) {
                size_t n_written = f.write(decoded, decoded_len);
                if (n_written != (size_t)decoded_len) {
                    error = true;
                    Serial.printf("UPLOAD_ERR write_failed wrote=%u expected=%d\n",
                                  (unsigned)n_written, decoded_len);
                } else {
                    bytes_written += (long)decoded_len;
                    // ACK de flow-control: confirma que a linha foi gravada.
                    // O host NÃO envia a próxima linha até receber este ACK.
                    Serial.write('+');
                    Serial.write('\n');
                }
            }

            if (error) break;
            hex_pos = 0;
        } else {
            if (hex_pos < (int)(sizeof(hex_line) - 1)) {
                hex_line[hex_pos++] = c;
            } else {
                // Linha hex mais longa que o buffer — erro de protocolo
                error = true;
                Serial.println("UPLOAD_ERR line_overflow");
                break;
            }
        }
    }

    f.close();

    if (error) {
        // Remove arquivo incompleto
        LittleFS.remove(path);
        // Erro já impresso acima
        return;
    }

    bool timeout = (millis() - last_byte_ms >= TIMEOUT_MS);
    if (timeout) {
        LittleFS.remove(path);
        Serial.printf("UPLOAD_ERR timeout after %ld bytes\n", bytes_written);
        return;
    }

    if (bytes_written != expected_bytes) {
        LittleFS.remove(path);
        Serial.printf("UPLOAD_ERR size_mismatch got=%ld expected=%ld\n",
                      bytes_written, expected_bytes);
        return;
    }

    Serial.printf("UPLOAD_OK %ld\n", bytes_written);
    Serial.printf("[INFO] %s gravado: %ld bytes  LittleFS usado: %u/%u\n",
                  path, bytes_written,
                  (unsigned)LittleFS.usedBytes(),
                  (unsigned)LittleFS.totalBytes());
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
            // Exporta dataset_priv.bin como hex para o PC
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
            Serial.printf("\n  bytes_rx    : %lu\n", g_bytes_rx);
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
    // Aumenta o buffer de RX do UART para 1024 bytes ANTES de Serial.begin().
    // Sem isso, o buffer padrão (128 bytes) pode transbordar durante um stall
    // de gravação do flash LittleFS (~10-50 ms por página de 256 bytes), o que
    // causa perda de bytes e consequentemente "odd_hex_len" durante o UPLOAD.
    Serial.setRxBufferSize(1024);
    Serial.begin(USB_BAUD);
    delay(300);

    pinMode(OUT_PIN, INPUT);

    if (!LittleFS.begin(false)) {
        Serial.println("[FATAL] LittleFS nao montou. Rode Upload Filesystem Image primeiro.");
        while (true) delay(1000);
    }

    if (LittleFS.exists(PRIV_PATH)) {
        File f = LittleFS.open(PRIV_PATH, "r");
        if (f) {
            long sz = f.size();
            f.close();
            int n = (int)(sz / BPR);
            Serial.printf("[INFO] dataset_priv.bin existente: %d amostras (%ld bytes)\n", n, sz);
            Serial.println("  Envie 'r' para apagar e reiniciar, ou continue coletando.");
        }
    }

    g.eng_ok = radar.begin(SENSOR_RX_PIN, SENSOR_TX_PIN, SENSOR_BAUD);

    Serial.println("\n" + String('=', 55));
    Serial.println("FedKD-MR — Firmware de Coleta de Dados Reais");
    Serial.printf("Engineering Mode: %s\n", g.eng_ok ? "OK" : "NAO (modo basico)");
    Serial.printf("LittleFS: total=%u  usado=%u bytes\n",
                  (unsigned)LittleFS.totalBytes(),
                  (unsigned)LittleFS.usedBytes());
    Serial.println(String('=', 55));
    Serial.println("Comandos: e=EMPTY  s=STATIONARY  a=APPROACHING  l=LEAVING");
    Serial.println("          p=Pausar/Retomar  d=Finalizar  r=Apagar  ?=Status");
    Serial.println("          x=Exportar hex  UPLOAD /path size=Receber arquivo");
    Serial.println("Aguardando label inicial...");  // NÃO usar \n extra — println já adiciona \r\n
}

// ── loop ──────────────────────────────────────────────────────────────────────
/**
 * Loop com buffer de linha para comandos Serial.
 *
 * Dois modos de operação:
 *
 * 1) Comando single-char (e/s/a/l/p/d/r/?/x/b/v):
 *    Processado imediatamente quando o char chega, sem esperar '\n'.
 *    Excepção: 'U'/'u' aguarda a linha inteira (pode ser início de "UPLOAD ...").
 *
 * 2) Comando multi-palavra "UPLOAD /caminho tamanho":
 *    Acumulado em s_line_buf até '\n', depois despachado para handle_upload().
 */
void loop() {
    // ── Leitura de comandos Serial com buffer de linha ────────────────────────
    while (Serial.available()) {
        char c = (char)Serial.read();

        if (c == '\r') continue;

        if (c == '\n') {
            // Processa linha acumulada
            s_line_buf[s_line_pos] = '\0';
            if (s_line_pos >= 7 && strncmp(s_line_buf, "UPLOAD ", 7) == 0) {
                handle_upload(s_line_buf);
            } else if (s_line_pos == 1) {
                // Char único enviado com '\n' (ex: terminal que adiciona LF)
                handle_command(s_line_buf[0]);
            }
            s_line_pos = 0;

        } else {
            // Acumula no buffer
            if (s_line_pos < (int)(sizeof(s_line_buf) - 1)) {
                s_line_buf[s_line_pos++] = c;
            }

            // Processa comandos single-char IMEDIATAMENTE ao chegar o 1º char,
            // EXCETO 'U'/'u' que pode ser início de "UPLOAD ...".
            if (s_line_pos == 1 && c != 'U' && c != 'u') {
                handle_command(c);
                s_line_pos = 0;   // reseta buffer — char processado
            }
        }
    }

    // ── Finalização de coleta ─────────────────────────────────────────────────
    if (g.done) {
        print_done();
        g.done = false;
        g.current_label = 0;
        return;
    }

    // ── Leitura do sensor ─────────────────────────────────────────────────────
    g_bytes_rx += Serial2.available();
    if (!radar.update()) return;
    g_frames_ok++;

    const LD2410C_Frame& fr = radar.getFrame();
    if (!fr.valid) return;
    if (!fr.engineering) {
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

    radar.fillFeatureVector(frame_buf[frame_idx]);
    frame_idx++;

    if (frame_idx < T_WINDOW) return;

    frame_idx = 0;
    if (!check_direction(g.current_label)) return;
    if (write_sample(g.current_label)) {
        g.counts[g.current_label - 1]++;
        g.total++;
    }
}
