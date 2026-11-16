/**
 * FedKD-MR — firmware temporario somente para transferencia serial.
 * Nao inicia sensor, nao coleta amostras e nao aceita comandos destrutivos.
 * Compativel com o UPLOAD do fedkd_orchestrator.py, com ACK + por linha.
 */
#include <Arduino.h>
#include <LittleFS.h>
#include <stdio.h>
#include <string.h>

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

    // Allowlist: nao permite modificar dataset privado, checkpoints ou outros arquivos.
    const char* allowed[] = {
        "/dataset_train.bin", "/dataset_test.bin", "/dataset_pub.bin",
        "/metadata.json", "/teacher_probs.bin", "/teacher_probs_meta.json"
    };
    bool permitted = false;
    for (const char* item : allowed) {
        if (strcmp(path, item) == 0) { permitted = true; break; }
    }
    if (!permitted) {
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
    String tmp = "/fedkd_up.tmp";
    File f = LittleFS.open(tmp.c_str(), "w");
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
    const uint32_t TIMEOUT_MS = 120000UL;

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
                    Serial.flush();  // garante transmissão imediata antes de aguardar próxima linha
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
        LittleFS.remove(tmp.c_str());
        // Erro já impresso acima
        return;
    }

    bool timeout = (millis() - last_byte_ms >= TIMEOUT_MS);
    if (timeout) {
        LittleFS.remove(tmp.c_str());
        Serial.printf("UPLOAD_ERR timeout after %ld bytes\n", bytes_written);
        return;
    }

    if (bytes_written != expected_bytes) {
        LittleFS.remove(tmp.c_str());
        Serial.printf("UPLOAD_ERR size_mismatch got=%ld expected=%ld\n",
                      bytes_written, expected_bytes);
        return;
    }

    // Arquivo completo antes da troca. Nunca altera checkpoints.
    if (LittleFS.exists(path) && !LittleFS.remove(path)) {
        LittleFS.remove(tmp.c_str());
        Serial.println("UPLOAD_ERR replace_failed");
        return;
    }
    if (!LittleFS.rename(tmp.c_str(), path)) {
        LittleFS.remove(tmp.c_str());
        Serial.println("UPLOAD_ERR rename_failed");
        return;
    }

    Serial.printf("UPLOAD_OK %ld\n", bytes_written);
    Serial.printf("[INFO] %s gravado: %ld bytes  LittleFS usado: %u/%u\n",
                  path, bytes_written,
                  (unsigned)LittleFS.usedBytes(),
                  (unsigned)LittleFS.totalBytes());
}


static char command_line[256];
static size_t command_len = 0;

void setup() {
    Serial.setRxBufferSize(1024);
    Serial.begin(115200);
    delay(300);
    if (!LittleFS.begin(false)) {
        Serial.println("FATAL LittleFS mount failed; no format attempted");
        while (true) delay(1000);
    }
    Serial.printf("UPLOAD_ONLY_READY LittleFS: total=%u usado=%u bytes\n",
                  (unsigned)LittleFS.totalBytes(), (unsigned)LittleFS.usedBytes());
    Serial.println("Somente UPLOAD autorizado; coleta desabilitada.");
}

void loop() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\r') continue;
        if (c == '\n') {
            command_line[command_len] = 0;
            if (command_len >= 7 && strncmp(command_line, "UPLOAD ", 7) == 0) {
                handle_upload(command_line);
            } else if (command_len) {
                Serial.println("UPLOAD_ERR unsupported_command");
            }
            command_len = 0;
        } else if (command_len < sizeof(command_line)-1) {
            command_line[command_len++] = c;
        } else {
            command_len = 0;
            Serial.println("UPLOAD_ERR command_too_long");
        }
    }
    delay(1);
}
