/**
 * LD2410C.cpp — Implementação do parser HLK-LD2410C
 * ==================================================
 * Protocolo binário HLK, Engineering Mode.
 *
 * Estrutura dos frames (Engineering Mode):
 *   [FD FC FB FA] [len_lo len_hi]
 *   [01 00]         ← tipo: engineering (0x0002 = básico)
 *   [AA]            ← marcador de início de dados
 *   [target_state]
 *   [mov_dist_lo mov_dist_hi] [mov_energy]
 *   [stat_dist_lo stat_dist_hi] [stat_energy]
 *   [detect_dist_lo detect_dist_hi]
 *   [mg0..mg8]      ← 9 bytes de energia por gate móvel
 *   [sg0..sg8]      ← 9 bytes de energia por gate estático
 *   [55] [00]       ← tail + checksum (checksum não verificado nesta impl.)
 *   [04 03 02 01]
 *
 * Estrutura de frame básico (sem Engineering Mode):
 *   Mesmo que acima mas tipo = [02 00], sem bytes de gate, len total menor.
 *
 * Referência: documentação HLK-LD2410C v1.02 + biblioteca ncmreynolds/LD2410.
 */

#include "LD2410C.h"

// ── Constantes de protocolo ───────────────────────────────────────────────────
static const uint8_t FRAME_HEADER[4] = { 0xFD, 0xFC, 0xFB, 0xFA };
static const uint8_t FRAME_FOOTER[4] = { 0x04, 0x03, 0x02, 0x01 };

// Comprimentos esperados de dados (entre len e footer, excluindo próprio len)
static const uint16_t ENG_FRAME_DATA_LEN   = 32;   // Engineering Mode
static const uint16_t BASIC_FRAME_DATA_LEN = 14;   // Modo básico

// Offsets dentro de _buf (após os 2 bytes de tipo)
static const int OFF_TYPE_LO = 0;
static const int OFF_TYPE_HI = 1;
static const int OFF_HEAD    = 2;   // deve ser 0xAA
static const int OFF_STATE   = 3;
static const int OFF_MOV_LO  = 4;
static const int OFF_MOV_HI  = 5;
static const int OFF_MOV_E   = 6;
static const int OFF_STAT_LO = 7;
static const int OFF_STAT_HI = 8;
static const int OFF_STAT_E  = 9;
static const int OFF_DET_LO  = 10;
static const int OFF_DET_HI  = 11;
static const int OFF_MGATE   = 12;  // [12..20] moving gates 0-8
static const int OFF_SGATE   = 21;  // [21..29] static gates 0-8
// [30] = 0x55 tail, [31] = 0x00 check

// ── begin ─────────────────────────────────────────────────────────────────────
bool LD2410C::begin(int rx_pin, int tx_pin, uint32_t baud) {
    _serial.begin(baud, SERIAL_8N1, rx_pin, tx_pin);
    delay(100);

    // Tenta habilitar Engineering Mode até 3 vezes
    for (int attempt = 1; attempt <= 3; attempt++) {
        if (enableEngineeringMode()) {
            Serial.printf("[LD2410C] Engineering Mode habilitado (tentativa %d)\n", attempt);
            return true;
        }
        delay(200);
    }
    Serial.println("[LD2410C] AVISO: Engineering Mode não confirmado — modo básico ativo");
    return false;  // sensor pode ainda estar funcionando em modo básico
}

// ── enableEngineeringMode ─────────────────────────────────────────────────────
bool LD2410C::enableEngineeringMode() {
    // Passo 1: entra em modo de configuração
    // Payload: FF 00 (cmd) + 01 00 (parâmetro = "enable config")
    static const uint8_t enter_cfg[] = { 0xFF, 0x00, 0x01, 0x00 };
    _sendCmd(enter_cfg, sizeof(enter_cfg));
    if (!_waitAck(0x00FF)) return false;
    delay(50);

    // Passo 2: habilita Engineering Mode
    // Payload: 62 00
    static const uint8_t start_eng[] = { 0x62, 0x00 };
    _sendCmd(start_eng, sizeof(start_eng));
    if (!_waitAck(0x0062)) return false;
    delay(50);

    // Passo 3: sai do modo de configuração
    // Payload: FE 00
    static const uint8_t close_cfg[] = { 0xFE, 0x00 };
    _sendCmd(close_cfg, sizeof(close_cfg));
    if (!_waitAck(0x00FE)) return false;

    return true;
}

// ── disableEngineeringMode ────────────────────────────────────────────────────
bool LD2410C::disableEngineeringMode() {
    static const uint8_t enter_cfg[] = { 0xFF, 0x00, 0x01, 0x00 };
    _sendCmd(enter_cfg, sizeof(enter_cfg));
    if (!_waitAck(0x00FF)) return false;
    delay(50);

    static const uint8_t stop_eng[] = { 0x63, 0x00 };
    _sendCmd(stop_eng, sizeof(stop_eng));
    if (!_waitAck(0x0063)) return false;
    delay(50);

    static const uint8_t close_cfg[] = { 0xFE, 0x00 };
    _sendCmd(close_cfg, sizeof(close_cfg));
    if (!_waitAck(0x00FE)) return false;

    return true;
}

// ── update ────────────────────────────────────────────────────────────────────
bool LD2410C::update() {
    while (_serial.available()) {
        uint8_t b = _serial.read();

        switch (_state) {
            // ── Sincronização do header: FD FC FB FA ──────────────────────
            case 0:
                if (b == 0xFD) _state = 1;
                break;
            case 1:
                if (b == 0xFC) _state = 2;
                else _state = (b == 0xFD) ? 1 : 0;
                break;
            case 2:
                if (b == 0xFB) _state = 3;
                else _state = (b == 0xFD) ? 1 : 0;
                break;
            case 3:
                if (b == 0xFA) _state = 4;
                else _state = (b == 0xFD) ? 1 : 0;
                break;

            // ── Leitura do campo length (2 bytes, little-endian) ──────────
            case 4:
                _data_len  = b;
                _state     = 5;
                break;
            case 5:
                _data_len |= (uint16_t)b << 8;
                _data_idx  = 0;
                if (_data_len == 0 || _data_len > BUF_SIZE) {
                    // Frame com comprimento inválido — reseta
                    _state = 0;
                } else {
                    _state = 6;
                }
                break;

            // ── Leitura dos bytes de dados ────────────────────────────────
            case 6:
                _buf[_data_idx++] = b;
                if (_data_idx >= _data_len) _state = 7;
                break;

            // ── Sincronização do footer: 04 03 02 01 ─────────────────────
            case 7:
                _state = (b == 0x04) ? 8 : 0;
                break;
            case 8:
                _state = (b == 0x03) ? 9 : 0;
                break;
            case 9:
                _state = (b == 0x02) ? 10 : 0;
                break;
            case 10:
                _state = 0;
                if (b == 0x01) {
                    return _parseBuffer();
                }
                break;

            default:
                _state = 0;
                break;
        }
    }
    return false;
}

// ── fillFeatureVector ─────────────────────────────────────────────────────────
void LD2410C::fillFeatureVector(float feat[12]) const {
    // Limites físicos do sensor:
    //   distância: 0–500 cm (alcance máximo do LD2410C)
    //   energia:   0–100 (escala nativa do sensor)
    // ATENÇÃO: o prepare_dataset.py com dados reais deve usar os mesmos limites.
    static constexpr float MAX_DIST   = 500.0f;
    static constexpr float MAX_ENERGY = 100.0f;

    feat[0] = _frame.moving_dist_cm      / MAX_DIST;
    feat[1] = _frame.moving_energy        / MAX_ENERGY;
    feat[2] = _frame.stationary_dist_cm   / MAX_DIST;
    feat[3] = _frame.stationary_energy    / MAX_ENERGY;
    for (int i = 0; i < 4; i++) {
        feat[4 + i] = _frame.moving_gate[i]     / MAX_ENERGY;
        feat[8 + i] = _frame.stationary_gate[i] / MAX_ENERGY;
    }

    // Clamp conservador — sensor pode ocasionalmente reportar valores acima do esperado
    for (int i = 0; i < 12; i++) {
        if (feat[i] < 0.0f) feat[i] = 0.0f;
        if (feat[i] > 1.0f) feat[i] = 1.0f;
    }
}

// ── _sendCmd ──────────────────────────────────────────────────────────────────
bool LD2410C::_sendCmd(const uint8_t* payload, uint8_t len) {
    // Frame de configuração: [header(4)] [len_lo len_hi] [payload] [footer(4)]
    uint8_t frame[16] = {};
    frame[0] = 0xFD; frame[1] = 0xFC; frame[2] = 0xFB; frame[3] = 0xFA;
    frame[4] = len;  frame[5] = 0x00;
    memcpy(frame + 6, payload, len);
    frame[6 + len] = 0x04;
    frame[7 + len] = 0x03;
    frame[8 + len] = 0x02;
    frame[9 + len] = 0x01;
    _serial.write(frame, (size_t)(len + 10));
    return true;
}

// ── _waitAck ──────────────────────────────────────────────────────────────────
bool LD2410C::_waitAck(uint16_t cmd_word, uint32_t timeout_ms) {
    // O ACK tem o mesmo header/footer que o comando, com os bytes de tipo
    // modificados: tipo_lo = cmd_lo, tipo_hi = cmd_hi | 0x01
    // Exemplos:
    //   Enter config (0x00FF) → ACK tipo: [FF 01]
    //   Start eng   (0x0062) → ACK tipo: [62 01]
    //   Close config (0x00FE) → ACK tipo: [FE 01]
    uint8_t expected_lo = (uint8_t)(cmd_word & 0xFF);
    uint8_t expected_hi = (uint8_t)((cmd_word >> 8) | 0x01);

    uint32_t deadline = millis() + timeout_ms;
    uint8_t  st       = 0;

    while (millis() < deadline) {
        if (!_serial.available()) { delay(1); continue; }
        uint8_t b = _serial.read();
        switch (st) {
            case 0: st = (b == 0xFD) ? 1 : 0;            break;
            case 1: st = (b == 0xFC) ? 2 : (b==0xFD?1:0); break;
            case 2: st = (b == 0xFB) ? 3 : (b==0xFD?1:0); break;
            case 3: st = (b == 0xFA) ? 4 : (b==0xFD?1:0); break;
            case 4: st = 5; break;  // len_lo — ignora
            case 5: st = 6; break;  // len_hi — ignora
            case 6: st = (b == expected_lo) ? 7 : 0; break;
            case 7:
                if (b == expected_hi) return true;
                st = 0;
                break;
            default: st = 0;
        }
    }
    return false;
}

// ── _parseBuffer ─────────────────────────────────────────────────────────────
bool LD2410C::_parseBuffer() {
    if (_data_len < 2) return false;

    uint8_t type_lo = _buf[OFF_TYPE_LO];

    _frame.valid       = false;
    _frame.engineering = false;

    if (type_lo == 0x01) {
        // ── Engineering Mode frame ──────────────────────────────────────
        if (_data_len < ENG_FRAME_DATA_LEN) return false;
        if (_buf[OFF_HEAD] != 0xAA)         return false;

        _frame.target_state        = _buf[OFF_STATE];
        _frame.moving_dist_cm      = (uint16_t)_buf[OFF_MOV_LO] | ((uint16_t)_buf[OFF_MOV_HI] << 8);
        _frame.moving_energy       = _buf[OFF_MOV_E];
        _frame.stationary_dist_cm  = (uint16_t)_buf[OFF_STAT_LO] | ((uint16_t)_buf[OFF_STAT_HI] << 8);
        _frame.stationary_energy   = _buf[OFF_STAT_E];
        _frame.detection_dist_cm   = (uint16_t)_buf[OFF_DET_LO] | ((uint16_t)_buf[OFF_DET_HI] << 8);
        for (int i = 0; i < 9; i++) {
            _frame.moving_gate[i]     = _buf[OFF_MGATE + i];
            _frame.stationary_gate[i] = _buf[OFF_SGATE + i];
        }
        _frame.engineering = true;
        _frame.valid       = true;
    }
    else if (type_lo == 0x02) {
        // ── Modo básico (sem dados de gate) ───────────────────────────
        if (_data_len < BASIC_FRAME_DATA_LEN) return false;
        if (_buf[OFF_HEAD] != 0xAA)           return false;

        _frame.target_state        = _buf[OFF_STATE];
        _frame.moving_dist_cm      = (uint16_t)_buf[OFF_MOV_LO] | ((uint16_t)_buf[OFF_MOV_HI] << 8);
        _frame.moving_energy       = _buf[OFF_MOV_E];
        _frame.stationary_dist_cm  = (uint16_t)_buf[OFF_STAT_LO] | ((uint16_t)_buf[OFF_STAT_HI] << 8);
        _frame.stationary_energy   = _buf[OFF_STAT_E];
        _frame.detection_dist_cm   = (uint16_t)_buf[OFF_DET_LO] | ((uint16_t)_buf[OFF_DET_HI] << 8);
        memset(_frame.moving_gate,     0, sizeof(_frame.moving_gate));
        memset(_frame.stationary_gate, 0, sizeof(_frame.stationary_gate));
        _frame.engineering = false;
        _frame.valid       = true;
    }
    // Tipo 0xFF = ACK de configuração — ignorado aqui (tratado em _waitAck)

    return _frame.valid;
}
