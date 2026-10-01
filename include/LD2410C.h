#pragma once
/**
 * LD2410C.h — Parser UART para o radar HLK-LD2410C
 * =================================================
 * Protocolo binário proprietário HLK, 256000 baud, 8N1.
 * Requer Engineering Mode para obter energias por gate (N_FEATURES=12).
 *
 * Uso rápido:
 *   LD2410C radar(Serial2);
 *   radar.begin(LD2410C_RX_PIN, LD2410C_TX_PIN);   // pinos em Config.h
 *   radar.enableEngineeringMode();
 *
 *   float feat[12];
 *   while (true) {
 *     if (radar.update()) {
 *       radar.fillFeatureVector(feat);  // feat[] normalizados [0,1]
 *       // acumule T_WINDOW amostras e passe ao classificador
 *     }
 *   }
 */

#include <Arduino.h>

// ── Frame de dados ────────────────────────────────────────────────────────────
struct LD2410C_Frame {
    uint8_t  target_state;           // 0=nenhum 1=móvel 2=estático 3=ambos
    uint16_t moving_dist_cm;         // distância do alvo em movimento (cm)
    uint8_t  moving_energy;          // energia do alvo em movimento (0–100)
    uint16_t stationary_dist_cm;     // distância do alvo estático (cm)
    uint8_t  stationary_energy;      // energia do alvo estático (0–100)
    uint16_t detection_dist_cm;      // distância de detecção geral (cm)
    uint8_t  moving_gate[9];         // energia por gate móvel (gates 0–8)
    uint8_t  stationary_gate[9];     // energia por gate estático (gates 0–8)
    bool     engineering;            // true = frame de Engineering Mode
    bool     valid;                  // true = frame completo e parseado
};

// ── Classe principal ──────────────────────────────────────────────────────────
class LD2410C {
public:
    explicit LD2410C(HardwareSerial& serial) : _serial(serial) {}

    /**
     * Inicializa UART2 e tenta habilitar Engineering Mode.
     * @param rx_pin  GPIO de RX (ex: 16)
     * @param tx_pin  GPIO de TX (ex: 17)
     * @param baud    baud rate do sensor (padrão: 256000)
     * @return true se Engineering Mode foi habilitado com sucesso
     */
    bool begin(int rx_pin, int tx_pin, uint32_t baud = 256000);

    /** Habilita Engineering Mode: sequência EnterConfig → Start → CloseConfig. */
    bool enableEngineeringMode();

    /** Desabilita Engineering Mode (volta ao modo básico). */
    bool disableEngineeringMode();

    /**
     * Processa bytes disponíveis no UART. Chame o mais frequente possível.
     * @return true quando um frame completo e válido for recebido
     */
    bool update();

    /** Retorna o último frame válido recebido. */
    const LD2410C_Frame& getFrame() const { return _frame; }

    /** Retorna true se o último frame recebido tem dados de gate (Engineering Mode). */
    bool hasEngineeringData() const { return _frame.valid && _frame.engineering; }

    /**
     * Preenche feat[12] com features normalizadas [0,1]:
     *   feat[0]   moving_distance   / 500 cm
     *   feat[1]   moving_energy     / 100
     *   feat[2]   stationary_dist   / 500 cm
     *   feat[3]   stationary_energy / 100
     *   feat[4-7] moving_gate[0-3]  / 100
     *   feat[8-11] stationary_gate[0-3] / 100
     *
     * NOTA: as mesmas bounds devem ser usadas no prepare_dataset.py ao
     * treinar com dados reais (MAX_DIST=500, MAX_ENERGY=100).
     * Chame apenas quando hasEngineeringData() == true.
     */
    void fillFeatureVector(float feat[12]) const;

private:
    HardwareSerial& _serial;
    LD2410C_Frame   _frame {};

    // ── Máquina de estados do parser ─────────────────────────────────────────
    // 0-3: sincronização de header (FD FC FB FA)
    // 4-5: leitura dos 2 bytes de comprimento
    // 6:   leitura de _data_len bytes para _buf
    // 7-10: sincronização de footer (04 03 02 01)
    uint8_t  _state    = 0;
    uint16_t _data_len = 0;
    uint16_t _data_idx = 0;

    static const size_t BUF_SIZE = 64;
    uint8_t _buf[BUF_SIZE];

    // Envia um comando encapsulado no frame de configuração HLK.
    // payload: bytes do corpo do comando (sem header/footer/len).
    bool _sendCmd(const uint8_t* payload, uint8_t len);

    // Aguarda ACK de um comando. cmd_word: word do comando (ex: 0x00FF).
    bool _waitAck(uint16_t cmd_word, uint32_t timeout_ms = 500);

    // Interpreta os bytes em _buf como frame de dados ou ACK.
    bool _parseBuffer();
};

// ── Acumulador de janela deslizante ──────────────────────────────────────────
/**
 * SensorWindow mantém um buffer circular de T_WINDOW leituras do LD2410C e
 * entrega o vetor achatado (T_WINDOW × 12 = 96 features) ao classificador.
 *
 * Uso:
 *   SensorWindow<8, 12> window;
 *   float flat[96];
 *   if (window.push(feat12) && window.isFull()) {
 *     window.flatten(flat);
 *     // passe flat[] ao NeuralNetwork::FeedForward()
 *   }
 */
template<int T_WIN, int N_FEAT>
class SensorWindow {
public:
    /** Adiciona uma leitura. Retorna true se a janela está cheia. */
    bool push(const float feat[N_FEAT]) {
        memcpy(_buf[_head], feat, N_FEAT * sizeof(float));
        _head = (_head + 1) % T_WIN;
        if (_count < T_WIN) _count++;
        return _count == T_WIN;
    }

    bool isFull() const { return _count == T_WIN; }

    /** Copia a janela em ordem cronológica para out[T_WIN × N_FEAT]. */
    void flatten(float out[T_WIN * N_FEAT]) const {
        for (int i = 0; i < T_WIN; i++) {
            int src = (_head + i) % T_WIN;
            memcpy(out + i * N_FEAT, _buf[src], N_FEAT * sizeof(float));
        }
    }

    void reset() { _head = 0; _count = 0; }

private:
    float _buf[T_WIN][N_FEAT] {};
    int   _head  = 0;
    int   _count = 0;
};
