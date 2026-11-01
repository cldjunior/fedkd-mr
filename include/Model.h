#pragma once
/**
 * Model.h — Interface abstrata para modelos locais FedKD-MR
 * ==========================================================
 * Permite que train_supervised, train_kd, evaluate e
 * publish_logits_mqtt operem sem saber se o modelo é
 * MLP, CNN1D ou GRU.
 *
 * Contrato:
 *   - feedForward(x)      : recebe x[N_INPUT] flat, retorna ptr para probs[N_CLASSES]
 *   - backProp(y)         : gradiente em relação ao último feedForward; y[N_CLASSES]
 *   - getMSE(n)           : retorna sum_sq_err / (N_CLASSES * n) e zera acumulador
 *   - setLR(lrw, lrb)    : ajusta taxa de aprendizado de pesos e biases
 *   - clampParams(max)    : clipa todos os parâmetros a [-max, max] e zera NaN/Inf
 *   - saveWeights(f)      : serializa todos os pesos/biases para o arquivo aberto
 *   - loadWeights(f)      : carrega pesos/biases do arquivo aberto (posição 0)
 */

#include "Config.h"
#include <FS.h>      // File (incluso transitivamente via LittleFS.h — explicitado aqui
                     // para que Model.h compile em contextos sem LittleFS.h no path)
#include <math.h>

class Model {
public:
    virtual ~Model() {}

    // Forward pass — retorna ponteiro interno (válido até próximo feedForward)
    virtual float* feedForward(const float* x) = 0;

    // Backward pass — deve ser chamado logo após feedForward
    virtual void backProp(const float* y_target) = 0;

    // MSE acumulado desde o último reset; n = número de amostras
    virtual float getMSE(int n) = 0;

    // Ajusta learning rate
    virtual void setLR(float lr_weights, float lr_biases) = 0;

    // Clipa parâmetros e zera NaN/Inf
    virtual void clampParams(float max_val) = 0;

    /**
     * Serializa todos os pesos e biases para o File já aberto em modo escrita.
     * Retorna true se bem-sucedido.
     * O chamador é responsável por abrir e fechar o arquivo.
     */
    virtual bool saveWeights(File& f) = 0;

    /**
     * Carrega pesos e biases do File já aberto em modo leitura (posição 0).
     * Retorna true se bem-sucedido.
     * O chamador é responsável por abrir e fechar o arquivo.
     */
    virtual bool loadWeights(File& f) = 0;

protected:
    // Utilitário: softmax in-place sobre vetor de tamanho n
    static void softmax(float* v, int n) {
        float mx = v[0];
        for (int i = 1; i < n; i++) if (v[i] > mx) mx = v[i];
        float sum = 0.f;
        for (int i = 0; i < n; i++) { v[i] = expf(v[i] - mx); sum += v[i]; }
        for (int i = 0; i < n; i++) v[i] /= sum;
    }

    // Utilitário: relu e sua derivada
    static inline float relu(float x)       { return x > 0.f ? x : 0.f; }
    static inline float relu_d(float x)     { return x > 0.f ? 1.f : 0.f; }

    // Utilitário: tanh e sua derivada
    static inline float tanh_d(float t)     { return 1.f - t * t; }  // t = tanh(x)

    // Utilitário: sigmoid e sua derivada
    static inline float sigmoid(float x)    { return 1.f / (1.f + expf(-x)); }
    static inline float sigmoid_d(float s)  { return s * (1.f - s); }   // s = sigma(x)

    // Clamp seguro
    static inline float clamp(float v, float lo, float hi) {
        if (!isfinite(v)) return 0.f;
        if (v > hi) return hi;
        if (v < lo) return lo;
        return v;
    }

    // Auxiliar de serialização: grava bloco de floats; retorna bytes escritos
    static size_t write_floats(File& f, const float* data, size_t n) {
        return f.write((const uint8_t*)data, n * sizeof(float));
    }

    // Auxiliar de serialização: lê bloco de floats; retorna bytes lidos
    static size_t read_floats(File& f, float* data, size_t n) {
        return f.read((uint8_t*)data, n * sizeof(float));
    }
};
