#pragma once
/**
 * Model.h — Interface abstrata para modelos locais FedKD-MR
 * ==========================================================
 * Permite que train_supervised, train_kd, evaluate e
 * publish_logits_mqtt operem sem saber se o modelo é
 * MLP, CNN1D ou GRU.
 *
 * Contrato:
 *   - feedForward(x)  : recebe x[N_INPUT] flat, retorna ptr para probs[N_CLASSES]
 *   - backProp(y)     : gradiente em relação ao último feedForward; y[N_CLASSES]
 *   - getMSE(n)       : retorna sum_sq_err / (N_CLASSES * n) e zera acumulador
 *   - setLR(lrw, lrb): ajusta taxa de aprendizado de pesos e biases
 *   - clampParams(max): clipa todos os parâmetros a [-max, max] e zera NaN/Inf
 */

#include "Config.h"
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
};
