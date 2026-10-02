#pragma once
/**
 * CNN1DModel.h — Cliente 2: 1D-CNN para HAR no ESP32
 * ====================================================
 * Arquitetura (baseada no notebook FedKD_MR.ipynb, Cell 25):
 *
 *   Input: x[96] flat → reshape → seq[T=8][C=12]  (channels-first: [C=12][T=8])
 *
 *   Conv1D(in_ch=12, out_ch=32, kernel=3, pad=1) → ReLU  → out: [32][8]
 *   Conv1D(in_ch=32, out_ch=64, kernel=3, pad=1) → ReLU  → out: [64][8]
 *   GlobalMaxPool                                          → out: [64]
 *   Dense(64 → N_CLASSES=4)                       → Softmax
 *
 * Padding=SAME (pad=1) mantém comprimento temporal T=8 em ambas as camadas.
 * Sem dropout (ESP32 — treino on-device).
 *
 * Backprop implementado manualmente:
 *   Softmax+CE → Dense → GlobalMaxPool → Conv2 → Conv1
 *
 * Memória estimada (float32):
 *   Conv1 W: 12×32×3 = 1152 f + 32 bias
 *   Conv2 W: 32×64×3 = 6144 f + 64 bias
 *   Dense W: 64×4    =  256 f +  4 bias
 *   Total:   ~29 KB  — ok para ESP32 (~320 KB heap)
 */

#include "Model.h"
#include <stdlib.h>
#include <string.h>

// ── Hiperparâmetros fixos ─────────────────────────────────────────────────────
static const int CNN_T  = 8;         // T_WINDOW
static const int CNN_C  = 12;        // N_FEATURES
static const int CNN_K  = 3;         // kernel size
static const int CNN_CH1 = 32;       // filtros camada 1
static const int CNN_CH2 = 64;       // filtros camada 2
static const int CNN_CLS = N_CLASSES;// 4

class CNN1DModel : public Model {
public:
    // ── Parâmetros treináveis ─────────────────────────────────────────────────
    // Conv1: weights[out_ch][in_ch][k], bias[out_ch]
    float w1[CNN_CH1][CNN_C ][CNN_K];
    float b1[CNN_CH1];

    // Conv2: weights[out_ch][in_ch][k], bias[out_ch]
    float w2[CNN_CH2][CNN_CH1][CNN_K];
    float b2[CNN_CH2];

    // Dense: weights[out][in], bias[out]
    float wd[CNN_CLS][CNN_CH2];
    float bd[CNN_CLS];

    // ── Activações intermediárias (para backprop) ─────────────────────────────
    float a1[CNN_CH1][CNN_T];   // saída ReLU conv1
    float z1[CNN_CH1][CNN_T];   // pré-ReLU conv1
    float a2[CNN_CH2][CNN_T];   // saída ReLU conv2
    float z2[CNN_CH2][CNN_T];   // pré-ReLU conv2
    float pool[CNN_CH2];         // GlobalMaxPool
    int   pool_pos[CNN_CH2];     // posição do max (para backward)
    float logits[CNN_CLS];       // pré-Softmax
    float probs[CNN_CLS];        // Softmax output

    // ── Variáveis de entrada (para backprop da conv1) ─────────────────────────
    float inp[CNN_C][CNN_T];     // reshape de x[96]

    // ── Gradientes (membros heap para não estourar pilha loopTask de 8 KB) ───
    // dA2+dZ2 = 2×(64×8×4) = 4 KB; dA1+dZ1 = 2×(32×8×4) = 2 KB
    float dA2_m[CNN_CH2][CNN_T];
    float dZ2_m[CNN_CH2][CNN_T];
    float dA1_m[CNN_CH1][CNN_T];
    float dZ1_m[CNN_CH1][CNN_T];

    // ── Learning rates ───────────────────────────────────────────────────────
    float lr_w = 0.003f;
    float lr_b = 0.003f;

    // ── MSE acumulado ────────────────────────────────────────────────────────
    float sum_sq_err = 0.f;

    // ── Construtor: inicialização He/Xavier ──────────────────────────────────
    CNN1DModel() {
        // He init para ReLU: std = sqrt(2 / fan_in)
        float std1 = sqrtf(2.f / (CNN_C  * CNN_K));
        float std2 = sqrtf(2.f / (CNN_CH1 * CNN_K));
        float stdD = sqrtf(2.f / CNN_CH2);

        for (int o = 0; o < CNN_CH1; o++) {
            b1[o] = 0.f;
            for (int i = 0; i < CNN_C; i++)
                for (int k = 0; k < CNN_K; k++)
                    w1[o][i][k] = randf() * std1;
        }
        for (int o = 0; o < CNN_CH2; o++) {
            b2[o] = 0.f;
            for (int i = 0; i < CNN_CH1; i++)
                for (int k = 0; k < CNN_K; k++)
                    w2[o][i][k] = randf() * std2;
        }
        for (int o = 0; o < CNN_CLS; o++) {
            bd[o] = 0.f;
            for (int i = 0; i < CNN_CH2; i++)
                wd[o][i] = randf() * stdD;
        }
        sum_sq_err = 0.f;
    }

    // ── Interface Model ──────────────────────────────────────────────────────

    float* feedForward(const float* x) override {
        // Reshape x[96] → inp[12][8]  (inp[feature][timestep])
        for (int c = 0; c < CNN_C; c++)
            for (int t = 0; t < CNN_T; t++)
                inp[c][t] = x[t * CNN_C + c];

        // ── Conv1 + ReLU ──────────────────────────────────────────────────────
        // pad=1 (zero-padding) → saída mesmo tamanho T=8
        for (int o = 0; o < CNN_CH1; o++) {
            for (int t = 0; t < CNN_T; t++) {
                float sum = b1[o];
                for (int i = 0; i < CNN_C; i++) {
                    for (int k = 0; k < CNN_K; k++) {
                        int ts = t + k - 1;         // -1 = padding left
                        if (ts >= 0 && ts < CNN_T)
                            sum += w1[o][i][k] * inp[i][ts];
                    }
                }
                z1[o][t] = sum;
                a1[o][t] = relu(sum);
            }
        }

        // ── Conv2 + ReLU ──────────────────────────────────────────────────────
        for (int o = 0; o < CNN_CH2; o++) {
            for (int t = 0; t < CNN_T; t++) {
                float sum = b2[o];
                for (int i = 0; i < CNN_CH1; i++) {
                    for (int k = 0; k < CNN_K; k++) {
                        int ts = t + k - 1;
                        if (ts >= 0 && ts < CNN_T)
                            sum += w2[o][i][k] * a1[i][ts];
                    }
                }
                z2[o][t] = sum;
                a2[o][t] = relu(sum);
            }
        }

        // ── GlobalMaxPool ─────────────────────────────────────────────────────
        for (int o = 0; o < CNN_CH2; o++) {
            float mx = a2[o][0]; int pos = 0;
            for (int t = 1; t < CNN_T; t++)
                if (a2[o][t] > mx) { mx = a2[o][t]; pos = t; }
            pool[o]     = mx;
            pool_pos[o] = pos;
        }

        // ── Dense → Softmax ───────────────────────────────────────────────────
        for (int o = 0; o < CNN_CLS; o++) {
            float sum = bd[o];
            for (int i = 0; i < CNN_CH2; i++) sum += wd[o][i] * pool[i];
            logits[o] = sum;
            probs[o]  = sum;  // copiado antes do softmax
        }
        softmax(probs, CNN_CLS);

        return probs;
    }

    void backProp(const float* y_target) override {
        // ── Gradiente Softmax+CE: dL/dlogits = probs - y_target ──────────────
        float dLogits[CNN_CLS];
        float sq_err = 0.f;
        for (int k = 0; k < CNN_CLS; k++) {
            float diff = probs[k] - y_target[k];
            dLogits[k] = diff;
            sq_err += diff * diff;
        }
        sum_sq_err += sq_err;

        // ── Dense backward ────────────────────────────────────────────────────
        float dPool[CNN_CH2] = {};
        for (int o = 0; o < CNN_CLS; o++) {
            bd[o] -= lr_b * dLogits[o];
            for (int i = 0; i < CNN_CH2; i++) {
                dPool[i] += wd[o][i] * dLogits[o];
                wd[o][i] -= lr_w * dLogits[o] * pool[i];
            }
        }

        // ── GlobalMaxPool backward (grad só passa pela posição do max) ────────
        memset(dA2_m, 0, sizeof(dA2_m));
        for (int o = 0; o < CNN_CH2; o++)
            dA2_m[o][pool_pos[o]] = dPool[o];

        // ── Conv2 backward ────────────────────────────────────────────────────
        // dZ2 = dA2 * relu'(z2)
        for (int o = 0; o < CNN_CH2; o++)
            for (int t = 0; t < CNN_T; t++)
                dZ2_m[o][t] = dA2_m[o][t] * relu_d(z2[o][t]);

        // dA1[i][ts] += sum_o sum_k w2[o][i][k] * dZ2[o][t]  (t = ts - k + 1)
        memset(dA1_m, 0, sizeof(dA1_m));
        for (int o = 0; o < CNN_CH2; o++) {
            b2[o] -= lr_b * _sum(dZ2_m[o], CNN_T);
            for (int i = 0; i < CNN_CH1; i++) {
                for (int k = 0; k < CNN_K; k++) {
                    float dw = 0.f;
                    for (int t = 0; t < CNN_T; t++) {
                        int ts = t + k - 1;
                        if (ts >= 0 && ts < CNN_T) {
                            dw += dZ2_m[o][t] * a1[i][ts];
                            dA1_m[i][ts] += w2[o][i][k] * dZ2_m[o][t];
                        }
                    }
                    w2[o][i][k] -= lr_w * dw;
                }
            }
        }

        // ── Conv1 backward ────────────────────────────────────────────────────
        for (int o = 0; o < CNN_CH1; o++)
            for (int t = 0; t < CNN_T; t++)
                dZ1_m[o][t] = dA1_m[o][t] * relu_d(z1[o][t]);

        for (int o = 0; o < CNN_CH1; o++) {
            b1[o] -= lr_b * _sum(dZ1_m[o], CNN_T);
            for (int i = 0; i < CNN_C; i++) {
                for (int k = 0; k < CNN_K; k++) {
                    float dw = 0.f;
                    for (int t = 0; t < CNN_T; t++) {
                        int ts = t + k - 1;
                        if (ts >= 0 && ts < CNN_T)
                            dw += dZ1_m[o][t] * inp[i][ts];
                    }
                    w1[o][i][k] -= lr_w * dw;
                }
            }
        }
    }

    float getMSE(int n) override {
        float mse = (n > 0) ? sum_sq_err / ((float)CNN_CLS * n) : 0.f;
        sum_sq_err = 0.f;
        return mse;
    }

    void setLR(float lr_weights, float lr_biases) override {
        lr_w = lr_weights;
        lr_b = lr_biases;
    }

    void clampParams(float max_val) override {
        // Conv1
        for (int o = 0; o < CNN_CH1; o++) {
            b1[o] = clamp(b1[o], -max_val, max_val);
            for (int i = 0; i < CNN_C;  i++)
                for (int k = 0; k < CNN_K; k++)
                    w1[o][i][k] = clamp(w1[o][i][k], -max_val, max_val);
        }
        // Conv2
        for (int o = 0; o < CNN_CH2; o++) {
            b2[o] = clamp(b2[o], -max_val, max_val);
            for (int i = 0; i < CNN_CH1; i++)
                for (int k = 0; k < CNN_K; k++)
                    w2[o][i][k] = clamp(w2[o][i][k], -max_val, max_val);
        }
        // Dense
        for (int o = 0; o < CNN_CLS; o++) {
            bd[o] = clamp(bd[o], -max_val, max_val);
            for (int i = 0; i < CNN_CH2; i++)
                wd[o][i] = clamp(wd[o][i], -max_val, max_val);
        }
    }

private:
    // Número aleatório em [-1, 1]
    static float randf() {
        return ((float)random(-1000, 1001)) / 1000.f;
    }

    // Soma de um vetor de tamanho n
    static float _sum(const float* v, int n) {
        float s = 0.f;
        for (int i = 0; i < n; i++) s += v[i];
        return s;
    }
};
