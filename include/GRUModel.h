#pragma once
/**
 * GRUModel.h — Cliente 3: GRU para HAR no ESP32
 * ===============================================
 * Arquitetura (baseada no notebook FedKD_MR.ipynb, Cell 25):
 *
 *   Input: x[96] flat → seq[T=8][C=12]
 *   GRU(input_size=12, hidden_size=64)  — reset_after=False (Keras default)
 *   Dense(64 → N_CLASSES=4) → Softmax
 *
 * Implementação GRU simplificado (sem reset_after):
 *   z_t = σ(Wz·x_t + Uz·h_{t-1} + bz)   [update gate]
 *   r_t = σ(Wr·x_t + Ur·h_{t-1} + br)   [reset gate]
 *   h̃_t = tanh(Wh·x_t + Uh·(r_t⊙h_{t-1}) + bh)  [candidate]
 *   h_t = z_t ⊙ h_{t-1} + (1 - z_t) ⊙ h̃_t
 *
 * BPTT através de T=8 timesteps.
 * Sem dropout (treino on-device).
 *
 * Memória estimada (float32):
 *   Wz/Wr/Wh: 3 × (12×64) = 2 304 f
 *   Uz/Ur/Uh: 3 × (64×64) = 12 288 f
 *   bz/br/bh: 3 × 64       =   192 f
 *   Dense W:  64×4          =   256 f
 *   Total:   ~59 KB  — ok para ESP32-S3 com PS-RAM ou ~320 KB heap padrão
 */

#include "Model.h"

// ── Dimensões ─────────────────────────────────────────────────────────────────
static const int GRU_T   = 8;    // T_WINDOW
static const int GRU_IN  = 12;   // N_FEATURES
static const int GRU_H   = 64;   // hidden_size
static const int GRU_CLS = N_CLASSES; // 4

class GRUModel : public Model {
public:
    // ── Parâmetros GRU ────────────────────────────────────────────────────────
    // Matrizes de entrada [hidden × input]
    float Wz[GRU_H][GRU_IN], Wr[GRU_H][GRU_IN], Wh[GRU_H][GRU_IN];
    // Matrizes recorrentes [hidden × hidden]
    float Uz[GRU_H][GRU_H],  Ur[GRU_H][GRU_H],  Uh[GRU_H][GRU_H];
    // Biases
    float bz[GRU_H], br[GRU_H], bh[GRU_H];

    // ── Camada densa final ────────────────────────────────────────────────────
    float Wd[GRU_CLS][GRU_H];
    float bd[GRU_CLS];

    // ── Activações por timestep (para BPTT) ──────────────────────────────────
    float z_t[GRU_T][GRU_H];   // update gate (após sigmoid)
    float r_t[GRU_T][GRU_H];   // reset gate  (após sigmoid)
    float hc_t[GRU_T][GRU_H];  // candidate hidden state (após tanh)
    float h_t[GRU_T][GRU_H];   // hidden state
    float h_prev[GRU_T][GRU_H];// h_{t-1} para cada t (salvo durante forward)
    float r_h[GRU_T][GRU_H];   // r_t ⊙ h_{t-1} (para BPTT de Wh/Uh)

    // ── Entrada salva (para BPTT) ─────────────────────────────────────────────
    float x_seq[GRU_T][GRU_IN];

    // ── Saída final ───────────────────────────────────────────────────────────
    float probs[GRU_CLS];

    // ── Learning rates e MSE ─────────────────────────────────────────────────
    float lr_w = 0.003f;
    float lr_b = 0.003f;
    float sum_sq_err = 0.f;

    // ── Construtor ───────────────────────────────────────────────────────────
    GRUModel() {
        float std_in  = sqrtf(2.f / GRU_IN);
        float std_rec = sqrtf(2.f / GRU_H);
        float std_d   = sqrtf(2.f / GRU_H);

        for (int i = 0; i < GRU_H; i++) {
            bz[i] = br[i] = bh[i] = 0.f;
            for (int j = 0; j < GRU_IN; j++) {
                Wz[i][j] = randf() * std_in;
                Wr[i][j] = randf() * std_in;
                Wh[i][j] = randf() * std_in;
            }
            for (int j = 0; j < GRU_H; j++) {
                Uz[i][j] = randf() * std_rec;
                Ur[i][j] = randf() * std_rec;
                Uh[i][j] = randf() * std_rec;
            }
        }
        for (int o = 0; o < GRU_CLS; o++) {
            bd[o] = 0.f;
            for (int j = 0; j < GRU_H; j++)
                Wd[o][j] = randf() * std_d;
        }
    }

    // ── Interface Model ──────────────────────────────────────────────────────

    float* feedForward(const float* x) override {
        // Reshape x[96] → x_seq[8][12]
        for (int t = 0; t < GRU_T; t++)
            for (int c = 0; c < GRU_IN; c++)
                x_seq[t][c] = x[t * GRU_IN + c];

        // GRU forward através de T timesteps
        float h[GRU_H] = {};  // h_0 = zeros

        for (int t = 0; t < GRU_T; t++) {
            const float* xt = x_seq[t];

            // Salva h_{t-1}
            for (int i = 0; i < GRU_H; i++) h_prev[t][i] = h[i];

            // Update gate: z = σ(Wz·x + Uz·h + bz)
            for (int i = 0; i < GRU_H; i++) {
                float s = bz[i];
                for (int j = 0; j < GRU_IN; j++) s += Wz[i][j] * xt[j];
                for (int j = 0; j < GRU_H;  j++) s += Uz[i][j] * h[j];
                z_t[t][i] = sigmoid(s);
            }

            // Reset gate: r = σ(Wr·x + Ur·h + br)
            for (int i = 0; i < GRU_H; i++) {
                float s = br[i];
                for (int j = 0; j < GRU_IN; j++) s += Wr[i][j] * xt[j];
                for (int j = 0; j < GRU_H;  j++) s += Ur[i][j] * h[j];
                r_t[t][i] = sigmoid(s);
            }

            // r ⊙ h_{t-1}
            for (int i = 0; i < GRU_H; i++) r_h[t][i] = r_t[t][i] * h[i];

            // Candidate: h̃ = tanh(Wh·x + Uh·(r⊙h) + bh)
            for (int i = 0; i < GRU_H; i++) {
                float s = bh[i];
                for (int j = 0; j < GRU_IN; j++) s += Wh[i][j] * xt[j];
                for (int j = 0; j < GRU_H;  j++) s += Uh[i][j] * r_h[t][j];
                hc_t[t][i] = tanhf(s);
            }

            // New hidden: h = z ⊙ h_old + (1-z) ⊙ h̃
            for (int i = 0; i < GRU_H; i++) {
                h_t[t][i] = z_t[t][i] * h[i] + (1.f - z_t[t][i]) * hc_t[t][i];
                h[i] = h_t[t][i];
            }
        }

        // Dense na última hidden state h_{T-1}
        for (int o = 0; o < GRU_CLS; o++) {
            float s = bd[o];
            for (int j = 0; j < GRU_H; j++) s += Wd[o][j] * h[j];
            probs[o] = s;
        }
        softmax(probs, GRU_CLS);
        return probs;
    }

    void backProp(const float* y_target) override {
        // ── Gradiente final: dL/dprobs (Softmax+CE) ──────────────────────────
        float dLogits[GRU_CLS];
        float sq_err = 0.f;
        for (int k = 0; k < GRU_CLS; k++) {
            float diff = probs[k] - y_target[k];
            dLogits[k] = diff;
            sq_err += diff * diff;
        }
        sum_sq_err += sq_err;

        // Última hidden state
        const float* hT = h_t[GRU_T - 1];

        // ── Dense backward ────────────────────────────────────────────────────
        float dH[GRU_H] = {};  // gradiente que chega na hidden state final
        for (int o = 0; o < GRU_CLS; o++) {
            bd[o] -= lr_b * dLogits[o];
            for (int j = 0; j < GRU_H; j++) {
                dH[j] += Wd[o][j] * dLogits[o];
                Wd[o][j] -= lr_w * dLogits[o] * hT[j];
            }
        }

        // ── BPTT pela sequência ───────────────────────────────────────────────
        float dH_next[GRU_H];
        for (int i = 0; i < GRU_H; i++) dH_next[i] = dH[i];

        for (int t = GRU_T - 1; t >= 0; t--) {
            const float* xt    = x_seq[t];
            const float* hp    = h_prev[t];    // h_{t-1}
            const float* zt    = z_t[t];
            const float* rt    = r_t[t];
            const float* hct   = hc_t[t];
            const float* rht   = r_h[t];       // r_t ⊙ h_{t-1}
            float* dH_curr     = dH_next;      // recebe gradiente de cima

            // dh_t/dz_t, dh_t/dhc_t
            float dZ[GRU_H], dHc[GRU_H];
            for (int i = 0; i < GRU_H; i++) {
                dZ[i]  = dH_curr[i] * (hp[i] - hct[i]) * sigmoid_d(zt[i]);
                dHc[i] = dH_curr[i] * (1.f - zt[i]) * tanh_d(hct[i]);
            }

            // dL/dbz, dL/dbh
            for (int i = 0; i < GRU_H; i++) {
                bz[i] -= lr_b * dZ[i];
                bh[i] -= lr_b * dHc[i];
            }

            // dL/dWz, dL/dUz, dL/dWh, dL/dUh
            // dUh está na camada candidata: Uh opara em rh = r_t ⊙ h_{t-1}
            float dRH[GRU_H] = {};  // grad em r_t ⊙ h_{t-1}
            for (int i = 0; i < GRU_H; i++) {
                for (int j = 0; j < GRU_IN; j++) {
                    Wz[i][j] -= lr_w * dZ[i]  * xt[j];
                    Wh[i][j] -= lr_w * dHc[i] * xt[j];
                }
                for (int j = 0; j < GRU_H; j++) {
                    Uz[i][j] -= lr_w * dZ[i]  * hp[j];
                    Uh[i][j] -= lr_w * dHc[i] * rht[j];
                    dRH[j]   += dHc[i] * Uh[i][j];
                }
            }

            // grad passa por r_t ⊙ h_{t-1}: dR = dRH ⊙ h_{t-1}
            float dR[GRU_H];
            for (int i = 0; i < GRU_H; i++)
                dR[i] = dRH[i] * hp[i] * sigmoid_d(rt[i]);

            for (int i = 0; i < GRU_H; i++) {
                br[i] -= lr_b * dR[i];
                for (int j = 0; j < GRU_IN; j++)
                    Wr[i][j] -= lr_w * dR[i] * xt[j];
                for (int j = 0; j < GRU_H; j++)
                    Ur[i][j] -= lr_w * dR[i] * hp[j];
            }

            // Gradiente para h_{t-1}: contribuições de z_t, r_t e h_t
            float dH_prev[GRU_H] = {};
            for (int i = 0; i < GRU_H; i++) {
                // via h_t = z ⊙ h_old + ...
                dH_prev[i] += dH_curr[i] * zt[i];
                // via Uz em z gate
                for (int j = 0; j < GRU_H; j++) dH_prev[i] += dZ[j]  * Uz[j][i];
                // via Ur em r gate
                for (int j = 0; j < GRU_H; j++) dH_prev[i] += dR[j]  * Ur[j][i];
                // via rh = r ⊙ h_old → Uh em candidate
                dH_prev[i] += dRH[i] * rt[i];
            }

            for (int i = 0; i < GRU_H; i++) dH_next[i] = dH_prev[i];
        }
    }

    float getMSE(int n) override {
        float mse = (n > 0) ? sum_sq_err / ((float)GRU_CLS * n) : 0.f;
        sum_sq_err = 0.f;
        return mse;
    }

    void setLR(float lr_weights, float lr_biases) override {
        lr_w = lr_weights;
        lr_b = lr_biases;
    }

    void clampParams(float max_val) override {
#define CLAMP_MAT(M, R, C) \
    for (int _i = 0; _i < (R); _i++) \
        for (int _j = 0; _j < (C); _j++) \
            (M)[_i][_j] = clamp((M)[_i][_j], -max_val, max_val);
#define CLAMP_VEC(V, N) \
    for (int _i = 0; _i < (N); _i++) \
        (V)[_i] = clamp((V)[_i], -max_val, max_val);

        CLAMP_MAT(Wz, GRU_H, GRU_IN)  CLAMP_MAT(Wr, GRU_H, GRU_IN)  CLAMP_MAT(Wh, GRU_H, GRU_IN)
        CLAMP_MAT(Uz, GRU_H, GRU_H)   CLAMP_MAT(Ur, GRU_H, GRU_H)   CLAMP_MAT(Uh, GRU_H, GRU_H)
        CLAMP_VEC(bz, GRU_H)           CLAMP_VEC(br, GRU_H)           CLAMP_VEC(bh, GRU_H)
        CLAMP_MAT(Wd, GRU_CLS, GRU_H) CLAMP_VEC(bd, GRU_CLS)

#undef CLAMP_MAT
#undef CLAMP_VEC
    }

private:
    static float randf() {
        return ((float)random(-1000, 1001)) / 1000.f;
    }
};
