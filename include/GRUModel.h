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
 * Formato de serialização (saveWeights / loadWeights):
 *   [4 bytes] magic = 0x47525531  ('GRU1')
 *   Wz[64][12], Wr[64][12], Wh[64][12]  → 3 × 3072 = 9216 bytes
 *   Uz[64][64], Ur[64][64], Uh[64][64]  → 3 × 16384 = 49152 bytes
 *   bz[64], br[64], bh[64]              → 3 × 256 = 768 bytes
 *   Wd[4][64], bd[4]                    → 1024 + 16 = 1040 bytes
 *   Total: ~60 KB
 */

#include "Model.h"

// Magic para validação do arquivo de pesos GRU
static constexpr uint32_t GRU_WEIGHT_MAGIC = 0x47525531u;  // 'GRU1'

// ── Dimensões ─────────────────────────────────────────────────────────────────
static const int GRU_T   = 8;    // T_WINDOW
static const int GRU_IN  = 12;   // N_FEATURES
static const int GRU_H   = 64;   // hidden_size
static const int GRU_CLS = N_CLASSES; // 4

class GRUModel : public Model {
public:
    // ── Parâmetros GRU ────────────────────────────────────────────────────────
    float Wz[GRU_H][GRU_IN], Wr[GRU_H][GRU_IN], Wh[GRU_H][GRU_IN];
    float Uz[GRU_H][GRU_H],  Ur[GRU_H][GRU_H],  Uh[GRU_H][GRU_H];
    float bz[GRU_H], br[GRU_H], bh[GRU_H];

    // ── Camada densa final ────────────────────────────────────────────────────
    float Wd[GRU_CLS][GRU_H];
    float bd[GRU_CLS];

    // ── Activações por timestep (para BPTT) ──────────────────────────────────
    float z_t[GRU_T][GRU_H];
    float r_t[GRU_T][GRU_H];
    float hc_t[GRU_T][GRU_H];
    float h_t[GRU_T][GRU_H];
    float h_prev[GRU_T][GRU_H];
    float r_h[GRU_T][GRU_H];

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
        for (int t = 0; t < GRU_T; t++)
            for (int c = 0; c < GRU_IN; c++)
                x_seq[t][c] = x[t * GRU_IN + c];

        float h[GRU_H] = {};

        for (int t = 0; t < GRU_T; t++) {
            const float* xt = x_seq[t];
            for (int i = 0; i < GRU_H; i++) h_prev[t][i] = h[i];

            for (int i = 0; i < GRU_H; i++) {
                float s = bz[i];
                for (int j = 0; j < GRU_IN; j++) s += Wz[i][j] * xt[j];
                for (int j = 0; j < GRU_H;  j++) s += Uz[i][j] * h[j];
                z_t[t][i] = sigmoid(s);
            }

            for (int i = 0; i < GRU_H; i++) {
                float s = br[i];
                for (int j = 0; j < GRU_IN; j++) s += Wr[i][j] * xt[j];
                for (int j = 0; j < GRU_H;  j++) s += Ur[i][j] * h[j];
                r_t[t][i] = sigmoid(s);
            }

            for (int i = 0; i < GRU_H; i++) r_h[t][i] = r_t[t][i] * h[i];

            for (int i = 0; i < GRU_H; i++) {
                float s = bh[i];
                for (int j = 0; j < GRU_IN; j++) s += Wh[i][j] * xt[j];
                for (int j = 0; j < GRU_H;  j++) s += Uh[i][j] * r_h[t][j];
                hc_t[t][i] = tanhf(s);
            }

            for (int i = 0; i < GRU_H; i++) {
                h_t[t][i] = z_t[t][i] * h[i] + (1.f - z_t[t][i]) * hc_t[t][i];
                h[i] = h_t[t][i];
            }
        }

        for (int o = 0; o < GRU_CLS; o++) {
            float s = bd[o];
            for (int j = 0; j < GRU_H; j++) s += Wd[o][j] * h[j];
            probs[o] = s;
        }
        softmax(probs, GRU_CLS);
        return probs;
    }

    void backProp(const float* y_target) override {
        float dLogits[GRU_CLS];
        float sq_err = 0.f;
        for (int k = 0; k < GRU_CLS; k++) {
            float diff = probs[k] - y_target[k];
            dLogits[k] = diff;
            sq_err += diff * diff;
        }
        sum_sq_err += sq_err;

        const float* hT = h_t[GRU_T - 1];

        float dH[GRU_H] = {};
        for (int o = 0; o < GRU_CLS; o++) {
            bd[o] -= lr_b * dLogits[o];
            for (int j = 0; j < GRU_H; j++) {
                dH[j] += Wd[o][j] * dLogits[o];
                Wd[o][j] -= lr_w * dLogits[o] * hT[j];
            }
        }

        float dH_next[GRU_H];
        for (int i = 0; i < GRU_H; i++) dH_next[i] = dH[i];

        for (int t = GRU_T - 1; t >= 0; t--) {
            const float* xt  = x_seq[t];
            const float* hp  = h_prev[t];
            const float* zt  = z_t[t];
            const float* rt  = r_t[t];
            const float* hct = hc_t[t];
            const float* rht = r_h[t];
            float* dH_curr   = dH_next;

            float dZ[GRU_H], dHc[GRU_H];
            for (int i = 0; i < GRU_H; i++) {
                dZ[i]  = dH_curr[i] * (hp[i] - hct[i]) * sigmoid_d(zt[i]);
                dHc[i] = dH_curr[i] * (1.f - zt[i]) * tanh_d(hct[i]);
            }

            for (int i = 0; i < GRU_H; i++) {
                bz[i] -= lr_b * dZ[i];
                bh[i] -= lr_b * dHc[i];
            }

            float dRH[GRU_H] = {};
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

            float dH_prev[GRU_H] = {};
            for (int i = 0; i < GRU_H; i++) {
                dH_prev[i] += dH_curr[i] * zt[i];
                for (int j = 0; j < GRU_H; j++) dH_prev[i] += dZ[j]  * Uz[j][i];
                for (int j = 0; j < GRU_H; j++) dH_prev[i] += dR[j]  * Ur[j][i];
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

    // ── Serialização de pesos ────────────────────────────────────────────────

    bool saveWeights(File& f) override {
        uint32_t magic = GRU_WEIGHT_MAGIC;
        if (f.write((uint8_t*)&magic, 4) != 4) return false;
        // Matrizes de entrada
        if (write_floats(f, &Wz[0][0], GRU_H * GRU_IN) != GRU_H * GRU_IN * 4u) return false;
        if (write_floats(f, &Wr[0][0], GRU_H * GRU_IN) != GRU_H * GRU_IN * 4u) return false;
        if (write_floats(f, &Wh[0][0], GRU_H * GRU_IN) != GRU_H * GRU_IN * 4u) return false;
        // Matrizes recorrentes
        if (write_floats(f, &Uz[0][0], GRU_H * GRU_H)  != GRU_H * GRU_H  * 4u) return false;
        if (write_floats(f, &Ur[0][0], GRU_H * GRU_H)  != GRU_H * GRU_H  * 4u) return false;
        if (write_floats(f, &Uh[0][0], GRU_H * GRU_H)  != GRU_H * GRU_H  * 4u) return false;
        // Biases GRU
        if (write_floats(f, bz, GRU_H) != GRU_H * 4u) return false;
        if (write_floats(f, br, GRU_H) != GRU_H * 4u) return false;
        if (write_floats(f, bh, GRU_H) != GRU_H * 4u) return false;
        // Dense
        if (write_floats(f, &Wd[0][0], GRU_CLS * GRU_H) != (size_t)GRU_CLS * GRU_H * 4u) return false;
        if (write_floats(f, bd,         GRU_CLS)          != (size_t)GRU_CLS          * 4u) return false;
        return true;
    }

    bool loadWeights(File& f) override {
        uint32_t magic = 0;
        if (f.read((uint8_t*)&magic, 4) != 4 || magic != GRU_WEIGHT_MAGIC) {
            Serial.printf("[GRUModel] loadWeights: magic inválido (0x%08X)\n", magic);
            return false;
        }
        if (read_floats(f, &Wz[0][0], GRU_H * GRU_IN) != GRU_H * GRU_IN * 4u) return false;
        if (read_floats(f, &Wr[0][0], GRU_H * GRU_IN) != GRU_H * GRU_IN * 4u) return false;
        if (read_floats(f, &Wh[0][0], GRU_H * GRU_IN) != GRU_H * GRU_IN * 4u) return false;
        if (read_floats(f, &Uz[0][0], GRU_H * GRU_H)  != GRU_H * GRU_H  * 4u) return false;
        if (read_floats(f, &Ur[0][0], GRU_H * GRU_H)  != GRU_H * GRU_H  * 4u) return false;
        if (read_floats(f, &Uh[0][0], GRU_H * GRU_H)  != GRU_H * GRU_H  * 4u) return false;
        if (read_floats(f, bz, GRU_H) != GRU_H * 4u) return false;
        if (read_floats(f, br, GRU_H) != GRU_H * 4u) return false;
        if (read_floats(f, bh, GRU_H) != GRU_H * 4u) return false;
        if (read_floats(f, &Wd[0][0], GRU_CLS * GRU_H) != (size_t)GRU_CLS * GRU_H * 4u) return false;
        if (read_floats(f, bd,         GRU_CLS)          != (size_t)GRU_CLS          * 4u) return false;
        return true;
    }

private:
    static float randf() {
        return ((float)random(-1000, 1001)) / 1000.f;
    }
};
