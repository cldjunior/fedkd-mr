#pragma once
/**
 * MLPModel.h — Wrapper do Cliente 1 (MLP via NeuralNetwork.h)
 * ============================================================
 * Adapta a interface NeuralNetwork para a interface Model,
 * permitindo que main.cpp use Model* sem saber a arquitetura.
 *
 * Ativa com: #define CLIENT_ID 1
 *
 * Nota: NeuralNetwork.h deve ser incluído ANTES deste header,
 * com os defines de activação (#define ReLU, Softmax, etc.)
 * já declarados em main.cpp.
 *
 * Formato de serialização (saveWeights / loadWeights):
 *   [4 bytes] magic = 0x4D4C5031  ('MLP1')
 *   Para cada camada l = 1..numberOflayers-1:
 *     [nout × 4 bytes] biases
 *     [nout × nin × 4 bytes] weights  (linha i = todos os pesos do neurônio i)
 */

#include "Model.h"
#include <NeuralNetwork.h>

// Magic para validação do arquivo de pesos MLP
static constexpr uint32_t MLP_WEIGHT_MAGIC = 0x4D4C5031u;  // 'MLP1'

class MLPModel : public Model {
    // NeuralNetwork.h armazena o ponteiro `layers` internamente.
    // Mantemos cópias heap aqui para que sobrevivam até o destrutor.
    unsigned int* _layers_mem;
    byte*         _actv_mem;

public:
    NeuralNetwork* nn;

    /**
     * Toma posse dos arrays alocados pelo chamador — NÃO os delete externamente.
     * @param layers   array heap de tamanhos de camadas (ex: {96, 64, 4})
     * @param n_layers número de elementos no array
     * @param actv     array heap de activações (ex: {2, 6} = ReLU, Softmax)
     */
    MLPModel(unsigned int* layers, unsigned int n_layers, byte* actv)
        : _layers_mem(layers), _actv_mem(actv)
    {
        nn = new NeuralNetwork(layers, n_layers, actv);
    }

    ~MLPModel() override {
        delete nn;
        delete[] _layers_mem;
        delete[] _actv_mem;
    }

    // ── Interface Model ──────────────────────────────────────────

    float* feedForward(const float* x) override {
        return (float*)nn->FeedForward((IDFLOAT*)x);
    }

    void backProp(const float* y_target) override {
        nn->BackProp((IDFLOAT*)y_target);
    }

    float getMSE(int n) override {
        return (float)nn->getMeanSqrdError(n > 0 ? (unsigned int)n : 1u);
    }

    void setLR(float lr_weights, float lr_biases) override {
        nn->LearningRateOfWeights = lr_weights;
        nn->LearningRateOfBiases  = lr_biases;
    }

    void clampParams(float max_val) override {
        for (unsigned int l = 1; l < nn->numberOflayers; l++) {
            unsigned int nout = nn->layers[l]._numberOfOutputs;
            unsigned int nin  = nn->layers[l]._numberOfInputs;
            for (unsigned int i = 0; i < nout; i++) {
                float b = (float)nn->layers[l].bias[i];
                nn->layers[l].bias[i] = (IDFLOAT)clamp(b, -max_val, max_val);
                for (unsigned int j = 0; j < nin; j++) {
                    float w = (float)nn->layers[l].weights[i][j];
                    nn->layers[l].weights[i][j] = (IDFLOAT)clamp(w, -max_val, max_val);
                }
            }
        }
    }

    // ── Serialização de pesos ────────────────────────────────────

    bool saveWeights(File& f) override {
        // Magic
        uint32_t magic = MLP_WEIGHT_MAGIC;
        if (f.write((uint8_t*)&magic, 4) != 4) return false;

        for (unsigned int l = 1; l < nn->numberOflayers; l++) {
            unsigned int nout = nn->layers[l]._numberOfOutputs;
            unsigned int nin  = nn->layers[l]._numberOfInputs;

            // Biases
            for (unsigned int i = 0; i < nout; i++) {
                float b = (float)nn->layers[l].bias[i];
                if (f.write((uint8_t*)&b, 4) != 4) return false;
            }

            // Weights [nout][nin]
            for (unsigned int i = 0; i < nout; i++) {
                for (unsigned int j = 0; j < nin; j++) {
                    float w = (float)nn->layers[l].weights[i][j];
                    if (f.write((uint8_t*)&w, 4) != 4) return false;
                }
            }
        }
        return true;
    }

    bool loadWeights(File& f) override {
        // Valida magic
        uint32_t magic = 0;
        if (f.read((uint8_t*)&magic, 4) != 4 || magic != MLP_WEIGHT_MAGIC) {
            Serial.printf("[MLPModel] loadWeights: magic inválido (0x%08X)\n", magic);
            return false;
        }

        for (unsigned int l = 1; l < nn->numberOflayers; l++) {
            unsigned int nout = nn->layers[l]._numberOfOutputs;
            unsigned int nin  = nn->layers[l]._numberOfInputs;

            // Biases
            for (unsigned int i = 0; i < nout; i++) {
                float b = 0.f;
                if (f.read((uint8_t*)&b, 4) != 4) return false;
                nn->layers[l].bias[i] = (IDFLOAT)b;
            }

            // Weights [nout][nin]
            for (unsigned int i = 0; i < nout; i++) {
                for (unsigned int j = 0; j < nin; j++) {
                    float w = 0.f;
                    if (f.read((uint8_t*)&w, 4) != 4) return false;
                    nn->layers[l].weights[i][j] = (IDFLOAT)w;
                }
            }
        }
        return true;
    }
};
