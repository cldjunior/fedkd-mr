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
 */

#include "Model.h"
#include <NeuralNetwork.h>

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
        // NeuralNetwork::FeedForward aceita IDFLOAT* (= float*)
        // Casting direto é seguro pois IDFLOAT == float no ESP32
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
        // l=0 é a camada de entrada — sem weights/bias alocados; começa em 1
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
};
