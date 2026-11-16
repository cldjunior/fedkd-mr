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
 *   [4 bytes] magic = 0x4D4C5032 ('MLP2'; incompatível com MLP1 incompleto)
 *   Para cada camada treinável l = 0..numberOflayers-1:
 *     [1 × 4 bytes] bias no modo SINGLE_BIAS_PER_LAYER (atual)
 *     [nout × 4 bytes] biases no modo MULTIPLE_BIASES_PER_LAYER
 *     [nout × nin × 4 bytes] weights (linha i = pesos do neurônio i)
 *   MLP 96-64-4 com SINGLE_BIAS_PER_LAYER: 25612 bytes (float32).
 */

#include "Model.h"
#include <NeuralNetwork.h>

// Magic para validação do arquivo de pesos MLP
static constexpr uint32_t MLP_WEIGHT_MAGIC = 0x4D4C5032u;  // 'MLP2'

#if !defined(SINGLE_BIAS_PER_LAYER) && !defined(MULTIPLE_BIASES_PER_LAYER) && !defined(NO_BIAS)
#error "MLPModel: formato de bias desconhecido"
#endif

class MLPModel : public Model {
    // NeuralNetwork.h armazena o ponteiro `layers` internamente.
    // Mantemos cópias heap aqui para que sobrevivam até o destrutor.
    unsigned int* _layers_mem;
    byte*         _actv_mem;

public:
    NeuralNetwork* nn;
    float _last_logits[N_CLASSES] = {};
    float _last_input[N_INPUT] = {};
    bool _logits_ready = false;

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
        for(int i=0;i<N_INPUT;++i) _last_input[i]=x[i];
        float* result = (float*)nn->FeedForward((IDFLOAT*)x);
        // Esta configuracao MLP usa ReLU na camada oculta e Softmax na saida.
        // Reconstruir a ultima transformacao afim SEM aplicar softmax,
        // utilizando os parametros treinados reais da NeuralNetwork.h.
        const unsigned int last = nn->numberOflayers - 1;
        if (last != 1 || nn->layers[last]._numberOfOutputs != N_CLASSES ||
            nn->layers[0]._numberOfOutputs != 64) {
            _logits_ready = false;
            return result;
        }
        // Reconstroi a camada escondida a partir da entrada. Isso evita depender
        // de buffers internos de ativacao que a biblioteca pode descartar.
        float hidden[64];
        for (unsigned int j=0; j<64; ++j) {
#if defined(SINGLE_BIAS_PER_LAYER)
            float a = (float)nn->layers[0].bias[0];
#elif defined(MULTIPLE_BIASES_PER_LAYER)
            float a = (float)nn->layers[0].bias[j];
#else
            float a = 0.f;
#endif
            for (unsigned int i=0; i<nn->layers[0]._numberOfInputs; ++i)
                a += (float)nn->layers[0].weights[j][i] * x[i];
            hidden[j] = a > 0.f ? a : 0.f; // ReLU
        }
        for (unsigned int k=0; k<N_CLASSES; ++k) {
#if defined(SINGLE_BIAS_PER_LAYER)
            float z = (float)nn->layers[last].bias[0];
#elif defined(MULTIPLE_BIASES_PER_LAYER)
            float z = (float)nn->layers[last].bias[k];
#else
            float z = 0.f;
#endif
            for (unsigned int j=0; j<64; ++j)
                z += (float)nn->layers[last].weights[k][j] * hidden[j];
            _last_logits[k] = z;
        }
        _logits_ready = true;
        return result;
    }

    bool getLogits(float out[N_CLASSES]) const override {
        if (!_logits_ready) return false;
        for (int k=0; k<N_CLASSES; ++k) out[k] = _last_logits[k];
        return true;
    }

    void backProp(const float* y_target) override {
        nn->BackProp((IDFLOAT*)y_target);
    }

    float backPropKD(const float* q, float temperature) override {
        if (!_logits_ready || nn->numberOflayers != 2) return NAN;
        float dz[N_CLASSES], loss=0.f;
        if (!KDMath::lossGradient(_last_logits,q,N_CLASSES,temperature,dz,loss)) return NAN;
        auto &hidden = nn->layers[0];
        auto &output = nn->layers[1];
        const unsigned int nh=hidden._numberOfOutputs;
        if(nh != 64) return NAN;
        // Use the original weights when propagating gradient to the hidden layer.
        float dh[64]={};
        for(unsigned int j=0;j<nh;++j)
            for(int k=0;k<N_CLASSES;++k) dh[j]+=(float)output.weights[k][j]*dz[k];
        float dbout=0.f;
        for(int k=0;k<N_CLASSES;++k) {
            dbout+=dz[k];
            for(unsigned int j=0;j<nh;++j)
                output.weights[k][j]-=(IDFLOAT)(nn->LearningRateOfWeights*dz[k]*(float)hidden.outputs[j]);
        }
#if defined(SINGLE_BIAS_PER_LAYER)
        output.bias[0]-=(IDFLOAT)(nn->LearningRateOfBiases*dbout);
#elif defined(MULTIPLE_BIASES_PER_LAYER)
        for(int k=0;k<N_CLASSES;++k) output.bias[k]-=(IDFLOAT)(nn->LearningRateOfBiases*dz[k]);
#endif
        float dbhidden=0.f;
        const float *input=(const float*)_last_input;
        for(unsigned int j=0;j<nh;++j) {
            float d=hidden.outputs[j]>0.f?dh[j]:0.f;
            dbhidden+=d;
            for(unsigned int i=0;i<hidden._numberOfInputs;++i)
                hidden.weights[j][i]-=(IDFLOAT)(nn->LearningRateOfWeights*d*input[i]);
#if defined(MULTIPLE_BIASES_PER_LAYER)
            hidden.bias[j]-=(IDFLOAT)(nn->LearningRateOfBiases*d);
#endif
        }
#if defined(SINGLE_BIAS_PER_LAYER)
        hidden.bias[0]-=(IDFLOAT)(nn->LearningRateOfBiases*dbhidden);
#endif
        _logits_ready=false;
        return loss;
    }

    float getMSE(int n) override {
        return (float)nn->getMeanSqrdError(n > 0 ? (unsigned int)n : 1u);
    }

    void setLR(float lr_weights, float lr_biases) override {
        nn->LearningRateOfWeights = lr_weights;
        nn->LearningRateOfBiases  = lr_biases;
    }

    void clampParams(float max_val) override {
        for (unsigned int l=0;l<nn->numberOflayers;++l) {
            auto &layer=nn->layers[l];
            const unsigned int nout=layer._numberOfOutputs, nin=layer._numberOfInputs;
#if defined(SINGLE_BIAS_PER_LAYER)
            layer.bias[0]=(IDFLOAT)clamp((float)layer.bias[0],-max_val,max_val);
#elif defined(MULTIPLE_BIASES_PER_LAYER)
            for(unsigned int i=0;i<nout;++i)
                layer.bias[i]=(IDFLOAT)clamp((float)layer.bias[i],-max_val,max_val);
#endif
            for(unsigned int i=0;i<nout;++i)
                for(unsigned int j=0;j<nin;++j)
                    layer.weights[i][j]=(IDFLOAT)clamp((float)layer.weights[i][j],-max_val,max_val);
        }
    }

    // ── Serialização de pesos ────────────────────────────────────
    // Contrato MLP2: todas as camadas treináveis, na ordem da biblioteca.
    // NÃO modifica o modelo durante o salvamento.

    size_t serializedSize() const {
        size_t total = sizeof(uint32_t);
        for (unsigned int l = 0; l < nn->numberOflayers; ++l) {
            const unsigned int nout = nn->layers[l]._numberOfOutputs;
            const unsigned int nin  = nn->layers[l]._numberOfInputs;
#if defined(SINGLE_BIAS_PER_LAYER)
            total += sizeof(float);
#elif defined(MULTIPLE_BIASES_PER_LAYER)
            total += (size_t)nout * sizeof(float);
#endif
            total += (size_t)nout * nin * sizeof(float);
        }
        return total;
    }

    bool saveWeights(File& f) override {
        if (!f) return false;
        const uint32_t magic = MLP_WEIGHT_MAGIC;
        if (f.write((const uint8_t*)&magic, sizeof(magic)) != sizeof(magic)) return false;

        for (unsigned int l = 0; l < nn->numberOflayers; ++l) {
            const unsigned int nout = nn->layers[l]._numberOfOutputs;
            const unsigned int nin  = nn->layers[l]._numberOfInputs;
#if defined(SINGLE_BIAS_PER_LAYER)
            const float b = (float)nn->layers[l].bias[0];
            if (f.write((const uint8_t*)&b, sizeof(b)) != sizeof(b)) return false;
#elif defined(MULTIPLE_BIASES_PER_LAYER)
            for (unsigned int i = 0; i < nout; ++i) {
                const float b = (float)nn->layers[l].bias[i];
                if (f.write((const uint8_t*)&b, sizeof(b)) != sizeof(b)) return false;
            }
#endif
            for (unsigned int i = 0; i < nout; ++i) {
                for (unsigned int j = 0; j < nin; ++j) {
                    const float w = (float)nn->layers[l].weights[i][j];
                    if (f.write((const uint8_t*)&w, sizeof(w)) != sizeof(w)) return false;
                }
            }
        }
        return true;
    }

    bool loadWeights(File& f) override {
        if (!f) return false;
        // Rejeita arquivo truncado/de formato antigo antes de modificar parâmetros.
        const size_t start = f.position();
        const size_t expected = serializedSize();
        if (f.size() < start || f.size() - start != expected) {
            Serial.printf("[MLPModel] loadWeights: tamanho incorreto (%u; esperado %u)\n",
                          (unsigned)(f.size() - (f.size() < start ? f.size() : start)),
                          (unsigned)expected);
            return false;
        }
        uint32_t magic = 0;
        if (f.read((uint8_t*)&magic, sizeof(magic)) != sizeof(magic) || magic != MLP_WEIGHT_MAGIC) {
            Serial.printf("[MLPModel] loadWeights: formato nao suportado (0x%08lX)\n",
                          (unsigned long)magic);
            return false;
        }

        for (unsigned int l = 0; l < nn->numberOflayers; ++l) {
            const unsigned int nout = nn->layers[l]._numberOfOutputs;
            const unsigned int nin  = nn->layers[l]._numberOfInputs;
#if defined(SINGLE_BIAS_PER_LAYER)
            float b = 0.f;
            if (f.read((uint8_t*)&b, sizeof(b)) != sizeof(b)) return false;
            nn->layers[l].bias[0] = (IDFLOAT)b;
#elif defined(MULTIPLE_BIASES_PER_LAYER)
            for (unsigned int i = 0; i < nout; ++i) {
                float b = 0.f;
                if (f.read((uint8_t*)&b, sizeof(b)) != sizeof(b)) return false;
                nn->layers[l].bias[i] = (IDFLOAT)b;
            }
#endif
            for (unsigned int i = 0; i < nout; ++i) {
                for (unsigned int j = 0; j < nin; ++j) {
                    float w = 0.f;
                    if (f.read((uint8_t*)&w, sizeof(w)) != sizeof(w)) return false;
                    nn->layers[l].weights[i][j] = (IDFLOAT)w;
                }
            }
        }
        return true;
    }
};
