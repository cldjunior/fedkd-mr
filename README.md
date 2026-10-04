# FedKD-MR — Federated Knowledge Distillation with Model Randomization

Sistema de aprendizado federado heterogêneo para detecção de presença com radar HLK-LD2410C.  
Combina ESP32s (firmware C++) e Raspberry Pis / notebooks (Python + PyTorch) em uma federação coordenada por MQTT.

---

## Visão geral da arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                    Broker MQTT (Mosquitto)               │
│                      192.168.0.12:1883                  │
└───────┬──────────────────────┬──────────────────────────┘
        │                      │
        ▼                      ▼
┌───────────────┐    ┌──────────────────────────────────┐
│  fedkd_server │    │  fedkd_monitor                   │
│  (Python)     │    │  (Python — dashboard http :8080) │
└───────┬───────┘    └──────────────────────────────────┘
        │ publica teacher_probs + cmd/start
        │ recebe logits + cmd/done
        │
   ┌────┴─────────────────────────────────────────┐
   │                                              │
   ▼                                              ▼
ESP32 #1-3 (MLP em C++)          RPi #4-8 (Python + PyTorch)
  client_id 1, 2, 3                GRU · LSTM · CNN1D · CNN2D · MLP
  firmware/main.cpp                rpi_client/fedkd_client.py
```

### Protocolo MQTT por tópico

| Tópico | Direção | Formato | Conteúdo |
|--------|---------|---------|----------|
| `fedkd/client/{id}/logits/push` | cliente → servidor | binary float32 | probabilidades (n_pub × n_classes) |
| `fedkd/client/{id}/teacher/pull` | servidor → cliente | binary float32 | teacher_probs agregados |
| `fedkd/client/{id}/cmd/pull` | servidor → cliente | JSON | `{"cmd":"start","round":N}` |
| `fedkd/client/{id}/cmd/push` | cliente → servidor | JSON | done + métricas completas |

---

## Estrutura de arquivos

```
FedKD-MR/
│
├── prepare_dataset.py          # gera todos os .bin + metadata.json
│
├── server/
│   ├── fedkd_server.py         # servidor de agregação (não modificar)
│   └── fedkd_monitor.py        # dashboard web em tempo real
│
├── rpi_client/
│   ├── fedkd_client.py         # pipeline principal do cliente Python
│   ├── mqtt_handler.py         # protocolo MQTT (serialização binária)
│   ├── models.py               # GRU · LSTM · CNN1D · CNN2D · MLP
│   ├── dataset.py              # leitura de .bin + DataLoader
│   ├── config.py               # carregamento de YAML + defaults
│   ├── requirements.txt
│   └── configs/
│       ├── client_gru.yaml     # RPi #4 — GRU
│       ├── client_lstm.yaml    # RPi #5 — LSTM
│       ├── client_cnn1d.yaml   # RPi #6 — CNN1D
│       ├── client_cnn2d.yaml   # RPi #7 — CNN2D
│       └── client_mlp.yaml     # RPi #8 — MLP
│
├── firmware/
│   ├── main.cpp                # firmware ESP32 (PlatformIO)
│   ├── Config.h / Config_*.h   # configuração por cliente ESP32
│   ├── Model.h / MLPModel.h / GRUModel.h / CNN1DModel.h
│   ├── platformio.ini
│   └── README.md               # instruções específicas do firmware
│
└── output_fedkd/               # gerado pelo prepare_dataset.py
    ├── client_{1..8}/
    │   ├── dataset_priv.bin    # D_priv do cliente (treino local)
    │   └── metadata.json
    ├── shared/
    │   ├── dataset_pub.bin     # D_pub (mesmo para todos os clientes)
    │   ├── metadata.json
    │   ├── teacher_probs.bin   # consenso inicial (simulado)
    │   └── teacher_probs_meta.json
    └── test/
        └── dataset_test.bin    # avaliação independente
```

---

## Requisitos

### Servidor e monitor (notebook / PC)
```bash
pip install paho-mqtt websockets
```

### Cliente Python (RPi ou notebook simulando RPi)
```bash
cd rpi_client/
pip install -r requirements.txt
# PyTorch em ARM (Raspberry Pi 4 aarch64):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Broker MQTT
```bash
sudo apt install mosquitto mosquitto-clients
sudo systemctl enable --now mosquitto
```

---

## Passo a passo completo

### 1. Gerar os datasets

```bash
# Sintético — 200 amostras/classe, 8 clientes, não-IID, ruído realista
python prepare_dataset.py \
    --n_synth 200 \
    --n_clients 8 \
    --non_iid \
    --noise_scale 2.5 \
    --n_pub_per_class 20 \
    --output output_fedkd

# Com Excel real:
python prepare_dataset.py \
    --input ld2410c_dataset.xlsx \
    --n_clients 8 \
    --non_iid \
    --n_pub_per_class 20 \
    --output output_fedkd
```

> **`--noise_scale`**: 1.0 = classes bem separadas (acurácia ~100%),
> 2.0–3.0 = classes sobrepostas (acurácia 70–85%, mais realista para avaliar KD).  
> **`--n_pub_per_class 20`**: gera 80 amostras no D_pub (20 × 4 classes), necessário para um KD efetivo nos ESP32.

Saída: `output_fedkd/client_{1..8}/`, `output_fedkd/shared/`, `output_fedkd/test/`.

### 2. Distribuir os dados

**Para os ESP32** (via PlatformIO → Upload Filesystem Image):
```
output_fedkd/client_1/dataset_priv.bin      → firmware/data/dataset_priv.bin
output_fedkd/client_1/metadata.json         → firmware/data/metadata.json
output_fedkd/shared/dataset_pub.bin         → firmware/data/dataset_pub.bin
output_fedkd/shared/teacher_probs.bin       → firmware/data/teacher_probs.bin
output_fedkd/shared/teacher_probs_meta.json → firmware/data/teacher_probs_meta.json
```
Repita para ESP32 #2 (`client_2/`) e #3 (`client_3/`). Use sempre o `metadata.json` do `client_X/` — ele tem prioridade sobre o do `shared/`.

**Para cada instância RPi** (ou pasta local no notebook):
```bash
# Exemplo para o cliente GRU (id=4):
mkdir -p rpi_client/data_client4/
cp output_fedkd/client_4/dataset_priv.bin  rpi_client/data_client4/
cp output_fedkd/client_4/metadata.json     rpi_client/data_client4/
cp output_fedkd/shared/dataset_pub.bin     rpi_client/data_client4/
cp output_fedkd/shared/metadata.json       rpi_client/data_client4/  # mesmo arquivo
cp output_fedkd/test/dataset_test.bin      rpi_client/data_client4/  # avaliação
```

Aponte o `data_dir` no YAML para a pasta correspondente:
```yaml
data_dir: data_client4/
```

### 3. Iniciar o servidor

```bash
cd server/
python fedkd_server.py
```

O servidor aguarda conexões dos clientes registrados antes de enviar o primeiro `cmd/start`.

### 4. Abrir o monitor (opcional, mas recomendado)

```bash
cd server/
python fedkd_monitor.py --broker 192.168.0.12 --http-port 8080
```

Abra `http://localhost:8080` no navegador. O monitor é passivo — apenas observa.

### 5. Iniciar os clientes Python

Cada cliente em um terminal separado:

```bash
cd rpi_client/
python fedkd_client.py --config configs/client_gru.yaml   # RPi #4
python fedkd_client.py --config configs/client_lstm.yaml  # RPi #5
python fedkd_client.py --config configs/client_cnn1d.yaml # RPi #6
python fedkd_client.py --config configs/client_cnn2d.yaml # RPi #7
python fedkd_client.py --config configs/client_mlp.yaml   # RPi #8
```

Ou todos de uma vez em background:
```bash
for cfg in gru lstm cnn1d cnn2d mlp; do
    python fedkd_client.py --config configs/client_${cfg}.yaml \
        > logs/client_${cfg}.log 2>&1 &
done
```

---

## Configuração YAML dos clientes RPi

Todos os YAMLs compartilham os mesmos campos base:

```yaml
client_id:  4              # único por cliente (4–8 para RPi)
model:      gru            # gru | lstm | cnn1d | cnn2d | mlp

data_dir:   data_client4/  # pasta com os .bin deste cliente
train_file: dataset_priv.bin
pub_file:   dataset_pub.bin
meta_file:  metadata.json

t_window:   8              # timesteps por janela
n_features: 12             # features por timestep
n_classes:  4              # EMPTY · STATIONARY · APPROACHING · LEAVING

local_epochs: 50;  local_lr: 0.001
kd_epochs:    20;  kd_lr:    0.0005;  kd_temperature: 1.0
ft_epochs:    10;  ft_lr:    0.0005

batch_size: 32

broker:          "192.168.0.12"
port:            1883
keepalive:       600
teacher_timeout: 600

device: cpu
seed:   42
```

Parâmetros específicos por arquitetura:

| Modelo | Parâmetros extras |
|--------|-------------------|
| GRU / LSTM | `rnn_hidden: 128`, `rnn_layers: 2`, `rnn_dropout: 0.3`, `rnn_bidirectional: false` |
| CNN1D / CNN2D | `cnn_channels: [32, 64, 128]`, `cnn_kernel: 3`, `cnn_dropout: 0.3` |
| MLP | `mlp_hidden: [128, 64]` |

---

## Pipeline por round (cliente Python)

```
Round N
  │
  ├─ [1] Treino local supervisionado  (D_priv,  local_epochs)
  ├─ [2] Inferência em D_pub          → publica softmax probs via MQTT
  ├─ [3] Aguarda teacher_probs        ← recebe agregação do servidor
  ├─ [4] Knowledge Distillation       (D_pub + teacher_probs, kd_epochs)
  ├─ [5] Fine-tuning supervisionado   (D_priv,  ft_epochs)
  └─ [6] Avaliação em D_test + publica done → métricas completas
```

> **KD Loss**: soft cross-entropy `−Σ teacher_probs · log_p_student · T²`  
> Numericamente estável: `nan_to_num` + `log_softmax.clamp(min=-100)` + gradient clip (norm=5).

---

## Métricas geradas

O servidor grava dois CSVs em `logs/`:

### `logs/metrics.csv` — por cliente por round

| Campo | Descrição |
|-------|-----------|
| `acc_local/kd/ft` | Acurácia no **D_test** após cada fase |
| `f1_local/kd/ft` | F1 macro no **D_test** após cada fase |
| `loss_local/kd/ft` | Loss média da última época de cada fase |
| `t_local/kd/ft_ms` | Duração de cada fase em milissegundos |
| `delta_acc_kd` | `acc_kd − acc_local` (ganho do KD) |
| `delta_acc_ft` | `acc_ft − acc_kd` (ganho do fine-tuning) |

> Campos ESP32 (`heap_free`) chegam com valor; campos RPi sem equivalente chegam como `null`.

### `logs/consensus.csv` — por round (agregação global)

| Campo | Descrição |
|-------|-----------|
| `entropy_mean/min/max` | Entropia das teacher_probs (↓ = mais confiante) |
| `kl_mean/max` | KL(cliente ‖ teacher) — divergência do consenso |
| `cosine_mean/min` | Similaridade cosseno entre logits dos clientes |
| `t_round_ms` | Duração total do round |

**Convergência esperada**: `kl_mean` cai ~10× em 6 rounds; `cosine_mean` → 1.0.

---

## Identificação de clientes

| ID | Tipo | Modelo | `client_name` |
|----|------|--------|---------------|
| 1 | ESP32 | MLP pequeno | `FedKD-ESP32-1` |
| 2 | ESP32 | MLP médio | `FedKD-ESP32-2` |
| 3 | ESP32 | MLP maior | `FedKD-ESP32-3` |
| 4 | RPi/Python | GRU | `FedKD-RPi-4` |
| 5 | RPi/Python | LSTM | `FedKD-RPi-5` |
| 6 | RPi/Python | CNN1D | `FedKD-RPi-6` |
| 7 | RPi/Python | CNN2D | `FedKD-RPi-7` |
| 8 | RPi/Python | MLP | `FedKD-RPi-8` |

---

## Dicas e troubleshooting

**Acurácia sempre 100%?**  
O dataset sintético padrão é fácil. Use `--noise_scale 2.5` ou maior para forçar sobreposição entre classes e obter métricas mais informativas.

**NaN na KD loss?**  
Já tratado no código: `nan_to_num` nos teacher_probs + `log_softmax.clamp(-100)` + gradient clip. Se aparecer, verifique se o servidor está enviando probabilidades (0–1) e não logits brutos.

**Cliente não recebe `cmd/start`?**  
Verifique se o `client_id` no YAML bate com o registrado no servidor (`fedkd_server.py` → lista de clientes esperados).

**Timeout aguardando teacher_probs?**  
Aumente `teacher_timeout` no YAML (padrão 600 s para GRU/LSTM). O servidor só envia após receber logits de **todos** os clientes registrados.

**RPi com PyTorch lento?**  
Normal para modelos recorrentes em CPU. Reduza `local_epochs` / `kd_epochs` ou use `rnn_layers: 1` no YAML.

---

## Referências

- Dataset sintético: gerador HLK-LD2410C em `prepare_dataset.py` (`generate_synthetic_windows`)
- Protocolo binário: formato Atlântico — float32 row-major, label uint8 (1-based)
- KD: Hinton et al., *Distilling the Knowledge in a Neural Network* (2015)
- Agregação: média ponderada de log-probs + softmax com temperatura
