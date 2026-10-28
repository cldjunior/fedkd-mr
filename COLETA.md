# FedKD-MR — Coleta de Dados Reais com o HLK-LD2410C

## 1. Conexão hardware (obrigatória antes de qualquer coisa)

```
  HLK-LD2410C                ESP32 DevKit
  ─────────────────────────────────────────────────────────────
  Pino 1  TX (UART out) ───→  GPIO 16  (RX2)  [nível 3.3 V — direta]
  Pino 2  RX (UART in)  ←───  GPIO 17  (TX2)  [nível 3.3 V — direta]
  Pino 3  OUT           ───→  GPIO 18  (input, opcional)
  Pino 4  GND           ───   GND
  Pino 5  VCC (5 V)     ───   VIN  (5 V quando ESP32 alimentado por USB)
  ─────────────────────────────────────────────────────────────
```

> **Atenção sobre alimentação:** O sensor precisa de 5 V / >200 mA.
> Use um cabo USB de qualidade diretamente no computador (não hub sem
> alimentação). Se o sensor resetar sozinho, providencie alimentação
> externa de 5 V ligada ao GND comum.
>
> **Nível lógico:** O sensor opera em 3.3 V — sem divisor de tensão
> necessário. A conexão TX/RX é direta ao ESP32.

### Identificação dos pinos na placa

A serigrafia no módulo LD2410C já indica: `TX RX OUT GND VCC`.
Os pinos estão na ordem da esquerda para a direita quando o radar
(patch de antena) está voltado para você.

---

## 2. Estrutura dos arquivos entregues

```
firmware/collector/
  main_collector.cpp          ← firmware de coleta (on-device)
  platformio_collector.ini    ← configuração PlatformIO separada

collect_dataset.py            ← alternativa: coleta via PC (ver seção 5)
```

---

## 3. Preparação do firmware de coleta

### 3.1 Copie o platformio_collector.ini

Coloque-o **na raiz do projeto** (onde está o `platformio.ini` atual)
como um ambiente extra. Edite o `platformio.ini` existente e **adicione**
o bloco `[env:collector]`:

```ini
; — Cole no final do seu platformio.ini existente —

[env:collector]
platform     = espressif32
board        = esp32dev
framework    = arduino
monitor_speed = 115200
upload_speed  = 921600

build_src_filter =
    -<*>
    +<firmware/LD2410C.cpp>
    +<firmware/collector/main_collector.cpp>

lib_deps =
    bblanchon/ArduinoJson @ ^7.0.0

board_build.filesystem = littlefs
build_flags = -DCORE_DEBUG_LEVEL=0
```

### 3.2 Grave o LittleFS vazio (só na primeira vez)

Se o ESP32 ainda não tem a partição LittleFS inicializada:

```bash
pio run -e collector -t uploadfs   # formata e grava LittleFS vazio
```

> **Cuidado:** `uploadfs` apaga tudo no LittleFS.
> Se já existirem arquivos de datasets sintéticos que você quer
> preservar, pule este passo — o firmware de coleta abre o arquivo
> em modo append.

### 3.3 Grave o firmware de coleta

```bash
pio run -e collector -t upload
```

---

## 4. Coletando dados

### 4.1 Abra o monitor serial

```bash
pio device monitor -e collector    # ou use o Serial Monitor da IDE
```

Você verá a mensagem de boot e a confirmação de Engineering Mode.

### 4.2 Comandos durante a coleta

| Tecla | Ação |
|-------|------|
| `e`   | Label: EMPTY (sala vazia) |
| `s`   | Label: STATIONARY (pessoa parada) |
| `a`   | Label: APPROACHING (pessoa se aproximando) |
| `l`   | Label: LEAVING (pessoa se afastando) |
| `p`   | Pausar / retomar |
| `d`   | Finalizar (imprime estatísticas) |
| `r`   | Apagar dataset_priv.bin (recomeçar) |
| `?`   | Ver status atual |

### 4.3 Procedimento de coleta recomendado

Para cada classe, recomenda-se **mínimo 50 amostras** (idealmente 100+).
Cada janela = 8 frames × ~100 ms = ~0.8 segundos de sensor.

```
EMPTY (e):
  - Retire-se do campo do sensor (>5 m de distância ou fora do ângulo ±60°)
  - Envie 'e' → aguarde 50-100 janelas serem gravadas
  - Envie 'p' para pausar

STATIONARY (s):
  - Posicione-se a ~1-2 m do sensor, fique imóvel (respire normalmente)
  - Envie 's' → aguarde 50-100 janelas
  - Tente distâncias variadas (0.5m, 1m, 2m, 3m) para cobrir a variação

APPROACHING (a):
  - Inicie a ~3-4 m do sensor, caminhe em direção a ele até ~0.5 m
  - Envie 'a' → cada passagem contribui com algumas janelas
  - Repita o percurso até coletar 50+ janelas

LEAVING (l):
  - Inicie a ~0.5 m do sensor, caminhe se afastando até ~3-4 m
  - Envie 'l' → repita até 50+ janelas

Dica: pause (p) sempre que trocar de cena para evitar janelas
      com transição de comportamento.
```

### 4.4 Verificar progresso

Envie `?` a qualquer momento para ver contagem por classe e tamanho do arquivo.

### 4.5 Finalizar

Quando satisfeito com a quantidade de dados por classe, envie `d`.
O firmware imprime um resumo e para de gravar.

---

## 5. Pós-coleta: transição para o firmware FedKD

### 5.1 Grave o firmware principal sem apagar o LittleFS

```bash
# Escolha o ambiente do seu ESP32 (mlp, cnn1d ou gru)
pio run -e mlp -t upload           # NÃO use -t uploadfs aqui!
```

O `dataset_priv.bin` gravado durante a coleta permanece intacto na flash.

### 5.2 D_pub e teacher_probs (ainda sintéticos)

O `dataset_pub.bin` e `teacher_probs.bin` compartilhados continuam
sendo gerados pelo `prepare_dataset.py`. Na primeira rodada com dados
reais, use:

```bash
python prepare_dataset.py \
    --n_synth 200 \
    --n_clients 8 \
    --non_iid \
    --noise_scale 2.5 \
    --n_pub_per_class 20 \
    --output output_fedkd
```

Depois copie **somente** `output_fedkd/shared/` para a LittleFS de cada
ESP32 (via `uploadfs` separado, mas isso sobrescreveria o `dataset_priv.bin`).

**Solução prática:** grave o `dataset_pub.bin` e `teacher_probs.bin`
pelo script de upload parcial abaixo, ou use o `collect_dataset.py`
(seção 5.3) que gera tudo junto.

#### Upload seletivo de arquivos para LittleFS (sem apagar o restante)

Use `ampy` ou `littlefs-python` para enviar arquivos individualmente:

```bash
pip install adafruit-ampy
# Sobe apenas o D_pub (sem apagar dataset_priv.bin):
ampy --port COM3 put output_fedkd/shared/dataset_pub.bin /dataset_pub.bin
ampy --port COM3 put output_fedkd/shared/teacher_probs.bin /teacher_probs.bin
ampy --port COM3 put output_fedkd/shared/teacher_probs_meta.json /teacher_probs_meta.json
ampy --port COM3 put output_fedkd/client_1/metadata.json /metadata.json
```

> Ajuste `COM3` para a porta do seu ESP32 e `client_1` para o ID correto.

### 5.3 Alternativa: collect_dataset.py (coleta via PC)

Se preferir coletar os dados no PC em vez de on-device:

```bash
pip install pyserial numpy scikit-learn
python collect_dataset.py --port COM3 --n_clients 8 --n_pub_per_class 20
```

O script gera o `output_fedkd_real/` completo (incluindo D_pub sintético
e teacher_probs) e você distribui os arquivos manualmente para cada ESP32.

---

## 6. Features coletadas

O firmware extrai 12 features por timestep do frame Engineering Mode do LD2410C:

| Índice | Feature | Normalização |
|--------|---------|-------------|
| 0 | `moving_distance_cm` | ÷ 500 cm |
| 1 | `moving_energy` | ÷ 100 |
| 2 | `stationary_distance_cm` | ÷ 500 cm |
| 3 | `stationary_energy` | ÷ 100 |
| 4 | `moving_gate[0]` (0–0.75 m) | ÷ 100 |
| 5 | `moving_gate[1]` (0.75–1.5 m) | ÷ 100 |
| 6 | `moving_gate[2]` (1.5–2.25 m) | ÷ 100 |
| 7 | `moving_gate[3]` (2.25–3 m) | ÷ 100 |
| 8 | `stationary_gate[0]` | ÷ 100 |
| 9 | `stationary_gate[1]` | ÷ 100 |
| 10 | `stationary_gate[2]` | ÷ 100 |
| 11 | `stationary_gate[3]` | ÷ 100 |

**Engineering Mode é obrigatório** para os dados de gate. O firmware
tenta habilitá-lo automaticamente no `begin()`. Se a saída mostrar
"Engineering Mode: NAO", verifique a conexão RX do sensor (GPIO 17 → TX do sensor).

---

## 7. Verificação do dataset coletado

Após a coleta, verifique o arquivo via Python:

```python
import numpy as np, struct

BPR = 96 * 4 + 1  # 385 bytes por amostra

with open("dataset_priv.bin", "rb") as f:
    data = f.read()

n = len(data) // BPR
print(f"Amostras: {n}")

for i in range(min(3, n)):
    row = data[i*BPR:(i+1)*BPR]
    feats = struct.unpack_from("<96f", row, 0)
    label = struct.unpack_from("<B", row, 384)[0]
    print(f"  [{i}] label={label}  feat[0]={feats[0]:.3f}  feat[1]={feats[1]:.3f}")
```

---

*Gerado para o projeto FedKD-MR — UFBA-WISER*
