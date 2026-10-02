#ifndef CONFIG_H_
#define CONFIG_H_

// ============================================================
// Config.h — FedKD-MR  (valores vêm de platformio.ini)
// ============================================================
// NADA é hardcoded aqui. Todos os parâmetros são definidos
// como build_flags no platformio.ini.
// Este arquivo só:
//   1. Valida que os obrigatórios foram passados
//   2. Fornece defaults para os opcionais
//   3. Deriva constantes calculadas
// ============================================================

// ── Obrigatório: CLIENT_ID ───────────────────────────────────
// Definido em platformio.ini: build_flags = -D CLIENT_ID=1
#ifndef CLIENT_ID
  #error "CLIENT_ID não definido. Adicione -D CLIENT_ID=1 (ou 2, 3) em platformio.ini"
#endif

#define _STR(x)   #x
#define _TOSTR(x) _STR(x)
#define CLIENT_NAME "FedKD-ESP32-" _TOSTR(CLIENT_ID)

// ── Arquitetura por cliente ──────────────────────────────────
#define MODEL_ARCH_MLP   1
#define MODEL_ARCH_CNN1D 2
#define MODEL_ARCH_GRU   3

#if   CLIENT_ID == 1
  #define MODEL_ARCH       MODEL_ARCH_MLP
  #define MODEL_LAYERS     {N_INPUT, 64, N_CLASSES}
  #define MODEL_N_LAYERS   3
  #define MODEL_ACTV       {2, 6}   // ReLU, Softmax
#elif CLIENT_ID == 2
  #define MODEL_ARCH       MODEL_ARCH_CNN1D
#elif CLIENT_ID == 3
  #define MODEL_ARCH       MODEL_ARCH_GRU
#else
  #error "CLIENT_ID inválido. Use 1 (MLP), 2 (CNN1D) ou 3 (GRU)."
#endif

// ── Dataset / Janelamento ────────────────────────────────────
// Sobrescrevíveis por build_flags se mudar de dataset
#ifndef T_WINDOW
  #define T_WINDOW    8
#endif
#ifndef N_FEATURES
  #define N_FEATURES  12
#endif
#define N_INPUT   (T_WINDOW * N_FEATURES)
#define N_CLASSES 4

// ── Hiperparâmetros — treino local ───────────────────────────
#ifndef LOCAL_EPOCHS
  #define LOCAL_EPOCHS      30
#endif
#ifndef LOCAL_LR_WEIGHTS
  #define LOCAL_LR_WEIGHTS  0.003f
#endif
#ifndef LOCAL_LR_BIASES
  #define LOCAL_LR_BIASES   0.003f
#endif

// ── Hiperparâmetros — KD ─────────────────────────────────────
#ifndef KD_EPOCHS
  #define KD_EPOCHS         10
#endif
#ifndef KD_LR_WEIGHTS
  #define KD_LR_WEIGHTS     0.001f
#endif
#ifndef KD_LR_BIASES
  #define KD_LR_BIASES      0.001f
#endif

// ── Hiperparâmetros — fine-tuning ────────────────────────────
#ifndef FT_EPOCHS
  #define FT_EPOCHS         5
#endif
#ifndef FT_LR_WEIGHTS
  #define FT_LR_WEIGHTS     0.001f
#endif
#ifndef FT_LR_BIASES
  #define FT_LR_BIASES      0.001f
#endif

// ── Caminhos LittleFS ────────────────────────────────────────
#define XY_TRAIN_PATH           "/dataset_priv.bin"
#define METADATA_JSON_PATH      "/metadata.json"
#define XY_PUB_PATH             "/dataset_pub.bin"
#define TEACHER_PROBS_PATH      "/teacher_probs.bin"
#define TEACHER_PROBS_META_PATH "/teacher_probs_meta.json"
#define MODEL_PATH              "/model_fedkd.nn"

// ── Rede / MQTT ──────────────────────────────────────────────
// Obrigatórios quando ENABLE_MQTT está ativo
#ifndef WIFI_SSID
  #define WIFI_SSID     "MINHA_REDE"
#endif
#ifndef WIFI_PASSWORD
  #define WIFI_PASSWORD "MINHA_SENHA"
#endif
#ifndef MQTT_BROKER
  #define MQTT_BROKER   "192.168.0.1"
#endif
#ifndef MQTT_PORT
  #define MQTT_PORT     1883
#endif
#ifndef CONNECTION_TIMEOUT
  #define CONNECTION_TIMEOUT 30000
#endif

#define TOPIC_LOGITS_PUSH  "fedkd/logits/push/" CLIENT_NAME
#define TOPIC_TEACHER_PULL "fedkd/teacher/pull"
#define TOPIC_CMD_PULL     "fedkd/cmd/pull"
#define TOPIC_CMD_PUSH     "fedkd/cmd/push/" CLIENT_NAME

// ── Pinos UART — HLK-LD2410C ────────────────────────────────
#ifndef LD2410C_RX_PIN
  #define LD2410C_RX_PIN  16
#endif
#ifndef LD2410C_TX_PIN
  #define LD2410C_TX_PIN  17
#endif
#ifndef LD2410C_BAUD
  #define LD2410C_BAUD    256000
#endif
#ifndef WINDOW_INTERVAL_MS
  #define WINDOW_INTERVAL_MS 100
#endif

// ── Watchdog / Serial ────────────────────────────────────────
#ifndef WDT_TIMEOUT_SEC
  #define WDT_TIMEOUT_SEC 120
#endif
#ifndef SERIAL_BAUD
  #define SERIAL_BAUD     115200
#endif

// ── Flags ────────────────────────────────────────────────────
#define DATASET_BINARY 1
// ENABLE_MQTT e SAVE_MODEL vêm de build_flags no platformio.ini

#endif /* CONFIG_H_ */
