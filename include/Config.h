#ifndef CONFIG_H_
#define CONFIG_H_

// ============================================================
// Config.h — FedKD-MR com ESP32 + HLK-LD2410C
// ============================================================
// Substitui o Config.h original do projeto Atlântico.
// Mantém compatibilidade com ModelUtil.cpp onde necessário.
// ============================================================

// ── Identificação do cliente ─────────────────────────────────
// Altere antes de gravar em cada ESP32: 1, 2 ou 3
#define CLIENT_ID   1
#define _STR(x)     #x
#define _TOSTR(x)   _STR(x)
#define CLIENT_NAME "FedKD-ESP32-" _TOSTR(CLIENT_ID)

// ── Dataset / Janelamento ────────────────────────────────────
#define T_WINDOW      8     // timesteps por janela
#define N_FEATURES    12    // features por timestep
#define N_INPUT       96    // T_WINDOW × N_FEATURES (entrada do MLP)
#define N_CLASSES     4     // EMPTY=1, STATIONARY=2, APPROACHING=3, LEAVING=4

// ── Arquitetura do modelo local (por cliente) ────────────────
// Modelos heterogêneos — defina APENAS o CLIENT_ID acima e
// a arquitetura correspondente é selecionada automaticamente.
#if   CLIENT_ID == 1
  // MLP pequeno: ~3.5 KB de parâmetros
  #define MODEL_LAYERS       {N_INPUT, 32, 16, N_CLASSES}
  #define MODEL_N_LAYERS     4
  #define MODEL_ACTV         {1, 1, 6}   // Tanh, Tanh, Softmax
#elif CLIENT_ID == 2
  // MLP médio: ~5.5 KB de parâmetros
  #define MODEL_LAYERS       {N_INPUT, 48, 24, N_CLASSES}
  #define MODEL_N_LAYERS     4
  #define MODEL_ACTV         {1, 1, 6}
#elif CLIENT_ID == 3
  // MLP maior (verificar RAM disponível no ESP32 específico)
  #define MODEL_LAYERS       {N_INPUT, 64, 32, N_CLASSES}
  #define MODEL_N_LAYERS     4
  #define MODEL_ACTV         {1, 1, 6}
#endif

// ── Hiperparâmetros — Treino local supervisionado ────────────
// 30 épocas necessárias com dados sintéticos para convergir.
// LR 0.003 acelera convergência sem explodir (clamp_params protege).
#define LOCAL_EPOCHS         30
#define LOCAL_LR_WEIGHTS     0.003f
#define LOCAL_LR_BIASES      0.003f

// ── Hiperparâmetros — Knowledge Distillation (KD) ────────────
#define KD_EPOCHS            10
#define KD_LR_WEIGHTS        0.001f
#define KD_LR_BIASES         0.001f

// ── Hiperparâmetros — Fine-tuning pós-KD ────────────────────
#define FT_EPOCHS            5
#define FT_LR_WEIGHTS        0.001f
#define FT_LR_BIASES         0.001f

// ── Caminhos LittleFS ────────────────────────────────────────
// Gerados por prepare_dataset.py e gravados via PlatformIO FS Upload

// Dados privados do cliente (treino local e fine-tuning)
#define XY_TRAIN_PATH           "/dataset_priv.bin"
#define METADATA_JSON_PATH      "/metadata.json"

// Dataset público compartilhado (geração de logits e KD)
#define XY_PUB_PATH             "/dataset_pub.bin"

// Consenso do servidor (teacher_probs) — pré-computado offline
// ou recebido via MQTT na fase de federação real
#define TEACHER_PROBS_PATH      "/teacher_probs.bin"
#define TEACHER_PROBS_META_PATH "/teacher_probs_meta.json"

// Modelo persistido em flash
#define MODEL_PATH              "/model_fedkd.nn"

// ── Rede / MQTT (para fase 2 — federação real) ───────────────
#define WIFI_SSID               "SUA_REDE"
#define WIFI_PASSWORD           "SUA_SENHA"
#define MQTT_BROKER             "192.168.1.100"   // IP do servidor/workstation
#define MQTT_PORT               1883
#define CONNECTION_TIMEOUT      30000             // ms

// Tópicos FedKD-MR (novos — não conflitam com Atlântico)
#define TOPIC_LOGITS_PUSH       "fedkd/logits/push/" CLIENT_NAME
#define TOPIC_TEACHER_PULL      "fedkd/teacher/pull"
#define TOPIC_CMD_PULL          "fedkd/cmd/pull"
#define TOPIC_CMD_PUSH          "fedkd/cmd/push/" CLIENT_NAME

// ── Watchdog / Temporização ──────────────────────────────────
#define WDT_TIMEOUT_SEC         120   // segundos; treino pode demorar
#define SERIAL_BAUD             115200

// ── Flags de compilação ──────────────────────────────────────
#define DATASET_BINARY 1      // usa trainModelFromBinaryDataset
// #define ENABLE_MQTT    1   // descomente para habilitar MQTT (fase 2)
// #define SAVE_MODEL     1   // descomente para persistir modelo em flash

#endif /* CONFIG_H_ */
