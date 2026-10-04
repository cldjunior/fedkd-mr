"""
config.py — FedKD-MR Cliente Raspberry Pi
==========================================
Carrega configuração em duas camadas (prioridade crescente):
  1. Arquivo YAML (--config client_gru.yaml)   ← padrão de operação
  2. Argumentos CLI (--local-epochs 100 ...)   ← sobrescrevem o YAML

Uso normal (sem CLI extra):
    python fedkd_client.py --config configs/client_gru.yaml

Sobrescrita pontual:
    python fedkd_client.py --config configs/client_gru.yaml --local-epochs 100

Os arquivos YAML em configs/ definem: model, hyperparâmetros, rede e dataset.
Exemplo: configs/client_gru.yaml
"""

import argparse
import sys
from pathlib import Path

# PyYAML é opcional — usa json como fallback se YAML não estiver instalado
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    import json as _json_fallback
    _YAML_AVAILABLE = False


# ── Defaults globais (sobrescritos pelo YAML e depois pelo CLI) ──────────────

DEFAULTS = dict(
    # Identificação
    client_id       = 4,
    model           = "mlp",          # mlp | cnn1d | cnn2d | gru | lstm

    # Dataset
    data_dir        = "data/",
    train_file      = "dataset_priv.bin",
    pub_file        = "dataset_pub.bin",
    meta_file       = "metadata.json",

    # Janela temporal
    t_window        = 8,
    n_features      = 12,
    n_classes       = 4,

    # CNN2D — None = usa t_window e n_features automaticamente
    cnn2d_h         = None,
    cnn2d_w         = None,

    # Treino local
    local_epochs    = 50,
    local_lr        = 1e-3,

    # Knowledge Distillation
    kd_epochs       = 20,
    kd_lr           = 5e-4,
    kd_temperature  = 1.0,

    # Fine-tuning
    ft_epochs       = 10,
    ft_lr           = 5e-4,

    # Batching
    batch_size      = 32,

    # MLP
    mlp_hidden      = [128, 64],

    # CNN1D / CNN2D
    cnn_channels    = [32, 64, 128],
    cnn_kernel      = 3,
    cnn_dropout     = 0.3,

    # GRU / LSTM
    rnn_hidden        = 128,
    rnn_layers        = 2,
    rnn_dropout       = 0.3,
    rnn_bidirectional = False,

    # MQTT
    broker           = "192.168.0.12",
    port             = 1883,
    keepalive        = 300,
    teacher_timeout  = 300,

    # Runtime
    device           = "cpu",
    seed             = 42,
)


def _load_yaml(path: str) -> dict:
    """Lê YAML (ou JSON como fallback)."""
    text = Path(path).read_text()
    if _YAML_AVAILABLE:
        return yaml.safe_load(text) or {}
    # Tenta JSON — muitos configs simples são JSON válido
    try:
        return _json_fallback.loads(text)
    except Exception:
        raise RuntimeError(
            "PyYAML não instalado e o arquivo não é JSON válido. "
            "Instale com: pip install pyyaml"
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FedKD-MR — Cliente Raspberry Pi (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   metavar="FILE",
                   help="Arquivo YAML de configuração do cliente. "
                        "Ex: configs/client_gru.yaml")

    # Todos os outros argumentos são opcionais e sobrescrevem o YAML
    p.add_argument("--client-id",    type=int)
    p.add_argument("--model",        type=str,
                   choices=["mlp", "cnn1d", "cnn2d", "gru", "lstm"])
    p.add_argument("--data-dir",     type=str)
    p.add_argument("--train-file",   type=str)
    p.add_argument("--pub-file",     type=str)
    p.add_argument("--meta-file",    type=str)
    p.add_argument("--t-window",     type=int)
    p.add_argument("--n-features",   type=int)
    p.add_argument("--n-classes",    type=int)
    p.add_argument("--cnn2d-h",      type=int)
    p.add_argument("--cnn2d-w",      type=int)
    p.add_argument("--local-epochs", type=int)
    p.add_argument("--local-lr",     type=float)
    p.add_argument("--kd-epochs",    type=int)
    p.add_argument("--kd-lr",        type=float)
    p.add_argument("--kd-temperature", type=float)
    p.add_argument("--ft-epochs",    type=int)
    p.add_argument("--ft-lr",        type=float)
    p.add_argument("--batch-size",   type=int)
    p.add_argument("--mlp-hidden",   type=int, nargs="+", metavar="DIM")
    p.add_argument("--cnn-channels", type=int, nargs="+", metavar="CH")
    p.add_argument("--cnn-kernel",   type=int)
    p.add_argument("--cnn-dropout",  type=float)
    p.add_argument("--rnn-hidden",   type=int)
    p.add_argument("--rnn-layers",   type=int)
    p.add_argument("--rnn-dropout",  type=float)
    p.add_argument("--rnn-bidirectional", action="store_true", default=None)
    p.add_argument("--broker",        type=str)
    p.add_argument("--port",          type=int)
    p.add_argument("--keepalive",     type=int)
    p.add_argument("--teacher-timeout", type=int)
    p.add_argument("--device",        type=str)
    p.add_argument("--seed",          type=int)
    return p


def get_config():
    """
    Retorna namespace de configuração mesclando (prioridade crescente):
      DEFAULTS → YAML → argumentos CLI
    """
    parser = _build_parser()
    args   = parser.parse_args()

    # 1. Começa com defaults
    merged = dict(DEFAULTS)

    # 2. Sobrescreve com YAML se fornecido
    if args.config:
        yaml_cfg = _load_yaml(args.config)
        # Normaliza chaves: substitui '-' por '_'
        yaml_cfg = {k.replace("-", "_"): v for k, v in yaml_cfg.items()}
        merged.update(yaml_cfg)

    # 3. Sobrescreve com CLI (apenas os argumentos explicitamente passados)
    cli_dict = vars(args)
    cli_dict.pop("config", None)
    for k, v in cli_dict.items():
        key = k.replace("-", "_")
        if v is not None:
            merged[key] = v

    # 4. Converte para namespace simples
    import types
    cfg = types.SimpleNamespace(**merged)

    # 5. Deriva constantes calculadas
    cfg.n_input      = cfg.t_window * cfg.n_features
    cfg.client_name  = f"FedKD-RPi-{cfg.client_id}"
    cfg.cnn2d_h      = cfg.cnn2d_h or cfg.t_window
    cfg.cnn2d_w      = cfg.cnn2d_w or cfg.n_features

    # Tópicos MQTT — mesmo esquema dos ESP32s
    cfg.topic_logits_push  = f"fedkd/logits/push/{cfg.client_name}"
    cfg.topic_teacher_pull = "fedkd/teacher/pull"
    cfg.topic_cmd_pull     = "fedkd/cmd/pull"
    cfg.topic_cmd_push     = f"fedkd/cmd/push/{cfg.client_name}"

    return cfg
