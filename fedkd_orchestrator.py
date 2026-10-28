#!/usr/bin/env python3
"""
fedkd_orchestrator.py — FedKD-MR Automated Transition Orchestrator
====================================================================
Automatiza a transição da coleta de dados para a federação em 5 fases:

  Fase 1 — Exporta dataset_priv.bin de cada ESP32 (hex dump via Serial)
  Fase 2 — Gera arquivos compartilhados (dataset_pub.bin, metadata.json,
            teacher_probs.bin, teacher_probs_meta.json)
  Fase 3 — Faz upload dos arquivos compartilhados para cada ESP32
            (protocolo UPLOAD serial customizado — NÃO usa ampy/uploadfs)
  Fase 4 — Grava firmware de federação via PlatformIO (sem apagar LittleFS)
  Fase 5 — Inicia fedkd_server.py como subprocess

RESTRIÇÃO INVIOLÁVEL: fedkd_server.py NUNCA é modificado.

Uso:
  python fedkd_orchestrator.py --config orchestrator_example.yaml
  python fedkd_orchestrator.py --config orchestrator_example.yaml --skip-flash
  python fedkd_orchestrator.py --config orchestrator_example.yaml --dry-run
  python fedkd_orchestrator.py --config orchestrator_example.yaml --phase 1
  python fedkd_orchestrator.py --config orchestrator_example.yaml --phase 2 3

Requisitos:
  pip install pyserial numpy scikit-learn pyyaml
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[ERRO] pyserial não instalado. Execute: pip install pyserial")
    sys.exit(1)

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("[ERRO] scikit-learn não instalado. Execute: pip install scikit-learn")
    sys.exit(1)


# ── Constantes do firmware ────────────────────────────────────────────────────
N_FEATURES   = 12        # features por timestep
T_WINDOW     = 8         # timesteps por janela
N_INPUT      = N_FEATURES * T_WINDOW   # 96 float32
BPR          = N_INPUT * 4 + 1         # 385 bytes/amostra (96×float32 + uint8 label)

# Nomes das features (ordem de fillFeatureVector() no LD2410C)
FEATURE_NAMES = []
for _t in range(T_WINDOW):
    FEATURE_NAMES.extend([
        f"t{_t}_moving_dist",
        f"t{_t}_moving_energy",
        f"t{_t}_stationary_dist",
        f"t{_t}_stationary_energy",
        f"t{_t}_moving_g0",
        f"t{_t}_moving_g1",
        f"t{_t}_moving_g2",
        f"t{_t}_moving_g3",
        f"t{_t}_stationary_g0",
        f"t{_t}_stationary_g1",
        f"t{_t}_stationary_g2",
        f"t{_t}_stationary_g3",
    ])

CLASS_NAMES = ["EMPTY", "STATIONARY", "APPROACHING", "LEAVING"]

# Caminhos dos arquivos no LittleFS do ESP32 (conforme Config.h do main.cpp)
FS_PATHS = {
    "dataset_pub":      "/dataset_pub.bin",
    "metadata":         "/metadata.json",
    "teacher_probs":    "/teacher_probs.bin",
    "teacher_probs_meta": "/teacher_probs_meta.json",
}


# ── Estruturas de dados ───────────────────────────────────────────────────────

@dataclass
class ClientConfig:
    client_id:   int
    client_name: str
    port:        str
    env:         str


@dataclass
class OrchestratorConfig:
    clients:         List[ClientConfig]
    broker_host:     str
    broker_port:     int
    project_dir:     Path
    server_script:   str
    output_dir:      Path
    serial_baud:     int   = 115200
    n_pub_per_class: int   = 5
    n_classes:       int   = 4
    temperature:     float = 2.0
    # teacher MLP params
    mlp_hidden:      tuple = (64, 32)
    mlp_activation:  str   = "relu"
    mlp_max_iter:    int   = 300
    mlp_seed:        int   = 42
    # federation params
    fed_rounds:      Optional[int] = None
    fed_timeout:     int   = 120
    fed_interval:    float = 5.0
    # transfer params
    export_timeout:  int   = 60
    upload_timeout:  int   = 120
    inter_line_ms:   int   = 10
    hex_line_bytes:  int   = 32
    open_retries:    int   = 3
    open_retry_delay: float = 2.0


# ── Carregamento de configuração ──────────────────────────────────────────────

def load_config(path: str) -> OrchestratorConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    clients = [
        ClientConfig(
            client_id=c["client_id"],
            client_name=c["client_name"],
            port=c["port"],
            env=c["env"],
        )
        for c in raw["clients"]
    ]

    mlp_cfg = raw.get("teacher_mlp", {})
    fed_cfg = raw.get("federation", {})
    xfr_cfg = raw.get("transfer", {})

    return OrchestratorConfig(
        clients=clients,
        broker_host=raw["broker"]["host"],
        broker_port=raw["broker"].get("port", 1883),
        project_dir=Path(raw.get("project_dir", ".")),
        server_script=raw.get("server_script", "fedkd_server.py"),
        output_dir=Path(raw.get("output_dir", "orchestrator_output")),
        serial_baud=raw.get("serial_baud", 115200),
        n_pub_per_class=raw.get("n_pub_per_class", 5),
        n_classes=raw.get("n_classes", 4),
        temperature=raw.get("temperature", 2.0),
        mlp_hidden=tuple(mlp_cfg.get("hidden_layer_sizes", [64, 32])),
        mlp_activation=mlp_cfg.get("activation", "relu"),
        mlp_max_iter=mlp_cfg.get("max_iter", 300),
        mlp_seed=mlp_cfg.get("random_state", 42),
        fed_rounds=fed_cfg.get("rounds", None),
        fed_timeout=fed_cfg.get("timeout", 120),
        fed_interval=fed_cfg.get("inter_round_s", 5.0),
        export_timeout=xfr_cfg.get("export_timeout_s", 60),
        upload_timeout=xfr_cfg.get("upload_timeout_s", 120),
        inter_line_ms=xfr_cfg.get("inter_line_ms", 10),
        hex_line_bytes=xfr_cfg.get("hex_line_bytes", 32),
        open_retries=xfr_cfg.get("open_retry_count", 3),
        open_retry_delay=xfr_cfg.get("open_retry_delay_s", 2.0),
    )


# ── Utilitários Serial ────────────────────────────────────────────────────────

def _open_serial(port: str, baud: int, retries: int, retry_delay: float) -> serial.Serial:
    """Abre a porta serial SEM resetar o ESP32 (dtr=False, rts=False)."""
    for attempt in range(1, retries + 1):
        try:
            ser = serial.Serial()
            ser.port = port
            ser.baudrate = baud
            ser.timeout = 2.0
            ser.dtr = False   # CRÍTICO: não reseta o ESP32
            ser.rts = False
            ser.open()
            _info(f"Porta {port} aberta (tentativa {attempt}/{retries})")
            return ser
        except serial.SerialException as e:
            _warn(f"Tentativa {attempt}/{retries}: {e}")
            if attempt < retries:
                time.sleep(retry_delay)
    raise RuntimeError(f"Não foi possível abrir {port} após {retries} tentativas")


def _readline_timeout(ser: serial.Serial, timeout_s: float) -> Optional[str]:
    """Lê uma linha da serial com timeout absoluto."""
    deadline = time.monotonic() + timeout_s
    buf = bytearray()
    while time.monotonic() < deadline:
        if ser.in_waiting:
            b = ser.read(1)
            if b == b'\n':
                return buf.decode("ascii", errors="replace").strip()
            if b != b'\r':
                buf.extend(b)
        else:
            time.sleep(0.005)
    return None


# ── Logging ───────────────────────────────────────────────────────────────────

_start_time = time.monotonic()


def _ts() -> str:
    elapsed = time.monotonic() - _start_time
    m, s = divmod(int(elapsed), 60)
    return f"+{m:02d}:{s:02d}"


def _info(msg: str):
    print(f"  [{_ts()}] ✓ {msg}")


def _warn(msg: str):
    print(f"  [{_ts()}] ! {msg}", file=sys.stderr)


def _err(msg: str):
    print(f"  [{_ts()}] ✗ {msg}", file=sys.stderr)


def _section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Exportar dataset_priv.bin de cada ESP32
# ═══════════════════════════════════════════════════════════════════════════════

def export_dataset_priv(
    port: str,
    baud: int,
    retries: int,
    retry_delay: float,
    timeout_s: float,
) -> bytes:
    """
    Envia o comando 'x' para o firmware collector e recebe dataset_priv.bin
    via dump hex (protocolo DUMP_START / hex lines / DUMP_END).

    Retorna os bytes do arquivo, ou lança RuntimeError em caso de falha.
    """
    ser = _open_serial(port, baud, retries, retry_delay)
    try:
        # Limpa buffer antes de enviar
        ser.reset_input_buffer()
        time.sleep(0.2)

        _info(f"  Enviando 'x' para dump de {port}...")
        ser.write(b'x\n')
        ser.flush()

        # Aguarda DUMP_START <size>
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            line = _readline_timeout(ser, 5.0)
            if line is None:
                continue
            if line.startswith("DUMP_START"):
                parts = line.split()
                if len(parts) < 2:
                    raise RuntimeError(f"DUMP_START mal formado: '{line}'")
                expected_size = int(parts[1])
                _info(f"  DUMP_START: {expected_size} bytes esperados")
                break
        else:
            raise RuntimeError(f"Timeout aguardando DUMP_START de {port}")

        # Lê linhas hex até DUMP_END
        hex_data = bytearray()
        while time.monotonic() < deadline:
            line = _readline_timeout(ser, 10.0)
            if line is None:
                raise RuntimeError(f"Timeout durante dump de {port}")
            if line == "DUMP_END":
                break
            # Decodifica hex
            try:
                hex_data.extend(bytes.fromhex(line))
            except ValueError as e:
                raise RuntimeError(f"Linha hex inválida '{line[:20]}...': {e}")
        else:
            raise RuntimeError(f"Timeout aguardando DUMP_END de {port}")

        if len(hex_data) != expected_size:
            raise RuntimeError(
                f"Tamanho incorreto: recebeu {len(hex_data)}, esperado {expected_size}"
            )

        n_samples = len(hex_data) // BPR
        _info(f"  Exportado: {len(hex_data)} bytes = {n_samples} amostras")
        return bytes(hex_data)

    finally:
        ser.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — Gerar arquivos compartilhados
# ═══════════════════════════════════════════════════════════════════════════════

def read_bin_data(data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lê dados no formato binário do firmware (96×float32 LE + uint8 label 1-based).
    Retorna (X, y) onde X.shape=(n,96) float32 e y.shape=(n,) int (1-based).
    """
    n = len(data) // BPR
    if n == 0:
        return np.zeros((0, N_INPUT), dtype=np.float32), np.zeros(0, dtype=np.int32)

    raw = np.frombuffer(data, dtype=np.uint8)
    rows = raw[:n * BPR].reshape(n, BPR)

    # Features: bytes 0..383 como float32 LE
    feat_bytes = rows[:, :N_INPUT * 4].copy()
    X = feat_bytes.view(np.float32).reshape(n, N_INPUT).copy()

    # Label: byte 384
    y = rows[:, N_INPUT * 4].astype(np.int32)
    return X, y


def write_bin_data(path: Path, X: np.ndarray, y: np.ndarray):
    """Escreve dados no formato binário do firmware."""
    with open(path, 'wb') as f:
        for xi, yi in zip(X, y):
            f.write(xi.astype('<f4').tobytes())  # 96 × float32 LE
            f.write(bytes([int(yi)]))             # 1 uint8 label (1-based)


def build_metadata_json(
    n_pub: int,
    n_classes: int,
    client_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> dict:
    """
    Gera metadata.json compatível com o parser load_schema() do main.cpp.
    Os offsets de byte são determinísticos pelo layout do firmware:
      feature i → offset i*4
      label     → offset 96*4 = 384
    """
    schema = []
    for i, name in enumerate(FEATURE_NAMES):
        schema.append({
            "name":   name,
            "type":   "float32",
            "offset": i * 4,
        })

    # Coluna de label (uint8, 1-based)
    schema.append({
        "name":   "label",
        "type":   "uint8",
        "offset": N_INPUT * 4,   # 384
    })

    # Calcula min/max reais dos dados (informativo — ESP32 não usa normalização)
    all_X = np.concatenate([v[0] for v in client_data.values()], axis=0)
    feat_min = all_X.min(axis=0).tolist()
    feat_max = all_X.max(axis=0).tolist()
    normalization = {
        "method": "firmware_scale",
        "note": "Values already divided by MAX_DIST=500 / MAX_ENERGY=100 in firmware",
        "mins": feat_min,
        "maxs": feat_max,
    }

    # Mapa de classes
    label_map = {str(i + 1): CLASS_NAMES[i] for i in range(n_classes)}

    return {
        "bytes_per_row":   BPR,
        "label_column":    "label",
        "label_map":       label_map,
        "n_classes":       n_classes,
        "n_pub":           n_pub,
        "schema":          schema,
        "normalization":   normalization,
        "generated_by":    "fedkd_orchestrator.py",
        "generated_at":    datetime.now().isoformat(timespec="seconds"),
    }


def generate_teacher_probs(
    X_pub: np.ndarray,
    y_pub: np.ndarray,
    client_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    temperature: float,
    n_classes: int,
    mlp_hidden: tuple,
    mlp_activation: str,
    mlp_max_iter: int,
    mlp_seed: int,
) -> np.ndarray:
    """
    Treina um MLP sklearn por cliente e gera teacher_probs para D_pub.

    Passos:
      1. Para cada cliente: treina MLP em D_priv e obtém log-probs em D_pub
      2. Média ponderada dos log-probs entre clientes (peso = n_samples do cliente)
      3. Temperature scaling e softmax → teacher_probs (n_pub, n_classes)

    Retorna: (n_pub, n_classes) float32
    """
    n_pub = len(X_pub)
    all_classes = list(range(1, n_classes + 1))  # labels 1-based

    log_probs_list = []
    weights = []

    for client_name, (X_priv, y_priv) in client_data.items():
        if len(X_priv) < max(5, n_classes):
            _warn(f"  Cliente {client_name}: apenas {len(X_priv)} amostras — pulando MLP")
            continue

        present = set(y_priv.tolist())
        missing = set(all_classes) - present
        if missing:
            _warn(f"  Cliente {client_name}: classes ausentes {missing} — pulando MLP")
            continue

        _info(f"  Treinando MLP para {client_name} ({len(X_priv)} amostras)...")
        mlp = MLPClassifier(
            hidden_layer_sizes=mlp_hidden,
            activation=mlp_activation,
            max_iter=mlp_max_iter,
            random_state=mlp_seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
        mlp.fit(X_priv, y_priv)

        # log_proba retorna colunas na ordem de mlp.classes_
        lp = mlp.predict_log_proba(X_pub)   # (n_pub, n_classes_present)
        # Garante ordem canônica (classe 1..n_classes)
        class_order = list(mlp.classes_)
        lp_ordered = np.zeros((n_pub, n_classes), dtype=np.float64)
        for k, cls in enumerate(class_order):
            idx = cls - 1  # 1-based → 0-based index
            if 0 <= idx < n_classes:
                lp_ordered[:, idx] = lp[:, k]

        log_probs_list.append(lp_ordered)
        weights.append(len(X_priv))
        _info(f"    Pronto. Score treino: {mlp.score(X_priv, y_priv):.3f}")

    if not log_probs_list:
        _warn("  Nenhum MLP treinado com sucesso — usando distribuição uniforme")
        return np.full((n_pub, n_classes), 1.0 / n_classes, dtype=np.float32)

    # Média ponderada dos log-probs
    total_w = sum(weights)
    avg_log = np.zeros((n_pub, n_classes), dtype=np.float64)
    for lp, w in zip(log_probs_list, weights):
        avg_log += lp * (w / total_w)

    # Temperature scaling e softmax estabilizado
    scaled = avg_log / temperature
    scaled -= scaled.max(axis=1, keepdims=True)   # estabilidade numérica
    exp_s = np.exp(scaled)
    teacher_probs = (exp_s / exp_s.sum(axis=1, keepdims=True)).astype(np.float32)

    # Diagnóstico
    H = -np.sum(teacher_probs * np.log(teacher_probs + 1e-9), axis=1)
    _info(f"  Teacher probs: H_médio={H.mean():.3f}  "
          f"prob_range=[{teacher_probs.min():.3f}, {teacher_probs.max():.3f}]")

    return teacher_probs


def generate_shared_files(
    exported: Dict[str, bytes],
    cfg: OrchestratorConfig,
    output_dir: Path,
) -> Path:
    """
    Gera os 4 arquivos compartilhados a partir dos dados exportados.

    Retorna o caminho do diretório com os arquivos gerados.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Lê dados binários de cada cliente ────────────────────────────────────
    client_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, raw in exported.items():
        X, y = read_bin_data(raw)
        if len(X) == 0:
            raise RuntimeError(f"Cliente {name}: dataset_priv.bin vazio")
        client_data[name] = (X, y)
        counts = {cls: int((y == i + 1).sum()) for i, cls in enumerate(CLASS_NAMES)}
        _info(f"  {name}: {len(X)} amostras — {counts}")

    # ── Seleciona D_pub (estratificado, n_pub_per_class por classe) ───────────
    # Coleta amostras de todos os clientes combinados
    all_X = np.concatenate([v[0] for v in client_data.values()], axis=0)
    all_y = np.concatenate([v[1] for v in client_data.values()], axis=0)

    pub_idx = []
    for cls in range(1, cfg.n_classes + 1):
        cls_idx = np.where(all_y == cls)[0]
        if len(cls_idx) < cfg.n_pub_per_class:
            raise RuntimeError(
                f"Classe {CLASS_NAMES[cls-1]}: apenas {len(cls_idx)} amostras no "
                f"pool total; necessário n_pub_per_class={cfg.n_pub_per_class}"
            )
        rng = np.random.default_rng(seed=42 + cls)
        chosen = rng.choice(cls_idx, size=cfg.n_pub_per_class, replace=False)
        pub_idx.extend(chosen.tolist())

    pub_idx_set = set(pub_idx)
    X_pub = all_X[pub_idx]
    y_pub = all_y[pub_idx]
    n_pub = len(X_pub)
    _info(f"  D_pub: {n_pub} amostras ({cfg.n_pub_per_class} × {cfg.n_classes} classes)")

    # ── Gera teacher_probs ────────────────────────────────────────────────────
    _info("  Gerando teacher_probs (MLP ensemble)...")
    teacher_probs = generate_teacher_probs(
        X_pub, y_pub,
        client_data=client_data,
        temperature=cfg.temperature,
        n_classes=cfg.n_classes,
        mlp_hidden=cfg.mlp_hidden,
        mlp_activation=cfg.mlp_activation,
        mlp_max_iter=cfg.mlp_max_iter,
        mlp_seed=cfg.mlp_seed,
    )

    # ── Escreve dataset_pub.bin ───────────────────────────────────────────────
    pub_bin_path = output_dir / "dataset_pub.bin"
    write_bin_data(pub_bin_path, X_pub, y_pub)
    _info(f"  dataset_pub.bin: {pub_bin_path.stat().st_size} bytes")

    # ── Escreve teacher_probs.bin ─────────────────────────────────────────────
    tp_path = output_dir / "teacher_probs.bin"
    tp_path.write_bytes(teacher_probs.astype('<f4').tobytes())
    _info(f"  teacher_probs.bin: {tp_path.stat().st_size} bytes "
          f"({n_pub} × {cfg.n_classes} × 4)")

    # ── Escreve teacher_probs_meta.json ───────────────────────────────────────
    tp_meta = {
        "n_pub":       n_pub,
        "n_classes":   cfg.n_classes,
        "temperature": cfg.temperature,
        "bytes_per_row": cfg.n_classes * 4,
        "total_bytes":   n_pub * cfg.n_classes * 4,
    }
    tp_meta_path = output_dir / "teacher_probs_meta.json"
    tp_meta_path.write_text(json.dumps(tp_meta, indent=2))
    _info(f"  teacher_probs_meta.json: {tp_meta}")

    # ── Escreve metadata.json ─────────────────────────────────────────────────
    meta = build_metadata_json(n_pub, cfg.n_classes, client_data)
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    _info(f"  metadata.json: {len(meta['schema'])} colunas, bytes_per_row={BPR}")

    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — Upload de arquivos compartilhados para o ESP32
# ═══════════════════════════════════════════════════════════════════════════════

def _send_file_to_esp32(
    ser: serial.Serial,
    remote_path: str,
    data: bytes,
    inter_line_ms: int,
    hex_line_bytes: int,
    upload_timeout_s: float,
) -> bool:
    """
    Envia um arquivo para o ESP32 via protocolo UPLOAD serial.

    Protocolo:
      PC → ESP32:  "UPLOAD /path size\\n"
      ESP32 → PC:  "UPLOAD_READY\\n"
      PC → ESP32:  "<linha hex 64 chars>\\n"  (repetir)
      PC → ESP32:  "UPLOAD_END\\n"
      ESP32 → PC:  "UPLOAD_OK size\\n"  OU  "UPLOAD_ERR reason\\n"

    Retorna True se UPLOAD_OK recebido.
    """
    n_bytes = len(data)
    cmd = f"UPLOAD {remote_path} {n_bytes}\n"
    _info(f"    → UPLOAD {remote_path} ({n_bytes} bytes)")

    ser.reset_input_buffer()
    ser.write(cmd.encode("ascii"))
    ser.flush()

    # Aguarda UPLOAD_READY — o firmware pode misturar print_status() (sem \n)
    # com a resposta UPLOAD_READY/UPLOAD_ERR na mesma "linha" serial, pois o
    # print_status usa \r sem \n e o handle_upload() adiciona \n à sua resposta.
    # Por isso verificamos com 'in' em vez de '==' para detectar a resposta mesmo
    # que ela venha concatenada com o status anterior.
    found_ready = False
    deadline_ready = time.monotonic() + 10.0
    last_line = None
    while time.monotonic() < deadline_ready:
        remaining = deadline_ready - time.monotonic()
        if remaining <= 0:
            break
        line = _readline_timeout(ser, min(remaining, 2.0))
        if line is None:
            continue  # nenhum \n por até 2s — continua aguardando (handle_upload pode ser lento)
        last_line = line
        if "UPLOAD_READY" in line:
            found_ready = True
            break
        if "UPLOAD_ERR" in line:
            # Extrai a parte do erro mesmo que precedida de lixo de status
            err_start = line.index("UPLOAD_ERR")
            _err(f"    ESP32 reportou erro: '{line[err_start:]}'")
            break
        if line:
            _info(f"    ESP32 (descartado antes de UPLOAD_READY): '{line}'")

    if not found_ready:
        _err(f"    Timeout aguardando UPLOAD_READY (última linha recebida: '{last_line}')")
        return False

    # Envia dados em linhas hex
    hex_chars = hex_line_bytes * 2  # chars hex por linha
    offset = 0
    total_lines = (n_bytes + hex_line_bytes - 1) // hex_line_bytes
    deadline = time.monotonic() + upload_timeout_s

    for line_n in range(total_lines):
        if time.monotonic() > deadline:
            _err(f"    Timeout durante upload ({line_n}/{total_lines} linhas)")
            return False

        chunk = data[offset:offset + hex_line_bytes]
        hex_line = chunk.hex().upper() + "\n"
        ser.write(hex_line.encode("ascii"))
        ser.flush()
        offset += len(chunk)

        # ── Aguarda ACK de flow-control ('+\n') antes de enviar próxima linha ──
        # O firmware envia '+\n' após gravar cada linha com sucesso.
        # Isso garante que nunca enviamos dados mais rápido do que o ESP32 processa,
        # eliminando overflow do buffer UART independente do tempo de flash.
        ack_deadline = min(deadline, time.monotonic() + 10.0)
        ack = None
        while time.monotonic() < ack_deadline:
            remaining_ack = ack_deadline - time.monotonic()
            if remaining_ack <= 0:
                break
            ack = _readline_timeout(ser, min(remaining_ack, 2.0))
            if ack is not None:
                break
        if ack is None:
            _err(f"    Timeout esperando ACK da linha {line_n}/{total_lines}")
            return False
        if "UPLOAD_ERR" in ack:
            err_start = ack.index("UPLOAD_ERR")
            _err(f"    ESP32 reportou erro na linha {line_n}: '{ack[err_start:]}'")
            return False
        if ack != "+":
            # ACK inesperado mas não fatal — pode ser debug output do firmware
            _info(f"    ACK inesperado (linha {line_n}): '{ack}'")

        # Progresso a cada 10%
        if total_lines > 10 and line_n % max(1, total_lines // 10) == 0:
            pct = 100 * line_n // total_lines
            print(f"\r      {pct:3d}%  ({line_n}/{total_lines} linhas)   ", end="", flush=True)

    print()  # nova linha após progresso

    # Pausa curta antes de UPLOAD_END (ACK já garante que última linha foi processada)
    time.sleep(0.05)
    ser.write(b"UPLOAD_END\n")
    ser.flush()

    # Aguarda UPLOAD_OK / UPLOAD_ERR
    response = _readline_timeout(ser, 30.0)
    if response is None:
        _err("    Timeout aguardando resposta pós-UPLOAD_END")
        return False

    if response.startswith("UPLOAD_OK"):
        parts = response.split()
        recv_bytes = int(parts[1]) if len(parts) > 1 else -1
        if recv_bytes != n_bytes:
            _err(f"    UPLOAD_OK mas tamanho diverge: {recv_bytes} ≠ {n_bytes}")
            return False
        _info(f"    UPLOAD_OK {recv_bytes} bytes ✓")
        return True
    else:
        _err(f"    ESP32 respondeu: '{response}'")
        return False


def upload_shared_files(
    port: str,
    client_name: str,
    shared_dir: Path,
    cfg: OrchestratorConfig,
    dry_run: bool = False,
) -> bool:
    """
    Faz upload dos 4 arquivos compartilhados para um ESP32.
    Não apaga dataset_priv.bin.
    """
    files_to_upload = [
        ("dataset_pub",       shared_dir / "dataset_pub.bin"),
        ("metadata",          shared_dir / "metadata.json"),
        ("teacher_probs",     shared_dir / "teacher_probs.bin"),
        ("teacher_probs_meta", shared_dir / "teacher_probs_meta.json"),
    ]

    if dry_run:
        _info(f"  [DRY-RUN] Pulando upload para {client_name} ({port})")
        return True

    ser = _open_serial(port, cfg.serial_baud, cfg.open_retries, cfg.open_retry_delay)
    try:
        # Aguarda ESP32 terminar de bootar caso tenha resetado ao abrir a porta.
        # radar.begin() (firmware) aguarda ACK do sensor, podendo levar 2-3s.
        # Usamos 4s para garantir que TODO o output de setup() já chegou e foi descartado.
        _info("  Aguardando ESP32 inicializar (4s)...")
        time.sleep(4.0)
        ser.reset_input_buffer()  # descarta todo o boot message
        time.sleep(0.1)           # drena qualquer byte residual que chega logo após

        for key, local_path in files_to_upload:
            if not local_path.exists():
                _err(f"  Arquivo local não encontrado: {local_path}")
                return False
            data = local_path.read_bytes()
            ok = _send_file_to_esp32(
                ser,
                remote_path=FS_PATHS[key],
                data=data,
                inter_line_ms=cfg.inter_line_ms,
                hex_line_bytes=cfg.hex_line_bytes,
                upload_timeout_s=cfg.upload_timeout,
            )
            if not ok:
                _err(f"  Falha ao enviar {local_path.name} para {client_name}")
                return False
            time.sleep(0.5)  # pausa entre arquivos
    finally:
        ser.close()

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 4 — Gravar firmware de federação
# ═══════════════════════════════════════════════════════════════════════════════

def flash_firmware(
    port: str,
    env: str,
    project_dir: Path,
    dry_run: bool = False,
) -> bool:
    """
    Executa: pio run -e {env} -t upload --upload-port {port}
    NÃO usa 'pio run ... -t uploadfs' para preservar LittleFS.
    """
    cmd = ["pio", "run", "-e", env, "-t", "upload", f"--upload-port={port}"]
    _info(f"  $ {' '.join(cmd)}")

    if dry_run:
        _info(f"  [DRY-RUN] Pulando flash de {port} (env={env})")
        return True

    result = subprocess.run(
        cmd,
        cwd=str(project_dir),
        capture_output=False,   # deixa stdout/stderr aparecerem diretamente
    )

    if result.returncode != 0:
        _err(f"  pio run falhou (code {result.returncode})")
        return False

    _info(f"  Firmware '{env}' gravado com sucesso em {port}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 5 — Iniciar fedkd_server.py
# ═══════════════════════════════════════════════════════════════════════════════

def start_server(cfg: OrchestratorConfig, dry_run: bool = False) -> Optional[subprocess.Popen]:
    """
    Inicia fedkd_server.py como subprocess com os parâmetros do config.
    NUNCA modifica fedkd_server.py — apenas o invoca.
    """
    server_path = cfg.project_dir / cfg.server_script
    if not server_path.exists():
        _err(f"  Não encontrou {server_path}")
        return None

    cmd = [
        sys.executable,
        str(server_path),
        "--host",      cfg.broker_host,
        "--port",      str(cfg.broker_port),
        "--n-clients", str(len(cfg.clients)),
        "--timeout",   str(cfg.fed_timeout),
        "--temperature", str(cfg.temperature),
        "--interval",  str(cfg.fed_interval),
    ]
    if cfg.fed_rounds is not None:
        cmd += ["--rounds", str(cfg.fed_rounds)]

    _info(f"  Iniciando: {' '.join(cmd)}")

    if dry_run:
        _info(f"  [DRY-RUN] Servidor não iniciado")
        return None

    proc = subprocess.Popen(cmd, cwd=str(cfg.project_dir))
    _info(f"  fedkd_server.py PID={proc.pid}")
    return proc


# ═══════════════════════════════════════════════════════════════════════════════
# Orquestrador principal
# ═══════════════════════════════════════════════════════════════════════════════

class FedKDOrchestrator:
    def __init__(self, cfg: OrchestratorConfig, dry_run: bool = False):
        self.cfg     = cfg
        self.dry_run = dry_run

    def run(self, phases: Optional[List[int]] = None, skip_flash: bool = False):
        """
        Executa as fases especificadas (padrão: todas).
        phases=[1,2,3,4,5] ou subconjunto.
        """
        if phases is None:
            phases = [1, 2, 3, 4, 5]

        _section("FedKD-MR Orchestrator")
        print(f"  Clientes  : {len(self.cfg.clients)}")
        print(f"  Broker    : {self.cfg.broker_host}:{self.cfg.broker_port}")
        print(f"  ProjDir   : {self.cfg.project_dir}")
        print(f"  OutDir    : {self.cfg.output_dir}")
        print(f"  Fases     : {phases}")
        print(f"  Dry-run   : {self.dry_run}")

        exported: Dict[str, bytes] = {}
        shared_dir: Optional[Path] = None
        server_proc: Optional[subprocess.Popen] = None

        # ── Fase 1: Exportar dataset_priv.bin ────────────────────────────────
        if 1 in phases:
            _section("Fase 1 — Exportar dataset_priv.bin")
            if not self.dry_run:
                input("\n  Pressione ENTER quando a coleta de dados estiver concluída em TODOS os ESP32...\n")
            for client in self.cfg.clients:
                _info(f"Exportando de {client.client_name} ({client.port})...")
                if self.dry_run:
                    # Simula com dados fictícios
                    exported[client.client_name] = _make_dummy_data()
                else:
                    data = export_dataset_priv(
                        port=client.port,
                        baud=self.cfg.serial_baud,
                        retries=self.cfg.open_retries,
                        retry_delay=self.cfg.open_retry_delay,
                        timeout_s=self.cfg.export_timeout,
                    )
                    exported[client.client_name] = data
                    # Salva cópia local
                    backup = self.cfg.output_dir / f"dataset_priv_{client.client_name}.bin"
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(data)
                    _info(f"  Backup salvo: {backup}")
            _info("Fase 1 concluída.")
        else:
            # Tenta carregar de backups anteriores
            _info("Fase 1 pulada — carregando backups de dataset_priv...")
            for client in self.cfg.clients:
                backup = self.cfg.output_dir / f"dataset_priv_{client.client_name}.bin"
                if backup.exists():
                    exported[client.client_name] = backup.read_bytes()
                    n = len(exported[client.client_name]) // BPR
                    _info(f"  {client.client_name}: {n} amostras do backup")
                else:
                    _warn(f"  Backup não encontrado: {backup}")

        # ── Fase 2: Gerar arquivos compartilhados ─────────────────────────────
        if 2 in phases:
            _section("Fase 2 — Gerar arquivos compartilhados")
            if not exported:
                raise RuntimeError("Nenhum dado exportado disponível para a Fase 2")
            shared_dir = generate_shared_files(exported, self.cfg, self.cfg.output_dir)
            _info(f"Fase 2 concluída → {shared_dir}")
        else:
            shared_dir = self.cfg.output_dir
            _info(f"Fase 2 pulada — usando {shared_dir}")

        # ── Fase 3: Upload para cada ESP32 ────────────────────────────────────
        if 3 in phases:
            _section("Fase 3 — Upload de arquivos compartilhados")
            for client in self.cfg.clients:
                _info(f"Enviando para {client.client_name} ({client.port})...")
                ok = upload_shared_files(
                    port=client.port,
                    client_name=client.client_name,
                    shared_dir=shared_dir,
                    cfg=self.cfg,
                    dry_run=self.dry_run,
                )
                if not ok:
                    raise RuntimeError(f"Upload falhou para {client.client_name}")
            _info("Fase 3 concluída.")

        # ── Fase 4: Gravar firmware ───────────────────────────────────────────
        if 4 in phases and not skip_flash:
            _section("Fase 4 — Gravar firmware de federação")
            for client in self.cfg.clients:
                _info(f"Gravando '{client.env}' em {client.client_name} ({client.port})...")
                ok = flash_firmware(
                    port=client.port,
                    env=client.env,
                    project_dir=self.cfg.project_dir,
                    dry_run=self.dry_run,
                )
                if not ok:
                    raise RuntimeError(f"Flash falhou para {client.client_name}")
                time.sleep(2.0)  # pausa entre dispositivos
            _info("Fase 4 concluída.")
        elif skip_flash:
            _info("Fase 4 pulada (--skip-flash).")

        # ── Fase 5: Iniciar servidor ──────────────────────────────────────────
        if 5 in phases:
            _section("Fase 5 — Iniciar fedkd_server.py")
            # Aguarda ESP32s reiniciarem após flash
            if 4 in phases and not skip_flash and not self.dry_run:
                _info("Aguardando ESP32s inicializarem (5s)...")
                time.sleep(5.0)
            server_proc = start_server(self.cfg, dry_run=self.dry_run)
            if server_proc and not self.dry_run:
                _info("Servidor iniciado. Ctrl+C para encerrar.")
                try:
                    server_proc.wait()
                except KeyboardInterrupt:
                    _info("Interrompido pelo usuário. Encerrando servidor...")
                    server_proc.terminate()

        _section("Orquestrador concluído")


# ── Utilitário dry-run ────────────────────────────────────────────────────────

def _make_dummy_data(n_per_class: int = 30, n_classes: int = 4) -> bytes:
    """Gera dados sintéticos para dry-run."""
    rng = np.random.default_rng(0)
    buf = bytearray()
    for cls in range(1, n_classes + 1):
        for _ in range(n_per_class):
            x = rng.uniform(0.0, 0.5, N_INPUT).astype(np.float32)
            buf.extend(x.tobytes())
            buf.append(cls)
    return bytes(buf)


# ── Ponto de entrada ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FedKD-MR Orchestrator — Transição automática coleta → federação"
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Arquivo YAML de configuração (ex: orchestrator_example.yaml)"
    )
    parser.add_argument(
        "--phase", "-p", type=int, nargs="+", metavar="N",
        help="Executar apenas fases específicas (ex: --phase 1 2 3). Padrão: todas."
    )
    parser.add_argument(
        "--skip-flash", action="store_true",
        help="Pular a Fase 4 (gravar firmware). Útil quando o firmware já está gravado."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simula a execução sem abrir portas serial nem chamar PlatformIO."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FedKD-MR Orchestrator")
    print(f"  Config: {args.config}")
    print("=" * 60)

    cfg = load_config(args.config)
    orch = FedKDOrchestrator(cfg, dry_run=args.dry_run)
    orch.run(phases=args.phase, skip_flash=args.skip_flash)


if __name__ == "__main__":
    main()
