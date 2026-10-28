"""
collect_dataset.py — FedKD-MR: Coleta de Dados Reais do HLK-LD2410C
=====================================================================
Lê frames JSON do ESP32 via Serial USB e gera os mesmos arquivos .bin +
metadata.json que prepare_dataset.py, prontos para uso no FedKD-MR.

Uso:
    pip install pyserial numpy scikit-learn
    python collect_dataset.py --port COM3          # Windows
    python collect_dataset.py --port /dev/ttyUSB0  # Linux/Mac

Teclas durante a coleta:
    E  →  EMPTY        (sala vazia)
    S  →  STATIONARY   (pessoa parada)
    A  →  APPROACHING  (pessoa se aproximando)
    L  →  LEAVING      (pessoa se afastando)
    P  →  Pausar/Retomar coleta
    Q  →  Finalizar e gerar dataset

Requisitos de conexão:
    ESP32 flashed com firmware/collector/main_collector.cpp
    (ver instruções em COLETA.md)
"""

import argparse
import json
import os
import struct
import sys
import time
import threading
import warnings
from collections import deque, defaultdict

import numpy as np
import serial

try:
    import termios
    import tty
    POSIX = True
except ImportError:
    POSIX = False
    import msvcrt  # Windows

warnings.filterwarnings("ignore")

# ── Configuração do dataset (igual ao prepare_dataset.py) ────────────────────
T_WINDOW   = 8
N_FEATURES = 12
N_INPUT    = T_WINDOW * N_FEATURES  # 96

FEATURE_COLS = [
    "moving_distance_cm", "moving_energy",
    "stationary_distance_cm", "stationary_energy",
    "moving_g0", "moving_g1", "moving_g2", "moving_g3",
    "stationary_g0", "stationary_g1", "stationary_g2", "stationary_g3",
]

CLASSES   = ["EMPTY", "STATIONARY", "APPROACHING", "LEAVING"]
N_CLASSES = len(CLASSES)
LABEL_MAP = {c: i + 1 for i, c in enumerate(CLASSES)}  # 1-based (Atlântico)

KEY_MAP = {
    'e': 'EMPTY', 'E': 'EMPTY',
    's': 'STATIONARY', 'S': 'STATIONARY',
    'a': 'APPROACHING', 'A': 'APPROACHING',
    'l': 'LEAVING', 'L': 'LEAVING',
}

SEED = 42
np.random.seed(SEED)

# ── Leitura de tecla não-bloqueante ──────────────────────────────────────────

def _getch_posix(timeout=0.05):
    """Retorna tecla pressionada ou '' se nenhuma em timeout segundos."""
    import select
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            return sys.stdin.read(1)
        return ''
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def _getch_win(timeout=0.05):
    """Retorna tecla pressionada ou '' (Windows)."""
    end = time.time() + timeout
    while time.time() < end:
        if msvcrt.kbhit():
            return msvcrt.getwch()
        time.sleep(0.01)
    return ''

getch = _getch_posix if POSIX else _getch_win

# ── Parser de frame JSON do sensor ───────────────────────────────────────────

def parse_frame(line: str):
    """
    Converte JSON do firmware para vetor de N_FEATURES floats.
    Retorna None se o frame for inválido.

    Normalização inline (igual à fillFeatureVector do LD2410C.cpp):
        distância / 500.0   →  [0, 1]
        energia   / 100.0   →  [0, 1]
    Usamos floats brutos aqui; a normalização global (min-max sobre o
    conjunto de treino) é aplicada depois, como no prepare_dataset.py.
    """
    try:
        d = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None

    if 'boot' in d:
        return None  # mensagem de boot, não é frame

    try:
        feat = [
            float(d['md']),   # moving_distance_cm
            float(d['me']),   # moving_energy
            float(d['sd']),   # stationary_distance_cm
            float(d['se']),   # stationary_energy
            float(d['mg0']),  # moving_gate 0
            float(d['mg1']),  # moving_gate 1
            float(d['mg2']),  # moving_gate 2
            float(d['mg3']),  # moving_gate 3
            float(d['sg0']),  # stationary_gate 0
            float(d['sg1']),  # stationary_gate 1
            float(d['sg2']),  # stationary_gate 2
            float(d['sg3']),  # stationary_gate 3
        ]
        return feat
    except KeyError:
        return None

# ── Funções de geração de dataset (copiadas de prepare_dataset.py) ────────────

def build_feature_names():
    names = []
    for t in range(T_WINDOW):
        for col in FEATURE_COLS:
            names.append(f"t{t}_{col}")
    return names

def build_metadata(feature_names):
    schema = []
    offset = 0
    for name in feature_names:
        schema.append({"name": name, "type": "float32", "c_type": "float",
                        "bytes": 4, "offset": offset})
        offset += 4
    schema.append({"name": "label", "type": "uint8", "c_type": "uint8_t",
                   "bytes": 1, "offset": offset})
    bpr = offset + 1
    meta = {
        "label_column": "label",
        "label_map": LABEL_MAP,
        "schema": schema,
        "bytes_per_row": bpr,
        "n_input": N_INPUT,
        "n_classes": N_CLASSES,
        "t_window": T_WINDOW,
        "n_features_per_step": N_FEATURES,
        "feature_columns": feature_names,
        "classes": CLASSES,
    }
    return meta, bpr

def normalize(X_train, X_other_list):
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    rng  = maxs - mins
    rng[rng == 0] = 1.0
    X_norm = [(X - mins) / rng for X in [X_train] + X_other_list]
    return X_norm, {"mins": mins.tolist(), "maxs": maxs.tolist()}

def write_bin(path, X, y_str, bpr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        for xi, yi in zip(X, y_str):
            f.write(struct.pack(f"<{N_INPUT}f", *xi.tolist()))
            f.write(struct.pack("<B", LABEL_MAP[yi]))
    print(f"  Escrito: {path}  ({len(X)} amostras, {os.path.getsize(path)} bytes)")

def split_and_export(windows_X, windows_y, output_dir, n_clients,
                     n_pub_per_class, test_ratio, non_iid):
    """
    Divide as janelas coletadas e exporta no formato FedKD-MR.
    Usa a mesma lógica de split_dataset do prepare_dataset.py.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder

    X = np.array(windows_X, dtype=np.float32)
    y = np.array(windows_y)
    print(f"\n  Total coletado: {len(X)} janelas")
    for c in CLASSES:
        print(f"    {c}: {(y == c).sum()}")

    # ── D_pub, D_test, D_priv ────────────────────────────────────────────────
    pub_idx, test_idx, priv_idx = [], [], []
    for cls in CLASSES:
        idx = np.where(y == cls)[0].tolist()
        np.random.shuffle(idx)
        n_pub  = min(n_pub_per_class, len(idx))
        n_test = max(1, int(len(idx) * test_ratio)) if len(idx) > n_pub else 0
        pub_idx.extend(idx[:n_pub])
        test_idx.extend(idx[n_pub:n_pub + n_test])
        priv_idx.extend(idx[n_pub + n_test:])

    # Split D_priv entre clientes (IID simples para dataset real)
    client_idx = {i + 1: [] for i in range(n_clients)}
    np.random.shuffle(priv_idx)
    for k, idx in enumerate(priv_idx):
        client_idx[(k % n_clients) + 1].append(idx)

    if not priv_idx:
        print("  [AVISO] Sem amostras privadas — usando D_pub como fallback.")
        for c in client_idx:
            client_idx[c] = pub_idx.copy()

    X_pub  = X[pub_idx];  y_pub  = y[pub_idx]
    X_test = X[test_idx]; y_test = y[test_idx]
    X_c    = {c: X[client_idx[c]] for c in client_idx}
    y_c    = {c: y[client_idx[c]] for c in client_idx}

    # ── Normalização ─────────────────────────────────────────────────────────
    all_priv = np.concatenate(list(X_c.values()), axis=0) if priv_idx else X
    extras   = [X_pub, X_test] + [X_c[c] for c in sorted(X_c)]
    X_all, norm_stats = normalize(all_priv, extras)
    X_pub_n  = X_all[1]
    X_test_n = X_all[2]
    X_c_n    = {c: X_all[3 + i] for i, c in enumerate(sorted(X_c))}

    # ── Metadata ─────────────────────────────────────────────────────────────
    feat_names = build_feature_names()
    meta, bpr  = build_metadata(feat_names)
    meta["normalization"] = norm_stats
    meta["rows_pub"]  = int(len(X_pub_n))
    meta["rows_test"] = int(len(X_test_n))
    meta["source"]    = "real_sensor"

    shared_dir = os.path.join(output_dir, "shared")
    os.makedirs(shared_dir, exist_ok=True)
    with open(os.path.join(shared_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Arquivos .bin ─────────────────────────────────────────────────────────
    write_bin(os.path.join(shared_dir, "dataset_pub.bin"), X_pub_n, y_pub, bpr)

    test_dir = os.path.join(output_dir, "test")
    if len(X_test_n) > 0:
        write_bin(os.path.join(test_dir, "dataset_test.bin"), X_test_n, y_test, bpr)
    else:
        os.makedirs(test_dir, exist_ok=True)
        write_bin(os.path.join(test_dir, "dataset_test.bin"), X_pub_n, y_pub, bpr)

    for c in sorted(X_c_n):
        cdir = os.path.join(output_dir, f"client_{c}")
        os.makedirs(cdir, exist_ok=True)
        write_bin(os.path.join(cdir, "dataset_priv.bin"), X_c_n[c], y_c[c], bpr)
        with open(os.path.join(cdir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ── Teacher probs (MLP simulado sobre dados reais) ────────────────────────
    print(f"\n  Treinando MLPs para teacher_probs...")
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    arch_map = {1:(96,32,16,4),2:(96,48,24,4),3:(96,64,32,4),
                4:(96,128,64,32,4),5:(96,128,64,32,4),
                6:(96,64,32,4),7:(96,96,48,4),8:(96,128,64,4)}

    sum_logits = np.zeros((len(X_pub_n), N_CLASSES), dtype=np.float64)
    total_w = 0.0
    for c in sorted(X_c_n):
        if len(X_c_n[c]) == 0:
            continue
        le = LabelEncoder(); le.fit(CLASSES)
        arch = arch_map.get(c, (96, 32, 16, 4))
        clf = MLPClassifier(hidden_layer_sizes=arch[1:-1], activation="tanh",
                            max_iter=500, random_state=SEED + c)
        try:
            clf.fit(X_c_n[c], le.transform(y_c[c]))
            probs = clf.predict_proba(X_pub_n)
            probs_ord = np.zeros((len(X_pub_n), N_CLASSES))
            for j, cn in enumerate(le.classes_):
                probs_ord[:, CLASSES.index(cn)] = probs[:, j]
            probs_ord = np.clip(probs_ord, 1e-8, 1.0)
            w = float(len(X_c_n[c]))
            sum_logits += w * np.log(probs_ord)
            total_w += w
            print(f"    Cliente {c}: OK")
        except Exception as ex:
            print(f"    Cliente {c}: falhou ({ex})")

    if total_w > 0:
        avg = sum_logits / total_w
        avg /= 2.0  # temperatura=2
        avg -= avg.max(axis=1, keepdims=True)
        exp_a = np.exp(avg)
        tp = (exp_a / exp_a.sum(axis=1, keepdims=True)).astype(np.float32)
    else:
        tp = np.full((len(X_pub_n), N_CLASSES), 1.0/N_CLASSES, dtype=np.float32)

    tp_path = os.path.join(shared_dir, "teacher_probs.bin")
    with open(tp_path, "wb") as f:
        for row in tp:
            f.write(struct.pack(f"<{N_CLASSES}f", *row.tolist()))
    tp_meta = {"n_pub": len(X_pub_n), "n_classes": N_CLASSES,
               "temperature": 2.0, "bytes_per_row": N_CLASSES * 4,
               "total_bytes": len(X_pub_n) * N_CLASSES * 4}
    with open(os.path.join(shared_dir, "teacher_probs_meta.json"), "w") as f:
        json.dump(tp_meta, f, indent=2)
    print(f"  teacher_probs.bin: {len(X_pub_n)} amostras")

    print(f"\n{'='*55}")
    print(f"Dataset real gerado em: {output_dir}/")
    print(f"  D_pub : {len(X_pub_n)} amostras")
    print(f"  D_test: {len(X_test_n)} amostras")
    for c in sorted(X_c_n):
        print(f"  Cliente {c}: {len(X_c_n[c])} amostras")
    print(f"{'='*55}")

# ── Interface de coleta ───────────────────────────────────────────────────────

def clear_line():
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

def print_status(label, counts, frame_buf_size, paused):
    bar = {c: "█" * min(counts[c], 30) for c in CLASSES}
    label_str = label if label else "(nenhum — pressione E/S/A/L)"
    status = "PAUSADO" if paused else "GRAVANDO"
    sys.stdout.write(
        f"\r[{status}] Label: {label_str:12s}  "
        f"E:{counts['EMPTY']:3d} S:{counts['STATIONARY']:3d} "
        f"A:{counts['APPROACHING']:3d} L:{counts['LEAVING']:3d}  "
        f"buf:{frame_buf_size}/{T_WINDOW}  "
    )
    sys.stdout.flush()

def collect_loop(ser, windows_X, windows_y, stop_event):
    """
    Thread de leitura do sensor. Preenche windows_X / windows_y
    com base no label atual (compartilhado via lista mutável).
    """
    frame_buf = deque(maxlen=T_WINDOW)
    counts = {c: 0 for c in CLASSES}
    current_label = [None]   # mutável compartilhado
    paused = [False]
    eng_ok = [False]

    # Aguarda linha de boot
    print("\nAguardando ESP32...", end="", flush=True)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if line:
            try:
                d = json.loads(line)
                if "boot" in d:
                    eng_ok[0] = bool(d.get("eng", 0))
                    print(f" OK! Engineering Mode: {'SIM' if eng_ok[0] else 'NÃO (modo básico)'}")
                    break
            except Exception:
                pass
    else:
        print(" timeout — verifique a conexão e reinicie o ESP32.")
        stop_event.set()
        return

    print("\n" + "="*55)
    print("  Teclas:  E=EMPTY  S=STATIONARY  A=APPROACHING  L=LEAVING")
    print("           P=Pausar/Retomar  Q=Finalizar")
    print("="*55)
    print("Dica: mantenha a cena estável por 2s antes de mudar o label.\n")

    while not stop_event.is_set():
        # ── Leitura de tecla ─────────────────────────────────────────────────
        key = getch(timeout=0.02)
        if key:
            ku = key.upper()
            if ku == 'Q':
                stop_event.set()
                break
            elif ku == 'P':
                paused[0] = not paused[0]
                frame_buf.clear()  # descarta buffer ao pausar/retomar
            elif ku in ('E','S','A','L'):
                new_label = KEY_MAP[ku]
                if new_label != current_label[0]:
                    current_label[0] = new_label
                    frame_buf.clear()  # descarta buffer ao trocar de label
                    clear_line()
                    print(f"\n  → Label: {new_label}  (aguardando {T_WINDOW} frames...)\n")

        # ── Leitura de frame do sensor ───────────────────────────────────────
        if ser.in_waiting:
            try:
                line = ser.readline().decode("ascii", errors="ignore").strip()
            except Exception:
                continue
            if not line:
                continue
            feat = parse_frame(line)
            if feat is None:
                continue

            if paused[0] or current_label[0] is None:
                print_status(current_label[0], counts, len(frame_buf), paused[0])
                continue

            frame_buf.append(feat)

            if len(frame_buf) == T_WINDOW:
                # Janela completa — salva e avança (sliding window com passo T_WINDOW)
                window_flat = [v for frame in frame_buf for v in frame]
                windows_X.append(window_flat)
                windows_y.append(current_label[0])
                counts[current_label[0]] += 1
                frame_buf.clear()  # sem sobreposição: janelas independentes

            print_status(current_label[0], counts, len(frame_buf), paused[0])

    clear_line()
    print(f"\n  Coleta encerrada.")
    print(f"  Janelas coletadas: {len(windows_X)}")
    for c in CLASSES:
        print(f"    {c}: {counts[c]}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Coleta dados reais do HLK-LD2410C via ESP32 e gera dataset FedKD-MR"
    )
    parser.add_argument("--port", required=True,
                        help="Porta serial do ESP32 (ex: COM3 ou /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baudrate USB (padrão 115200)")
    parser.add_argument("--output", default="output_fedkd_real",
                        help="Diretório de saída (padrão: output_fedkd_real)")
    parser.add_argument("--n_clients", type=int, default=8,
                        help="Número de clientes FedKD (padrão 8)")
    parser.add_argument("--n_pub_per_class", type=int, default=20,
                        help="Amostras públicas por classe (padrão 20)")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Fração de teste (padrão 0.2)")
    args = parser.parse_args()

    print(f"\nFedKD-MR — Coleta de Dados Reais")
    print(f"  Porta: {args.port}  Baud: {args.baud}")
    print(f"  Saída: {args.output}/\n")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"[ERRO] Não foi possível abrir {args.port}: {e}")
        sys.exit(1)

    windows_X = []
    windows_y = []
    stop_event = threading.Event()

    # Coleta em thread separada para não bloquear a leitura de teclas
    t = threading.Thread(target=collect_loop,
                         args=(ser, windows_X, windows_y, stop_event),
                         daemon=True)
    t.start()
    t.join()
    ser.close()

    if len(windows_X) == 0:
        print("\nNenhuma janela coletada. Verifique a conexão.")
        sys.exit(0)

    # Verifica mínimo por classe
    counts = {c: sum(1 for y in windows_y if y == c) for c in CLASSES}
    min_count = min(counts.values())
    if min_count < args.n_pub_per_class + 2:
        print(f"\n[AVISO] Classe(s) com poucos dados: {counts}")
        print(f"  Mínimo recomendado por classe: {args.n_pub_per_class + 2} janelas.")
        print(f"  Ajuste --n_pub_per_class ou colete mais dados.")

    print(f"\n{'='*55}")
    print("Gerando dataset FedKD...")
    split_and_export(
        windows_X, windows_y,
        output_dir=args.output,
        n_clients=args.n_clients,
        n_pub_per_class=args.n_pub_per_class,
        test_ratio=args.test_ratio,
        non_iid=False,  # dataset real: split IID (mais simples e justo)
    )

    print(f"\nPróximos passos:")
    print(f"  1. Distribua os arquivos de output_fedkd_real/ para ESP32 e RPi")
    print(f"     (mesma estrutura que output_fedkd/ gerado pelo prepare_dataset.py)")
    print(f"  2. Inicie o servidor: cd server/ && python fedkd_server.py")
    print(f"  3. Inicie os clientes RPi: python rpi_client/fedkd_client.py --config ...")


if __name__ == "__main__":
    main()
