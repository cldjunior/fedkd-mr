"""
prepare_dataset.py — FedKD-MR com ESP32 + HLK-LD2410C
======================================================
Converte o dataset simulado (Excel) em arquivos .bin + metadata.json
compatíveis com trainModelFromBinaryDataset() do projeto Atlântico.

Saída por cliente:
  output/client_{1,2,3}/
    dataset_priv.bin      ← dados privados do cliente (treino local)
    metadata.json         ← esquema de colunas (compartilhado)
  output/shared/
    dataset_pub.bin       ← conjunto público (mesma repr., 1 amostra/classe)
    teacher_probs.bin     ← consenso de logits (N_pub × N_CLASSES float32)
    teacher_probs_meta.json
  output/test/
    dataset_test.bin      ← conjunto de teste independente

Uso (Excel — dataset simulado existente):
  pip install openpyxl numpy scikit-learn
  python prepare_dataset.py --input ld2410c_dataset_simulado.xlsx [--n_clients 3]

Uso (sintético — sem Excel, para validar o pipeline com N amostras por classe):
  python prepare_dataset.py --n_synth 50   # gera 50 amostras/classe = 200 total
  python prepare_dataset.py --n_synth 100  # 100 amostras/classe = 400 total

NOTA: o dataset simulado do Excel tem apenas 8 amostras (2 por classe).
      Use --n_synth para gerar um dataset maior enquanto os sensores reais
      não chegam. As features sintéticas imitam o padrão do HLK-LD2410C:
        EMPTY:       distâncias ~0, energias ~0
        STATIONARY:  distância_estacionária > 0, energia_estacionária alta
        APPROACHING: distância_móvel decrescendo ao longo dos timesteps
        LEAVING:     distância_móvel crescendo ao longo dos timesteps
"""

import argparse
import json
import os
import struct
import warnings
import numpy as np
import openpyxl
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Configuração do dataset ─────────────────────────────────────────────────

T_WINDOW   = 8      # timesteps por janela (ajustar conforme sensor real)
N_FEATURES = 12     # features por timestep
N_INPUT    = T_WINDOW * N_FEATURES  # 96 — entrada achatada do MLP

FEATURE_COLS = [
    "moving_distance_cm", "moving_energy",
    "stationary_distance_cm", "stationary_energy",
    "moving_g0", "moving_g1", "moving_g2", "moving_g3",
    "stationary_g0", "stationary_g1", "stationary_g2", "stationary_g3",
]

CLASSES       = ["EMPTY", "STATIONARY", "APPROACHING", "LEAVING"]
N_CLASSES     = len(CLASSES)
LABEL_MAP     = {c: i + 1 for i, c in enumerate(CLASSES)}  # 1-based (Atlântico)

# Semente para reprodutibilidade
SEED = 42
np.random.seed(SEED)

# ── Arquiteturas dos 3 clientes (MLP heterogêneo) ───────────────────────────
# Refletem os modelos que serão usados no ESP32 (Config.h)
CLIENT_ARCHITECTURES = {
    1: (96, 32, 16, 4),   # MLP pequeno
    2: (96, 48, 24, 4),   # MLP médio
    3: (96, 64, 32, 4),   # MLP maior (subject to RAM check on device)
}

# ── Gerador sintético de janelas HLK-LD2410C ────────────────────────────────

def generate_synthetic_windows(n_per_class: int, noise_scale: float = 1.0) -> tuple:
    """
    Gera n_per_class amostras por classe (total 4 × n_per_class) sem Excel.

    Cada amostra é uma janela de T_WINDOW=8 timesteps × N_FEATURES=12 features:
      moving_distance_cm, moving_energy,
      stationary_distance_cm, stationary_energy,
      moving_g0..g3, stationary_g0..g3

    Padrão por classe:
      EMPTY:       todos perto de zero  (sala vazia)
      STATIONARY:  distância estacionária > 0, energia alta, sem movimento
      APPROACHING: distância móvel decrescendo 300→50 cm ao longo dos 8 steps
      LEAVING:     distância móvel crescendo 50→300 cm ao longo dos 8 steps

    noise_scale: multiplicador de ruído (1.0=padrão; 3.0=difícil, classes sobrepostas)
    """
    rng = np.random.RandomState(SEED)
    X_list, y_list = [], []
    # σ base escalonado: noise_scale=1→fácil, noise_scale=3→classes sobrepostas
    σ_base = 2.0 * noise_scale   # EMPTY baseline noise
    σ_dyn  = 6.0 * noise_scale   # APPROACHING/LEAVING distance noise
    σ_stat = 6.0 * noise_scale   # STATIONARY position noise

    for _ in range(n_per_class):

        # ── EMPTY ──────────────────────────────────────────────────────────
        # Sala vazia: todas as leituras perto de zero, sem alvo detectado.
        flat = []
        for _ in range(T_WINDOW):
            md = float(np.clip(rng.normal(0, σ_base), 0, None))
            me = float(np.clip(rng.normal(0, σ_base), 0, None))
            sd = float(np.clip(rng.normal(0, σ_base), 0, None))
            se = float(np.clip(rng.normal(0, σ_base), 0, None))
            mg = [float(np.clip(rng.normal(0, σ_base * 0.5), 0, None)) for _ in range(4)]
            sg = [float(np.clip(rng.normal(0, σ_base * 0.5), 0, None)) for _ in range(4)]
            flat.extend([md, me, sd, se] + mg + sg)
        X_list.append(flat); y_list.append("EMPTY")

        # ── STATIONARY ─────────────────────────────────────────────────────
        # Pessoa parada: sd constante e alta, se alta; md e me perto de zero.
        flat = []
        base_dist   = rng.uniform(60, 280)
        base_energy = rng.uniform(60, 100)
        gate_base   = rng.uniform(30, 70)
        for _ in range(T_WINDOW):
            md = float(np.clip(rng.normal(0, σ_base * 1.5), 0, None))
            me = float(np.clip(rng.normal(0, σ_base * 1.5), 0, None))
            sd = float(np.clip(rng.normal(base_dist, σ_stat), 0, None))
            se = float(np.clip(rng.normal(base_energy, σ_base * 2.5), 0, None))
            mg = [float(np.clip(rng.normal(0, σ_base), 0, None)) for _ in range(4)]
            sg = [float(np.clip(rng.normal(max(0, gate_base - j * 10), σ_base * 2.5), 0, None))
                  for j in range(4)]
            flat.extend([md, me, sd, se] + mg + sg)
        X_list.append(flat); y_list.append("STATIONARY")

        # ── APPROACHING ─────────────────────────────────────────────────────
        # Pessoa se aproximando: md decrescente de 250→40 cm.
        flat = []
        start_dist  = rng.uniform(220, 400)
        end_dist    = rng.uniform(25,  80)
        base_energy = rng.uniform(70,  100)
        gate_base   = rng.uniform(35,  75)
        for t in range(T_WINDOW):
            frac  = t / max(T_WINDOW - 1, 1)
            dist_t = start_dist * (1.0 - frac) + end_dist * frac
            md = float(np.clip(rng.normal(dist_t, σ_dyn), 0, None))
            me = float(np.clip(rng.normal(base_energy, σ_base * 2.5), 0, None))
            sd = float(np.clip(rng.normal(0, σ_base * 1.5), 0, None))
            se = float(np.clip(rng.normal(0, σ_base * 1.5), 0, None))
            mg = [float(np.clip(rng.normal(max(0, gate_base - j * 8), σ_base * 2.5), 0, None))
                  for j in range(4)]
            sg = [float(np.clip(rng.normal(0, σ_base), 0, None)) for _ in range(4)]
            flat.extend([md, me, sd, se] + mg + sg)
        X_list.append(flat); y_list.append("APPROACHING")

        # ── LEAVING ──────────────────────────────────────────────────────────
        # Pessoa se afastando: md crescente de 40→250 cm.
        flat = []
        start_dist  = rng.uniform(25,  80)
        end_dist    = rng.uniform(220, 400)
        base_energy = rng.uniform(70,  100)
        gate_base   = rng.uniform(35,  75)
        for t in range(T_WINDOW):
            frac  = t / max(T_WINDOW - 1, 1)
            dist_t = start_dist * (1.0 - frac) + end_dist * frac
            md = float(np.clip(rng.normal(dist_t, σ_dyn), 0, None))
            me = float(np.clip(rng.normal(base_energy, σ_base * 2.5), 0, None))
            sd = float(np.clip(rng.normal(0, σ_base * 1.5), 0, None))
            se = float(np.clip(rng.normal(0, σ_base * 1.5), 0, None))
            mg = [float(np.clip(rng.normal(max(0, gate_base - j * 8), σ_base * 2.5), 0, None))
                  for j in range(4)]
            sg = [float(np.clip(rng.normal(0, σ_base), 0, None)) for _ in range(4)]
            flat.extend([md, me, sd, se] + mg + sg)
        X_list.append(flat); y_list.append("LEAVING")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    print(f"  Geradas {len(X)} amostras sintéticas ({n_per_class}/classe, noise_scale={noise_scale:.1f})")
    print(f"  Distribuição: { {c: int((y==c).sum()) for c in CLASSES} }")
    print(f"  Faixa de valores: min={X.min():.2f}  max={X.max():.2f}")
    return X, y


# ── Funções auxiliares ───────────────────────────────────────────────────────

def load_excel(path: str):
    """Carrega o Excel e retorna lista de dicts com cada linha."""
    wb = openpyxl.load_workbook(path)
    ws = wb["dataset_simulado"]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        rows.append(dict(zip(headers, row)))
    return rows


def build_windows(rows):
    """
    Agrupa linhas por sample_id, ordena por t e achata a janela T×F.
    Retorna: X [N_samples, N_INPUT], y_str [N_samples] (rótulo string).
    """
    samples = defaultdict(list)
    for r in rows:
        samples[r["sample_id"]].append(r)

    X, y_str = [], []
    for sid in sorted(samples.keys()):
        window = sorted(samples[sid], key=lambda r: r["t"])
        if len(window) != T_WINDOW:
            print(f"  [AVISO] sample {sid} tem {len(window)} timesteps (esperado {T_WINDOW}), ignorando.")
            continue
        flat = []
        for step in window:
            for col in FEATURE_COLS:
                flat.append(float(step[col]))
        label = window[0]["label"]
        X.append(flat)
        y_str.append(label)

    return np.array(X, dtype=np.float32), np.array(y_str)


def normalize(X_train, X_other_list):
    """Min-max normalização baseada em X_train; aplica ao restante."""
    mins  = X_train.min(axis=0)
    maxs  = X_train.max(axis=0)
    rng   = maxs - mins
    rng[rng == 0] = 1.0  # evita divisão por zero

    X_norm = [(X - mins) / rng for X in [X_train] + X_other_list]
    norm_stats = {"mins": mins.tolist(), "maxs": maxs.tolist()}
    return X_norm, norm_stats


def split_dataset(X, y_str, n_clients=3, n_pub_per_class=1, test_ratio=0.25,
                  non_iid=False):
    """
    Divide as amostras em D_pub (balanceado), D_test e D_priv por cliente.

    non_iid=True: split Dirichlet-style — cada cliente recebe classes
    com proporções desiguais. Isso simula cenários federated realistas onde
    clientes têm distribuições diferentes (ex: cliente 1 vê principalmente
    STATIONARY+APPROACHING, cliente 2 vê LEAVING+EMPTY).

    Padrão para 3 clientes com non_iid=True:
      Cliente 1: STATIONARY e APPROACHING (biased 70%)
      Cliente 2: LEAVING e EMPTY          (biased 70%)
      Cliente 3: distribuição uniforme     (balanceado)
    Isso força o KD a transferir conhecimento de classes sub-representadas.
    """
    unique_classes = CLASSES
    pub_idx, test_idx, priv_idx = [], [], []

    for cls in unique_classes:
        cls_idx = np.where(y_str == cls)[0]
        np.random.shuffle(cls_idx)
        n_pub  = min(n_pub_per_class, len(cls_idx))
        n_test = max(1, int(len(cls_idx) * test_ratio)) if len(cls_idx) > n_pub else 0
        pub_idx.extend(cls_idx[:n_pub].tolist())
        test_idx.extend(cls_idx[n_pub:n_pub + n_test].tolist())
        priv_idx.extend(cls_idx[n_pub + n_test:].tolist())

    # Distribui D_priv entre clientes
    client_idx = {i+1: [] for i in range(n_clients)}

    if non_iid and n_clients >= 2:
        # Split não-IID: clientes com viés de classe
        # Construir dict de índices por classe (do pool privado)
        priv_by_class = {cls: [] for cls in CLASSES}
        for idx in priv_idx:
            priv_by_class[y_str[idx]].append(idx)

        # Definir proporções por cliente:
        # Padrão para 3 clientes — ajusta para qualquer número de clientes
        # Matriz [n_clients x n_classes] de frações (linhas somam ≈1 por classe)
        if n_clients == 2:
            # Cliente 1: forte em STATIONARY+APPROACHING
            # Cliente 2: forte em LEAVING+EMPTY
            fracs = np.array([
                [0.1, 0.7, 0.7, 0.1],   # cliente 1: EMPTY, STAT, APPR, LEAV
                [0.9, 0.3, 0.3, 0.9],   # cliente 2
            ], dtype=float)
        elif n_clients == 3:
            fracs = np.array([
                [0.1, 0.7, 0.7, 0.1],   # cliente 1: STAT+APPR dominant
                [0.7, 0.1, 0.1, 0.7],   # cliente 2: EMPT+LEAV dominant
                [0.2, 0.2, 0.2, 0.2],   # cliente 3: balanced
            ], dtype=float)
        else:
            # n_clients > 3: round-robin IID mas com Dirichlet
            alpha = 0.5  # baixo alpha → mais heterogêneo
            fracs_raw = np.random.dirichlet([alpha]*n_clients, size=N_CLASSES).T  # [n_clients, n_classes]
            fracs = fracs_raw

        # Normalizar por coluna para frações somarem 1 por classe
        col_sums = fracs.sum(axis=0)
        fracs = fracs / col_sums[np.newaxis, :]

        for cls_i, cls in enumerate(CLASSES):
            cls_pool = priv_by_class[cls].copy()
            np.random.shuffle(cls_pool)
            n_cls = len(cls_pool)
            if n_cls == 0:
                continue
            # Distribuir proporcionalmente
            cuts = [0]
            for c in range(n_clients - 1):
                cuts.append(cuts[-1] + int(round(fracs[c, cls_i] * n_cls)))
            cuts.append(n_cls)
            for c in range(n_clients):
                client_idx[c+1].extend(cls_pool[cuts[c]:cuts[c+1]])
    else:
        # Split IID: round-robin simples
        np.random.shuffle(priv_idx)
        for k, idx in enumerate(priv_idx):
            client_idx[(k % n_clients) + 1].append(idx)

    total = len(X)
    mode_str = "não-IID" if non_iid else "IID"
    print(f"\n  Dataset: {total} amostras totais  [{mode_str}]")
    print(f"  D_pub : {len(pub_idx)} amostras — classes: {list(y_str[pub_idx])[:8]}{'...' if len(pub_idx)>8 else ''}")
    print(f"  D_test: {len(test_idx)} amostras")
    for c, idxs in client_idx.items():
        dist = {cls: int((y_str[idxs] == cls).sum()) for cls in CLASSES} if idxs else {}
        print(f"  D_priv cliente {c}: {len(idxs)} amostras  {dist}")

    if len(priv_idx) == 0:
        print("\n  [AVISO] Sem amostras privadas! Dataset simulado muito pequeno.")
        print("  O pipeline será validado usando D_pub como D_priv (apenas para teste).")
        for c in client_idx:
            client_idx[c] = pub_idx.copy()

    return pub_idx, test_idx, client_idx


def build_feature_names():
    """Gera nomes das features achatadas: t0_moving_distance_cm, ..., t7_stationary_g3."""
    names = []
    for t in range(T_WINDOW):
        for col in FEATURE_COLS:
            names.append(f"t{t}_{col}")
    return names


def build_metadata(feature_names, label_column="label"):
    """Constrói o metadata.json no formato Atlântico."""
    schema = []
    offset = 0
    for name in feature_names:
        schema.append({
            "name": name,
            "type": "float32",
            "c_type": "float",
            "bytes": 4,
            "offset": offset,
        })
        offset += 4
    # Label: uint8 (encoded 1-based)
    schema.append({
        "name": label_column,
        "type": "uint8",
        "c_type": "uint8_t",
        "bytes": 1,
        "offset": offset,
    })
    bytes_per_row = offset + 1

    meta = {
        "label_column": label_column,
        "label_map": LABEL_MAP,   # activa encoded_labels no ESP32
        "schema": schema,
        "bytes_per_row": bytes_per_row,
        "n_input": N_INPUT,
        "n_classes": N_CLASSES,
        "t_window": T_WINDOW,
        "n_features_per_step": N_FEATURES,
        "feature_columns": feature_names,
        "classes": CLASSES,
    }
    return meta, bytes_per_row


def write_bin(path: str, X: np.ndarray, y_str: np.ndarray, bytes_per_row: int):
    """Escreve arquivo .bin no formato Atlântico: [f0..f95 float32, label uint8]."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        for xi, yi in zip(X, y_str):
            # 96 floats
            f.write(struct.pack(f"<{N_INPUT}f", *xi.tolist()))
            # label uint8 (1-based encoded)
            label_encoded = LABEL_MAP[yi]
            f.write(struct.pack("<B", label_encoded))
    size = os.path.getsize(path)
    print(f"  Escrito: {path}  ({len(X)} amostras, {size} bytes)")


def train_simulated_model(X_train, y_str_train, hidden_layers, client_id):
    """Treina um MLP sklearn para simular o modelo local do cliente."""
    le = LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y_str_train)

    hidden = hidden_layers[1:-1]  # remove input e output
    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="tanh",
        max_iter=500,
        random_state=SEED + client_id,
        learning_rate_init=0.01,
    )
    try:
        clf.fit(X_train, y_enc)
        train_acc = clf.score(X_train, y_enc)
        print(f"  Cliente {client_id} MLP{hidden}: acc_treino={train_acc:.3f}")
    except Exception as e:
        print(f"  [AVISO] Cliente {client_id} treino falhou ({e}), usando modelo aleatório")
        # Retorna probabilidades uniformes se treino falhar (dataset muito pequeno)
        clf = None
    return clf, le


def compute_teacher_probs(models_le, X_pub, client_n_samples, temperature=2.0):
    """
    Calcula teacher_global_probs como média ponderada dos logits dos clientes
    sobre o dataset público. Equivale à célula ID=15 do notebook.

    Args:
        models_le: lista de (clf, le) por cliente
        X_pub: features do dataset público [N_pub, N_INPUT]
        client_n_samples: lista de n_samples por cliente (para ponderação)
        temperature: temperatura de suavização

    Returns:
        teacher_probs: [N_pub, N_CLASSES] float32
    """
    N_pub = len(X_pub)
    sum_logits = np.zeros((N_pub, N_CLASSES), dtype=np.float64)
    sum_weights = 0.0

    for i, (clf, le) in enumerate(models_le):
        weight = float(client_n_samples[i]) if client_n_samples[i] > 0 else 1.0

        if clf is None:
            # Modelo falhou: contribui com logits uniformes
            logits = np.zeros((N_pub, N_CLASSES), dtype=np.float64)
        else:
            # sklearn retorna prob após softmax; convertemos de volta para logits
            # aproximados via log para permitir temperatura
            probs = clf.predict_proba(X_pub)
            # Garantir ordem das classes consistente com CLASSES
            cls_order = list(le.classes_)
            probs_ordered = np.zeros((N_pub, N_CLASSES), dtype=np.float64)
            for j, cls_name in enumerate(cls_order):
                target_col = CLASSES.index(cls_name)  # índice canônico (0-based)
                probs_ordered[:, target_col] = probs[:, j]
            probs_ordered = np.clip(probs_ordered, 1e-8, 1.0)
            logits = np.log(probs_ordered)

        sum_logits  += weight * logits
        sum_weights += weight

    avg_logits = sum_logits / sum_weights

    # Aplicar temperatura e softmax → teacher_probs
    scaled = avg_logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)  # estabilidade numérica
    exp_s = np.exp(scaled)
    teacher_probs = (exp_s / exp_s.sum(axis=1, keepdims=True)).astype(np.float32)

    print(f"\n  teacher_probs shape: {teacher_probs.shape}")
    print(f"  Exemplo (amostra 0): {teacher_probs[0].round(4)}")
    return teacher_probs


def write_teacher_probs(path: str, teacher_probs: np.ndarray):
    """Salva teacher_probs como [N_pub × N_CLASSES] float32 little-endian."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    N_pub = len(teacher_probs)
    with open(path, "wb") as f:
        for row in teacher_probs:
            f.write(struct.pack(f"<{N_CLASSES}f", *row.tolist()))
    size = os.path.getsize(path)
    print(f"  Escrito: {path}  ({N_pub} amostras, {size} bytes)")


def verify_bin(path: str, meta: dict):
    """Verifica consistência do arquivo .bin gerado."""
    bpr = meta["bytes_per_row"]
    size = os.path.getsize(path)
    if size % bpr != 0:
        print(f"  [ERRO] {path}: tamanho {size} não é múltiplo de {bpr}")
        return False
    n = size // bpr
    print(f"  OK: {path} — {n} amostras, {bpr} bytes/linha")
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepara datasets para FedKD-MR ESP32")
    parser.add_argument("--input", default="",
                        help="Caminho para o arquivo Excel do dataset simulado "
                             "(ignorado quando --n_synth é passado)")
    parser.add_argument("--n_synth", type=int, default=0,
                        help="Amostras sintéticas por classe (0 = usa Excel). "
                             "Ex.: --n_synth 50 gera 200 amostras totais.")
    parser.add_argument("--output", default="output_fedkd",
                        help="Diretório de saída")
    parser.add_argument("--n_clients", type=int, default=3,
                        help="Número de clientes ESP32")
    parser.add_argument("--n_pub_per_class", type=int, default=1,
                        help="Amostras públicas por classe")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Temperatura para suavização dos logits")
    parser.add_argument("--noise_scale", type=float, default=1.0,
                        help="Multiplicador de ruído para dados sintéticos "
                             "(1.0=fácil, 3.0=difícil, classes sobrepostas). "
                             "Use ≥2.0 para ver KD delta > 0.")
    parser.add_argument("--non_iid", action="store_true",
                        help="Distribuição não-IID entre clientes: cada cliente "
                             "recebe proporções diferentes por classe. "
                             "Melhor para avaliar o benefício da federação.")
    args = parser.parse_args()

    # Fallback: se --input não foi passado explicitamente e --n_synth também não,
    # usa o nome default do Excel (mantém compatibilidade com uso antigo)
    if not args.input and not args.n_synth:
        args.input = "ld2410c_dataset_simulado.xlsx"

    print("=" * 60)
    print("FedKD-MR — Preparação de Datasets para ESP32")
    print("=" * 60)

    # ── 1. Carregar / gerar dados ─────────────────────────────────
    if args.n_synth > 0:
        print(f"\n[1] Gerando {args.n_synth} amostras sintéticas por classe "
              f"(noise_scale={args.noise_scale})...")
        X, y_str = generate_synthetic_windows(args.n_synth, noise_scale=args.noise_scale)
    else:
        input_path = args.input or "ld2410c_dataset_simulado.xlsx"
        print(f"\n[1] Carregando {input_path}...")
        rows = load_excel(input_path)
        X, y_str = build_windows(rows)
        print(f"  {len(X)} janelas × {N_INPUT} features, {N_CLASSES} classes")
        print(f"  Distribuição: { {c: int((y_str==c).sum()) for c in CLASSES} }")

    # ── 2. Dividir em D_pub, D_test, D_priv por cliente ──────────
    print(f"\n[2] Dividindo dataset ({args.n_clients} clientes)...")
    pub_idx, test_idx, client_idx = split_dataset(
        X, y_str,
        n_clients=args.n_clients,
        n_pub_per_class=args.n_pub_per_class,
        non_iid=args.non_iid,
    )

    X_pub  = X[pub_idx];  y_pub  = y_str[pub_idx]
    X_test = X[test_idx]; y_test = y_str[test_idx]
    X_clients = {c: X[idxs] for c, idxs in client_idx.items()}
    y_clients = {c: y_str[idxs] for c, idxs in client_idx.items()}

    # ── 3. Normalizar (baseado em todos os dados privados reunidos) ─
    # normalize() devolve [all_priv_norm, *X_other_norm]: índice 0 = all_priv,
    # índice 1 = X_pub_n, índice 2 = X_test_n, índices 3.. = cada cliente.
    print(f"\n[3] Normalizando...")
    all_priv = np.concatenate(list(X_clients.values()), axis=0) \
               if any(len(v) > 0 for v in X_clients.values()) else X
    norm_inputs = [X_pub, X_test] + [X_clients[c] for c in sorted(X_clients)]
    X_norm_all, norm_stats = normalize(all_priv, norm_inputs)
    # índice 0 = all_priv normalizado (X_train) — descartado aqui
    X_pub_n  = X_norm_all[1]   # normalizado de X_pub
    X_test_n = X_norm_all[2]   # normalizado de X_test
    X_clients_n = {c: X_norm_all[3 + i] for i, c in enumerate(sorted(X_clients))}

    # ── 4. Construir metadata ─────────────────────────────────────
    feature_names = build_feature_names()
    meta, bytes_per_row = build_metadata(feature_names)
    meta["normalization"] = norm_stats
    meta["rows_pub"]  = int(len(X_pub_n))
    meta["rows_test"] = int(len(X_test_n))

    # Salvar metadata compartilhado
    shared_dir = os.path.join(args.output, "shared")
    os.makedirs(shared_dir, exist_ok=True)
    with open(os.path.join(shared_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  metadata.json: {bytes_per_row} bytes/linha, {N_INPUT} features + 1 label")

    # ── 5. Escrever arquivos .bin ──────────────────────────────────
    print(f"\n[4] Escrevendo arquivos binários...")

    # D_pub (compartilhado)
    write_bin(os.path.join(shared_dir, "dataset_pub.bin"), X_pub_n, y_pub, bytes_per_row)

    # D_test
    test_dir = os.path.join(args.output, "test")
    if len(X_test_n) > 0:
        write_bin(os.path.join(test_dir, "dataset_test.bin"), X_test_n, y_test, bytes_per_row)
    else:
        print("  [INFO] D_test vazio (dataset muito pequeno); usando D_pub como fallback de teste.")
        os.makedirs(test_dir, exist_ok=True)
        write_bin(os.path.join(test_dir, "dataset_test.bin"), X_pub_n, y_pub, bytes_per_row)

    # D_priv por cliente + copiar metadata
    for c in sorted(X_clients_n.keys()):
        cdir = os.path.join(args.output, f"client_{c}")
        os.makedirs(cdir, exist_ok=True)
        write_bin(os.path.join(cdir, "dataset_priv.bin"), X_clients_n[c], y_clients[c], bytes_per_row)
        # Cópia do metadata para cada cliente
        with open(os.path.join(cdir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ── 6. Treinar modelos simulados e gerar teacher_probs ─────────
    print(f"\n[5] Treinando {args.n_clients} MLPs para gerar teacher_probs...")
    models_le = []
    client_n_samples = []
    for c in sorted(X_clients_n.keys()):
        arch = CLIENT_ARCHITECTURES.get(c, (N_INPUT, 32, 16, N_CLASSES))
        clf, le = train_simulated_model(X_clients_n[c], y_clients[c], arch, c)
        models_le.append((clf, le))
        client_n_samples.append(len(X_clients_n[c]))

    print(f"\n[6] Calculando teacher_probs (temperatura={args.temperature})...")
    teacher_probs = compute_teacher_probs(models_le, X_pub_n, client_n_samples, args.temperature)

    write_teacher_probs(os.path.join(shared_dir, "teacher_probs.bin"), teacher_probs)

    # Metadata do teacher_probs (para o ESP32 saber N_pub e N_CLASSES)
    tp_meta = {
        "n_pub": int(len(X_pub_n)),
        "n_classes": N_CLASSES,
        "temperature": args.temperature,
        "bytes_per_row": N_CLASSES * 4,
        "total_bytes": int(len(X_pub_n)) * N_CLASSES * 4,
    }
    with open(os.path.join(shared_dir, "teacher_probs_meta.json"), "w") as f:
        json.dump(tp_meta, f, indent=2)
    print(f"  teacher_probs_meta.json: n_pub={tp_meta['n_pub']}, {tp_meta['total_bytes']} bytes total")

    # ── 7. Verificar integridade ───────────────────────────────────
    print(f"\n[7] Verificando arquivos gerados...")
    verify_bin(os.path.join(shared_dir, "dataset_pub.bin"), meta)
    verify_bin(os.path.join(test_dir, "dataset_test.bin"), meta)
    for c in sorted(X_clients_n.keys()):
        verify_bin(os.path.join(args.output, f"client_{c}", "dataset_priv.bin"), meta)

    # ── 8. Resumo ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Arquivos prontos para o ESP32:")
    print(f"  Para cada cliente {list(sorted(X_clients_n.keys()))}:")
    print(f"    output/client_N/dataset_priv.bin  → LittleFS: /dataset_priv.bin")
    print(f"    output/client_N/metadata.json     → LittleFS: /metadata.json")
    print(f"  Compartilhado (igual em todos):")
    print(f"    output/shared/dataset_pub.bin     → LittleFS: /dataset_pub.bin")
    print(f"    output/shared/metadata.json       (mesmo arquivo)")
    print(f"    output/shared/teacher_probs.bin   → LittleFS: /teacher_probs.bin")
    print(f"    output/shared/teacher_probs_meta.json → LittleFS: /teacher_probs_meta.json")
    print(f"\nUse o PlatformIO 'Upload Filesystem Image' para gravar em LittleFS.")
    print("Copie os arquivos do cliente correspondente para a pasta 'data/' do projeto.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
