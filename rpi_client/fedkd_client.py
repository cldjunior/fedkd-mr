"""
fedkd_client.py — FedKD-MR Cliente Raspberry Pi
=================================================
Pipeline completo de aprendizado federado com Knowledge Distillation.

Round por round:
  1. Treino local supervisionado (D_priv)
  2. Inferência em D_pub → publica logits via MQTT
  3. Aguarda teacher_probs do servidor (agregação dos logits de todos os clientes)
  4. Knowledge Distillation (D_pub + teacher_probs como soft targets)
  5. Fine-tuning supervisionado (D_priv)
  6. Avaliação e publicação de 'done'

Uso:
    python fedkd_client.py --config configs/client_gru.yaml
    python fedkd_client.py --config configs/client_gru.yaml --local-epochs 100
"""

import logging
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from config       import get_config
from dataset      import load_dataset, make_loader, iter_pub_samples
from models       import build_model
from mqtt_handler import MQTTHandler

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fedkd_client")


# ── Treino local supervisionado ───────────────────────────────────────────────

def train_supervised(model: nn.Module, loader, cfg, tag: str = "local") -> tuple[float, float]:
    """
    Treina o modelo com CrossEntropyLoss por `epochs` épocas.

    Returns:
        (acc, loss) médios na última época.
    """
    epochs = cfg.local_epochs if tag == "local" else cfg.ft_epochs
    lr     = cfg.local_lr     if tag == "local" else cfg.ft_lr

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device    = next(model.parameters()).device

    model.train()
    last_acc  = 0.0
    last_loss = float("nan")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total   = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds   = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += len(y_batch)

        last_acc  = correct / total if total else 0.0
        last_loss = total_loss / max(total, 1)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            logger.info("[%s] Epoch %3d/%d — loss=%.4f acc=%.4f",
                        tag, epoch, epochs, last_loss, last_acc)

    return last_acc, last_loss


# ── Knowledge Distillation ───────────────────────────────────────────────────

def kd_loss(student_logits: torch.Tensor,
            teacher_probs:  torch.Tensor,
            temperature:    float) -> torch.Tensor:
    """
    Soft cross-entropy KD loss — numericamente estável.

    Equivale a KL(teacher || student) + H(teacher), mas evita log(teacher_probs),
    que produz NaN quando o professor é confiante (valores próximos de 0 ou 1).

    student_logits: (B, C) — logits brutos do aluno
    teacher_probs:  (B, C) — probabilidades do professor (já normalizadas)
    temperature:    T ≥ 1 (suaviza as distribuições)
    """
    T = temperature
    # Garante que logits não contenham NaN/Inf antes de log_softmax
    student_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=50.0, neginf=-50.0)
    log_p = F.log_softmax(student_logits / T, dim=1)
    # Clamp: log_softmax retorna -inf quando softmax underflowa para 0.0 em float32.
    # Clamp em -100 ≈ probabilidade mínima e^{-100} ≈ 3.7e-44 → sem -inf → sem NaN.
    log_p = log_p.clamp(min=-100.0)
    loss = -(teacher_probs * log_p).sum(dim=1).mean()
    return loss * (T ** 2)


def train_kd(model: nn.Module, pub_loader, teacher_probs_np: np.ndarray,
             cfg) -> float:
    """
    Realiza Knowledge Distillation usando D_pub e teacher_probs do servidor.

    Args:
        model:            Modelo PyTorch a ser treinado.
        pub_loader:       DataLoader de D_pub.
        teacher_probs_np: np.ndarray (n_pub, n_classes) — soft targets.
        cfg:              Namespace de configuração.

    Returns:
        Loss KD médio na última época.
    """
    device    = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.kd_lr)
    teacher_t = torch.from_numpy(teacher_probs_np.astype(np.float32)).to(device)

    # ── Diagnóstico: loga estatísticas dos teacher_probs recebidos ────────────
    n_nan = torch.isnan(teacher_t).sum().item()
    n_inf = torch.isinf(teacher_t).sum().item()
    if n_nan > 0 or n_inf > 0:
        logger.warning("[KD] teacher_probs tem %d NaN e %d Inf — serão substituídos por 0",
                       n_nan, n_inf)
    else:
        logger.debug("[KD] teacher_probs OK — min=%.4f max=%.4f",
                     teacher_t.min().item(), teacher_t.max().item())

    # ── Sanitização completa ─────────────────────────────────────────────────
    # 1) nan_to_num ANTES do clamp: clamp(NaN) == NaN, portanto NaN deve ser
    #    eliminado antes de qualquer operação aritmética.
    teacher_t = torch.nan_to_num(teacher_t, nan=0.0, posinf=1.0, neginf=0.0)
    # 2) Epsilon em cada elemento: evita zeros exatos (0 × log_p = 0×(-inf)=NaN)
    teacher_t = teacher_t.clamp(min=1e-9)
    # 3) Renormaliza para que cada linha some 1
    row_sums  = teacher_t.sum(dim=1, keepdim=True)
    teacher_t = teacher_t / row_sums

    model.train()
    last_loss = float("nan")

    for epoch in range(1, cfg.kd_epochs + 1):
        total_loss = 0.0
        total      = 0
        # Itera sobre D_pub emparelhado com teacher_probs
        for batch_idx, (x_batch, _) in enumerate(pub_loader):
            x_batch = x_batch.to(device)

            # Fatia de teacher_probs correspondente ao batch
            start = batch_idx * pub_loader.batch_size
            end   = start + len(x_batch)
            tp_batch = teacher_t[start:end]

            optimizer.zero_grad()
            logits = model(x_batch)
            loss   = kd_loss(logits, tp_batch, cfg.kd_temperature)

            if torch.isnan(loss):
                logger.warning("[KD] loss=NaN detectado no batch %d — batch ignorado", batch_idx)
                continue

            loss.backward()
            # Clip de gradiente: evita explosão que corromperia pesos
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * len(x_batch)
            total      += len(x_batch)

        last_loss = total_loss / max(total, 1)
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.kd_epochs:
            logger.info("[KD] Epoch %3d/%d — kd_loss=%.6f", epoch, cfg.kd_epochs, last_loss)

    return last_loss


# ── Avaliação ────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader, cfg) -> tuple[float, float]:
    """
    Avalia o modelo no loader fornecido.

    Returns:
        (accuracy, macro_f1)
    """
    from collections import defaultdict

    device = next(model.parameters()).device
    model.eval()

    correct = 0
    total   = 0
    # Para F1 macro: TP, FP, FN por classe
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += len(y_batch)

            for pred, true in zip(preds.cpu().tolist(), y_batch.cpu().tolist()):
                if pred == true:
                    tp[true] += 1
                else:
                    fp[pred] += 1
                    fn[true] += 1

    acc = correct / max(total, 1)

    n_classes = cfg.n_classes
    f1s = []
    for c in range(n_classes):
        precision = tp[c] / max(tp[c] + fp[c], 1)
        recall    = tp[c] / max(tp[c] + fn[c], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))

    return acc, macro_f1


# ── Inferência em D_pub → logits ─────────────────────────────────────────────

def infer_pub_logits(model: nn.Module, pub_loader, cfg) -> np.ndarray:
    """
    Roda D_pub pelo modelo e retorna PROBABILIDADES (softmax dos logits).

    O servidor agrega via avg_probs e executa np.log(avg_probs + eps),
    portanto espera valores em (0, 1) — não logits brutos (que podem ser
    negativos e quebrar o log com NaN).

    Returns:
        np.ndarray de shape (n_pub, n_classes), dtype float32, valores em (0,1).
    """
    device = next(model.parameters()).device
    model.eval()
    all_logits = []

    with torch.no_grad():
        for x_batch, _ in pub_loader:
            x_batch = x_batch.to(device)
            logits  = model(x_batch)
            # Converte para probabilidades antes de enviar ao servidor
            probs   = F.softmax(logits, dim=1)
            all_logits.append(probs.cpu().numpy())

    return np.vstack(all_logits).astype(np.float32)


# ── Pipeline principal ───────────────────────────────────────────────────────

def main():
    cfg = get_config()

    # Seed global
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger.info("=== FedKD-MR Cliente Raspberry Pi ===")
    logger.info("Cliente: %s | Modelo: %s | Device: %s",
                cfg.client_name, cfg.model.upper(), cfg.device)

    device = torch.device(cfg.device)

    # ── Datasets ────────────────────────────────────────────────────────────
    data_dir = Path(cfg.data_dir)

    train_path = str(data_dir / cfg.train_file)
    pub_path   = str(data_dir / cfg.pub_file)
    meta_path  = str(data_dir / cfg.meta_file) if cfg.train_file.endswith(".bin") else None

    logger.info("Carregando D_priv: %s", train_path)
    train_ds   = load_dataset(train_path, meta_path)
    train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True)

    logger.info("Carregando D_pub: %s", pub_path)
    pub_meta   = meta_path  # mesma schema; None se CSV
    pub_ds     = load_dataset(pub_path, pub_meta)
    pub_loader = make_loader(pub_ds, cfg.batch_size, shuffle=False)

    logger.info("D_priv: %d amostras | D_pub: %d amostras",
                len(train_ds), len(pub_ds))

    # ── Dataset de teste (opcional — dataset_test.bin na mesma pasta) ────────
    test_path = str(data_dir / "dataset_test.bin")
    if os.path.exists(test_path):
        logger.info("Carregando D_test: %s", test_path)
        test_ds     = load_dataset(test_path, meta_path)
        test_loader = make_loader(test_ds, cfg.batch_size, shuffle=False)
        logger.info("D_test: %d amostras", len(test_ds))
    else:
        logger.warning("dataset_test.bin não encontrado — avaliação usará D_priv")
        test_loader = train_loader

    # ── Modelo ──────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    logger.info("Parâmetros do modelo: %s  (%.1f K)",
                cfg.model.upper(), model.n_params() / 1000)

    # ── MQTT ────────────────────────────────────────────────────────────────
    mqtt = MQTTHandler(cfg)
    mqtt.connect()
    # Aguarda até o broker estar estável
    time.sleep(1.5)

    round_n = 0
    try:
        while True:
            round_n = mqtt.wait_start()
            t_round = time.time()
            logger.info("══════════════════════════════")
            logger.info("ROUND %d — iniciando", round_n)

            # ── 1. Treino local ─────────────────────────────────────────────
            logger.info("[ 1 ] Treino local supervisionado (%d épocas, lr=%.4f)",
                        cfg.local_epochs, cfg.local_lr)
            t0 = time.time()
            _, loss_local = train_supervised(model, train_loader, cfg, tag="local")
            t_local_ms = int((time.time() - t0) * 1000)
            acc_local, f1_local = evaluate(model, test_loader, cfg)
            logger.info("[ 1 ] Treino local — acc=%.4f  F1=%.4f  loss=%.4f  (%dms)",
                        acc_local, f1_local, loss_local, t_local_ms)

            # ── 2. Publicação de logits ──────────────────────────────────────
            logger.info("[ 2 ] Inferência em D_pub e publicação de logits")
            logits_np = infer_pub_logits(model, pub_loader, cfg)
            mqtt.publish_logits(logits_np)

            # ── 3. Aguarda teacher_probs ────────────────────────────────────
            logger.info("[ 3 ] Aguardando teacher_probs do servidor…")
            teacher_probs = mqtt.wait_teacher()  # bloqueia

            # ── 4. Knowledge Distillation ───────────────────────────────────
            logger.info("[ 4 ] Knowledge Distillation (%d épocas, T=%.1f)",
                        cfg.kd_epochs, cfg.kd_temperature)
            t0 = time.time()
            loss_kd = train_kd(model, pub_loader, teacher_probs, cfg)
            t_kd_ms = int((time.time() - t0) * 1000)
            acc_kd, f1_kd = evaluate(model, test_loader, cfg)
            logger.info("[ 4 ] KD — acc=%.4f  F1=%.4f  kd_loss=%.6f  (%dms)",
                        acc_kd, f1_kd, loss_kd, t_kd_ms)

            # ── 5. Fine-tuning ──────────────────────────────────────────────
            logger.info("[ 5 ] Fine-tuning (%d épocas, lr=%.4f)",
                        cfg.ft_epochs, cfg.ft_lr)
            t0 = time.time()
            _, loss_ft = train_supervised(model, train_loader, cfg, tag="ft")
            t_ft_ms = int((time.time() - t0) * 1000)
            acc_ft, f1_ft = evaluate(model, test_loader, cfg)
            logger.info("[ 5 ] Fine-tuning — acc=%.4f  F1=%.4f  loss=%.4f  (%dms)",
                        acc_ft, f1_ft, loss_ft, t_ft_ms)

            elapsed = time.time() - t_round
            logger.info("ROUND %d concluído em %.1fs", round_n, elapsed)
            logger.info("══════════════════════════════")

            mqtt.publish_done(round_n,
                              acc_local=acc_local,  f1_local=f1_local,  loss_local=loss_local,
                              acc_kd=acc_kd,        f1_kd=f1_kd,        loss_kd=loss_kd,
                              acc_ft=acc_ft,        f1_ft=f1_ft,        loss_ft=loss_ft,
                              t_local_ms=t_local_ms, t_kd_ms=t_kd_ms,  t_ft_ms=t_ft_ms)

    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário.")
    except TimeoutError as e:
        logger.error("Timeout: %s", e)
        sys.exit(1)
    finally:
        mqtt.disconnect()
        logger.info("Cliente encerrado.")


if __name__ == "__main__":
    main()
