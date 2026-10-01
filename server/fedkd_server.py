"""
fedkd_server.py — Servidor de Federação FedKD-MR
=================================================
Recebe logits dos clientes ESP32 via MQTT, agrega com temperature scaling,
e publica teacher_probs de volta para os clientes.

Fluxo de um round:
  1. Servidor publica {"cmd":"start","round":N} em fedkd/cmd/pull
  2. Cada ESP32 treina localmente (Etapa 1)
  3. Cada ESP32 publica logits em fedkd/logits/push/<client_name>
     Payload: n_pub × N_CLASSES × float32 (binário, row-major, LE)
  4. Servidor recebe logits de todos os clientes (ou timeout)
  5. Servidor agrega: média ponderada + softmax com temperatura T
  6. Servidor publica teacher_probs em fedkd/teacher/pull
     Payload: n_pub × N_CLASSES × float32 (mesmo formato)
  7. Cada ESP32 recebe teacher_probs → executa KD + fine-tuning (Etapa 2+3)
  8. Cada ESP32 publica {"status":"done"} em fedkd/cmd/push/<client_name>
  9. Goto 1

Instalação:
  pip install paho-mqtt numpy

Uso:
  # Modo contínuo (rounds automáticos após todos os clientes responderem)
  python fedkd_server.py --host 192.168.1.100 --n-clients 3

  # N rounds fixos
  python fedkd_server.py --host 192.168.1.100 --n-clients 3 --rounds 5

  # Aguarda comando manual para cada round (--manual)
  python fedkd_server.py --host 192.168.1.100 --n-clients 3 --manual

Requisitos do broker MQTT:
  Instale o Mosquitto: https://mosquitto.org/download/
    Linux:   sudo apt install mosquitto mosquitto-clients
    macOS:   brew install mosquitto
    Windows: instalador em mosquitto.org
  Inicie o broker na porta 1883 (configuração padrão).
"""

import argparse
import csv
import json
import os
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("[ERRO] paho-mqtt não instalado. Execute: pip install paho-mqtt")
    sys.exit(1)

# ── Configuração padrão ───────────────────────────────────────────────────────

N_CLASSES    = 4
TEMPERATURE  = 2.0          # temperatura para suavização dos soft targets

TOPIC_LOGITS  = "fedkd/logits/push/+"     # wildcard — recebe de qualquer cliente
TOPIC_TEACHER = "fedkd/teacher/pull"      # publica teacher_probs para todos
TOPIC_CMD_OUT = "fedkd/cmd/pull"          # publica comandos para todos
TOPIC_CMD_IN  = "fedkd/cmd/push/+"        # recebe status de clientes

# ── Servidor de Federação ─────────────────────────────────────────────────────

class FedKDServer:
    def __init__(self, host: str, port: int, n_clients: int,
                 timeout_s: int, temperature: float, log_dir: str = "logs"):
        self.host        = host
        self.port        = port
        self.n_clients   = n_clients
        self.timeout_s   = timeout_s
        self.temperature = temperature

        self._logits: dict[str, np.ndarray] = {}  # {client_name: (n_pub, N_CLASSES)}
        self._done:   dict[str, bool]        = {}  # {client_name: done_flag}
        self._metrics: dict[str, dict]       = {}  # {client_name: parsed done JSON}
        self._lock = threading.Lock()
        self.round_n = 0

        # ── Diretório de logs ────────────────────────────────────────────────
        self.log_dir    = Path(log_dir)
        self.logits_dir = self.log_dir / "logits"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logits_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_csv  = self.log_dir / "metrics.csv"
        self._consensus_csv = self.log_dir / "consensus.csv"
        self._init_csvs()

        # Cache para salvar métricas após _wait_all_done
        self._last_teacher_probs  = None
        self._last_consensus      = {}
        self._last_t_round_start  = 0.0

        self._client = mqtt.Client(client_id="fedkd-server", clean_session=True)
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

    def _init_csvs(self):
        """Cria os CSVs com cabeçalho se ainda não existirem."""
        if not self._metrics_csv.exists():
            with open(self._metrics_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "round","ts","client",
                    "acc_local","acc_kd","acc_ft",
                    "f1_local","f1_kd","f1_ft",
                    "loss_local","loss_kd","loss_ft",
                    "t_local_ms","t_kd_ms","t_ft_ms","heap_free",
                    "delta_acc_kd","delta_acc_ft",
                ])
        if not self._consensus_csv.exists():
            with open(self._consensus_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "round","ts","n_clients","n_pub",
                    "entropy_mean","entropy_min","entropy_max",
                    "kl_mean","kl_max",
                    "cosine_mean","cosine_min",
                    "bytes_logits","bytes_teacher",
                    "t_round_ms",
                ])

    # ── Callbacks MQTT ───────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[SERVER] Conectado ao broker {self.host}:{self.port}")
            client.subscribe(TOPIC_LOGITS)
            client.subscribe(TOPIC_CMD_IN)
        else:
            print(f"[SERVER] Falha ao conectar: rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            print(f"[SERVER] Desconectado inesperadamente (rc={rc}). Reconectando...")

    def _on_message(self, client, userdata, msg):
        topic   = msg.topic
        payload = msg.payload

        if topic.startswith("fedkd/logits/push/"):
            client_name = topic.split("/")[-1]
            n_bytes = len(payload)
            if n_bytes == 0 or n_bytes % (N_CLASSES * 4) != 0:
                print(f"[WARN] {client_name}: payload inválido ({n_bytes} bytes). Ignorando.")
                return
            n_pub  = n_bytes // (N_CLASSES * 4)
            logits = np.frombuffer(payload, dtype=np.float32).reshape(n_pub, N_CLASSES).copy()
            with self._lock:
                self._logits[client_name] = logits
            print(f"[SERVER] Logits de {client_name}: {n_pub} amostras × {N_CLASSES} classes"
                  f"  (prob_range=[{logits.min():.3f}, {logits.max():.3f}])")

        elif topic.startswith("fedkd/cmd/push/"):
            client_name = topic.split("/")[-1]
            try:
                info   = json.loads(payload)
                status = info.get("status", "?")
                if status == "done":
                    with self._lock:
                        self._done[client_name]   = True
                        self._metrics[client_name] = info
                    acc_l = info.get("acc_local", float("nan"))
                    acc_k = info.get("acc_kd",    float("nan"))
                    acc_f = info.get("acc_ft",    float("nan"))
                    print(f"[SERVER] Done de {client_name}:"
                          f"  acc_local={acc_l:.3f}"
                          f"  acc_kd={acc_k:.3f}  (Δ={acc_k-acc_l:+.3f})"
                          f"  acc_ft={acc_f:.3f}  (Δ={acc_f-acc_l:+.3f})")
                else:
                    print(f"[SERVER] Status de {client_name}: {status}")
            except Exception:
                pass  # payload não-JSON — ignora

    # ── Agregação ────────────────────────────────────────────────────────────

    def _aggregate(self) -> np.ndarray:
        """
        Agrega probabilidades dos clientes com temperature scaling.

        Passos:
          1. Média simples das probabilidades de todos os clientes
          2. Converte para logits aproximados: log(p + ε)
          3. Aplica temperatura T: logits_t = logits / T
          4. Softmax estabilizado → teacher_probs

        A temperatura T > 1 suaviza a distribuição (dark knowledge).
        T = 2.0 é um valor clássico em KD (Hinton et al., 2015).
        """
        with self._lock:
            probs_list = [v.copy() for v in self._logits.values()]
            n_clients_received = len(probs_list)

        if n_clients_received == 0:
            return None

        # Média das probabilidades entre clientes
        avg_probs = np.mean(probs_list, axis=0)  # (n_pub, N_CLASSES)

        # Normaliza para garantir soma = 1 (robustez numérica)
        row_sums = avg_probs.sum(axis=1, keepdims=True).clip(min=1e-7)
        avg_probs = avg_probs / row_sums

        # Converte para logits aproximados e aplica temperatura
        eps         = 1e-7
        logits_approx = np.log(avg_probs + eps)          # (n_pub, N_CLASSES)
        logits_t      = logits_approx / self.temperature  # escala por T

        # Softmax estabilizado (subtrai max por linha para evitar overflow)
        logits_t -= logits_t.max(axis=1, keepdims=True)
        exp_l        = np.exp(logits_t)
        teacher_probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        return teacher_probs.astype(np.float32)

    # ── Métricas de consenso (calculadas no servidor com numpy) ─────────────

    @staticmethod
    def _entropy(probs: np.ndarray) -> np.ndarray:
        """Entropia de Shannon por amostra (em nats). Shape: (n,)"""
        eps  = 1e-9
        p    = np.clip(probs, eps, 1.0)
        return -np.sum(p * np.log(p), axis=1)

    @staticmethod
    def _kl_div(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """KL(p||q) por amostra. Shape: (n,)"""
        eps = 1e-9
        return np.sum(p * np.log((p + eps) / (q + eps)), axis=1)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity por amostra entre dois arrays (n, k)."""
        na = np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-9)
        nb = np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-9)
        return np.sum((a / na) * (b / nb), axis=1)

    def _consensus_metrics(self, teacher_probs: np.ndarray) -> dict:
        """
        Calcula métricas de qualidade do consenso:
          - Entropia do teacher_probs (quão suave/confiante é o ensemble)
          - KL divergência de cada cliente em relação ao teacher
          - Cosine similarity entre clientes (apenas se ≥2 clientes)
        """
        with self._lock:
            client_logits = {k: v.copy() for k, v in self._logits.items()}

        H     = self._entropy(teacher_probs)
        stats = {
            "entropy_mean": float(H.mean()),
            "entropy_min":  float(H.min()),
            "entropy_max":  float(H.max()),
        }

        # KL de cada cliente em relação ao teacher
        kl_vals = []
        for name, probs in client_logits.items():
            n = min(len(probs), len(teacher_probs))
            kl = self._kl_div(probs[:n], teacher_probs[:n])
            kl_vals.append(kl.mean())
            print(f"  KL({name}||teacher) = {kl.mean():.4f}  (max={kl.max():.4f})")
        if kl_vals:
            stats["kl_mean"] = float(np.mean(kl_vals))
            stats["kl_max"]  = float(np.max(kl_vals))
        else:
            stats["kl_mean"] = stats["kl_max"] = float("nan")

        # Cosine similarity entre pares de clientes
        names  = list(client_logits.keys())
        cos_vals = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = client_logits[names[i]]
                b = client_logits[names[j]]
                n = min(len(a), len(b))
                cos = self._cosine(a[:n], b[:n])
                cos_vals.append(cos.mean())
                print(f"  cosine({names[i]}, {names[j]}) = {cos.mean():.4f}")
        if cos_vals:
            stats["cosine_mean"] = float(np.mean(cos_vals))
            stats["cosine_min"]  = float(np.min(cos_vals))
        else:
            stats["cosine_mean"] = stats["cosine_min"] = float("nan")

        return stats

    def _save_round_metrics(self, round_n: int, teacher_probs: np.ndarray,
                             consensus: dict, t_round_ms: int):
        """Salva métricas do round nos CSVs e logits em .npy."""
        ts = datetime.now().isoformat(timespec="seconds")

        # ── consensus.csv ────────────────────────────────────────────────────
        with self._lock:
            n_clients  = len(self._logits)
            bytes_log  = sum(v.nbytes for v in self._logits.values())
        bytes_teacher  = teacher_probs.nbytes
        n_pub          = teacher_probs.shape[0]

        with open(self._consensus_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                round_n, ts, n_clients, n_pub,
                f"{consensus.get('entropy_mean', float('nan')):.5f}",
                f"{consensus.get('entropy_min',  float('nan')):.5f}",
                f"{consensus.get('entropy_max',  float('nan')):.5f}",
                f"{consensus.get('kl_mean',      float('nan')):.5f}",
                f"{consensus.get('kl_max',       float('nan')):.5f}",
                f"{consensus.get('cosine_mean',  float('nan')):.5f}",
                f"{consensus.get('cosine_min',   float('nan')):.5f}",
                bytes_log, bytes_teacher, t_round_ms,
            ])

        # ── metrics.csv (uma linha por cliente) ──────────────────────────────
        with self._lock:
            metrics_snap = dict(self._metrics)

        with open(self._metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            for client_name, info in metrics_snap.items():
                al = info.get("acc_local", float("nan"))
                ak = info.get("acc_kd",    float("nan"))
                af = info.get("acc_ft",    float("nan"))
                w.writerow([
                    round_n, ts, client_name,
                    f"{al:.4f}", f"{ak:.4f}", f"{af:.4f}",
                    f"{info.get('f1_local', float('nan')):.4f}",
                    f"{info.get('f1_kd',    float('nan')):.4f}",
                    f"{info.get('f1_ft',    float('nan')):.4f}",
                    f"{info.get('loss_local', float('nan')):.5f}",
                    f"{info.get('loss_kd',    float('nan')):.5f}",
                    f"{info.get('loss_ft',    float('nan')):.5f}",
                    info.get("t_local_ms", ""),
                    info.get("t_kd_ms",    ""),
                    info.get("t_ft_ms",    ""),
                    info.get("heap_free",  ""),
                    f"{ak - al:.4f}" if not (isinstance(ak, float) and isinstance(al, float) and (ak != ak or al != al)) else "",
                    f"{af - al:.4f}" if not (isinstance(af, float) and isinstance(al, float) and (af != af or al != al)) else "",
                ])

        # ── Salva logits e teacher_probs como .npy ───────────────────────────
        tag = f"round_{round_n:04d}"
        np.save(self.logits_dir / f"{tag}_teacher.npy", teacher_probs)
        with self._lock:
            for name, arr in self._logits.items():
                np.save(self.logits_dir / f"{tag}_{name}.npy", arr)

        print(f"[LOG] Round {round_n} salvo → {self._metrics_csv.name},"
              f" {self._consensus_csv.name}, logits/*.npy")

    # ── Execução de um round ─────────────────────────────────────────────────

    def run_round(self, start_delay_s: float = 1.0) -> bool:
        self.round_n += 1
        t_round_start = time.time()
        bar = "=" * 50
        print(f"\n{bar}")
        print(f"[SERVER] === Round {self.round_n} ===")
        print(f"{bar}")

        # Limpa estado do round anterior
        with self._lock:
            self._logits.clear()
            self._done.clear()
            self._metrics.clear()

        time.sleep(start_delay_s)

        # Envia comando de início
        cmd = json.dumps({"cmd": "start", "round": self.round_n}).encode()
        self._client.publish(TOPIC_CMD_OUT, cmd)
        print(f"[SERVER] Comando 'start' enviado → {TOPIC_CMD_OUT}")

        # Aguarda logits de todos os clientes (com timeout)
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            with self._lock:
                n_recv = len(self._logits)
            if n_recv >= self.n_clients:
                break
            elapsed  = time.time() - (deadline - self.timeout_s)
            remaining = deadline - time.time()
            print(f"\r[SERVER] Aguardando logits: {n_recv}/{self.n_clients} clientes"
                  f"  |  {remaining:.0f}s restantes   ", end="", flush=True)
            time.sleep(0.5)
        print()  # nova linha após \r

        with self._lock:
            n_recv    = len(self._logits)
            clients   = list(self._logits.keys())

        if n_recv == 0:
            print("[SERVER] Timeout: nenhum cliente respondeu neste round.")
            return False

        if n_recv < self.n_clients:
            print(f"[SERVER] Agregando com {n_recv}/{self.n_clients} clientes: {clients}")
        else:
            print(f"[SERVER] Todos os {n_recv} clientes responderam: {clients}")

        # Agrega e publica
        teacher_probs = self._aggregate()
        if teacher_probs is None:
            print("[SERVER] Erro na agregação.")
            return False

        n_pub = teacher_probs.shape[0]
        print(f"[SERVER] teacher_probs calculado: {n_pub} amostras × {N_CLASSES} classes"
              f"  (T={self.temperature})")
        for i in range(n_pub):
            probs_str = "  ".join(f"{p:.4f}" for p in teacher_probs[i])
            print(f"  amostra {i}: [{probs_str}]")

        payload = teacher_probs.tobytes()
        self._client.publish(TOPIC_TEACHER, payload)
        print(f"[SERVER] teacher_probs publicado → {TOPIC_TEACHER}  ({len(payload)} bytes)")

        # ── Métricas de consenso (calculadas agora — logits ainda disponíveis) ─
        print("[SERVER] Calculando métricas de consenso…")
        self._last_teacher_probs = teacher_probs
        self._last_consensus     = self._consensus_metrics(teacher_probs)
        self._last_t_round_start = t_round_start

        return True

    def _wait_all_done(self, timeout_s: float = 120.0):
        """Aguarda confirmação 'done' de todos os clientes (opcional)."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                n_done = sum(1 for v in self._done.values() if v)
            if n_done >= self.n_clients:
                print(f"[SERVER] Todos os clientes confirmaram 'done'.")
                return True
            time.sleep(1.0)
        print(f"[SERVER] Timeout aguardando 'done' dos clientes.")
        return False

    # ── Loop principal ───────────────────────────────────────────────────────

    def start(self, n_rounds: int = None, manual: bool = False,
              inter_round_s: float = 5.0):
        self._client.connect(self.host, self.port, keepalive=60)
        self._client.loop_start()

        time.sleep(1.5)  # aguarda conexão e subscribe confirmados

        round_count = 0
        try:
            while n_rounds is None or round_count < n_rounds:
                if manual and round_count > 0:
                    input("\n[SERVER] Pressione ENTER para iniciar o próximo round...")

                ok = self.run_round()
                round_count += 1

                if ok:
                    # Aguarda todos confirmarem 'done' (métricas de cliente chegam aqui)
                    self._wait_all_done(timeout_s=self.timeout_s * 2)
                    # Salva métricas agora que _metrics está preenchido
                    if self._last_teacher_probs is not None:
                        t_total_ms = int((time.time() - self._last_t_round_start) * 1000)
                        self._save_round_metrics(
                            self.round_n,
                            self._last_teacher_probs,
                            self._last_consensus,
                            t_total_ms,
                        )

                if n_rounds is None:
                    print(f"[SERVER] Round {self.round_n} concluído. "
                          f"Próximo em {inter_round_s:.0f}s... (Ctrl+C para parar)")
                    time.sleep(inter_round_s)

        except KeyboardInterrupt:
            print(f"\n[SERVER] Interrompido após {round_count} rounds.")
        finally:
            self._client.loop_stop()
            self._client.disconnect()
            print("[SERVER] Desconectado do broker.")


# ── Ponto de entrada ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Servidor de Federação FedKD-MR")
    parser.add_argument("--host",       default="localhost",
                        help="IP/hostname do broker MQTT (padrão: localhost)")
    parser.add_argument("--port",       type=int, default=1883,
                        help="Porta do broker MQTT (padrão: 1883)")
    parser.add_argument("--n-clients",  type=int, default=3,
                        help="Número de clientes ESP32 esperados (padrão: 3)")
    parser.add_argument("--rounds",     type=int, default=None,
                        help="Número de rounds a executar (padrão: infinito)")
    parser.add_argument("--timeout",    type=int, default=120,
                        help="Timeout em segundos aguardando logits (padrão: 120)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Temperatura de distilação (padrão: {TEMPERATURE})")
    parser.add_argument("--interval",   type=float, default=5.0,
                        help="Intervalo entre rounds automáticos em segundos (padrão: 5)")
    parser.add_argument("--manual",     action="store_true",
                        help="Aguarda ENTER entre rounds")
    args = parser.parse_args()

    print("=" * 55)
    print("  FedKD-MR — Servidor de Federação MQTT")
    print("=" * 55)
    print(f"  Broker     : {args.host}:{args.port}")
    print(f"  Clientes   : {args.n_clients}")
    print(f"  Rounds     : {'∞ (contínuo)' if args.rounds is None else args.rounds}")
    print(f"  Timeout    : {args.timeout}s")
    print(f"  Temperatura: {args.temperature}")
    print(f"  N_CLASSES  : {N_CLASSES}")
    print("=" * 55)
    print(f"  Tópicos:")
    print(f"    Recebe logits : fedkd/logits/push/<client>")
    print(f"    Publica cmds  : {TOPIC_CMD_OUT}")
    print(f"    Publica teacher: {TOPIC_TEACHER}")
    print("=" * 55)

    server = FedKDServer(
        host=args.host,
        port=args.port,
        n_clients=args.n_clients,
        timeout_s=args.timeout,
        temperature=args.temperature,
    )
    server.start(
        n_rounds=args.rounds,
        manual=args.manual,
        inter_round_s=args.interval,
    )


if __name__ == "__main__":
    main()
