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
from fedkd_wire import pack as pack_wire, unpack as unpack_wire

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("[ERRO] paho-mqtt não instalado. Execute: pip install paho-mqtt")
    sys.exit(1)

# ── Configuração padrão ───────────────────────────────────────────────────────

N_CLASSES    = 4
TEMPERATURE  = 2.0          # temperatura para suavização dos soft targets

TOPIC_IDENTITY = "fedkd/identity/+"
EXPECTED_ARCH = {"FedKD-ESP32-1":"MLP", "FedKD-ESP32-2":"CNN1D", "FedKD-ESP32-3":"GRU"}
TOPIC_LOGITS  = "fedkd/logits/push/+"     # wildcard — recebe de qualquer cliente
TOPIC_TEACHER = "fedkd/teacher/pull"      # publica teacher_probs para todos
TOPIC_CMD_OUT = "fedkd/cmd/pull"          # publica comandos para todos
TOPIC_CMD_IN  = "fedkd/cmd/push/+"        # recebe status de clientes

# ── Servidor de Federação ─────────────────────────────────────────────────────

class FedKDServer:
    def __init__(self, host: str, port: int, n_clients: int,
                 timeout_s: int, temperature: float, log_dir: str = "logs",
                 client_weights: dict = None, sharpening: float = 0.7):
        self.host        = host
        self.port        = port
        self.n_clients   = n_clients
        self.timeout_s   = timeout_s
        if not np.isfinite(temperature) or temperature <= 0: raise ValueError("temperature deve ser > 0")
        if not np.isfinite(sharpening) or sharpening <= 0: raise ValueError("sharpening deve ser > 0")
        if not client_weights: raise ValueError("Pesos n_i dos clientes sao obrigatorios")
        self.temperature = temperature
        self.sharpening = sharpening
        self.client_weights = dict(client_weights)
        if any(not isinstance(n,int) or isinstance(n,bool) or n<=0 for n in self.client_weights.values()):
            raise ValueError("Todos os pesos n_i devem ser inteiros positivos")

        self._logits: dict[str, np.ndarray] = {}  # {client_name: (n_pub, N_CLASSES)}
        self._done:   dict[str, bool]        = {}  # {client_name: done_flag}
        self._identity: dict[str, dict] = {}
        self._round_open=False
        self._teacher_published=False
        self._metrics: dict[str, dict]       = {}  # {client_name: parsed done JSON}
        self._lock = threading.Lock()
        self.round_n = 0
        self.experiment_id = None

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
            client.subscribe(TOPIC_IDENTITY)
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

        if topic.startswith("fedkd/identity/"):
            client_name=topic.rsplit("/",1)[-1]
            try:
                info=json.loads(payload)
            except (ValueError,UnicodeError):
                print(f"[IDENTIDADE] JSON invalido {client_name}");return
            if not isinstance(info,dict) or client_name not in self.client_weights:
                print(f"[IDENTIDADE] cliente desconhecido: {client_name}");return
            if info.get("client_id") != client_name or info.get("model_arch") != EXPECTED_ARCH.get(client_name):
                print(f"[IDENTIDADE] arquitetura/cliente incoerente: {client_name} {info.get('model_arch')}");return
            chip=info.get("device_id")
            if not isinstance(chip,str) or len(chip)!=12 or any(c not in '0123456789ABCDEFabcdef' for c in chip):
                print(f"[IDENTIDADE] chip invalido: {client_name}");return
            if (info.get('round')!=self.round_n or info.get('experiment_id')!=self.experiment_id):
                return  # boot anterior ou round diferente, nao e identidade da rodada ativa
            with self._lock:
                if not self._round_open:return
                if any(other!=client_name and old.get('device_id')==chip for other,old in self._identity.items()):
                    print(f"[IDENTIDADE] chip duplicado: {chip}");return
                self._identity[client_name]=info
            print(f"[IDENTIDADE] {client_name}={info['model_arch']} chip={chip} fw={info.get('firmware_version')}")

        elif topic.startswith("fedkd/logits/push/"):
            client_name = topic.split("/")[-1]
            if client_name not in self.client_weights:
                print(f"[WARN] Cliente desconhecido: {client_name}. Ignorando.");return
            with self._lock:
                if not self._round_open or client_name not in self._identity:
                    print(f"[WARN] {client_name}: sem identidade valida nesta rodada.");return
            try:
                raw=unpack_wire(payload,self.round_n,self.experiment_id)
            except ValueError as exc:
                print(f"[WIRE] Rejeitando logits de {client_name}: {exc}");return
            n_bytes=len(raw)
            if n_bytes==0 or n_bytes%(N_CLASSES*4):
                print(f"[WARN] {client_name}: payload invalido ({n_bytes} bytes)");return
            n_pub=n_bytes//(N_CLASSES*4)
            logits=np.frombuffer(raw,dtype='<f4').reshape(n_pub,N_CLASSES).copy()
            if not np.isfinite(logits).all():
                print(f"[WARN] {client_name}: NaN/Inf");return
            with self._lock:
                if not self._round_open:return
                if client_name in self._logits:
                    print(f"[WIRE] Logits duplicados de {client_name}; ignorados");return
                if self._logits and logits.shape!=next(iter(self._logits.values())).shape:
                    print(f"[WARN] {client_name}: D_pub divergente");return
                self._logits[client_name]=logits
            print(f"[SERVER] Logits de {client_name}: {n_pub} amostras × {N_CLASSES} classes"
                  f"  (logit_range=[{logits.min():.3f}, {logits.max():.3f}])")

        elif topic.startswith("fedkd/cmd/push/"):
            client_name = topic.split("/")[-1]
            if client_name not in self.client_weights:return
            try:
                info=json.loads(payload)
                if not isinstance(info,dict):return
                status=info.get('status')
                if status=='done':
                    if info.get('experiment_id')!=self.experiment_id or info.get('round')!=self.round_n:
                        print(f"[WIRE] done antigo/invalido de {client_name}");return
                    with self._lock:
                        if not self._round_open or not self._teacher_published or client_name not in self._logits or client_name not in self._identity:
                            print(f"[WIRE] done sem logits/identidade de {client_name}");return
                        if client_name in self._done:return
                        self._done[client_name]=True
                        self._metrics[client_name]=info
                    al=float(info.get('acc_local',float('nan')))
                    ak=float(info.get('acc_kd',float('nan')))
                    af=float(info.get('acc_ft',float('nan')))
                    print(f"[SERVER] Done de {client_name}: acc_local={al:.3f} acc_kd={ak:.3f} acc_ft={af:.3f}")
                else:
                    print(f"[SERVER] Status de {client_name}: {status}")
            except (ValueError,TypeError,UnicodeError) as exc:
                print(f"[WARN] done malformado de {client_name}: {exc}")

    # ── Agregação ────────────────────────────────────────────────────────────

    @staticmethod
    def _softmax_logits(x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64)
        y = y - y.max(axis=1, keepdims=True)
        e = np.exp(y)
        return e / e.sum(axis=1, keepdims=True)

    def _aggregate(self) -> np.ndarray:
        """Lbar = sum_i n_i L_i / sum_i n_i; P=softmax(Lbar/T);
        Q=normalize(P**(1/tau_s)) = softmax(Lbar/(T*tau_s)).
        tau_s < 1 afia o consenso. Nao usa medias historicas.
        """
        with self._lock:
            values = {k:v.copy() for k,v in self._logits.items()}
        if not values:
            return None
        names = sorted(values)
        shape = values[names[0]].shape
        if any(values[n].shape != shape for n in names):
            raise ValueError("Logits de clientes com formas inconsistentes")
        weights = np.asarray([self.client_weights[n] for n in names], dtype=np.float64)
        weights = weights / weights.sum()
        stack = np.stack([values[n].astype(np.float64) for n in names], axis=0)
        consensus_logits = np.tensordot(weights, stack, axes=(0,0))
        prob_T = self._softmax_logits(consensus_logits / self.temperature)
        teacher = self._softmax_logits(np.log(np.clip(prob_T,1e-300,1.0)) / self.sharpening)
        if not np.isfinite(teacher).all():
            raise ValueError("Consenso nao-finito")
        self._last_weight_summary = {n:int(self.client_weights[n]) for n in names}
        return teacher.astype(np.float32)

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
        for name, logits in client_logits.items():
            n = min(len(logits), len(teacher_probs))
            probs_T = self._softmax_logits(logits[:n] / self.temperature)
            kl = self._kl_div(probs_T, teacher_probs[:n])
            kl_vals.append(kl.mean())
            print(f"  KL({name}||teacher) = {kl.mean():.4f}  (max={kl.max():.4f})")
        if kl_vals:
            stats["kl_mean"] = float(np.mean(kl_vals))
            stats["kl_max"]  = float(np.max(kl_vals))
        else:
            stats["kl_mean"] = stats["kl_max"] = float("nan")

        # Cosine similarity entre pares de clientes
        # Similaridade em espaco de probabilidades (logits possuem deslocamento arbitrario).
        client_logits = {k: self._softmax_logits(v / self.temperature) for k,v in client_logits.items()}
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
            self._identity.clear()
            self._round_open=True
            self._teacher_published=False

        time.sleep(start_delay_s)

        # Envia comando de início
        cmd = json.dumps({"cmd": "start", "round": self.round_n,
                          "experiment_id": self.experiment_id}).encode()
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
            print(f"[SERVER] Faltam {self.n_clients-n_recv} clientes; abortando rodada sem agregar.")
            return False
        else:
            print(f"[SERVER] Todos os {n_recv} clientes responderam: {clients}")

        # Agrega e publica
        teacher_probs = self._aggregate()
        if teacher_probs is None:
            print("[SERVER] Erro na agregação.")
            return False

        n_pub = teacher_probs.shape[0]
        print(f"[SERVER] teacher_probs calculado: {n_pub} amostras × {N_CLASSES} classes"
              f"  (T={self.temperature}, tau_s={self.sharpening}, n_i={self._last_weight_summary})")
        for i in range(n_pub):
            probs_str = "  ".join(f"{p:.4f}" for p in teacher_probs[i])
            print(f"  amostra {i}: [{probs_str}]")

        payload = pack_wire(teacher_probs.tobytes(),self.round_n,self.experiment_id)
        info = self._client.publish(TOPIC_TEACHER, payload)
        if info.rc != 0:
            print(f"[WIRE] falha ao publicar teacher rc={info.rc}");return False
        with self._lock:
            self._teacher_published=True
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
                    if not self._wait_all_done(timeout_s=self.timeout_s * 2):
                        print("[SERVER] Abortando sessao: nem todos os clientes confirmaram done.")
                        break
                    # Salva métricas agora que _metrics está preenchido
                    if self._last_teacher_probs is not None:
                        t_total_ms = int((time.time() - self._last_t_round_start) * 1000)
                        self._save_round_metrics(
                            self.round_n,
                            self._last_teacher_probs,
                            self._last_consensus,
                            t_total_ms,
                        )

                with self._lock:
                    self._round_open=False
                    self._teacher_published=False
                if not ok:
                    print("[SERVER] Abortando sessao por rodada incompleta.")
                    break
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
    parser.add_argument("--experiment-id", required=True,
                        help="Identificador estavel do experimento (A-Z, a-z, 0-9, - e _)")
    parser.add_argument("--start-round", type=int, default=1,
                        help="Primeira rodada a executar (1 para novo experimento; N+1 para retomada)")
    parser.add_argument("--rounds",     type=int, default=None,
                        help="Número de rounds a executar (padrão: infinito)")
    parser.add_argument("--timeout",    type=int, default=120,
                        help="Timeout em segundos aguardando logits (padrão: 120)")
    parser.add_argument("--client-weights-json", required=True,
                        help='JSON {"FedKD-ESP32-1":399,...} com n_i efetivo de cada cliente')
    parser.add_argument("--sharpening", type=float, default=0.7,
                        help="tau_s > 0; < 1 afia o consenso (padrao 0.7)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Temperatura de distilação (padrão: {TEMPERATURE})")
    parser.add_argument("--interval",   type=float, default=5.0,
                        help="Intervalo entre rounds automáticos em segundos (padrão: 5)")
    parser.add_argument("--manual",     action="store_true",
                        help="Aguarda ENTER entre rounds")
    args = parser.parse_args()
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.experiment_id):
        parser.error("--experiment-id deve ter de 1 a 64 caracteres alfanumericos, - ou _")
    if args.start_round < 1:
        parser.error("--start-round deve ser >= 1")
    if args.rounds is not None and args.rounds < 1:
        parser.error("--rounds deve ser >= 1")

    try:
        weights = json.loads(args.client_weights_json)
        if not isinstance(weights, dict) or len(weights) != args.n_clients:
            parser.error("Pesos devem conter exatamente n_clients identidades")
        if any(not isinstance(n,int) or isinstance(n,bool) or n <= 0 for n in weights.values()):
            parser.error("Pesos devem ser inteiros positivos")
    except (ValueError, TypeError) as exc:
        parser.error(f"--client-weights-json invalido: {exc}")
    if not np.isfinite(args.sharpening) or args.sharpening<=0:
        parser.error("--sharpening deve ser positivo")
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
        client_weights=weights,
        sharpening=args.sharpening,
    )
    server.experiment_id = args.experiment_id
    server.round_n = args.start_round - 1
    print(f"  Experimento: {args.experiment_id} | primeira rodada: {args.start_round}")
    server.start(
        n_rounds=args.rounds,
        manual=args.manual,
        inter_round_s=args.interval,
    )


if __name__ == "__main__":
    main()
