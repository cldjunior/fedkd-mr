"""
mqtt_handler.py — FedKD-MR Cliente Raspberry Pi
=================================================
Wrapper paho-mqtt que replica o protocolo binário usado pelos ESP32s.

Protocolo de payload:
  • logits/push : float32 row-major  (n_pub × n_classes) — same as ESP32
  • teacher/pull: float32 row-major  (n_pub × n_classes) — same as ESP32
  • cmd/pull    : JSON  {"cmd":"start","round":N}
  • cmd/push    : JSON  {"status":"done","round":N}

Uso típico:
    mqtt = MQTTHandler(cfg)
    mqtt.connect()
    mqtt.wait_start()          # bloqueia até receber cmd=start
    mqtt.publish_logits(arr)   # arr: np.ndarray shape (n_pub, n_classes)
    tp = mqtt.wait_teacher()   # retorna np.ndarray shape (n_pub, n_classes)
    mqtt.publish_done(round_n)
    mqtt.disconnect()
"""

import json
import struct
import logging
import threading
import time
import numpy as np
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MQTTHandler:
    """
    Gerencia conexão MQTT e troca de mensagens com o servidor FedKD-MR.

    Args:
        cfg: Namespace retornado por config.get_config().
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # Tópicos (derivados pelo config.py)
        self.topic_logits_push  = cfg.topic_logits_push
        self.topic_teacher_pull = cfg.topic_teacher_pull
        self.topic_cmd_pull     = cfg.topic_cmd_pull
        self.topic_cmd_push     = cfg.topic_cmd_push

        # Eventos de sincronização
        self._evt_start   = threading.Event()
        self._evt_teacher = threading.Event()

        # Payloads recebidos
        self._start_round:   int            = 0
        self._teacher_probs: np.ndarray | None = None

        # Cliente paho
        self._client = mqtt.Client(client_id=cfg.client_name,
                                   clean_session=True)
        self._client.on_connect    = self._on_connect
        self._client.on_message    = self._on_message
        self._client.on_disconnect = self._on_disconnect

    # ── Ciclo de vida ────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Conecta ao broker e inicia loop em thread de background."""
        cfg = self.cfg
        logger.info("Conectando a %s:%d como %s", cfg.broker, cfg.port, cfg.client_name)
        self._client.connect(cfg.broker, cfg.port, keepalive=cfg.keepalive)
        self._client.loop_start()

    def disconnect(self) -> None:
        """Encerra loop e desconecta do broker."""
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("Desconectado do broker MQTT.")

    # ── Publicação ───────────────────────────────────────────────────────────

    def publish_logits(self, logits: np.ndarray) -> None:
        """
        Publica logits do modelo no tópico push.

        Args:
            logits: np.ndarray de shape (n_pub, n_classes), dtype float32.
                    Serializado como bytes row-major (igual ao ESP32).
        """
        payload = logits.astype(np.float32).tobytes()
        n_bytes = len(payload)
        logger.info("Publicando logits: %s floats → %d bytes",
                    "×".join(str(d) for d in logits.shape), n_bytes)
        result = self._client.publish(self.topic_logits_push, payload, qos=1)
        result.wait_for_publish()

    def publish_done(self, round_n: int,
                     acc_local:  float = float("nan"),
                     acc_kd:     float = float("nan"),
                     acc_ft:     float = float("nan"),
                     f1_local:   float = float("nan"),
                     f1_kd:      float = float("nan"),
                     f1_ft:      float = float("nan"),
                     loss_local: float = float("nan"),
                     loss_kd:    float = float("nan"),
                     loss_ft:    float = float("nan"),
                     t_local_ms: int   = 0,
                     t_kd_ms:    int   = 0,
                     t_ft_ms:    int   = 0) -> None:
        """
        Publica status 'done' ao servidor para sinalizar fim do round.

        Campos enviados (os campos ausentes no ESP32 chegam como null):
            acc_local/kd/ft   : acurácia no D_test após cada fase  (0-1)
            f1_local/kd/ft    : F1 macro no D_test após cada fase  (0-1)
            loss_local/kd/ft  : loss média da última época de cada fase
            t_local/kd/ft_ms  : duração de cada fase em milissegundos
        """
        import math

        def _safe_float(v: float, ndigits: int = 4):
            """NaN/Inf → null; float finito → arredondado."""
            if math.isnan(v) or math.isinf(v):
                return None
            return round(float(v), ndigits)

        def _safe_int(v: int):
            return int(v) if v > 0 else None

        payload = json.dumps({
            "status":     "done",
            "round":      round_n,
            # ── acurácia ──────────────────────────────────────
            "acc_local":  _safe_float(acc_local),
            "acc_kd":     _safe_float(acc_kd),
            "acc_ft":     _safe_float(acc_ft),
            # ── F1 macro ──────────────────────────────────────
            "f1_local":   _safe_float(f1_local),
            "f1_kd":      _safe_float(f1_kd),
            "f1_ft":      _safe_float(f1_ft),
            # ── loss ──────────────────────────────────────────
            "loss_local": _safe_float(loss_local, 6),
            "loss_kd":    _safe_float(loss_kd,    6),
            "loss_ft":    _safe_float(loss_ft,    6),
            # ── timing ────────────────────────────────────────
            "t_local_ms": _safe_int(t_local_ms),
            "t_kd_ms":    _safe_int(t_kd_ms),
            "t_ft_ms":    _safe_int(t_ft_ms),
        })

        def _d(v): return v if not math.isnan(v) else -1
        logger.info(
            "Publicando done — round %d | "
            "acc(l/k/f)=%.4f/%.4f/%.4f  f1(l/k/f)=%.4f/%.4f/%.4f  "
            "loss(l/k/f)=%.4f/%.6f/%.4f  t(l/k/f)=%dms/%dms/%dms",
            round_n,
            _d(acc_local),  _d(acc_kd),  _d(acc_ft),
            _d(f1_local),   _d(f1_kd),   _d(f1_ft),
            _d(loss_local), _d(loss_kd), _d(loss_ft),
            t_local_ms, t_kd_ms, t_ft_ms,
        )
        result = self._client.publish(self.topic_cmd_push, payload, qos=1)
        result.wait_for_publish()

    # ── Espera bloqueante ────────────────────────────────────────────────────

    def wait_start(self, timeout: float | None = None) -> int:
        """
        Bloqueia até receber o comando 'start' do servidor.

        Returns:
            Número do round recebido no payload JSON.
        Raises:
            TimeoutError: se timeout (segundos) expirar sem receber start.
        """
        logger.info("Aguardando comando 'start' do servidor…")
        self._evt_start.clear()
        ok = self._evt_start.wait(timeout=timeout)
        if not ok:
            raise TimeoutError("Timeout aguardando comando 'start' do servidor.")
        logger.info("Recebido 'start' — round %d", self._start_round)
        return self._start_round

    def wait_teacher(self, timeout: float | None = None) -> np.ndarray:
        """
        Bloqueia até receber teacher_probs do servidor.

        Returns:
            np.ndarray de shape (n_pub, n_classes), dtype float32.
        Raises:
            TimeoutError: se timeout (segundos) expirar.
        """
        cfg = self.cfg
        _timeout = timeout or cfg.teacher_timeout
        logger.info("Aguardando teacher_probs (timeout=%ds)…", _timeout)
        self._evt_teacher.clear()
        ok = self._evt_teacher.wait(timeout=_timeout)
        if not ok:
            raise TimeoutError("Timeout aguardando teacher_probs do servidor.")
        arr = self._teacher_probs
        self._teacher_probs = None
        logger.info("teacher_probs recebido: shape %s", arr.shape)
        return arr

    # ── Callbacks paho ───────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("MQTT conectado ao broker.")
            client.subscribe(self.topic_teacher_pull, qos=1)
            client.subscribe(self.topic_cmd_pull, qos=1)
            logger.debug("Inscrito em: %s, %s",
                         self.topic_teacher_pull, self.topic_cmd_pull)
        else:
            logger.error("Falha na conexão MQTT — código %d", rc)

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logger.warning("Desconexão inesperada do broker (rc=%d). "
                           "paho tentará reconectar.", rc)

    def _on_message(self, client, userdata, msg):
        topic   = msg.topic
        payload = msg.payload

        if topic == self.topic_cmd_pull:
            self._handle_cmd(payload)

        elif topic == self.topic_teacher_pull:
            self._handle_teacher(payload)

        else:
            logger.debug("Mensagem desconhecida no tópico %s", topic)

    # ── Handlers de mensagem ─────────────────────────────────────────────────

    def _handle_cmd(self, payload: bytes) -> None:
        """Processa JSON de comando vindo do servidor."""
        try:
            doc = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Payload de cmd inválido: %r", payload[:40])
            return

        cmd = doc.get("cmd", "")
        if cmd == "start":
            self._start_round = int(doc.get("round", 0))
            self._evt_start.set()
        else:
            logger.debug("Comando desconhecido: %s", cmd)

    def _handle_teacher(self, payload: bytes) -> None:
        """
        Deserializa teacher_probs.
        Formato: float32 row-major (n_pub × n_classes).
        n_pub e n_classes são inferidos do tamanho do payload.
        """
        cfg = self.cfg
        n_floats = len(payload) // 4
        if n_floats == 0 or len(payload) % 4 != 0:
            logger.error("teacher_probs com tamanho inválido: %d bytes", len(payload))
            return

        arr = np.frombuffer(payload, dtype=np.float32).copy()

        # Tenta reshape pelo número de classes conhecido
        n_classes = cfg.n_classes
        if n_floats % n_classes == 0:
            n_pub = n_floats // n_classes
            arr = arr.reshape(n_pub, n_classes)
        else:
            logger.warning("Não foi possível fazer reshape em (%d, %d). "
                           "Usando flat.", n_floats // n_classes, n_classes)

        self._teacher_probs = arr
        self._evt_teacher.set()
