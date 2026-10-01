#!/usr/bin/env python3
"""
fedkd_monitor.py — FedKD-MR Real-Time Monitor
==============================================
Dashboard web que monitora toda a comunicação servidor↔clientes em tempo real.

Subscreve todos os tópicos MQTT do FedKD, observa os CSVs de métricas e
serve um dashboard HTML acessível pelo navegador.
Não interfere no servidor nem nos clientes.

Requisitos:
  pip install paho-mqtt websockets

Uso:
  python fedkd_monitor.py
  python fedkd_monitor.py --broker 192.168.0.12 --http-port 8080

  # CSVs gerados pelo fedkd_server.py ficam em logs/ por padrão:
  python fedkd_monitor.py --metrics-csv logs/metrics.csv \
                           --consensus-csv logs/consensus.csv

Abra o navegador em: http://localhost:8080
"""

import argparse
import asyncio
import csv
import json
import os
import struct
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Set

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("[ERRO] paho-mqtt não instalado. Execute: pip install paho-mqtt")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("[ERRO] websockets não instalado. Execute: pip install websockets")
    sys.exit(1)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_BROKER        = "192.168.0.12"
DEFAULT_PORT          = 1883
DEFAULT_HTTP          = 8080
DEFAULT_WS            = 8765
DEFAULT_METRICS_CSV   = "logs/metrics.csv"
DEFAULT_CONSENSUS_CSV = "logs/consensus.csv"
N_CLASSES             = 4

# ── Estado compartilhado ──────────────────────────────────────────────────────
state = {
    "round":     0,
    "phase":     "aguardando",
    "clients":   {},
    "history":   [],
    "log":       [],
    "broker":    DEFAULT_BROKER,
    "connected": False,
}
state_lock = threading.Lock()

# Caminhos dos CSVs (definidos em main())
metrics_csv_path   = DEFAULT_METRICS_CSV
consensus_csv_path = DEFAULT_CONSENSUS_CSV

# ── Asyncio — WebSocket broadcast ─────────────────────────────────────────────
ws_loop: asyncio.AbstractEventLoop = None
ws_clients: Set = set()
_ws_queue: asyncio.Queue = None

# ── Tracking de round em curso ────────────────────────────────────────────────
_round_start_ts = {}


def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _log(msg: str, level: str = "info"):
    entry = {"ts": _ts(), "level": level, "msg": msg}
    with state_lock:
        state["log"].append(entry)
        if len(state["log"]) > 300:
            state["log"] = state["log"][-300:]
    icons = {"info": "·", "success": "✓", "warn": "!", "error": "✗", "debug": "…"}
    print(f"  [{entry['ts']}] {icons.get(level,'·')} {msg}")
    _broadcast_push({"type": "log", "entry": entry})


def _broadcast_push(data: dict):
    if ws_loop and _ws_queue:
        ws_loop.call_soon_threadsafe(_ws_queue.put_nowait, json.dumps(data))


def _broadcast_state():
    with state_lock:
        snap = {
            "type":      "state",
            "round":     state["round"],
            "phase":     state["phase"],
            "clients":   dict(state["clients"]),
            "history":   list(state["history"][-30:]),
            "connected": state["connected"],
            "broker":    state["broker"],
        }
    _broadcast_push(snap)


# ── Leitura dos CSVs ──────────────────────────────────────────────────────────

def _read_csv(path: str) -> list:
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"  [WARN] Erro ao ler {path}: {e}")
        return []


def _broadcast_metrics():
    """Lê os CSVs e envia para todos os browsers conectados."""
    D = _read_csv(metrics_csv_path)
    C = _read_csv(consensus_csv_path)
    _broadcast_push({"type": "metrics", "D": D, "C": C})


# ── CSV Watcher ───────────────────────────────────────────────────────────────

def csv_watcher():
    """Thread que detecta mudanças nos CSVs e emite métricas atualizadas."""
    last_mt = {metrics_csv_path: 0, consensus_csv_path: 0}
    while True:
        changed = False
        for path in (metrics_csv_path, consensus_csv_path):
            try:
                mt = os.path.getmtime(path)
                if mt != last_mt[path]:
                    last_mt[path] = mt
                    changed = True
            except FileNotFoundError:
                pass
        if changed:
            _broadcast_metrics()
        time.sleep(2)


# ── Callbacks MQTT ─────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        with state_lock:
            state["connected"] = True
        _log(f"Conectado ao broker MQTT {state['broker']}:{DEFAULT_PORT}", "success")
        client.subscribe("fedkd/#")
        _broadcast_state()
    else:
        _log(f"Falha ao conectar ao broker MQTT (rc={rc})", "error")


def on_disconnect(client, userdata, rc):
    with state_lock:
        state["connected"] = False
    if rc != 0:
        _log(f"Desconectado do broker MQTT (rc={rc}) — reconectando…", "warn")
    _broadcast_state()


def on_message(client, userdata, msg):
    topic   = msg.topic
    payload = msg.payload

    if topic == "fedkd/cmd/pull":
        try:
            doc     = json.loads(payload.decode())
            cmd     = doc.get("cmd", "")
            round_n = int(doc.get("round", 0))
            if cmd == "start":
                ts = _ts()
                with state_lock:
                    state["round"] = round_n
                    state["phase"] = "local_train"
                _round_start_ts[round_n] = ts
                _log(f"[Round {round_n}] → START enviado a todos os clientes", "info")
                _broadcast_state()
        except Exception as e:
            _log(f"[cmd/pull] erro de parse: {e}", "error")

    elif topic.startswith("fedkd/logits/push/"):
        client_name = topic.split("/")[-1]
        n_bytes     = len(payload)
        n_samples   = n_bytes // (N_CLASSES * 4) if n_bytes >= N_CLASSES * 4 else 0
        prob_range  = ""
        if n_samples > 0:
            try:
                arr = struct.unpack_from(f"{n_samples * N_CLASSES}f", payload)
                prob_range = f"  prob=[{min(arr):.3f}…{max(arr):.3f}]"
            except Exception:
                pass
        with state_lock:
            cur_round = state["round"]
            if client_name not in state["clients"]:
                state["clients"][client_name] = {}
            state["clients"][client_name].update({
                "last_seen": _ts(), "round": cur_round,
                "logits_n": n_samples, "status": "logits_ok",
            })
            state["phase"] = "agregando"
        _log(f"[{client_name}] logits recebidos — {n_samples} amostras × {N_CLASSES} classes"
             f"  ({n_bytes} B){prob_range}", "info")
        _broadcast_state()

    elif topic == "fedkd/teacher/pull":
        n_bytes   = len(payload)
        n_samples = n_bytes // (N_CLASSES * 4) if n_bytes >= N_CLASSES * 4 else 0
        with state_lock:
            state["phase"] = "kd_finetune"
        _log(f"Teacher probs publicados — {n_samples} amostras × {N_CLASSES} classes"
             f"  ({n_bytes} B)", "success")
        _broadcast_state()

    elif topic.startswith("fedkd/cmd/push/"):
        client_name = topic.split("/")[-1]
        try:
            doc    = json.loads(payload.decode())
            status = doc.get("status", "?")
            if status == "done":
                rnd       = int(doc.get("round", state["round"]))
                acc_local = doc.get("acc_local", None)
                acc_kd    = doc.get("acc_kd",    None)
                acc_ft    = doc.get("acc_ft",    None)
                if acc_ft is None:
                    acc_ft = doc.get("accuracy", None)
                with state_lock:
                    if client_name not in state["clients"]:
                        state["clients"][client_name] = {}
                    state["clients"][client_name].update({
                        "last_seen": _ts(), "round": rnd, "status": "done",
                        "acc_local": acc_local, "acc_kd": acc_kd, "acc_ft": acc_ft,
                        "t_local_ms": doc.get("t_local_ms"),
                        "t_kd_ms":    doc.get("t_kd_ms"),
                        "heap_free":  doc.get("heap_free"),
                    })
                def _pct(v): return f"{v*100:.1f}%" if v is not None else "—"
                delta = (f"  Δkd={acc_kd-acc_local:+.3f}"
                         if acc_kd is not None and acc_local is not None else "")
                _log(f"[{client_name}] Round {rnd} concluído"
                     f"  local={_pct(acc_local)} kd={_pct(acc_kd)} ft={_pct(acc_ft)}{delta}",
                     "success")
                with state_lock:
                    all_done = all(
                        c.get("status") == "done"
                        for c in state["clients"].values()
                        if c.get("round") == rnd
                    )
                    if all_done:
                        ts_end  = _ts()
                        ts_start = _round_start_ts.get(rnd, "—")
                        state["phase"] = "aguardando"
                        n_clients_rnd = len([
                            c for c in state["clients"].values()
                            if c.get("round") == rnd
                        ])
                        state["history"].append({
                            "round": rnd, "ts_start": ts_start,
                            "ts_end": ts_end, "clients": n_clients_rnd,
                        })
                        if len(state["history"]) > 100:
                            state["history"] = state["history"][-100:]
                _broadcast_state()
                # Round concluído → aguarda o servidor gravar os CSVs e empurra
                time.sleep(0.5)
                _broadcast_metrics()
            else:
                _log(f"[{client_name}] status: {status}", "debug")
        except Exception as e:
            _log(f"[cmd/push] erro de parse: {e}", "error")

    else:
        _log(f"[{topic}]  {len(payload)} bytes", "debug")


# ── MQTT thread ───────────────────────────────────────────────────────────────

def run_mqtt(broker: str, port: int):
    with state_lock:
        state["broker"] = broker
    mc = mqtt.Client(client_id="fedkd-monitor", clean_session=True)
    mc.on_connect    = on_connect
    mc.on_disconnect = on_disconnect
    mc.on_message    = on_message
    while True:
        try:
            mc.connect(broker, port, keepalive=60)
            mc.loop_forever()
        except Exception as e:
            _log(f"Erro MQTT: {e} — tentando em 5s…", "error")
            time.sleep(5)


# ── WebSocket server ──────────────────────────────────────────────────────────

async def ws_handler(websocket):
    ws_clients.add(websocket)
    _log(f"Browser conectado (WS) — {len(ws_clients)} cliente(s)", "debug")
    with state_lock:
        snap = {
            "type": "state", "round": state["round"], "phase": state["phase"],
            "clients": dict(state["clients"]), "history": list(state["history"][-30:]),
            "log": list(state["log"][-100:]), "connected": state["connected"],
            "broker": state["broker"],
        }
    await websocket.send(json.dumps(snap))
    # Envia métricas atuais imediatamente
    D = _read_csv(metrics_csv_path)
    C = _read_csv(consensus_csv_path)
    await websocket.send(json.dumps({"type": "metrics", "D": D, "C": C}))
    try:
        async for _ in websocket:
            pass
    finally:
        ws_clients.discard(websocket)
        _log(f"Browser desconectado — {len(ws_clients)} cliente(s)", "debug")


async def broadcast_worker():
    while True:
        msg = await _ws_queue.get()
        if ws_clients:
            await asyncio.gather(
                *[ws.send(msg) for ws in list(ws_clients)],
                return_exceptions=True,
            )


async def run_ws_server(host: str, port: int):
    global _ws_queue
    _ws_queue = asyncio.Queue()
    asyncio.create_task(broadcast_worker())
    async with websockets.serve(ws_handler, host, port):
        _log(f"WebSocket server em ws://{host}:{port}", "info")
        await asyncio.Future()


def start_ws_thread(host: str, port: int):
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    ws_loop.run_until_complete(run_ws_server(host, port))



# ── Dashboard HTML ────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FedKD-MR Monitor</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:#30363d;--text:#e6edf3;--muted:#8b949e;
  --blue:#58a6ff;--green:#3fb950;--yellow:#d29922;--red:#f85149;--purple:#bc8cff;--orange:#ffa657;--teal:#39d353;--r:7px}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;display:flex;flex-direction:column;height:100vh;overflow:hidden}
header{background:var(--bg2);border-bottom:1px solid var(--border);padding:10px 18px;display:flex;align-items:center;gap:12px;flex-shrink:0}
header h1{font-size:15px;font-weight:700;color:var(--blue);white-space:nowrap}
#broker-badge{font-size:11px;color:var(--muted);background:var(--bg3);border:1px solid var(--border);border-radius:20px;padding:2px 9px}
#conn-dot{width:9px;height:9px;border-radius:50%;background:var(--red);transition:all .4s;flex-shrink:0}
#conn-dot.ok{background:var(--green);box-shadow:0 0 6px var(--green)}
#phase-pill{padding:3px 12px;border-radius:20px;font-size:11px;font-weight:600;background:var(--bg3);border:1px solid var(--border);color:var(--muted);white-space:nowrap;transition:all .3s}
#phase-pill.local_train{border-color:var(--orange);color:var(--orange)}
#phase-pill.agregando{border-color:var(--purple);color:var(--purple)}
#phase-pill.kd_finetune{border-color:var(--blue);color:var(--blue)}
#round-badge{font-size:12px;font-weight:700;white-space:nowrap}
.ml{margin-left:auto;display:flex;align-items:center;gap:8px}
#ws-pill{font-size:10px;color:var(--muted);padding:2px 8px;border:1px solid var(--border);border-radius:10px}
.body-wrap{display:flex;flex:1;min-height:0}
.col-charts{flex:1;min-width:0;overflow-y:auto;padding:14px 12px 14px 16px;display:flex;flex-direction:column;gap:12px}
.col-side{width:280px;flex-shrink:0;border-left:1px solid var(--border);overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:12px}
.kpi-row{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}
.kpi{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:10px 12px}
.kpi-val{font-size:20px;font-weight:700;line-height:1}
.kpi-lbl{font-size:10px;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:.4px}
.kpi-sub{font-size:10px;color:var(--muted);margin-top:3px}
.badge{display:inline-block;padding:1px 6px;border-radius:9px;font-size:10px;font-weight:600;margin-top:4px}
.bg{background:rgba(63,185,80,.15);color:var(--green);border:1px solid rgba(63,185,80,.3)}
.by{background:rgba(210,153,34,.15);color:var(--yellow);border:1px solid rgba(210,153,34,.3)}
.bb{background:rgba(88,166,255,.15);color:var(--blue);border:1px solid rgba(88,166,255,.3)}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:12px 14px}
.card h2{font-size:10px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.strip{background:var(--bg3);border-left:3px solid var(--blue);border-radius:4px;padding:5px 9px;margin-top:7px;font-size:10px;color:var(--muted);line-height:1.4}
.strip.ok{border-left-color:var(--green)}.strip.warn{border-left-color:var(--yellow)}
.strip strong{color:var(--text)}
table{width:100%;border-collapse:collapse;font-size:11px}
th{color:var(--muted);font-weight:600;text-align:right;padding:5px 6px;border-bottom:1px solid var(--border);white-space:nowrap}
th:first-child{text-align:center}
td{padding:4px 6px;border-bottom:1px solid rgba(48,54,61,.4);text-align:right;font-variant-numeric:tabular-nums}
td:first-child{text-align:center;font-weight:600;color:var(--muted)}
tr.hi td{background:rgba(63,185,80,.07)}tr:hover td{background:var(--bg3)}
.dp{color:var(--green);font-weight:700}.dn{color:var(--red)}.dz{color:var(--muted)}
.client-card{background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);padding:10px 12px;margin-bottom:8px;transition:border-color .3s}
.client-card.logits_ok{border-color:var(--purple)}.client-card.done{border-color:var(--green)}
.client-name{font-size:12px;font-weight:700;margin-bottom:5px}
.client-stat{font-size:11px;color:var(--muted);line-height:1.8}.client-stat span{color:var(--text)}
.sb{display:inline-block;font-size:10px;font-weight:600;padding:1px 7px;border-radius:9px;margin-top:4px}
.sb-idle{background:#1e2329;color:var(--muted)}.sb-logits{background:#2a1f40;color:var(--purple)}.sb-done{background:#0d2318;color:var(--green)}
#hist-table{width:100%;font-size:11px;border-collapse:collapse}
#hist-table th{color:var(--muted);font-weight:600;padding:3px 5px;border-bottom:1px solid var(--border);text-align:left}
#hist-table td{padding:3px 5px;border-bottom:1px solid rgba(48,54,61,.3)}
#log-wrap{max-height:260px;overflow-y:auto;font-family:'Cascadia Code','Fira Code',monospace;font-size:11px;line-height:1.6}
.le{display:flex;gap:6px;padding:1px 0;border-bottom:1px solid #1c2128}
.lt{color:var(--muted);flex-shrink:0;font-size:10px}.lm{word-break:break-all}
.li .lm{color:var(--text)}.ls .lm{color:var(--green)}.lw .lm{color:var(--yellow)}.lr .lm{color:var(--red)}.ld .lm{color:var(--muted);font-style:italic}
#waiting{text-align:center;color:var(--muted);padding:40px 0}
@media(max-width:900px){.kpi-row{grid-template-columns:repeat(3,1fr)}.g2{grid-template-columns:1fr}.col-side{display:none}}
</style>
</head>
<body>
<header>
  <div id="conn-dot"></div>
  <h1>FedKD-MR Monitor</h1>
  <div id="round-badge">—</div>
  <div id="phase-pill" class="aguardando">aguardando</div>
  <div class="ml">
    <div id="broker-badge">—</div>
    <div id="ws-pill">⬤ conectando…</div>
  </div>
</header>
<div class="body-wrap">
  <div class="col-charts">
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-val" id="kpi-rounds" style="color:var(--green)">—</div><div class="kpi-lbl">Rounds</div><span class="badge bb" id="kpi-rounds-b">—</span></div>
      <div class="kpi"><div class="kpi-val" id="kpi-acc" style="color:var(--blue)">—</div><div class="kpi-lbl">Acc local média</div><div class="kpi-sub" id="kpi-acc-s">—</div></div>
      <div class="kpi"><div class="kpi-val" id="kpi-dkd" style="color:var(--teal)">—</div><div class="kpi-lbl">Δ KD médio</div><span class="badge by" id="kpi-dkd-b">—</span></div>
      <div class="kpi"><div class="kpi-val" id="kpi-tkd" style="color:var(--orange)">—</div><div class="kpi-lbl">t_KD médio</div><div class="kpi-sub" id="kpi-tkd-s">—</div></div>
      <div class="kpi"><div class="kpi-val" id="kpi-f1" style="color:var(--purple)">—</div><div class="kpi-lbl">F1 macro médio</div><div class="kpi-sub" id="kpi-f1-s">—</div></div>
      <div class="kpi"><div class="kpi-val" id="kpi-heap" style="color:var(--muted)">—</div><div class="kpi-lbl">Heap livre</div><span class="badge bg" id="kpi-heap-b">—</span></div>
    </div>
    <div id="waiting"><em>Aguardando dados do primeiro round…</em></div>
    <div id="charts-area" style="display:none;flex-direction:column;gap:12px">
      <div class="g2">
        <div class="card"><h2>Acurácia por round — local / pós-KD / pós-FT</h2><canvas id="cAcc" height="150"></canvas><div class="strip" id="s-acc"></div></div>
        <div class="card"><h2>Δ KD por round — ganho sobre local</h2><canvas id="cDelta" height="150"></canvas><div class="strip" id="s-delta">Verde = KD melhorou. Cinza = sem diferença.</div></div>
      </div>
      <div class="g2">
        <div class="card"><h2>Loss — local / KD / FT</h2><canvas id="cLoss" height="140"></canvas></div>
        <div class="card"><h2>F1 macro — local / pós-KD / pós-FT</h2><canvas id="cF1" height="140"></canvas></div>
      </div>
      <div class="g2">
        <div class="card"><h2>Timing por estágio (ms) — stacked</h2><canvas id="cTime" height="130"></canvas></div>
        <div class="card"><h2>Entropia teacher &amp; KL cliente→teacher</h2><canvas id="cEnt" height="130"></canvas></div>
      </div>
      <div class="card"><h2>Heap livre ESP32 (bytes)</h2><canvas id="cHeap" height="90"></canvas></div>
      <div class="card">
        <h2>Detalhamento por round</h2>
        <table><thead><tr><th>#</th><th>Acc local</th><th>Acc KD</th><th>Acc FT</th><th>Δ KD</th><th>Δ FT</th><th>F1 local</th><th>F1 KD</th><th>Loss local</th><th>Loss KD</th><th>t_local</th><th>t_KD</th><th>t_FT</th><th>Heap</th></tr></thead>
        <tbody id="tBody"></tbody></table>
      </div>
    </div>
  </div>
  <div class="col-side">
    <div class="card"><h2>Clientes ESP32</h2><div id="clients-wrap"><em style="color:var(--muted);font-size:12px">Nenhum cliente visto ainda.</em></div></div>
    <div class="card"><h2>Histórico de rounds</h2>
      <div id="hist-empty" style="color:var(--muted);font-size:12px">Nenhum round concluído.</div>
      <table id="hist-table" style="display:none"><thead><tr><th>Rnd</th><th>Início</th><th>Fim</th><th>Clnt</th></tr></thead><tbody id="hist-body"></tbody></table>
    </div>
    <div class="card" style="flex:1;display:flex;flex-direction:column"><h2>Log de eventos</h2><div id="log-wrap"></div></div>
  </div>
</div>
<script>
const WS_PORT=__WS_PORT__;
const wsUrl=`ws://${location.hostname}:${WS_PORT}`;
Chart.defaults.color='#8b949e';Chart.defaults.borderColor='#30363d';
Chart.defaults.font.family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
Chart.defaults.font.size=10;
let CI={};
function destroyCharts(){Object.values(CI).forEach(c=>{try{c.destroy()}catch(e){}});CI={};}
const baseOpts=(ymin,ymax,ylabel)=>({
  responsive:true,animation:false,
  plugins:{legend:{position:'top',labels:{boxWidth:9,padding:6,font:{size:10}}},tooltip:{mode:'index',intersect:false,padding:7}},
  scales:{x:{grid:{color:'#1c2128'},ticks:{maxTicksLimit:10}},y:{grid:{color:'#1c2128'},min:ymin,max:ymax,title:{display:!!ylabel,text:ylabel,color:'#8b949e',font:{size:9}}}}
});
function parseMetrics(rawD,rawC){
  const byRound={};
  rawD.forEach(row=>{const r=+row.round;if(!byRound[r])byRound[r]=[];byRound[r].push(row);});
  const D=Object.keys(byRound).map(Number).sort((a,b)=>a-b).map(r=>{
    const rows=byRound[r];
    const avg=f=>rows.reduce((s,row)=>s+(+row[f]||0),0)/rows.length;
    return{r,al:avg('acc_local'),ak:avg('acc_kd'),af:avg('acc_ft'),
      fl:avg('f1_local'),fk:avg('f1_kd'),ff:avg('f1_ft'),
      ll:avg('loss_local'),lk:avg('loss_kd'),lf:avg('loss_ft'),
      tl:avg('t_local_ms'),tk:avg('t_kd_ms'),tf:avg('t_ft_ms'),
      h:avg('heap_free'),dk:avg('delta_acc_kd'),df:avg('delta_acc_ft')};
  });
  const C=rawC.map(row=>({r:+row.round,em:+row.entropy_mean||0,emin:+row.entropy_min||0,emax:+row.entropy_max||0,kl:+row.kl_mean||0}));
  return{D,C};
}
function renderKPIs(D){
  if(!D.length)return;
  const avgAl=D.reduce((s,d)=>s+d.al,0)/D.length;
  const avgDk=D.reduce((s,d)=>s+d.dk,0)/D.length;
  const avgTk=D.reduce((s,d)=>s+d.tk,0)/D.length;
  const avgF1=D.reduce((s,d)=>s+d.fl,0)/D.length;
  const nPos=D.filter(d=>d.dk>0).length;
  const last=D[D.length-1];
  set('kpi-rounds',D.length);set('kpi-rounds-b','sessão ativa');
  set('kpi-acc',(avgAl*100).toFixed(1)+'%');
  set('kpi-acc-s',`${(Math.min(...D.map(d=>d.al))*100).toFixed(1)} – ${(Math.max(...D.map(d=>d.al))*100).toFixed(1)}%`);
  set('kpi-dkd',(avgDk>=0?'+':'')+(avgDk*100).toFixed(2)+'%');
  const db=el('kpi-dkd-b');db.textContent=`${nPos}/${D.length} rounds ↑`;db.className=`badge ${nPos>0?'bg':'by'}`;
  set('kpi-tkd',Math.round(avgTk)+' ms');
  set('kpi-tkd-s',`${Math.round(Math.min(...D.map(d=>d.tk)))} – ${Math.round(Math.max(...D.map(d=>d.tk)))} ms`);
  set('kpi-f1',avgF1.toFixed(3));
  set('kpi-f1-s',`${Math.min(...D.map(d=>d.fl)).toFixed(2)} – ${Math.max(...D.map(d=>d.fl)).toFixed(2)}`);
  set('kpi-heap',Math.round(last.h/1024)+' KB');set('kpi-heap-b','estável');
  const posR=D.filter(d=>d.dk>0);
  const sa=el('s-acc');
  if(posR.length){sa.className='strip ok';sa.innerHTML=`<strong>${posR.map(d=>`R${d.r} (+${(d.dk*100).toFixed(2)}%)`).join(', ')}:</strong> teacher global melhora o aluno local.`;}
  else{sa.className='strip warn';sa.innerHTML='Nenhum Δ KD positivo ainda — aguardando mais rounds.';}
}
function renderCharts(D,C){
  destroyCharts();if(!D.length)return;
  const Rs=D.map(d=>d.r),Rl=D.map(d=>`R${d.r}`);
  const accMin=Math.max(0,Math.floor(Math.min(...D.map(d=>d.al))*100)-5);
  CI.acc=new Chart('cAcc',{type:'line',data:{labels:Rs,datasets:[
    {label:'Acc local',data:D.map(d=>+(d.al*100).toFixed(2)),borderColor:'#58a6ff',backgroundColor:'rgba(88,166,255,.07)',fill:true,tension:.3,pointRadius:4,pointHoverRadius:6},
    {label:'Acc pós-KD',data:D.map(d=>+(d.ak*100).toFixed(2)),borderColor:'#3fb950',fill:false,tension:.3,pointRadius:4,borderDash:[5,3]},
    {label:'Acc pós-FT',data:D.map(d=>+(d.af*100).toFixed(2)),borderColor:'#bc8cff',fill:false,tension:.3,pointRadius:4,borderDash:[2,2]},
  ]},options:baseOpts(accMin,101,'%')});
  CI.delta=new Chart('cDelta',{type:'bar',data:{labels:Rl,datasets:[
    {label:'Δ KD (%)',data:D.map(d=>+(d.dk*100).toFixed(2)),
     backgroundColor:D.map(d=>d.dk>0?'rgba(63,185,80,.75)':'rgba(88,166,255,.2)'),
     borderColor:D.map(d=>d.dk>0?'#3fb950':'#58a6ff'),borderWidth:1,borderRadius:4},
  ]},options:{...baseOpts(null,null,'%'),plugins:{legend:{display:false},tooltip:{mode:'index',intersect:false}}}});
  CI.loss=new Chart('cLoss',{type:'line',data:{labels:Rs,datasets:[
    {label:'Loss local',data:D.map(d=>d.ll),borderColor:'#58a6ff',fill:false,tension:.3,pointRadius:3},
    {label:'Loss KD',data:D.map(d=>d.lk),borderColor:'#ffa657',fill:false,tension:.3,pointRadius:3,borderDash:[5,3]},
    {label:'Loss FT',data:D.map(d=>d.lf),borderColor:'#bc8cff',fill:false,tension:.3,pointRadius:3,borderDash:[2,2]},
  ]},options:baseOpts(0,null,'')});
  CI.f1=new Chart('cF1',{type:'line',data:{labels:Rs,datasets:[
    {label:'F1 local',data:D.map(d=>d.fl),borderColor:'#58a6ff',fill:false,tension:.3,pointRadius:3},
    {label:'F1 KD',data:D.map(d=>d.fk),borderColor:'#3fb950',fill:false,tension:.3,pointRadius:3,borderDash:[5,3]},
    {label:'F1 FT',data:D.map(d=>d.ff),borderColor:'#bc8cff',fill:false,tension:.3,pointRadius:3,borderDash:[2,2]},
  ]},options:baseOpts(.5,1.02,'')});
  CI.time=new Chart('cTime',{type:'bar',data:{labels:Rl,datasets:[
    {label:'t_local',data:D.map(d=>d.tl),backgroundColor:'rgba(88,166,255,.7)',stack:'a'},
    {label:'t_KD',data:D.map(d=>d.tk),backgroundColor:'rgba(255,166,87,.75)',stack:'a'},
    {label:'t_FT',data:D.map(d=>d.tf),backgroundColor:'rgba(188,140,255,.7)',stack:'a'},
  ]},options:{...baseOpts(0,null,'ms'),scales:{x:{stacked:true,grid:{color:'#1c2128'}},y:{stacked:true,grid:{color:'#1c2128'}}}}});
  if(C.length){
    CI.ent=new Chart('cEnt',{type:'line',data:{labels:C.map(c=>c.r),datasets:[
      {label:'H média',data:C.map(c=>+c.em.toFixed(4)),borderColor:'#bc8cff',fill:false,tension:.3,pointRadius:3,yAxisID:'y'},
      {label:'H min',data:C.map(c=>+c.emin.toFixed(4)),borderColor:'#3fb950',fill:false,tension:.3,pointRadius:2,borderDash:[3,3],yAxisID:'y'},
      {label:'H max',data:C.map(c=>+c.emax.toFixed(4)),borderColor:'#f85149',fill:false,tension:.3,pointRadius:2,borderDash:[3,3],yAxisID:'y'},
      {label:'KL',data:C.map(c=>+c.kl.toFixed(5)),borderColor:'#ffa657',fill:false,tension:.3,pointRadius:3,borderDash:[1,1],yAxisID:'y2'},
    ]},options:{responsive:true,animation:false,
      plugins:{legend:{position:'top',labels:{boxWidth:9,padding:6,font:{size:10}}},tooltip:{mode:'index',intersect:false}},
      scales:{x:{grid:{color:'#1c2128'}},
        y:{grid:{color:'#1c2128'},title:{display:true,text:'nats (H)',color:'#8b949e',font:{size:9}},position:'left'},
        y2:{grid:{drawOnChartArea:false},title:{display:true,text:'nats (KL)',color:'#ffa657',font:{size:9}},position:'right'}}}});
  }
  CI.heap=new Chart('cHeap',{type:'line',data:{labels:Rs,datasets:[
    {label:'Heap livre',data:D.map(d=>d.h),borderColor:'#39d353',backgroundColor:'rgba(57,211,83,.07)',fill:true,tension:.3,pointRadius:3},
  ]},options:baseOpts(0,null,'bytes')});
}
function renderTable(D){
  const tb=el('tBody');if(!tb)return;
  const pct=v=>(v*100).toFixed(1)+'%';
  const dpct=v=>{const s=(v*100).toFixed(2);return(v>0?'+':'')+s+'%';};
  const dcls=v=>v>0?'dp':v<0?'dn':'dz';
  tb.innerHTML='';
  [...D].reverse().forEach(d=>{
    tb.innerHTML+=`<tr class="${d.dk>0?'hi':''}"><td>${d.r}</td>
      <td>${pct(d.al)}</td><td>${pct(d.ak)}</td><td>${pct(d.af)}</td>
      <td class="${dcls(d.dk)}">${dpct(d.dk)}</td><td class="${dcls(d.df)}">${dpct(d.df)}</td>
      <td>${d.fl.toFixed(4)}</td><td>${d.fk.toFixed(4)}</td>
      <td>${d.ll.toFixed(5)}</td><td>${d.lk.toFixed(5)}</td>
      <td>${Math.round(d.tl).toLocaleString()}</td><td>${Math.round(d.tk).toLocaleString()}</td>
      <td>${Math.round(d.tf).toLocaleString()}</td><td>${Math.round(d.h).toLocaleString()}</td></tr>`;
  });
}
function applyMetrics(rawD,rawC){
  const{D,C}=parseMetrics(rawD,rawC);
  const has=D.length>0;
  el('waiting').style.display=has?'none':'block';
  el('charts-area').style.display=has?'flex':'none';
  if(!has)return;
  renderKPIs(D);renderCharts(D,C);renderTable(D);
}
const PHASE_LABEL={aguardando:'aguardando próximo round',local_train:'treino local (etapa 1)',agregando:'aguardando / agregando logits',kd_finetune:'KD + fine-tuning (etapas 2–3)',done:'round concluído'};
function applyState(s){
  el('broker-badge').textContent=s.broker||'—';
  el('conn-dot').className=s.connected?'ok':'';
  el('round-badge').textContent=s.round>0?`Round ${s.round}`:'—';
  const pill=el('phase-pill');pill.className=`phase-pill ${s.phase}`;pill.textContent=PHASE_LABEL[s.phase]||s.phase;
  renderClients(s.clients||{});
  if(s.history)renderHistory(s.history);
  if(s.log){el('log-wrap').innerHTML='';s.log.forEach(appendLog);}
}
function renderClients(clients){
  const wrap=el('clients-wrap');const keys=Object.keys(clients||{});
  if(!keys.length){wrap.innerHTML='<em style="color:var(--muted);font-size:12px">Nenhum cliente visto ainda.</em>';return;}
  wrap.innerHTML='';keys.sort().forEach(name=>{
    const c=clients[name];
    const sc=c.status==='done'?'done':c.status==='logits_ok'?'logits_ok':'';
    const sb=c.status==='done'?'sb-done':c.status==='logits_ok'?'sb-logits':'sb-idle';
    const sl=c.status==='done'?'✓ done':c.status==='logits_ok'?'📤 logits':'⏳ aguardando';
    const pct=v=>v!=null?(v*100).toFixed(1)+'%':'—';
    const delta=(k,l)=>k!=null&&l!=null?`<span style="color:${k>=l?'#3fb950':'#f85149'}">(${k>=l?'+':''}${((k-l)*100).toFixed(1)}%)</span>`:'';
    wrap.innerHTML+=`<div class="client-card ${sc}"><div class="client-name">${name}</div>
      <div class="client-stat">Round: <span>${c.round??'—'}</span></div>
      <div class="client-stat">Logits: <span>${c.logits_n!=null?c.logits_n+' amostras':'—'}</span></div>
      <div class="client-stat">Acc local: <span>${pct(c.acc_local)}</span></div>
      <div class="client-stat">Acc pós-KD: <span>${pct(c.acc_kd)} ${delta(c.acc_kd,c.acc_local)}</span></div>
      <div class="client-stat">Heap: <span>${c.heap_free?Math.round(c.heap_free/1024)+'KB':'—'}</span></div>
      <span class="sb ${sb}">${sl}</span></div>`;
  });
}
function renderHistory(history){
  if(!history||!history.length)return;
  el('hist-empty').style.display='none';el('hist-table').style.display='';
  el('hist-body').innerHTML='';
  [...history].reverse().forEach(h=>{
    el('hist-body').innerHTML+=`<tr><td><b>${h.round}</b></td><td>${h.ts_start??'—'}</td><td>${h.ts_end??'—'}</td><td>${h.clients??'—'}</td></tr>`;
  });
}
const logWrap=document.getElementById('log-wrap');let autoScroll=true;
logWrap.addEventListener('scroll',()=>{autoScroll=logWrap.scrollTop+logWrap.clientHeight>=logWrap.scrollHeight-20;});
function appendLog(e){
  const cls={info:'li',success:'ls',warn:'lw',error:'lr',debug:'ld'}[e.level]||'li';
  const div=document.createElement('div');div.className=`le ${cls}`;
  div.innerHTML=`<span class="lt">${e.ts}</span><span class="lm">${esc(e.msg)}</span>`;
  logWrap.appendChild(div);if(logWrap.children.length>400)logWrap.removeChild(logWrap.firstChild);
  if(autoScroll)logWrap.scrollTop=logWrap.scrollHeight;
}
function el(id){return document.getElementById(id);}
function set(id,v){const e=el(id);if(e)e.textContent=v;}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
let ws,reconnTimer;
function connect(){
  ws=new WebSocket(wsUrl);
  ws.onopen=()=>{el('ws-pill').textContent='⬤ dashboard conectado';el('ws-pill').style.color='var(--green)';clearTimeout(reconnTimer);};
  ws.onclose=ws.onerror=()=>{el('ws-pill').textContent='⬤ reconectando…';el('ws-pill').style.color='var(--yellow)';reconnTimer=setTimeout(connect,2000);};
  ws.onmessage=e=>{const msg=JSON.parse(e.data);if(msg.type==='state')applyState(msg);else if(msg.type==='metrics')applyMetrics(msg.D,msg.C);else if(msg.type==='log')appendLog(msg.entry);};
}
connect();
</script>
</body>
</html>"""



# ── HTTP server ───────────────────────────────────────────────────────────────

class DashboardHandler(BaseHTTPRequestHandler):
    ws_port = DEFAULT_WS

    def do_GET(self):
        html = DASHBOARD_HTML.replace("__WS_PORT__", str(self.ws_port))
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


def run_http(http_port: int, ws_port: int):
    DashboardHandler.ws_port = ws_port
    httpd = HTTPServer(("0.0.0.0", http_port), DashboardHandler)
    print(f"  Dashboard HTTP em  http://localhost:{http_port}")
    httpd.serve_forever()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global metrics_csv_path, consensus_csv_path

    parser = argparse.ArgumentParser(description="FedKD-MR Real-Time Monitor")
    parser.add_argument("--broker",        default=DEFAULT_BROKER,
                        help=f"IP do broker MQTT (padrão: {DEFAULT_BROKER})")
    parser.add_argument("--port",          type=int, default=DEFAULT_PORT,
                        help=f"Porta MQTT (padrão: {DEFAULT_PORT})")
    parser.add_argument("--http-port",     type=int, default=DEFAULT_HTTP,
                        help=f"Porta HTTP do dashboard (padrão: {DEFAULT_HTTP})")
    parser.add_argument("--ws-port",       type=int, default=DEFAULT_WS,
                        help=f"Porta WebSocket (padrão: {DEFAULT_WS})")
    parser.add_argument("--metrics-csv",   default=DEFAULT_METRICS_CSV,
                        help=f"Caminho do metrics.csv (padrão: {DEFAULT_METRICS_CSV})")
    parser.add_argument("--consensus-csv", default=DEFAULT_CONSENSUS_CSV,
                        help=f"Caminho do consensus.csv (padrão: {DEFAULT_CONSENSUS_CSV})")
    args = parser.parse_args()

    metrics_csv_path   = args.metrics_csv
    consensus_csv_path = args.consensus_csv

    print("=" * 55)
    print("  FedKD-MR — Monitor em Tempo Real")
    print("=" * 55)
    print(f"  Broker MQTT : {args.broker}:{args.port}")
    print(f"  WebSocket   : ws://0.0.0.0:{args.ws_port}")
    print(f"  Dashboard   : http://localhost:{args.http_port}")
    print(f"  metrics.csv : {metrics_csv_path}")
    print(f"  consensus   : {consensus_csv_path}")
    print("=" * 55)

    ws_thread = threading.Thread(
        target=start_ws_thread, args=("0.0.0.0", args.ws_port), daemon=True)
    ws_thread.start()

    http_thread = threading.Thread(
        target=run_http, args=(args.http_port, args.ws_port), daemon=True)
    http_thread.start()

    watcher_thread = threading.Thread(target=csv_watcher, daemon=True)
    watcher_thread.start()

    time.sleep(0.5)

    print(f"\n  Abrindo http://localhost:{args.http_port} no navegador…")
    print("  (Ctrl+C para encerrar)\n")

    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{args.http_port}")
    except Exception:
        pass

    run_mqtt(args.broker, args.port)


if __name__ == "__main__":
    main()
