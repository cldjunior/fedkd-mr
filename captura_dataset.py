#!/usr/bin/env python3
"""
captura_dataset.py — Exporta dataset_priv.bin do ESP32 para o PC.

Uso:
    python3 captura_dataset.py          # detecta porta automaticamente
    python3 captura_dataset.py /dev/ttyUSB0

O script envia o comando 'x' para o firmware de coleta, lê o dump hex
e salva dataset_priv.bin no diretório atual.
"""

import sys
import time
import binascii
from pathlib import Path

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("Instale pyserial:  pip install pyserial")
    sys.exit(1)

BAUD     = 115200
OUT_FILE = "dataset_priv.bin"
BPR      = 385   # bytes por amostra (96 float32 + 1 uint8)


def find_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        if any(k in desc for k in ("usb", "uart", "cp210", "ch340", "ftdi", "silabs")):
            return p.device
    if ports:
        return ports[0].device
    return None


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else find_port()
    if not port:
        print("Porta serial não encontrada. Passe como argumento: python3 captura_dataset.py /dev/ttyUSB0")
        sys.exit(1)

    print(f"Conectando em {port} @ {BAUD}...")
    # Configura dtr/rts antes de abrir para evitar reset automático do ESP32
    ser = serial.Serial()
    ser.port     = port
    ser.baudrate = BAUD
    ser.timeout  = 10
    ser.dtr      = False
    ser.rts      = False
    ser.open()
    time.sleep(1.0)
    ser.reset_input_buffer()

    print("Enviando comando 'x' para o ESP32...")
    ser.write(b'x')

    # Aguarda DUMP_START
    expected_bytes = None
    print("Aguardando início do dump...", end="", flush=True)
    deadline = time.time() + 5
    while time.time() < deadline:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if line.startswith("DUMP_START"):
            parts = line.split()
            if len(parts) > 1:
                expected_bytes = int(parts[1])
            print(f" OK ({expected_bytes} bytes esperados)")
            break
    else:
        print("\nTimeout aguardando DUMP_START. Verifique se o firmware de coleta está ativo.")
        ser.close()
        sys.exit(1)

    # Lê linhas hex até DUMP_END
    hex_str = ""
    print("Lendo dump", end="", flush=True)
    while True:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if line == "DUMP_END":
            print(" concluído.")
            break
        if line:
            hex_str += line
            print(".", end="", flush=True)

    ser.close()

    if not hex_str:
        print("Dump vazio — nenhum dado recebido.")
        sys.exit(1)

    try:
        data = binascii.unhexlify(hex_str)
    except Exception as e:
        print(f"Erro ao decodificar hex: {e}")
        sys.exit(1)

    n_samples = len(data) // BPR
    remainder = len(data) % BPR

    print(f"\nBytes recebidos : {len(data)}")
    print(f"Amostras        : {n_samples}  (+ {remainder} bytes extras — ignorados)")
    if expected_bytes and len(data) != expected_bytes:
        print(f"AVISO: esperado {expected_bytes} bytes, recebido {len(data)} bytes")

    # Salva
    Path(OUT_FILE).write_bytes(data[:n_samples * BPR])
    print(f"Arquivo salvo   : {OUT_FILE}")
    print("\nAnexe o dataset_priv.bin ao chat para análise.")


if __name__ == "__main__":
    main()
