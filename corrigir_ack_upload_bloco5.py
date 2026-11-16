#!/usr/bin/env python3
"""Correção pontual do recebimento da confirmação UPLOAD no Bloco 5.

Executar na raiz do projeto: python3 corrigir_ack_upload_bloco5.py
Cria backup fedkd_orchestrator.py.antes_ack_bloco5.bak; não altera firmware ou datasets.
"""
from pathlib import Path
import py_compile

file = Path('fedkd_orchestrator.py')
if not file.is_file():
    raise SystemExit('ERRO: execute na raiz do projeto, onde está fedkd_orchestrator.py')
source = file.read_text(encoding='utf-8')
old = '''    # Aguarda UPLOAD_OK / UPLOAD_ERR
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
'''
new = '''    # Espera pela resposta de protocolo; o LittleFS pode emitir mensagens
    # diagnosticas antes do UPLOAD_OK/UPLOAD_ERR, inclusive ao consultar exists().
    deadline_final = time.monotonic() + 30.0
    while time.monotonic() < deadline_final:
        response = _readline_timeout(ser, min(2.0, max(0.01, deadline_final - time.monotonic())))
        if response is None:
            continue
        if "UPLOAD_ERR" in response:
            _err(f"    ESP32 reportou erro: '{response[response.index('UPLOAD_ERR'):]}'")
            return False
        if "UPLOAD_OK" in response:
            confirmed = response[response.index("UPLOAD_OK"):]
            parts = confirmed.split()
            try:
                recv_bytes = int(parts[1])
            except (IndexError, ValueError):
                _err(f"    Resposta UPLOAD_OK inválida: '{response}'")
                return False
            if recv_bytes != n_bytes:
                _err(f"    UPLOAD_OK mas tamanho diverge: {recv_bytes} ≠ {n_bytes}")
                return False
            _info(f"    UPLOAD_OK {recv_bytes} bytes ✓")
            return True
        if response.strip():
            _info(f"    Diagnóstico ESP32 (não é ACK): '{response}'")
    _err("    Timeout aguardando UPLOAD_OK/UPLOAD_ERR após UPLOAD_END")
    return False
'''
if old not in source:
    if new in source:
        raise SystemExit('A correção já está instalada; nada foi alterado.')
    raise SystemExit('ERRO: trecho esperado não encontrado; nenhum arquivo foi alterado. Envie a função de upload para análise.')
updated = source.replace(old, new, 1)
# Validate before touching project
import tempfile
with tempfile.TemporaryDirectory() as td:
    preview = Path(td)/'fedkd_orchestrator.py'
    preview.write_text(updated, encoding='utf-8')
    py_compile.compile(str(preview), doraise=True)
backup = file.with_name(file.name + '.antes_ack_bloco5.bak')
if backup.exists():
    raise SystemExit(f'ERRO: backup {backup} já existe; não sobrescrevi.')
backup.write_text(source, encoding='utf-8')
file.write_text(updated, encoding='utf-8')
print('PASS: leitura final do upload corrigida e sintaxe Python verificada.')
print(f'Backup: {backup}')
print('Nenhum firmware, dataset ou checkpoint foi alterado.')
