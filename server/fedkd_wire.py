"""Envelope FedKD-MR Bloco 6. Protocolo de consistencia (nao autentica MQTT)."""
import struct

MAGIC = b'FKD6'
HEADER = struct.Struct('<4sIQ')
HEADER_LEN = HEADER.size


def experiment_hash(experiment_id: str) -> int:
    value = 0xcbf29ce484222325
    for b in experiment_id.encode('utf-8'):
        value = ((value ^ b) * 0x100000001b3) & 0xffffffffffffffff
    return value


def pack(payload: bytes, round_number: int, experiment_id: str) -> bytes:
    if not round_number > 0 or not experiment_id:
        raise ValueError('rodada e experimento obrigatorios')
    return HEADER.pack(MAGIC, round_number, experiment_hash(experiment_id)) + payload


def unpack(payload: bytes, round_number: int, experiment_id: str) -> bytes:
    if len(payload) < HEADER_LEN or not experiment_id:
        raise ValueError('envelope ausente')
    magic, found_round, found_hash = HEADER.unpack_from(payload)
    if magic != MAGIC or found_round != round_number or found_hash != experiment_hash(experiment_id):
        raise ValueError('rodada ou experimento diferente')
    return payload[HEADER_LEN:]
