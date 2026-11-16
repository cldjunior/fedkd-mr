#!/usr/bin/env python3
"""Gera partições disjuntas de FedKD-MR a partir dos backups; nunca altera originais.

Formato dos registros: 96 float32 little-endian + uint8 label; 385 bytes.
Sem metadados temporais, esta é somente uma separação por equivalência exata de features.
Não substitui split por sessão/grupo temporal.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROW = 385
NCLASS = 4

def load(path):
    raw = path.read_bytes()
    if not raw or len(raw) % ROW:
        raise ValueError(f'{path}: tamanho inválido ({len(raw)} bytes)')
    rows = [raw[i:i+ROW] for i in range(0, len(raw), ROW)]
    bad = {r[-1] for r in rows if not 1 <= r[-1] <= NCLASS}
    if bad:
        raise ValueError(f'{path}: rótulos inválidos: {bad}')
    return rows

def feat(row):
    return hashlib.sha256(row[:-1]).hexdigest()

def write_once(path, rows):
    if path.exists():
        raise FileExistsError(f'Arquivo já existe, não sobrescrever: {path}')
    with path.open('xb') as f:
        for row in rows:
            f.write(row)

def count(rows):
    return {str(i): sum(r[-1] == i for r in rows) for i in range(1, NCLASS+1)}

def run(src, dst, fraction, seed):
    if not (0.05 <= fraction <= 0.45):
        raise ValueError('test_fraction precisa estar entre 0.05 e 0.45')
    paths = sorted(src.glob('dataset_priv_*.bin'))
    if not paths:
        raise FileNotFoundError('Nenhum dataset_priv_*.bin encontrado')
    pub_path = src / 'dataset_pub.bin'
    if not pub_path.is_file():
        raise FileNotFoundError(f'Ausente: {pub_path}')
    if dst.exists():
        raise FileExistsError(f'Destino já existe: {dst}. Escolha outro nome; nenhuma sobrescrita é permitida.')
    originals = {p.stem.removeprefix('dataset_priv_'): load(p) for p in paths}
    pub = load(pub_path)
    known_labels = {}
    for row in pub + [r for rows in originals.values() for r in rows]:
        h = feat(row)
        if h in known_labels and known_labels[h] != row[-1]:
            raise ValueError(f'Features idênticas possuem rótulos conflitantes: hash={h[:16]}')
        known_labels[h] = row[-1]
    pub_features = set(map(feat, pub))
    if len(pub_features) != len(pub):
        raise ValueError('D_pub contém features duplicadas; verificar dados antes de continuar')
    # Elimina TODAS as ocorrências públicas, inclusive repetições internas ao cliente.
    retained = {name: [r for r in rows if feat(r) not in pub_features] for name, rows in originals.items()}
    # Um hash de features possui sempre o mesmo grupo (treino ou teste) entre clientes.
    # Atribuição determinística independente da ordem dos registros.
    all_features = {feat(r) for rows in retained.values() for r in rows}
    def assigned_test(h):
        digest = hashlib.sha256(f'{seed}:{h}'.encode()).digest()
        return int.from_bytes(digest[:8], 'big') / 2**64 < fraction
    test_features = {h for h in all_features if assigned_test(h)}
    train = {name: [r for r in rows if feat(r) not in test_features] for name, rows in retained.items()}
    test = {name: [r for r in rows if feat(r) in test_features] for name, rows in retained.items()}
    for name in originals:
        if not train[name] or not test[name]:
            raise ValueError(f'{name}: split vazio; selecione outra seed/fraction')
        missing = [str(cls) for cls in range(1, NCLASS+1) if count(train[name])[str(cls)] == 0 and count(originals[name])[str(cls)] > 0]
        if missing:
            raise ValueError(f'{name}: treinamento perdeu classes {missing}; altere seed/fraction')
    train_hashes = {feat(r) for rows in train.values() for r in rows}
    test_hashes = {feat(r) for rows in test.values() for r in rows}
    assert not (pub_features & train_hashes or pub_features & test_hashes or train_hashes & test_hashes)
    report = {'format': {'row_bytes': ROW, 'features_float32': 96, 'label_uint8': True},
              'source': str(src.resolve()), 'seed': seed, 'test_fraction_target': fraction,
              'warning': 'Sem IDs de sessão e timestamps: split sem colisões exatas, mas NÃO garante independência temporal ou entre janelas adjacentes.',
              'public': {'n': len(pub), 'per_class': count(pub)}, 'clients': {}}
    for name in originals:
        report['clients'][name] = {'original': len(originals[name]),
                                   'removed_public_occurrences': len(originals[name]) - len(retained[name]),
                                   'train_n': len(train[name]), 'test_n': len(test[name]),
                                   'train_per_class': count(train[name]), 'test_per_class': count(test[name]),
                                   'train_bytes': len(train[name])*ROW, 'test_bytes': len(test[name])*ROW}
    # Saída transacional: todos os testes ocorrem ANTES da criação da pasta.
    dst.mkdir(parents=True, exist_ok=False)
    write_once(dst / 'dataset_pub.bin', pub)
    for name in originals:
        write_once(dst / f'dataset_train_{name}.bin', train[name])
        write_once(dst / f'dataset_test_{name}.bin', test[name])
    (dst / 'manifesto.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('=== BLOCO 5 — PARTIÇÕES GERADAS (originais intocados) ===')
    print('D_pub:', len(pub), 'amostras')
    for name, data in report['clients'].items():
        print(f"{name}: original={data['original']} removidas_pub={data['removed_public_occurrences']} treino={data['train_n']} teste={data['test_n']}")
    print('COLISÕES EXATAS ENTRE PUBLIC / TRAIN / TEST: 0 (verificado globalmente)')
    print('ATENÇÃO: ausência de metadados de sessão impede validar independência temporal.')
    print('Arquivos gerados em:', dst)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', default='orchestrator_output', type=Path)
    p.add_argument('--out', default='bloco5_particoes_v1', type=Path)
    p.add_argument('--test-fraction', default=0.20, type=float)
    p.add_argument('--seed', default=42, type=int)
    a = p.parse_args()
    run(a.dir, a.out, a.test_fraction, a.seed)
