#!/usr/bin/env python3
"""Auditoria somente-leitura dos dados FedKD-MR: não altera arquivos de treino/checkpoint."""
import argparse, collections, hashlib, json, sys
from pathlib import Path

ROW = 385  # 96 float32 + uint8 rotulo
NCLASS = 4

def load(path):
    raw = path.read_bytes()
    if len(raw) % ROW:
        raise ValueError(f'{path}: {len(raw)} bytes não múltiplo de {ROW}')
    rows = [raw[i:i + ROW] for i in range(0, len(raw), ROW)]
    bad = sorted({r[-1] for r in rows if r[-1] not in range(1, NCLASS + 1)})
    if bad:
        raise ValueError(f'{path}: rótulos fora de 1..{NCLASS}: {bad}')
    return rows

def h(row):
    return hashlib.sha256(row).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', default='orchestrator_output', help='Pasta com dataset_priv_*.bin e dataset_pub.bin')
    ap.add_argument('--json', default=None, help='Caminho opcional para relatório JSON sem amostras')
    a = ap.parse_args()
    d = Path(a.dir)
    priv_paths = sorted(d.glob('dataset_priv_*.bin'))
    pubpath = d / 'dataset_pub.bin'
    if not priv_paths or not pubpath.exists():
        print(f'ERRO: não encontrei dataset_priv_*.bin e dataset_pub.bin em {d}', file=sys.stderr)
        return 2
    private = {p.stem.removeprefix('dataset_priv_'): load(p) for p in priv_paths}
    public = load(pubpath)
    pub_count = collections.Counter(map(h, public))
    private_counts = {name: collections.Counter(map(h, rows)) for name, rows in private.items()}
    overlap = {}
    all_private = collections.Counter()
    for name, counts in private_counts.items():
        overlap[name] = sum((counts & pub_count).values())
        all_private.update(counts)
    total_overlap = sum((all_private & pub_count).values())
    pub_unique_overlap = sum(1 for digest in pub_count if digest in all_private)
    # Possível mesmo vetor com rótulo divergente: computa hash dos primeiros 384 bytes.
    pub_feature_hashes = {h(r[:-1]) for r in public}
    feat_overlap = {name: sum(1 for r in rows if h(r[:-1]) in pub_feature_hashes) for name, rows in private.items()}
    test_paths = sorted(d.glob('*test*.bin')) + sorted(d.glob('*teste*.bin'))
    report = {
        'directory': str(d.resolve()),
        'row_bytes': ROW,
        'public_samples': len(public),
        'public_by_class': dict(sorted(collections.Counter(r[-1] for r in public).items())),
        'private_samples': {n: len(r) for n, r in private.items()},
        'private_by_class': {n: dict(sorted(collections.Counter(r[-1] for r in rows).items())) for n, rows in private.items()},
        'overlap_exact_rows_by_client': overlap,
        'overlap_exact_rows_total_multiset': total_overlap,
        'public_unique_rows_in_private': pub_unique_overlap,
        'overlap_same_features_by_client': feat_overlap,
        'test_files_found_in_output_dir': [p.name for p in test_paths],
        'note': 'Sobreposição exata não exclui duplicação semântica, janelas temporais adjacentes ou vazamento temporal. Ausência de arquivo de teste aqui não prova ausência em outros locais.'
    }
    print('=== AUDITORIA FEDKD-MR — BLOCO 5 (somente leitura) ===')
    print('D_pub:', len(public), 'amostras | classes:', report['public_by_class'])
    for n, rows in private.items():
        print(f'{n}: {len(rows)} privadas | sobreposição exata com D_pub: {overlap[n]} | mesma feature: {feat_overlap[n]}')
    print('D_pub com ao menos uma ocorrência idêntica no conjunto privado:', pub_unique_overlap, 'de', len(pub_count), 'registros públicos únicos')
    print('Arquivos de teste com nome reconhecido:', ', '.join(report['test_files_found_in_output_dir']) or 'nenhum')
    print('RESULTADO:', 'VAZAMENTO EXATO DETECTADO' if total_overlap else 'nenhuma sobreposição exata detectada (outros vazamentos não descartados)')
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print('Relatório:', a.json)
    return 0

if __name__ == '__main__':
    sys.exit(main())
