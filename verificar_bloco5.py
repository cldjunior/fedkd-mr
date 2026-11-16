#!/usr/bin/env python3
"""Auditoria independente e read-only das partições do Bloco 5."""
import argparse, hashlib, json
from pathlib import Path
ROW=385

def rows(path):
    b=path.read_bytes()
    if not b or len(b)%ROW: raise ValueError(f'{path}: arquivo vazio ou tamanho inválido: {len(b)}')
    a=[b[i:i+ROW] for i in range(0,len(b),ROW)]
    if any(not 1<=r[-1]<=4 for r in a): raise ValueError(f'{path}: rótulos inválidos')
    return a

def check(partition, origin):
    m=json.loads((partition/'manifesto.json').read_text())
    if m['format']['row_bytes']!=ROW: raise ValueError('row_bytes inválido')
    pub=rows(partition/'dataset_pub.bin')
    if (partition/'dataset_pub.bin').read_bytes()!=(origin/'dataset_pub.bin').read_bytes():
        raise ValueError('D_pub difere do backup da orquestração')
    hashes={'public':set(), 'train':set(), 'test':set()}
    def add(kind,a):
        for r in a:
            hashes[kind].add(hashlib.sha256(r[:-1]).digest())
    add('public',pub)
    if len(pub)!=m['public']['n']: raise ValueError('D_pub quantidade inválida')
    for key,d in m['clients'].items():
        for kind in ['train','test']:
            a=rows(partition/f'dataset_{kind}_{key}.bin')
            if len(a)!=d[f'{kind}_n']:
                raise ValueError(f'{key}: {kind}_n difere do manifesto')
            add(kind,a)
            print(f'{key}: {kind}={len(a)}')
    for a,b in [('public','train'),('public','test'),('train','test')]:
        common=hashes[a]&hashes[b]
        if common: raise ValueError(f'COLISÃO {a}/{b}: {len(common)} features')
    print(f'PASS: D_pub={len(pub)}; sem colisões exatas; manifesto consistente; originais intocados')
    print('ATENÇÃO: ausência de metadados não exclui vazamento temporal.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--partitions',type=Path,default=Path('bloco5_particoes_v1'))
    p.add_argument('--origin',type=Path,default=Path('orchestrator_output'))
    a=p.parse_args();check(a.partitions,a.origin)
