"""
dataset.py — FedKD-MR Cliente Raspberry Pi
===========================================
Loader flexível para datasets no formato:
  • Binário ESP32: .bin (row-major float32) + metadata.json
  • CSV:           colunas auto-detectadas; coluna 'label' obrigatória
  • Parquet:       idem ao CSV via pandas

Retorna torch.utils.data.Dataset compatível com DataLoader.

Formato binário (.bin + metadata.json):
  - Cada linha tem `bytes_per_row` bytes
  - Features: float32 em offsets declarados no schema
  - Label:    uint8 / int8 / int32 em `label_offset`
  - Labels são 1-indexed (1=EMPTY … 4=LEAVING); convertidos para 0-indexed aqui
"""

import json
import struct
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple


# ── Loader binário ───────────────────────────────────────────────────────────

class BinarySchema:
    """Parse de metadata.json no formato gerado pelo notebook FedKD_MR."""

    def __init__(self, meta_path: str):
        with open(meta_path, "r") as f:
            doc = json.load(f)

        self.bytes_per_row: int = doc["bytes_per_row"]
        label_col: str = doc.get("label_column", "label")
        self.label_map: Optional[dict] = doc.get("label_map")

        self.feat_offsets: list[int] = []
        self.label_offset: int = 0
        self.label_fmt: str = "B"   # struct format: B=uint8, b=int8, i=int32

        for col in doc["schema"]:
            name   = col["name"]
            dtype  = col.get("type", "float32")
            offset = col["offset"]

            if name == label_col:
                self.label_offset = offset
                if dtype == "uint8":
                    self.label_fmt = "B"
                elif dtype == "int8":
                    self.label_fmt = "b"
                else:   # int32 ou float32
                    self.label_fmt = "i"
            elif name != "timestamp":
                self.feat_offsets.append(offset)

        self.n_features: int = len(self.feat_offsets)

    def read_label(self, row: bytes) -> int:
        """Retorna label 0-indexed a partir de row bytes."""
        (val,) = struct.unpack_from(self.label_fmt, row, self.label_offset)
        return int(val) - 1   # converte 1-indexed → 0-indexed


class BinaryDataset(Dataset):
    """
    Lê um .bin row-major usando o schema do metadata.json.
    Cada amostra é um par (x: Tensor[n_input], y: int).
    """

    def __init__(self, bin_path: str, schema: BinarySchema, transform=None):
        self.schema    = schema
        self.transform = transform

        data = Path(bin_path).read_bytes()
        row_size = schema.bytes_per_row
        n_rows   = len(data) // row_size

        self.X = np.zeros((n_rows, schema.n_features), dtype=np.float32)
        self.Y = np.zeros(n_rows, dtype=np.int64)

        for i in range(n_rows):
            row = data[i * row_size: (i + 1) * row_size]
            for j, off in enumerate(schema.feat_offsets):
                (val,) = struct.unpack_from("f", row, off)
                self.X[i, j] = val
            self.Y[i] = schema.read_label(row)

        # Remove amostras com label inválido (fora de [0, n_classes-1])
        valid = (self.Y >= 0)
        self.X = self.X[valid]
        self.Y = self.Y[valid]

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = int(self.Y[idx])
        if self.transform:
            x = self.transform(x)
        return x, y


# ── Loader CSV / Parquet ─────────────────────────────────────────────────────

class TabularDataset(Dataset):
    """
    Lê um CSV ou Parquet.
    A coluna 'label' (ou nome configurável) deve conter inteiros 1-indexed.
    Todas as outras colunas numéricas são tratadas como features.
    """

    def __init__(self, path: str, label_col: str = "label",
                 exclude_cols: Optional[list] = None, transform=None):
        import pandas as pd

        p = Path(path)
        if p.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        exclude = set(exclude_cols or []) | {label_col, "timestamp"}
        feat_cols = [c for c in df.columns if c not in exclude
                     and np.issubdtype(df[c].dtype, np.number)]

        self.X = df[feat_cols].values.astype(np.float32)
        self.Y = (df[label_col].values.astype(np.int64) - 1)  # 1-indexed → 0-indexed
        self.transform = transform

        valid = self.Y >= 0
        self.X = self.X[valid]
        self.Y = self.Y[valid]

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = int(self.Y[idx])
        if self.transform:
            x = self.transform(x)
        return x, y


# ── Factory ──────────────────────────────────────────────────────────────────

def load_dataset(data_path: str, meta_path: Optional[str] = None,
                 label_col: str = "label") -> Dataset:
    """
    Detecta o formato pelo sufixo e retorna o Dataset adequado.

    Args:
        data_path: Caminho para .bin, .csv ou .parquet
        meta_path: Obrigatório para .bin; ignorado para os demais
        label_col: Nome da coluna de label (CSV/Parquet)
    """
    ext = Path(data_path).suffix.lower()

    if ext == ".bin":
        if meta_path is None:
            raise ValueError("metadata.json obrigatório para arquivos .bin")
        schema = BinarySchema(meta_path)
        return BinaryDataset(data_path, schema)

    elif ext in (".csv", ".parquet"):
        return TabularDataset(data_path, label_col=label_col)

    else:
        raise ValueError(f"Formato não suportado: {ext}. Use .bin, .csv ou .parquet.")


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool = True,
                num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False)


# ── Utilitário: lê D_pub e retorna logits como numpy ────────────────────────

def iter_pub_samples(data_path: str, meta_path: Optional[str] = None):
    """
    Gerador que emite (x_tensor,) para cada amostra do dataset público.
    Usado em publish_logits() sem precisar carregar tudo em memória.
    """
    ds = load_dataset(data_path, meta_path)
    for i in range(len(ds)):
        x, _ = ds[i]
        yield x


def dataset_n_features(data_path: str, meta_path: Optional[str] = None) -> int:
    """Retorna o número de features do dataset sem carregá-lo todo."""
    ext = Path(data_path).suffix.lower()
    if ext == ".bin":
        schema = BinarySchema(meta_path)
        return schema.n_features
    else:
        import pandas as pd
        if ext == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path, nrows=1)
        return len([c for c in df.columns
                    if c not in ("label", "timestamp")
                    and np.issubdtype(df[c].dtype, np.number)])
