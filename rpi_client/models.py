"""
models.py — FedKD-MR Cliente Raspberry Pi
==========================================
Modelos PyTorch para HAR. Todos aceitam input flat x[n_input] e
retornam logits não-normalizados (sem Softmax — CrossEntropyLoss aplica).

Arquiteturas disponíveis:
  MLP    — camadas densas totalmente conectadas
  CNN1D  — conv temporal sobre sequência [C, T]
  CNN2D  — conv espacial sobre mapa [1, H, W]  (H=T_WINDOW, W=N_FEATURES por padrão)
  GRU    — GRU empilhado, last hidden state → Dense
  LSTM   — LSTM empilhado, last hidden state → Dense

Todos herdam de FedKDModel que expõe get_flat_params() / set_flat_params()
para serialização (útil se quiser agregar pesos no futuro).
"""

import torch
import torch.nn as nn
from typing import List


# ── Classe base ──────────────────────────────────────────────────────────────

class FedKDModel(nn.Module):
    """Interface base com utilitários de serialização de parâmetros."""

    def get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat: torch.Tensor) -> None:
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[offset: offset + n].view_as(p))
            offset += n

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── MLP ──────────────────────────────────────────────────────────────────────

class MLP(FedKDModel):
    """
    Perceptron multi-camada.

    Args:
        n_input:    Tamanho do vetor de entrada (T_WINDOW × N_FEATURES).
        hidden:     Lista com dimensões das camadas ocultas. Ex: [128, 64]
        n_classes:  Número de classes de saída.
        dropout:    Taxa de dropout após cada camada oculta (0 = desabilitado).
    """

    def __init__(self, n_input: int, hidden: List[int],
                 n_classes: int, dropout: float = 0.3):
        super().__init__()
        dims   = [n_input] + hidden + [n_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # não aplica ReLU/dropout na saída
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_input)
        return self.net(x)


# ── CNN1D ─────────────────────────────────────────────────────────────────────

class CNN1D(FedKDModel):
    """
    Convolução temporal 1D.

    Input: x flat (B, n_input) → reshape → (B, C_in=N_FEATURES, T=T_WINDOW)
    Blocos Conv1d → BatchNorm → ReLU (→ Dropout) → GlobalMaxPool → Dense

    Args:
        t_window:   Número de timesteps.
        n_features: Features por timestep (= canais de entrada).
        channels:   Filtros por camada Conv1d. Ex: [32, 64, 128]
        kernel:     Tamanho do kernel (padding=kernel//2 → mantém comprimento).
        n_classes:  Classes de saída.
        dropout:    Taxa de dropout (0 = desabilitado).
    """

    def __init__(self, t_window: int, n_features: int,
                 channels: List[int], kernel: int, n_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        self.t_window   = t_window
        self.n_features = n_features

        in_ch  = n_features
        convs  = []
        for out_ch in channels:
            convs += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                          padding=kernel // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            if dropout > 0:
                convs.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.convs   = nn.Sequential(*convs)
        self.pool    = nn.AdaptiveMaxPool1d(1)   # GlobalMaxPool
        self.flatten = nn.Flatten()
        self.head    = nn.Linear(channels[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_input) → (B, N_FEATURES, T_WINDOW)
        B = x.size(0)
        x = x.view(B, self.t_window, self.n_features)   # (B, T, C)
        x = x.permute(0, 2, 1)                           # (B, C, T)
        x = self.convs(x)                                 # (B, out_ch, T)
        x = self.pool(x)                                  # (B, out_ch, 1)
        x = self.flatten(x)                               # (B, out_ch)
        return self.head(x)


# ── CNN2D ─────────────────────────────────────────────────────────────────────

class CNN2D(FedKDModel):
    """
    Convolução espacial 2D sobre o mapa tempo-frequência.

    Input: x flat (B, n_input) → reshape → (B, 1, H, W)
    H e W são configuráveis (default: H=T_WINDOW, W=N_FEATURES).

    Args:
        img_h, img_w: Dimensões da imagem 2D.
        channels:     Filtros por camada Conv2d. Ex: [32, 64, 128]
        kernel:       Tamanho do kernel quadrado. Ex: 3 → (3×3)
        n_classes:    Classes de saída.
        dropout:      Taxa de dropout.
    """

    def __init__(self, img_h: int, img_w: int,
                 channels: List[int], kernel: int, n_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w

        in_ch = 1
        convs = []
        for out_ch in channels:
            convs += [
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                          padding=kernel // 2, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ]
            if dropout > 0:
                convs.append(nn.Dropout2d(dropout))
            in_ch = out_ch

        self.convs   = nn.Sequential(*convs)
        self.pool    = nn.AdaptiveMaxPool2d(1)   # GlobalMaxPool2D
        self.flatten = nn.Flatten()
        self.head    = nn.Linear(channels[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_input) → (B, 1, H, W)
        B = x.size(0)
        x = x.view(B, 1, self.img_h, self.img_w)
        x = self.convs(x)                         # (B, out_ch, H', W')
        x = self.pool(x)                           # (B, out_ch, 1, 1)
        x = self.flatten(x)                        # (B, out_ch)
        return self.head(x)


# ── GRU ──────────────────────────────────────────────────────────────────────

class GRUModel(FedKDModel):
    """
    GRU empilhado com GlobalMaxPool temporal na saída → Dense.

    Input: x flat (B, n_input) → reshape → (B, T, C)

    Args:
        t_window:       Timesteps.
        n_features:     Features / timestep (= input_size do GRU).
        hidden_size:    Dimensão do estado oculto.
        num_layers:     Camadas GRU empilhadas.
        bidirectional:  Se True, dobra hidden_size na saída.
        dropout:        Dropout entre camadas GRU (num_layers > 1).
        n_classes:      Classes de saída.
    """

    def __init__(self, t_window: int, n_features: int,
                 hidden_size: int, num_layers: int,
                 bidirectional: bool, dropout: float, n_classes: int):
        super().__init__()
        self.t_window   = t_window
        self.n_features = n_features

        self.gru = nn.GRU(
            input_size    = n_features,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = bidirectional,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        out_size  = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_input) → (B, T, C)
        B = x.size(0)
        x = x.view(B, self.t_window, self.n_features)
        out, _ = self.gru(x)           # out: (B, T, out_size)
        # GlobalMaxPool temporal
        x = out.max(dim=1).values      # (B, out_size)
        return self.head(x)


# ── LSTM ─────────────────────────────────────────────────────────────────────

class LSTMModel(FedKDModel):
    """
    LSTM empilhado com GlobalMaxPool temporal na saída → Dense.

    Mesmos parâmetros do GRUModel.
    """

    def __init__(self, t_window: int, n_features: int,
                 hidden_size: int, num_layers: int,
                 bidirectional: bool, dropout: float, n_classes: int):
        super().__init__()
        self.t_window   = t_window
        self.n_features = n_features

        self.lstm = nn.LSTM(
            input_size    = n_features,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = bidirectional,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        out_size  = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.view(B, self.t_window, self.n_features)
        out, _ = self.lstm(x)          # out: (B, T, out_size)
        x = out.max(dim=1).values      # GlobalMaxPool temporal
        return self.head(x)


# ── Factory ──────────────────────────────────────────────────────────────────

def build_model(cfg) -> FedKDModel:
    """
    Instancia o modelo escolhido a partir do namespace de config.py.

    Args:
        cfg: Namespace retornado por get_config() em config.py.

    Returns:
        Modelo FedKDModel pronto para treino.
    """
    arch = cfg.model.lower()

    if arch == "mlp":
        return MLP(
            n_input   = cfg.n_input,
            hidden    = cfg.mlp_hidden,
            n_classes = cfg.n_classes,
            dropout   = cfg.cnn_dropout,   # reutiliza flag genérica
        )

    elif arch == "cnn1d":
        return CNN1D(
            t_window   = cfg.t_window,
            n_features = cfg.n_features,
            channels   = cfg.cnn_channels,
            kernel     = cfg.cnn_kernel,
            n_classes  = cfg.n_classes,
            dropout    = cfg.cnn_dropout,
        )

    elif arch == "cnn2d":
        return CNN2D(
            img_h     = cfg.cnn2d_h,
            img_w     = cfg.cnn2d_w,
            channels  = cfg.cnn_channels,
            kernel    = cfg.cnn_kernel,
            n_classes = cfg.n_classes,
            dropout   = cfg.cnn_dropout,
        )

    elif arch == "gru":
        return GRUModel(
            t_window      = cfg.t_window,
            n_features    = cfg.n_features,
            hidden_size   = cfg.rnn_hidden,
            num_layers    = cfg.rnn_layers,
            bidirectional = cfg.rnn_bidirectional,
            dropout       = cfg.rnn_dropout,
            n_classes     = cfg.n_classes,
        )

    elif arch == "lstm":
        return LSTMModel(
            t_window      = cfg.t_window,
            n_features    = cfg.n_features,
            hidden_size   = cfg.rnn_hidden,
            num_layers    = cfg.rnn_layers,
            bidirectional = cfg.rnn_bidirectional,
            dropout       = cfg.rnn_dropout,
            n_classes     = cfg.n_classes,
        )

    else:
        raise ValueError(f"Arquitetura desconhecida: {arch}")
