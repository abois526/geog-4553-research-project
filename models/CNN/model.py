"""
model.py  —  Downy Brome SDM
1D CNN binary classifier for multi-band spectral vectors.

Input  : (batch, n_bands)   e.g. (64, 64)
Output : (batch, 1)         sigmoid probability  [0, 1]
"""

import torch
import torch.nn as nn


class SpectralSDM(nn.Module):
    """
    1D CNN that treats the n_bands spectral vector as a 1D sequence.

    Architecture
    ------------
    Three conv blocks (Conv1d → BN → ReLU → Dropout) with increasing
    filter depth, followed by global average pooling and a two-layer MLP head.
    """

    def __init__(self, n_bands: int = 64, dropout: float = 0.4):
        super().__init__()

        # --- Convolutional backbone ---
        # Input shape: (batch, 1, n_bands)  — treat bands as 1-channel sequence
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(1,  32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Global average pool → collapses the band dimension
        self.gap = nn.AdaptiveAvgPool1d(1)   # (batch, 128, 1)

        # --- MLP head ---
        self.head = nn.Sequential(
            nn.Flatten(),                    # (batch, 128)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),                    # presence probability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_bands)
        x = x.unsqueeze(1)          # → (batch, 1, n_bands)
        x = self.conv_blocks(x)     # → (batch, 128, n_bands)
        x = self.gap(x)             # → (batch, 128, 1)
        x = self.head(x)            # → (batch, 1)
        return x


def get_model(n_bands: int = 64, dropout: float = 0.4) -> SpectralSDM:
    return SpectralSDM(n_bands=n_bands, dropout=dropout)
