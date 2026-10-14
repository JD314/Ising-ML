"""
Lightweight phase classifiers (P ~ 1-2 k parameters, so P << N_samples).

Both models map spins s in {-1, +1}^(L x L) (shape (B, L, L) or (B, 1, L, L))
to one raw logit per sample (shape (B, 1)); apply a sigmoid externally.
"""

from __future__ import annotations

import torch
from torch import nn


def _as_image(x: torch.Tensor) -> torch.Tensor:
    """(B, L, L) -> (B, 1, L, L) float; (B, 1, L, L) passes through."""
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return x.to(torch.get_default_dtype()) if not x.is_floating_point() else x


class SimpleMLP(nn.Module):
    """Flatten -> Linear(L^2, 16) -> ReLU -> Linear(16, 1)."""

    def __init__(self, L: int):
        super().__init__()
        self.L = L
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(L * L, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(_as_image(x))


class SimpleCNN(nn.Module):
    """
    Conv2d(1, 8, 3, circular) -> ReLU -> MaxPool2d(2)
    -> Conv2d(8, 16, 3, padding=1) -> ReLU -> AdaptiveAvgPool2d(1) -> Linear(16, 1).

    The first convolution wraps periodically (torus); the second uses the
    default zero padding, as specified.
    """

    def __init__(self, L: int):
        super().__init__()
        self.L = L
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(_as_image(x)).flatten(1))
