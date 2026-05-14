"""Small MLP used for engine smoke tests and checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn


class TestMLP(nn.Module):
    """Tiny MLP: ``[B, 128]`` -> ``[B, 10]``."""

    def __init__(self, in_features: int = 128, hidden: int = 64, out_features: int = 10) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
