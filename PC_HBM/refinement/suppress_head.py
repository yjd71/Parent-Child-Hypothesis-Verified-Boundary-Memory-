"""Suppression candidate helper for final adaptive mixture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuppressHead(nn.Module):
    """Predict positive suppression magnitude from compact context maps."""

    def __init__(self, in_ch: int = 4, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x))
