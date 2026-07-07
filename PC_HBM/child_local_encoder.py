"""Local p2 child patch encoder.

Input patches are ``[N,p2_ch,window,window]`` and output keys are
``[N,512]`` by default.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import normalize


class ChildLocalEncoder(nn.Module):
    """Encode p2 local patches into normalized child keys."""

    def __init__(self, in_ch: int, dim: int = 512, window: int = 5) -> None:
        super().__init__()
        hidden = max(64, dim // 2)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.numel() == 0:
            return patches.new_empty(0, self.proj.out_features)
        x = self.net(patches).flatten(1)
        return normalize(self.proj(x), dim=-1)
