"""Build query state tokens for PC-HCA from p3 and route context."""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import normalize


class QueryStateBuilder(nn.Module):
    """Fuse q3, q_child and route context into ``[M,512]`` states."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3 + 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, q3: torch.Tensor, q_child: torch.Tensor, route_context_token: torch.Tensor, c23: torch.Tensor, parent_entropy: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([q3, q_child, route_context_token, c23, parent_entropy.unsqueeze(1)], dim=1)
        return normalize(self.net(feat), dim=-1)
