"""Attention pooling for camouflage-context route descriptors."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..common.utils import normalize


class RouteAttentionPool(nn.Module):
    """Pool five route descriptors into a single ``[B,D]`` query."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: tokens ``[B,5,D]``. Returns route query and weights."""

        logits = self.score(tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * tokens, dim=1)
        return normalize(pooled, dim=-1), weights
