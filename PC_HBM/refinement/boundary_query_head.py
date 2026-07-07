"""Boundary query heads for p3/p2/p1 token selection.

Inputs are boundary feature maps ``[B,C,H,W]`` and outputs are score maps
``[B,1,H,W]`` plus batch-aware flat token indices.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..common.utils import finite_or_zero, token_indices_from_score


class BoundaryQueryHead(nn.Module):
    """Small convolutional boundary scorer with top-k/threshold selection."""

    def __init__(self, in_ch: int, hidden_ch: int = 32, top_ratio: float = 0.25, min_tokens: int = 1, max_tokens: int | None = None) -> None:
        super().__init__()
        self.top_ratio = float(top_ratio)
        self.min_tokens = int(min_tokens)
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), hidden_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, threshold: float | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        score = torch.sigmoid(finite_or_zero(self.net(x)))
        batch_ids, flat_indices, token_scores = token_indices_from_score(
            score,
            top_ratio=self.top_ratio,
            threshold=threshold,
            min_tokens=self.min_tokens,
            max_tokens=self.max_tokens,
        )
        return score, {
            "batch_ids": batch_ids,
            "flat_indices": flat_indices,
            "token_scores": token_scores,
            "height": torch.tensor(score.size(2), device=score.device),
            "width": torch.tensor(score.size(3), device=score.device),
        }


class BoundaryQueryHead3(BoundaryQueryHead):
    """p3 boundary scorer, input ``[B,5,40,40]``."""

    def __init__(self, top_ratio: float = 0.25, max_tokens: int | None = None) -> None:
        super().__init__(5, top_ratio=top_ratio, max_tokens=max_tokens)


class BoundaryQueryHead2(BoundaryQueryHead):
    """p2 boundary scorer, input ``[B,8,80,80]``."""

    def __init__(self, top_ratio: float = 0.25, max_tokens: int | None = None) -> None:
        super().__init__(8, top_ratio=top_ratio, max_tokens=max_tokens)


class BoundaryQueryHead1(BoundaryQueryHead):
    """p1 boundary scorer, input ``[B,8,160,160]``."""

    def __init__(self, top_ratio: float = 0.20, max_tokens: int | None = None) -> None:
        super().__init__(8, top_ratio=top_ratio, max_tokens=max_tokens)
