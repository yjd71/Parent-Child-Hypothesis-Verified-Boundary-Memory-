"""p3 gated residual correction from PC-HBM token states."""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import add_tokens_to_map


class P3GatedResidual(nn.Module):
    """Project PC tokens to p3 channels and add only at boundary token sites."""

    def __init__(self, dim: int = 512, p3_ch: int = 768) -> None:
        super().__init__()
        self.out = nn.Linear(dim, int(p3_ch))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, p3: torch.Tensor, batch_ids: torch.Tensor, flat_indices: torch.Tensor, z3_token: torch.Tensor, gate: torch.Tensor, gate_pc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.out(z3_token) * gate * gate_pc
        return add_tokens_to_map(p3, batch_ids, flat_indices, delta), delta
