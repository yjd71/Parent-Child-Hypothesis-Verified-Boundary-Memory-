"""P1 Pixel Refinement Attention (P1-PRA).

P1-PRA refines high-resolution p1 boundary tokens using local p2 references and
produces pixel-level gate/residual/offset/suppression maps for the final mixture.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boundary_query_head import BoundaryQueryHead1
from ..common.utils import (
    boundary_features_from_logits,
    finite_or_zero,
    gather_tokens,
    local_window_gather,
    masked_softmax,
    scatter_tokens,
)


class P1PixelRefinementAttention(nn.Module):
    """Retarget p2 refs to p1 tokens with local attention."""

    def __init__(self, p1_ch: int, dim: int = 512, window: int = 3, tau: float = 0.10, top_ratio: float = 0.20, detach_refs: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        self.window = int(window)
        self.tau = float(tau)
        self.detach_refs = bool(detach_refs)
        self.boundary_head = BoundaryQueryHead1(top_ratio=top_ratio)
        self.query_encoder = nn.Conv2d(int(p1_ch), self.dim, kernel_size=1, bias=False)
        self.ref_encoder = nn.Sequential(
            nn.Conv2d(self.dim + 1 + 1 + 2 + 1, self.dim, kernel_size=1),
            nn.GroupNorm(8, self.dim),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1),
        )
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.prior = nn.Linear(self.dim, 1)
        self.g_head = nn.Linear(self.dim, 1)
        self.r_head = nn.Linear(self.dim, 1)
        self.o_head = nn.Linear(self.dim, 2)
        self.sup_head = nn.Linear(self.dim, 1)

    def build_boundary_input(self, z_main: torch.Tensor, p1_hw, p2_aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        z160 = F.interpolate(z_main, size=p1_hw, mode="bilinear", align_corners=False)
        base = boundary_features_from_logits(z160)
        extras = [
            F.interpolate(p2_aux["B2_refined_map"], size=p1_hw, mode="bilinear", align_corners=False),
            F.interpolate(p2_aux["G2_refined_map"], size=p1_hw, mode="bilinear", align_corners=False),
            F.interpolate(p2_aux["valid2_map"], size=p1_hw, mode="bilinear", align_corners=False),
        ]
        return torch.cat([base, *extras], dim=1)

    def forward(self, p1: torch.Tensor, z_main: torch.Tensor, p2_aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bsz, _, h1, w1 = p1.shape
        b1_input = self.build_boundary_input(z_main, (h1, w1), p2_aux)
        B1, idx = self.boundary_head(b1_input)
        batch_ids = idx["batch_ids"]
        flat_indices = idx["flat_indices"]
        q_map = self.query_encoder(p1)
        q_tokens = gather_tokens(q_map, batch_ids, flat_indices)
        ref = torch.cat(
            [
                p2_aux["F2_ref_map"],
                p2_aux["B2_refined_map"],
                p2_aux["G2_refined_map"],
                p2_aux["O2_refined_map"],
                p2_aux["valid2_map"],
            ],
            dim=1,
        )
        if self.detach_refs:
            ref = ref.detach()
        R2_map = self.ref_encoder(ref)
        local_R2, local_mask2 = local_window_gather(R2_map, batch_ids, flat_indices, (h1, w1), tuple(R2_map.shape[-2:]), self.window)
        if q_tokens.numel() == 0:
            return self._empty(p1, B1, R2_map)
        q = self.q_proj(q_tokens).unsqueeze(1)
        k = self.k_proj(local_R2)
        v = self.v_proj(local_R2)
        logits = (q * k).sum(dim=-1) / (self.dim ** 0.5) / max(self.tau, 1e-6)
        logits = logits + self.prior(local_R2).squeeze(-1)
        attn = masked_softmax(logits, local_mask2, dim=1)
        r1_tokens = finite_or_zero((attn.unsqueeze(-1) * v).sum(dim=1))
        G1_map = scatter_tokens((bsz, 1, h1, w1), batch_ids, flat_indices, torch.sigmoid(self.g_head(r1_tokens)), reduce="replace")
        R1_map = scatter_tokens((bsz, 1, h1, w1), batch_ids, flat_indices, torch.tanh(self.r_head(r1_tokens)), reduce="replace")
        O1_map = scatter_tokens((bsz, 2, h1, w1), batch_ids, flat_indices, torch.tanh(self.o_head(r1_tokens)), reduce="replace")
        R_sup_map = scatter_tokens((bsz, 1, h1, w1), batch_ids, flat_indices, F.softplus(self.sup_head(r1_tokens)), reduce="replace")
        valid1_map = scatter_tokens((bsz, 1, h1, w1), batch_ids, flat_indices, torch.ones_like(G1_map.flatten(2).transpose(1, 2)[batch_ids, flat_indices]), reduce="replace")
        return {
            "G1_map": G1_map,
            "R1_map": R1_map,
            "O1_map": O1_map,
            "R_sup_map": R_sup_map,
            "valid1_map": valid1_map,
            "B1": B1,
            "boundary_indices1": idx,
            "R2_map": R2_map,
            "attn1": attn,
        }

    def _empty(self, p1: torch.Tensor, B1: torch.Tensor, R2_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, _, h1, w1 = p1.shape
        return {
            "G1_map": p1.new_zeros(bsz, 1, h1, w1),
            "R1_map": p1.new_zeros(bsz, 1, h1, w1),
            "O1_map": p1.new_zeros(bsz, 2, h1, w1),
            "R_sup_map": p1.new_zeros(bsz, 1, h1, w1),
            "valid1_map": p1.new_zeros(bsz, 1, h1, w1),
            "B1": B1,
            "boundary_indices1": {"batch_ids": torch.empty(0, device=p1.device, dtype=torch.long), "flat_indices": torch.empty(0, device=p1.device, dtype=torch.long)},
            "R2_map": R2_map,
            "attn1": p1.new_empty(0, self.window * self.window),
        }
