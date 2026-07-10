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


class P1LocalStructuredPrior(nn.Module):
    """Structured local prior for P1-PRA attention logits."""

    def __init__(self, in_dim: int = 6, hidden: int = 32) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.residual = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(
        self,
        valid: torch.Tensor,
        b2_refined: torch.Tensor,
        g2_refined: torch.Tensor,
        dist: torch.Tensor,
        offset_mag: torch.Tensor,
        residual_terms: torch.Tensor,
    ) -> torch.Tensor:
        base = valid + b2_refined + g2_refined - dist - 0.1 * offset_mag
        return base + self.gamma.tanh() * self.residual(residual_terms).squeeze(-1)


class P1PixelRefinementAttention(nn.Module):
    """Retarget p2 refs to p1 tokens with local attention."""

    def __init__(
        self,
        p1_ch: int,
        dim: int = 512,
        window: int = 3,
        tau: float = 0.10,
        top_ratio: float = 0.20,
        detach_refs: bool = True,
        max_tokens: int | None = 2500,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.window = int(window)
        self.tau = float(tau)
        self.detach_refs = bool(detach_refs)
        self.boundary_head = BoundaryQueryHead1(
            top_ratio=top_ratio,
            max_tokens=max_tokens,
        )
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
        self.prior_residual = nn.Linear(self.dim, 1)
        self.structured_prior = P1LocalStructuredPrior(in_dim=6)
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
        prior1 = self._structured_prior(p2_aux, batch_ids, flat_indices, (h1, w1), tuple(R2_map.shape[-2:]), local_mask2, local_R2)
        logits = logits + prior1 + self.prior_residual(local_R2).squeeze(-1)
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
            "prior1": prior1,
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
            "prior1": p1.new_empty(0, self.window * self.window),
        }

    def _structured_prior(self, p2_aux, batch_ids, flat_indices, query_hw, ref_hw, local_mask, local_R2):
        local_b2, _ = local_window_gather(p2_aux["B2_refined_map"], batch_ids, flat_indices, query_hw, ref_hw, self.window)
        local_g2, _ = local_window_gather(p2_aux["G2_refined_map"], batch_ids, flat_indices, query_hw, ref_hw, self.window)
        local_valid, _ = local_window_gather(p2_aux["valid2_map"], batch_ids, flat_indices, query_hw, ref_hw, self.window)
        local_offset, _ = local_window_gather(p2_aux["O2_refined_map"], batch_ids, flat_indices, query_hw, ref_hw, self.window)
        b2_refined = local_b2.squeeze(-1).clamp(0.0, 1.0)
        g2_refined = local_g2.squeeze(-1).clamp(0.0, 1.0)
        valid = local_valid.squeeze(-1).clamp(0.0, 1.0) * local_mask.to(dtype=local_R2.dtype)
        offset_mag = local_offset.norm(dim=-1).clamp(0.0, 2.0)
        dist = self._window_distance(local_R2.device, local_R2.dtype).unsqueeze(0).expand_as(valid)
        residual_terms = torch.stack([valid, b2_refined, g2_refined, dist, offset_mag, local_mask.to(dtype=local_R2.dtype)], dim=-1)
        return self.structured_prior(valid, b2_refined, g2_refined, dist, offset_mag, residual_terms)

    def _window_distance(self, device, dtype):
        radius = self.window // 2
        coords = []
        denom = max(float(radius), 1.0)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                coords.append(((dx * dx + dy * dy) ** 0.5) / denom)
        return torch.tensor(coords, device=device, dtype=dtype)
