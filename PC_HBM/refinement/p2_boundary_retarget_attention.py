"""P2 Boundary Retarget Attention (P2-BRA).

P2-BRA maps verified p3 PC-HBM references to p2 boundary tokens through local
window cross-attention.  Inputs keep TALNet decoder p2 shape ``[B,384,80,80]``
for swin_v1_l and PC embedding shape ``[B,512,H,W]``.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .boundary_query_head import BoundaryQueryHead2
from ..common.utils import (
    add_tokens_to_map,
    finite_or_zero,
    gather_tokens,
    local_window_gather,
    masked_softmax,
    scatter_tokens,
)


class P2BoundaryRetargetAttention(nn.Module):
    """Retarget p3 PC refs to p2 boundary tokens with local attention."""

    def __init__(self, p2_ch: int, dim: int = 512, window: int = 3, tau: float = 0.10, top_ratio: float = 0.25, detach_refs: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        self.window = int(window)
        self.tau = float(tau)
        self.detach_refs = bool(detach_refs)
        self.boundary_head = BoundaryQueryHead2(top_ratio=top_ratio)
        self.query_encoder = nn.Conv2d(int(p2_ch), self.dim, kernel_size=1, bias=False)
        self.ref_encoder = nn.Sequential(
            nn.Conv2d(self.dim + 8 + 6 + 1 + 1 + 1 + 2 + 1, self.dim, kernel_size=1),
            nn.GroupNorm(8, self.dim),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1),
        )
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.prior = nn.Linear(self.dim, 1)
        self.restore = nn.Linear(self.dim, int(p2_ch))
        self.b_head = nn.Linear(self.dim, 1)
        self.g_head = nn.Linear(self.dim, 1)
        self.o_head = nn.Linear(self.dim, 2)
        self.gate = nn.Linear(self.dim, 1)
        nn.init.zeros_(self.restore.weight)
        nn.init.zeros_(self.restore.bias)

    def build_boundary_input(self, prob2: torch.Tensor, pc_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        from ..common.utils import boundary_features_from_logits

        base = boundary_features_from_logits(torch.logit(prob2.clamp(1e-6, 1.0 - 1e-6)))
        target = prob2.shape[-2:]
        extras = [
            torch.nn.functional.interpolate(pc_maps["gate_pc_map"], size=target, mode="bilinear", align_corners=False),
            torch.nn.functional.interpolate(pc_maps["C23_map"], size=target, mode="bilinear", align_corners=False),
            torch.nn.functional.interpolate(pc_maps["M_pc_map"], size=target, mode="bilinear", align_corners=False),
        ]
        return torch.cat([base, *extras], dim=1)

    def forward(self, p2: torch.Tensor, prob2: torch.Tensor, pc_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bsz, _, h2, w2 = p2.shape
        b2_input = self.build_boundary_input(prob2, pc_maps)
        B2, idx = self.boundary_head(b2_input)
        batch_ids = idx["batch_ids"]
        flat_indices = idx["flat_indices"]
        q_map = self.query_encoder(p2)
        q_tokens = gather_tokens(q_map, batch_ids, flat_indices)
        ref = torch.cat(
            [
                pc_maps["Z3_map"],
                pc_maps["E_attn_map"],
                pc_maps["G_attn_map"],
                pc_maps["M_pc_map"],
                pc_maps["gate_pc_map"],
                pc_maps["C23_map"],
                pc_maps["O_pc_map"],
                pc_maps["valid3_map"],
            ],
            dim=1,
        )
        if self.detach_refs:
            ref = ref.detach()
        R3_map = self.ref_encoder(ref)
        local_R3, local_mask3 = local_window_gather(R3_map, batch_ids, flat_indices, (h2, w2), tuple(R3_map.shape[-2:]), self.window)
        if q_tokens.numel() == 0:
            return self._empty(p2, B2, R3_map)
        q = self.q_proj(q_tokens).unsqueeze(1)
        k = self.k_proj(local_R3)
        v = self.v_proj(local_R3)
        logits = (q * k).sum(dim=-1) / (self.dim ** 0.5) / max(self.tau, 1e-6)
        logits = logits + self.prior(local_R3).squeeze(-1)
        attn = masked_softmax(logits, local_mask3, dim=1)
        f2_tokens = finite_or_zero((attn.unsqueeze(-1) * v).sum(dim=1))
        gate2 = torch.sigmoid(self.gate(f2_tokens))
        restored = self.restore(f2_tokens) * gate2
        p2_refined = add_tokens_to_map(p2, batch_ids, flat_indices, restored)
        B2_refined_map = scatter_tokens((bsz, 1, h2, w2), batch_ids, flat_indices, torch.sigmoid(self.b_head(f2_tokens)), reduce="replace")
        G2_refined_map = scatter_tokens((bsz, 1, h2, w2), batch_ids, flat_indices, torch.sigmoid(self.g_head(f2_tokens)), reduce="replace")
        O2_refined_map = scatter_tokens((bsz, 2, h2, w2), batch_ids, flat_indices, torch.tanh(self.o_head(f2_tokens)), reduce="replace")
        F2_ref_map = scatter_tokens((bsz, self.dim, h2, w2), batch_ids, flat_indices, f2_tokens, reduce="replace")
        valid2_map = scatter_tokens((bsz, 1, h2, w2), batch_ids, flat_indices, torch.ones_like(gate2), reduce="replace")
        return {
            "p2_refined": p2_refined,
            "F2_ref_map": F2_ref_map,
            "B2_refined_map": B2_refined_map,
            "G2_refined_map": G2_refined_map,
            "O2_refined_map": O2_refined_map,
            "valid2_map": valid2_map,
            "B2": B2,
            "boundary_indices2": idx,
            "R3_map": R3_map,
            "attn2": attn,
        }

    def _empty(self, p2: torch.Tensor, B2: torch.Tensor, R3_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, _, h2, w2 = p2.shape
        return {
            "p2_refined": p2,
            "F2_ref_map": p2.new_zeros(bsz, self.dim, h2, w2),
            "B2_refined_map": p2.new_zeros(bsz, 1, h2, w2),
            "G2_refined_map": p2.new_zeros(bsz, 1, h2, w2),
            "O2_refined_map": p2.new_zeros(bsz, 2, h2, w2),
            "valid2_map": p2.new_zeros(bsz, 1, h2, w2),
            "B2": B2,
            "boundary_indices2": {"batch_ids": torch.empty(0, device=p2.device, dtype=torch.long), "flat_indices": torch.empty(0, device=p2.device, dtype=torch.long)},
            "R3_map": R3_map,
            "attn2": p2.new_empty(0, self.window * self.window),
        }
