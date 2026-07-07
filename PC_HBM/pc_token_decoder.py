"""Decode attended hypothesis evidence into token-level maps."""

from __future__ import annotations

import torch
import torch.nn as nn


class PCTokenDecoder(nn.Module):
    """Decode q3_new and attention-weighted hypotheses.

    Returns evidence/value geometry/map correction tokens for scatter.
    """

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.evidence = nn.Linear(dim, 8)
        self.mask = nn.Linear(dim, 1)
        self.offset = nn.Linear(dim, 2)

    def forward(self, q3_new: torch.Tensor, attn: torch.Tensor, parent_ret: dict, child_ver: dict) -> dict:
        weights = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        e_attn = (weights.unsqueeze(-1) * parent_ret["top_parent_values"]).sum(dim=1)
        g_attn = (weights.unsqueeze(-1) * parent_ret["top_parent_geo"]).sum(dim=1)
        g_child = (weights.unsqueeze(-1) * child_ver["G2_child_top"]).sum(dim=1)
        e_learned = self.evidence(q3_new)
        m_pc = torch.sigmoid(self.mask(q3_new))
        offset = torch.tanh(self.offset(q3_new))
        return {
            "E_attn": e_attn + e_learned * 0.1,
            "G_attn": g_attn,
            "G_child_attn": g_child,
            "M_pc_token": m_pc,
            "O_pc_token": offset,
            "Z3_token": q3_new,
        }
