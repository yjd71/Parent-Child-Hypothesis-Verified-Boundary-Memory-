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
        self.evidence_residual = nn.Linear(dim, 8)
        self.mask_residual = nn.Linear(dim, 1)
        self.offset = nn.Linear(dim + 6 + 6 + 1, 2)
        nn.init.zeros_(self.evidence_residual.weight)
        nn.init.zeros_(self.evidence_residual.bias)
        nn.init.zeros_(self.mask_residual.weight)
        nn.init.zeros_(self.mask_residual.bias)

    def forward(self, q3_new: torch.Tensor, attn: torch.Tensor, parent_ret: dict, child_ver: dict) -> dict:
        weights = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        e_attn = (weights.unsqueeze(-1) * parent_ret["top_parent_values"]).sum(dim=1)
        g_attn = (weights.unsqueeze(-1) * parent_ret["top_parent_geo"]).sum(dim=1)
        g_child = (weights.unsqueeze(-1) * child_ver["G2_child_top"]).sum(dim=1)
        e_attn = e_attn + 0.1 * self.evidence_residual(q3_new)
        s_fg = e_attn[:, 0] + e_attn[:, 1]
        s_bg = e_attn[:, 2] + e_attn[:, 3]
        m_evidence = s_fg - s_bg
        m_residual = 0.1 * torch.tanh(self.mask_residual(q3_new)).squeeze(-1)
        m_pc = (m_evidence + m_residual).clamp(-1.0, 1.0)
        offset_in = torch.cat([q3_new, g_attn, g_child, m_pc[:, None]], dim=-1)
        offset = torch.tanh(self.offset(offset_in))
        return {
            "E_attn": e_attn,
            "G_attn": g_attn,
            "G_child_attn": g_child,
            "M_pc_token": m_pc[:, None],
            "M_pc_evidence": m_evidence[:, None],
            "M_pc_residual": m_residual[:, None],
            "O_pc_token": offset,
            "Z3_token": q3_new,
        }
