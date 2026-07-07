"""Structured prior bias for PC-HBM hypothesis attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredPriorBiasNet(nn.Module):
    """Positive structured prior plus residual MLP with zero-initial gamma."""

    def __init__(self, value_dim: int = 8, geometry_dim: int = 6, hidden: int = 64) -> None:
        super().__init__()
        in_dim = value_dim + geometry_dim * 2 + 4
        self.residual = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.gamma_prior = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        parent_values: torch.Tensor,
        parent_geo: torch.Tensor,
        child_geo: torch.Tensor,
        s_child: torch.Tensor,
        s_geo: torch.Tensor,
    ) -> torch.Tensor:
        parent_fg = parent_values[..., 5]
        parent_bg = parent_values[..., 4]
        child_pos = torch.sigmoid(s_child)
        geo_pos = torch.sigmoid(s_geo)
        contradiction = parent_bg * child_pos + parent_fg * (1.0 - child_pos)
        prior_base = F.softplus(parent_fg) + F.softplus(child_pos) + F.softplus(geo_pos) - F.softplus(contradiction)
        geo_delta = (parent_geo - child_geo).abs().mean(dim=-1, keepdim=True)
        feat = torch.cat([parent_values, parent_geo, child_geo, s_child.unsqueeze(-1), s_geo.unsqueeze(-1), geo_delta, contradiction.unsqueeze(-1)], dim=-1)
        residual = self.residual(feat).squeeze(-1)
        return prior_base + self.gamma_prior.tanh() * residual
