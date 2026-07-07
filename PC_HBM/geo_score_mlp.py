"""Geometry compatibility scoring for parent-child hypotheses."""

from __future__ import annotations

import torch
import torch.nn as nn


class GeoScoreMLP(nn.Module):
    """Score geometry consistency for ``[M,K,6]`` parent/child/query values."""

    def __init__(self, geometry_dim: int = 6, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(geometry_dim * 3 + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, parent_geo: torch.Tensor, child_geo: torch.Tensor, query_geo: torch.Tensor) -> torch.Tensor:
        q = query_geo.unsqueeze(1).expand_as(parent_geo)
        delta_pc = (parent_geo - child_geo).abs().mean(dim=-1, keepdim=True)
        delta_pq = (parent_geo - q).abs().mean(dim=-1, keepdim=True)
        delta_cq = (child_geo - q).abs().mean(dim=-1, keepdim=True)
        feat = torch.cat([parent_geo, child_geo, q, delta_pc, delta_pq, delta_cq], dim=-1)
        return self.net(feat).squeeze(-1)
