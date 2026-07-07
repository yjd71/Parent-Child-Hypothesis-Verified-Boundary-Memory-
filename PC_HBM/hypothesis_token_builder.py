"""Encode parent-child hypotheses into attention tokens."""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import normalize


class HypothesisTokenBuilder(nn.Module):
    """Build ``H_tokens [M,K,512]`` from parent/child/evidence tensors."""

    def __init__(self, dim: int = 512, value_dim: int = 8, geometry_dim: int = 6, hidden: int = 512) -> None:
        super().__init__()
        self.region_embed = nn.Embedding(4, dim)
        in_dim = dim * 2 + value_dim + geometry_dim * 2 + 4 + dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, parent_ret: dict, child_ver: dict) -> torch.Tensor:
        values = parent_ret["top_parent_values"]
        region_id = values[..., :4].argmax(dim=-1).clamp(0, 3)
        region = self.region_embed(region_id)
        scores = torch.stack(
            [
                parent_ret["top_parent_scores"],
                child_ver["S_child"],
                child_ver["S_geo"],
                child_ver["prior_bias"],
            ],
            dim=-1,
        )
        feat = torch.cat(
            [
                parent_ret["top_parent_keys"],
                child_ver["K_child_top"],
                values,
                parent_ret["top_parent_geo"],
                child_ver["G2_child_top"],
                scores,
                region,
            ],
            dim=-1,
        )
        return normalize(self.net(feat), dim=-1)
