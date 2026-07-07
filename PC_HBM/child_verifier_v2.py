"""Dynamic p2 child verification for retrieved p3 parent hypotheses."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .geo_score_mlp import GeoScoreMLP
from .structured_prior_bias_net import StructuredPriorBiasNet
from .utils import EPS, js_divergence, normalize_prob


class ChildScoreMLP(nn.Module):
    """Feature compatibility score for ``q_child`` and child memory keys."""

    def __init__(self, dim: int = 512, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, q_child: torch.Tensor, child_keys: torch.Tensor) -> torch.Tensor:
        q = q_child.unsqueeze(1).expand_as(child_keys)
        feat = torch.cat([q, child_keys, (q - child_keys).abs(), q * child_keys], dim=-1)
        return self.net(feat).squeeze(-1)


class HypScoreNet(nn.Module):
    """Combine parent, child, geo and prior scores into hypothesis score."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, parent_scores: torch.Tensor, child_scores: torch.Tensor, geo_scores: torch.Tensor, prior_bias: torch.Tensor) -> torch.Tensor:
        feat = torch.stack([parent_scores, child_scores, geo_scores, prior_bias], dim=-1)
        return self.net(feat).squeeze(-1)


class ChildVerifierV2(nn.Module):
    """Verify parent hypotheses with dynamic p2 child query support."""

    def __init__(self, dim: int = 512, value_dim: int = 8, geometry_dim: int = 6) -> None:
        super().__init__()
        self.child_score = ChildScoreMLP(dim=dim)
        self.geo_score = GeoScoreMLP(geometry_dim=geometry_dim)
        self.prior = StructuredPriorBiasNet(value_dim=value_dim, geometry_dim=geometry_dim)
        self.hyp_score = HypScoreNet()

    def forward(
        self,
        q_child: torch.Tensor,
        g2_query: torch.Tensor,
        parent_ret: Dict[str, torch.Tensor],
        child_bank: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        child_keys = child_bank["p2_child_keys"]
        child_geo = child_bank["p2_child_geo"]
        s_child = self.child_score(q_child, child_keys)
        s_geo = self.geo_score(parent_ret["top_parent_geo"], child_geo, g2_query)
        prior_bias = self.prior(parent_ret["top_parent_values"], parent_ret["top_parent_geo"], child_geo, s_child, s_geo)
        s_hyp = self.hyp_score(parent_ret["top_parent_scores"], s_child, s_geo, prior_bias)
        hyp_attn = torch.softmax(s_hyp, dim=1)
        p_pc_group = (hyp_attn.unsqueeze(-1) * parent_ret["top_parent_values"][..., :4]).sum(dim=1)
        p_pc_group = normalize_prob(p_pc_group, dim=1)
        c23 = js_divergence(parent_ret["P3_group"], p_pc_group, dim=1).unsqueeze(1)
        child_entropy = -(hyp_attn * hyp_attn.clamp_min(EPS).log()).sum(dim=1)
        return {
            "S_child": s_child,
            "S_geo": s_geo,
            "prior_bias": prior_bias,
            "S_hyp": s_hyp,
            "P_pc_group": p_pc_group,
            "C23_token": c23,
            "child_entropy": child_entropy,
            "hyp_attn": hyp_attn,
            "K_child_top": child_keys,
            "G2_child_top": child_geo,
        }
