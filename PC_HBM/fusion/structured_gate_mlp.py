"""Structured GateMLP for PC-HBM token correction gating."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredGateMLP(nn.Module):
    """Structured base plus residual MLP gate for ``gate_pc_token [M,1]``."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Linear(12, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.gamma_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        confidence: torch.Tensor,
        c23: torch.Tensor,
        u_token: torch.Tensor,
        parent_entropy: torch.Tensor,
        child_entropy: torch.Tensor,
        child_scores: torch.Tensor,
        geo_scores: torch.Tensor,
        feature_group_dropout: torch.Tensor | None = None,
    ) -> torch.Tensor:
        child_prob = torch.sigmoid(child_scores)
        geo_prob = torch.sigmoid(geo_scores)
        feat = torch.stack(
            [
                confidence.squeeze(-1),
                c23.squeeze(-1),
                u_token.squeeze(-1),
                parent_entropy,
                child_entropy,
                child_prob.max(dim=1).values,
                child_prob.mean(dim=1),
                child_prob.std(dim=1, unbiased=False),
                geo_prob.max(dim=1).values,
                geo_prob.mean(dim=1),
                (1.0 - c23.squeeze(-1)).clamp(0.0, 1.0),
                confidence.squeeze(-1) * (1.0 - u_token.squeeze(-1)),
            ],
            dim=1,
        )
        if feature_group_dropout is not None:
            feat = feat * feature_group_dropout.to(dtype=feat.dtype)
        base = confidence.squeeze(-1) + 0.5 * child_prob.max(dim=1).values + 0.5 * geo_prob.max(dim=1).values
        base = base - c23.squeeze(-1) - 0.5 * parent_entropy - 0.5 * child_entropy
        residual = self.residual(feat).squeeze(-1)
        return torch.sigmoid(base + self.gamma_gate.tanh() * residual).unsqueeze(1)
