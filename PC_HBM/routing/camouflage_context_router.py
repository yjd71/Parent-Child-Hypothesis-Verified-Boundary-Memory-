"""x3 camouflage-context routing into labelled image memory."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .route_attention_pool import RouteAttentionPool
from ..common.utils import EPS, finite_or_zero, gradient_strength, normalize


class CamouflageContextRouter(nn.Module):
    """Build five x3 route descriptors and query PC-HBM route memory."""

    def __init__(self, x3_ch: int, dim: int = 512, top_img_k: int = 32) -> None:
        super().__init__()
        self.dim = int(dim)
        self.top_img_k = int(top_img_k)
        self.proj_x3 = nn.Conv2d(int(x3_ch), self.dim, kernel_size=1, bias=False)
        self.pool = RouteAttentionPool(self.dim)

    def encode_route_tokens(self, x3: torch.Tensor, prob3: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        z = self.proj_x3(x3)
        if prob3 is None:
            prob = torch.sigmoid(z[:, :1])
        else:
            prob = F.interpolate(prob3, size=z.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        unc = 4.0 * prob * (1.0 - prob)
        grad = gradient_strength(prob)
        bg = (prob < 0.35).float()
        env = 1.0 - grad.clamp(0.0, 1.0)
        descriptors = {
            "x3_global": self._masked_pool(z, torch.ones_like(prob)),
            "x3_boundary": self._masked_pool(z, grad + unc),
            "x3_uncertain": self._masked_pool(z, unc),
            "x3_bg_near": self._masked_pool(z, bg * (grad + 0.25)),
            "x3_environment": self._masked_pool(z, env),
        }
        stacked = torch.stack([descriptors[name] for name in ("x3_global", "x3_boundary", "x3_uncertain", "x3_bg_near", "x3_environment")], dim=1)
        route_embed, weights = self.pool(stacked)
        descriptors["route_embed"] = route_embed
        descriptors["route_weights"] = weights
        return descriptors

    def forward(self, x3: torch.Tensor, prob3: torch.Tensor, memory, top_img_k: int | None = None) -> Dict[str, object]:
        tokens = self.encode_route_tokens(x3, prob3)
        routed = memory.route_query(tokens["route_embed"], top_img_k or self.top_img_k)
        routed["route_context"] = tokens["route_embed"]
        routed["route_tokens"] = tokens
        return routed

    def _masked_pool(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = F.interpolate(mask, size=z.shape[-2:], mode="bilinear", align_corners=False).clamp_min(0.0)
        denom = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)
        pooled = (z * mask).sum(dim=(-2, -1), keepdim=True) / denom
        return normalize(finite_or_zero(pooled.flatten(1)), dim=-1)
