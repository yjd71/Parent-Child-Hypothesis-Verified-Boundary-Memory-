from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from CBM.context.affinity import js_divergence, normalize_distribution, unfold_neighbors


class ContextualBoundaryAggregator(nn.Module):
    """Stable local context aggregation for point-wise boundary retrieval maps."""

    def __init__(
        self,
        kernel_size: int = 3,
        tau_feat: float = 0.1,
        tau_prob: float = 0.2,
        tau_evi: float = 0.2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.tau_feat = float(tau_feat)
        self.tau_prob = float(tau_prob)
        self.tau_evi = float(tau_evi)
        self.eps = float(eps)

    def forward(
        self,
        p3: torch.Tensor,
        prob3: torch.Tensor,
        Y_map: torch.Tensor,
        R_map: torch.Tensor,
        valid_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_feature_map(p3, "p3")
        bsz, _, height, width = p3.shape
        self._validate_context_map(Y_map, "Y_map", bsz, height, width)
        self._validate_context_map(R_map, "R_map", bsz, height, width)

        prob3 = self._prepare_single_channel(prob3, "prob3", p3, mode="bilinear").clamp_(0.0, 1.0)
        valid_map = self._prepare_single_channel(valid_map, "valid_map", p3, mode="nearest").bool()
        valid_float = valid_map.to(dtype=p3.dtype)

        if not valid_map.any():
            return self._empty_outputs(Y_map, R_map, valid_float)

        kernel_elems = self.kernel_size * self.kernel_size
        p3_norm = F.normalize(p3, dim=1, eps=self.eps)
        p3_center = p3_norm.flatten(2)
        p3_neighbors = unfold_neighbors(p3_norm, self.kernel_size, self.padding)
        sim_feat = (p3_center.unsqueeze(2) * p3_neighbors).sum(dim=1)

        prob_center = prob3.flatten(2)
        prob_neighbors = F.unfold(prob3, kernel_size=self.kernel_size, padding=self.padding)
        sim_prob = -(prob_center - prob_neighbors).abs()

        y_center = Y_map[:, :4].flatten(2)
        y_neighbors = unfold_neighbors(Y_map[:, :4], self.kernel_size, self.padding)
        js_neighbors = js_divergence(y_center.unsqueeze(2), y_neighbors, eps=self.eps).detach()

        valid_center = valid_float.flatten(2)
        valid_neighbors = F.unfold(valid_float, kernel_size=self.kernel_size, padding=self.padding) > 0.5
        logits = (
            sim_feat / max(self.tau_feat, self.eps)
            + sim_prob / max(self.tau_prob, self.eps)
            - js_neighbors / max(self.tau_evi, self.eps)
        )
        logits = logits.masked_fill(~valid_neighbors, -1.0e4)
        weights = F.softmax(logits, dim=1) * valid_neighbors.to(dtype=p3.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        weights = weights * valid_center

        y_ctx_flat = self._aggregate_with_weights(Y_map, weights, kernel_elems)
        r_ctx_flat = self._aggregate_with_weights(R_map, weights, kernel_elems)
        y_ctx = self._flat_to_map(y_ctx_flat, height, width)
        r_ctx = self._flat_to_map(r_ctx_flat, height, width)

        cons_flat = 1.0 - js_divergence(y_center, y_ctx_flat[:, :4], eps=self.eps)
        cons_flat = cons_flat.clamp_(0.0, 1.0) * valid_center[:, 0]
        cons_map = cons_flat.reshape(bsz, 1, height, width)
        return y_ctx, r_ctx, cons_map

    def _empty_outputs(
        self,
        Y_map: torch.Tensor,
        R_map: torch.Tensor,
        valid_float: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.zeros_like(Y_map), torch.zeros_like(R_map), torch.zeros_like(valid_float)

    def _validate_feature_map(self, x: torch.Tensor, name: str) -> None:
        if x.dim() != 4:
            raise ValueError(f"{name} must have shape [B, C, H, W], got {tuple(x.shape)}")

    def _validate_context_map(self, x: torch.Tensor, name: str, bsz: int, height: int, width: int) -> None:
        self._validate_feature_map(x, name)
        if x.size(0) != bsz or tuple(x.shape[-2:]) != (height, width):
            raise ValueError(
                f"{name} must match p3 batch/spatial shape [B, C, H, W], "
                f"got {tuple(x.shape)} for p3 batch/spatial {(bsz, height, width)}"
            )

    def _prepare_single_channel(
        self,
        x: torch.Tensor,
        name: str,
        ref: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"{name} must have shape [B, 1, H, W] or [B, H, W], got {tuple(x.shape)}")
        if x.size(0) != ref.size(0):
            raise ValueError(f"{name} batch size must match p3, got {x.size(0)} and {ref.size(0)}")
        x = x.to(device=ref.device, dtype=ref.dtype)
        if tuple(x.shape[-2:]) == tuple(ref.shape[-2:]):
            return x
        if mode == "nearest":
            return F.interpolate(x, size=ref.shape[-2:], mode=mode)
        return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False)

    def _aggregate_with_weights(self, x: torch.Tensor, weights: torch.Tensor, kernel_elems: int) -> torch.Tensor:
        neighbors = unfold_neighbors(x, self.kernel_size, self.padding)
        return (weights.unsqueeze(1) * neighbors).sum(dim=2)

    def _flat_to_map(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return x.reshape(x.size(0), x.size(1), height, width)

    def _normalize_distribution(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_distribution(x, eps=self.eps)
