from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLogitFusion(nn.Module):
    """Fuse boundary memory logits into the main segmentation logit."""

    def __init__(self, lambda_logit: float = 0.5) -> None:
        super().__init__()
        self.lambda_logit = float(lambda_logit)

    def forward(
        self,
        z_main: torch.Tensor,
        z_mem3: torch.Tensor,
        B_query: torch.Tensor,
        gate3: torch.Tensor,
    ) -> torch.Tensor:
        z_main = self._validate_main_logit(z_main)
        z_mem_up = self._prepare_single_channel(z_mem3, "z_mem3", z_main).to(dtype=z_main.dtype)
        B_up = self._prepare_single_channel(B_query, "B_query", z_main).clamp(0.0, 1.0)
        gate_up = self._prepare_single_channel(gate3, "gate3", z_main).clamp(0.0, 1.0)
        return z_main + self.lambda_logit * B_up * gate_up * (z_mem_up - z_main)

    def _validate_main_logit(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"z_main must have shape [B, 1, H, W], got {tuple(x.shape)}")
        return x

    def _prepare_single_channel(self, x: torch.Tensor, name: str, ref: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"{name} must have shape [B, 1, H, W] or [B, H, W], got {tuple(x.shape)}")
        if x.size(0) != ref.size(0):
            raise ValueError(f"{name} batch size must match z_main, got {x.size(0)} and {ref.size(0)}")

        x = x.to(device=ref.device, dtype=ref.dtype)
        if tuple(x.shape[-2:]) == tuple(ref.shape[-2:]):
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
