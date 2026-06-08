from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def prepare_reliability(
    reliability: Optional[torch.Tensor],
    p3: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if reliability is None:
        return torch.ones(p3.size(0), 1, p3.size(2), p3.size(3), device=p3.device, dtype=dtype)
    if reliability.dim() == 3:
        reliability = reliability.unsqueeze(1)
    if reliability.dim() != 4 or reliability.size(1) != 1:
        raise ValueError(f"reliability must have shape [B, 1, H, W] or [B, H, W], got {tuple(reliability.shape)}")
    reliability = reliability.to(device=p3.device, dtype=dtype)
    if tuple(reliability.shape[-2:]) != tuple(p3.shape[-2:]):
        reliability = F.interpolate(reliability, size=p3.shape[-2:], mode="nearest")
    return reliability.clamp(0.0, 1.0)
