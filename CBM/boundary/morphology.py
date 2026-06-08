from __future__ import annotations

import torch
import torch.nn.functional as F


def as_4d_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4 or mask.size(1) != 1:
        raise ValueError(f"{name} must have shape [B, 1, H, W] or [B, H, W], got {tuple(mask.shape)}")
    return mask


def odd_kernel(kernel: int) -> int:
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError(f"kernel must be a positive odd integer, got {kernel}")
    return kernel


def dilate(mask: torch.Tensor, kernel: int = 3, iterations: int = 1) -> torch.Tensor:
    """Binary dilation implemented with max pooling."""
    kernel = odd_kernel(kernel)
    if iterations < 1:
        return mask
    mask = as_4d_mask(mask, "mask").float()
    pad = kernel // 2
    out = mask
    for _ in range(iterations):
        out = F.max_pool2d(out, kernel_size=kernel, stride=1, padding=pad)
    return out.clamp(0.0, 1.0)


def erode(mask: torch.Tensor, kernel: int = 3, iterations: int = 1) -> torch.Tensor:
    """Binary erosion implemented as the complement of dilation."""
    kernel = odd_kernel(kernel)
    if iterations < 1:
        return mask
    mask = as_4d_mask(mask, "mask").float()
    out = mask
    for _ in range(iterations):
        out = 1.0 - dilate(1.0 - out, kernel=kernel, iterations=1)
    return out.clamp(0.0, 1.0)


def gradient_magnitude(prob: torch.Tensor) -> torch.Tensor:
    grad_x = F.pad((prob[:, :, :, 1:] - prob[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    grad_y = F.pad((prob[:, :, 1:, :] - prob[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    grad = grad_x + grad_y
    denom = grad.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (grad / denom).clamp(0.0, 1.0)
