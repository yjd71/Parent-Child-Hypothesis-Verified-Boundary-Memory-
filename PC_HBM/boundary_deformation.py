"""Boundary deformation utilities for final PC-HBM mixture."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import make_normalized_grid


def deform_logits(z_main: torch.Tensor, offset_pix: torch.Tensor, mask_corr: torch.Tensor) -> torch.Tensor:
    """Warp logits with pixel offsets and ``align_corners=False`` grid sampling."""

    bsz, _, height, width = z_main.shape
    grid = make_normalized_grid(height, width, z_main.device, z_main.dtype).expand(bsz, height, width, 2)
    norm_x = offset_pix[:, 0:1] * mask_corr * (2.0 / max(width, 1))
    norm_y = offset_pix[:, 1:2] * mask_corr * (2.0 / max(height, 1))
    delta = torch.cat([norm_x, norm_y], dim=1).permute(0, 2, 3, 1)
    return F.grid_sample(z_main, grid + delta, mode="bilinear", padding_mode="border", align_corners=False)
