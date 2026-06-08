from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .morphology import as_4d_mask, dilate, erode


def _resize_prob_map(x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    x = as_4d_mask(x, "gt").float()
    if tuple(x.shape[-2:]) != tuple(target_size):
        x = F.interpolate(x, size=target_size, mode="nearest")
    return x


def build_gt_regions(gt: torch.Tensor, target_size: Tuple[int, int], kernel: int = 3) -> Dict[str, torch.Tensor]:
    """Build four-region GT labels used by PLAN_V4.2 dense boundary memory."""
    gt = _resize_prob_map(gt, target_size)
    fg = (gt >= 0.5).to(dtype=gt.dtype)
    bg = 1.0 - fg

    fg_dilate = dilate(fg, kernel=kernel)
    fg_erode = erode(fg, kernel=kernel)
    boundary = (fg_dilate - fg_erode).clamp(0.0, 1.0)

    fg_boundary = (fg * boundary).clamp(0.0, 1.0)
    fg_core = (fg * (1.0 - boundary)).clamp(0.0, 1.0)
    bg_near = (bg * fg_dilate).clamp(0.0, 1.0)
    bg_far = (bg * (1.0 - fg_dilate)).clamp(0.0, 1.0)

    region_label = torch.full_like(fg[:, 0], fill_value=3, dtype=torch.long)
    region_label[bg_near[:, 0].bool()] = 2
    region_label[fg_boundary[:, 0].bool()] = 1
    region_label[fg_core[:, 0].bool()] = 0

    sdf_approx = fg_core * 1.0 + fg_boundary * 0.3 - bg_near * 0.3 - bg_far * 1.0
    sdf_approx = sdf_approx.clamp(-1.0, 1.0)

    return {
        "fg_core": fg_core,
        "fg_boundary": fg_boundary,
        "bg_near": bg_near,
        "bg_far": bg_far,
        "region_label": region_label,
        "sdf_approx": sdf_approx,
    }
