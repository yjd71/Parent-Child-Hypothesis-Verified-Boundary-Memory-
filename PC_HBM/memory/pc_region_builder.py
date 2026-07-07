"""SDF-based four-region builder for labelled PC-HBM memory.

Input GT masks are ``[B,1,H,W]`` or ``[B,H,W]`` and output regions are resized
to the requested feature grid.  Geometry values are ordered as
``[sdf_norm, normal_x, normal_y, offset_x, offset_y, geo_reliability]``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..common.utils import EPS, gradient_strength


def build_pc_regions(gt: torch.Tensor, target_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    """Build fg_core/fg_boundary/bg_near/bg_far masks and geometry maps."""

    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    gt = gt.float()
    gt_small = F.interpolate(gt, size=target_size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    fg = gt_small >= 0.5
    boundary_soft = _morph_boundary_binary(fg.float(), 3)
    bg_near_soft = F.max_pool2d(fg.float(), kernel_size=7, stride=1, padding=3) - fg.float()
    bg_near = bg_near_soft > 0
    fg_boundary = boundary_soft > 0
    fg_core = fg & ~fg_boundary
    bg_far = (~fg) & ~bg_near
    sdf = _signed_distance_fallback(fg.float())
    max_abs = sdf.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    sdf_norm = (sdf / max_abs).clamp(-1.0, 1.0)
    dy = F.pad(sdf_norm[..., 1:, :] - sdf_norm[..., :-1, :], (0, 0, 0, 1))
    dx = F.pad(sdf_norm[..., :, 1:] - sdf_norm[..., :, :-1], (0, 1, 0, 0))
    norm = torch.sqrt(dx.square() + dy.square() + EPS)
    nx = dx / norm
    ny = dy / norm
    height, width = target_size
    yy = torch.linspace(-1.0, 1.0, height, device=gt.device, dtype=gt.dtype).view(1, 1, height, 1)
    xx = torch.linspace(-1.0, 1.0, width, device=gt.device, dtype=gt.dtype).view(1, 1, 1, width)
    off_x = (-sdf_norm * nx).clamp(-1.0, 1.0)
    off_y = (-sdf_norm * ny).clamp(-1.0, 1.0)
    reliability = (1.0 - boundary_soft * 0.25).clamp(0.0, 1.0)
    geometry = torch.cat([sdf_norm, nx, ny, off_x + 0.0 * xx, off_y + 0.0 * yy, reliability], dim=1)
    return {
        "fg_core": fg_core.float(),
        "fg_boundary": fg_boundary.float(),
        "bg_near": bg_near.float(),
        "bg_far": bg_far.float(),
        "sdf": sdf_norm,
        "geometry": geometry,
        "boundary": fg_boundary.float(),
    }


def _morph_boundary_binary(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    dil = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0)


def _signed_distance_fallback(mask: torch.Tensor, iters: int = 16) -> torch.Tensor:
    """Torch morphology fallback for signed distance when scipy is unavailable."""

    fg = mask.clamp(0.0, 1.0)
    bg = 1.0 - fg
    dist_fg = _iterative_distance(fg, iters)
    dist_bg = _iterative_distance(bg, iters)
    return dist_fg - dist_bg


def _iterative_distance(mask: torch.Tensor, iters: int) -> torch.Tensor:
    edge = gradient_strength(mask).gt(0).float()
    known = edge.clone()
    dist = torch.zeros_like(mask)
    frontier = edge
    for step in range(1, int(iters) + 1):
        frontier = F.max_pool2d(frontier, kernel_size=3, stride=1, padding=1)
        newly = (frontier > 0).float() * (known <= 0).float() * mask
        dist = dist + newly * float(step)
        known = torch.maximum(known, newly)
    dist = dist + (known <= 0).float() * mask * float(iters + 1)
    return dist
