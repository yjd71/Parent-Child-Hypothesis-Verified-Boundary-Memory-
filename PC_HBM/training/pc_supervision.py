"""GT-derived supervision targets for PC-HBM losses and diagnostics."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from ..common.utils import REGION_TO_ID
from ..memory.pc_region_builder import build_pc_regions

REGION_FG_CORE = 0
REGION_FG_BOUNDARY = 1
REGION_BG_NEAR = 2
REGION_BG_FAR = 3


def build_region_label_map(gt: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Return ``[B,H,W]`` long labels in ``{0,1,2,3}`` from GT masks."""

    regions = build_pc_regions(gt, size)
    label = torch.full(
        (gt.size(0), int(size[0]), int(size[1])),
        REGION_BG_FAR,
        device=gt.device,
        dtype=torch.long,
    )
    label[regions["bg_near"][:, 0] > 0.5] = REGION_BG_NEAR
    label[regions["fg_core"][:, 0] > 0.5] = REGION_FG_CORE
    label[regions["fg_boundary"][:, 0] > 0.5] = REGION_FG_BOUNDARY
    return label


def build_geometry_target(gt: torch.Tensor, size: tuple[int, int]) -> Dict[str, torch.Tensor]:
    """Return GT geometry maps at ``size``."""

    geometry = build_pc_regions(gt, size)["geometry"]
    return {
        "sdf": geometry[:, 0:1],
        "normal": geometry[:, 1:3],
        "offset": geometry[:, 3:5],
        "reliability": geometry[:, 5:6],
    }


def gather_by_boundary_indices(map_tensor: torch.Tensor, boundary_indices: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Gather ``[B,C,H,W]`` or ``[B,H,W]`` values at boundary token indices."""

    batch_ids = boundary_indices["batch_ids"].long()
    flat_indices = boundary_indices["flat_indices"].long()
    if map_tensor.dim() == 4:
        channels = int(map_tensor.size(1))
        if batch_ids.numel() == 0:
            return map_tensor.new_empty(0, channels)
        flat = map_tensor.flatten(2).transpose(1, 2)
        return flat[batch_ids, flat_indices]
    if map_tensor.dim() == 3:
        if batch_ids.numel() == 0:
            return map_tensor.new_empty(0)
        flat = map_tensor.flatten(1)
        return flat[batch_ids, flat_indices]
    raise ValueError(f"map_tensor must be [B,C,H,W] or [B,H,W], got {tuple(map_tensor.shape)}")


def build_need_correction_map(
    z_main: torch.Tensor,
    gt: torch.Tensor,
    size: tuple[int, int],
    threshold: float = 0.25,
) -> torch.Tensor:
    """Return ``[B,1,H,W]`` where the main prediction differs from GT."""

    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    p_main = torch.sigmoid(F.interpolate(z_main, size=size, mode="bilinear", align_corners=False))
    gt_small = F.interpolate(gt.float(), size=size, mode="nearest")
    return ((p_main - gt_small).abs() > float(threshold)).to(dtype=z_main.dtype)


def parent_meta_to_region_ids(top_parent_meta: Any, device, default: int = -1) -> torch.Tensor:
    """Convert nested parent meta dictionaries to a ``[M,K]`` long tensor."""

    rows = list(top_parent_meta or [])
    if not rows:
        return torch.empty(0, 0, device=device, dtype=torch.long)
    width = max((len(row) for row in rows), default=0)
    ids = torch.full((len(rows), width), int(default), device=device, dtype=torch.long)
    for row_idx, row in enumerate(rows):
        for col_idx, meta in enumerate(row):
            if not isinstance(meta, dict):
                continue
            raw = meta.get("region_id", REGION_TO_ID.get(str(meta.get("region", "")), default))
            try:
                ids[row_idx, col_idx] = int(raw)
            except (TypeError, ValueError):
                ids[row_idx, col_idx] = int(default)
    return ids
