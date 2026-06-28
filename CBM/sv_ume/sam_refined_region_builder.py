from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from CBM.boundary.morphology import dilate, erode
from CBM.memory.labels import REGION_NAMES, REGION_TO_ID


def _validate_target_size(target_size: Sequence[int]) -> Tuple[int, int]:
    if isinstance(target_size, (str, bytes)) or len(target_size) != 2:
        raise ValueError(f"target_size must contain two positive integers, got {target_size!r}")
    height, width = target_size
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        raise ValueError(f"target_size must contain two positive integers, got {target_size!r}")
    return height, width


def _validate_prob_map(
    value: torch.Tensor,
    name: str,
    *,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dim() != 4 or value.size(1) != 1:
        raise ValueError(f"{name} must have shape [B, 1, H, W], got {tuple(value.shape)}")
    if value.size(0) < 1 or value.size(2) < 1 or value.size(3) < 1:
        raise ValueError(f"{name} must have non-empty batch and spatial dimensions")
    if batch_size is not None and value.size(0) != batch_size:
        raise ValueError(
            f"{name} batch size must be {batch_size}, got {value.size(0)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    return value


def _validate_optional_map(
    value: Optional[torch.Tensor],
    name: str,
    *,
    batch_size: int,
) -> None:
    if value is None:
        return
    _validate_prob_map(value, name, batch_size=batch_size)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_aux_tensors(value: Any, name: str, batch_size: int) -> None:
    if isinstance(value, torch.Tensor):
        if value.dim() < 1:
            raise ValueError(f"{name} tensor must have a batch dimension")
        if value.size(0) != batch_size:
            raise ValueError(
                f"{name} tensor batch size must be {batch_size}, got {value.size(0)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} tensor must contain only finite values")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_aux_tensors(item, f"{name}.{key}", batch_size)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_aux_tensors(item, f"{name}[{index}]", batch_size)


def _clean_probability(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _resize_probability(value: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(value, size=target_size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)


def _resize_finite_mask(value: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    finite = torch.isfinite(value).to(dtype=torch.float32)
    return F.interpolate(finite, size=target_size, mode="nearest") > 0.5


@torch.no_grad()
def build_sam_refined_regions(
    p_ref: torch.Tensor,
    conf_ref: Optional[torch.Tensor] = None,
    *,
    c_ref: Optional[torch.Tensor] = None,
    target_size: Sequence[int] = (40, 40),
    kernel: int = 3,
    R_band: Optional[torch.Tensor] = None,
    p_raw: Optional[torch.Tensor] = None,
    p_sam: Optional[torch.Tensor] = None,
    retrieval_aux: Any = None,
) -> Dict[str, Any]:
    """Build mutually exclusive p3 pseudo regions from SAM-refined predictions.

    Optional evidence inputs are validated only. They do not affect the region
    partition or the structural validity mask in this stage.
    """

    if (conf_ref is None) == (c_ref is None):
        raise ValueError("exactly one of conf_ref and c_ref must be provided")

    target_size = _validate_target_size(target_size)
    p_ref = _validate_prob_map(p_ref, "p_ref")
    confidence = conf_ref if conf_ref is not None else c_ref
    confidence = _validate_prob_map(confidence, "conf_ref", batch_size=p_ref.size(0))
    if confidence.device != p_ref.device:
        raise ValueError("p_ref and conf_ref must be on the same device")

    batch_size = p_ref.size(0)
    _validate_optional_map(R_band, "R_band", batch_size=batch_size)
    _validate_optional_map(p_raw, "p_raw", batch_size=batch_size)
    _validate_optional_map(p_sam, "p_sam", batch_size=batch_size)
    if retrieval_aux is not None:
        if not isinstance(retrieval_aux, (torch.Tensor, Mapping)):
            raise TypeError("retrieval_aux must be a Tensor or Mapping when provided")
        _validate_aux_tensors(retrieval_aux, "retrieval_aux", batch_size)

    finite_ref3 = _resize_finite_mask(p_ref, target_size)
    finite_conf3 = _resize_finite_mask(confidence, target_size)
    p_ref3 = _resize_probability(_clean_probability(p_ref), target_size)
    conf_ref3 = _resize_probability(_clean_probability(confidence), target_size)

    p_bin3 = (p_ref3 > 0.5).to(dtype=p_ref3.dtype)
    fg = p_bin3
    bg = 1.0 - fg
    fg_dilate = dilate(fg, kernel=kernel).to(dtype=p_ref3.dtype)
    fg_erode = erode(fg, kernel=kernel).to(dtype=p_ref3.dtype)
    boundary = (fg_dilate - fg_erode).clamp(0.0, 1.0)

    regions = {
        "fg_core": fg * (1.0 - boundary),
        "fg_boundary": fg * boundary,
        "bg_near": bg * fg_dilate,
        "bg_far": bg * (1.0 - fg_dilate),
    }
    if tuple(regions) != tuple(REGION_NAMES):
        raise RuntimeError("SV-UME region order does not match CBM REGION_NAMES")

    region_stack = torch.stack([regions[name] for name in REGION_NAMES], dim=0)
    region_union = region_stack.sum(dim=0)
    active_count = (region_stack > 0.5).sum(dim=0)
    if not bool(torch.all(active_count == 1)) or not torch.allclose(
        region_union, torch.ones_like(region_union), atol=1e-6, rtol=0.0
    ):
        raise RuntimeError("SAM-refined regions must be mutually exclusive and exhaustive")

    region_id_map = torch.full_like(
        p_bin3, REGION_TO_ID["bg_far"], dtype=torch.long
    )
    for region in REGION_NAMES:
        region_id_map.masked_fill_(regions[region] > 0.5, REGION_TO_ID[region])

    sdf_values = {
        "fg_core": 1.0,
        "fg_boundary": 0.3,
        "bg_near": -0.3,
        "bg_far": -1.0,
    }
    sdf = sum(regions[name] * sdf_values[name] for name in REGION_NAMES)
    valid = (
        finite_ref3
        & finite_conf3
        & (conf_ref3 > 0.0)
        & (region_union > 0.5)
    ).to(dtype=p_ref3.dtype)

    return {
        "p_ref3": p_ref3.detach(),
        "conf_ref3": conf_ref3.detach(),
        "p_bin3": p_bin3.detach(),
        "regions": {name: regions[name].detach() for name in REGION_NAMES},
        "region_id_map": region_id_map.detach(),
        "sdf": sdf.detach(),
        "valid": valid.detach(),
    }


__all__ = ["build_sam_refined_regions"]
