from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from CBM.memory.labels import REGION_NAMES


DEFAULT_IMAGE_WEIGHTS = {
    "global_teacher_sam_agreement": 0.20,
    "cbm_supported_change_score": 0.30,
    "sam_prompt_stability": 0.20,
    "area_reasonable_score": 0.10,
    "diversity_gain": 0.20,
    "over_seg_penalty": 0.20,
}

DEFAULT_REGION_WEIGHTS = {
    "teacher_sam_region_agreement": 0.20,
    "cbm_region_agreement": 0.35,
    "sam_region_stability": 0.15,
    "region_density": 0.15,
    "region_diversity": 0.15,
}

DEFAULT_REGION_THRESHOLDS = {
    "fg_core": 0.85,
    "fg_boundary": 0.92,
    "bg_near": 0.94,
    "bg_far": 0.85,
}

DEFAULT_TOKEN_THRESHOLDS = dict(DEFAULT_REGION_THRESHOLDS)
DEFAULT_CBM_LOGIT_SCALE = 4.0


def _finite_by_batch(value: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(value).reshape(value.size(0), -1).all(dim=1)


def _validate_b1hw(
    value: Any,
    name: str,
    *,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dim() != 4 or value.size(1) != 1:
        raise ValueError(f"{name} must have shape [B, 1, H, W], got {tuple(value.shape)}")
    if value.size(0) < 1 or value.size(2) < 1 or value.size(3) < 1:
        raise ValueError(f"{name} must have non-empty batch and spatial dimensions")
    if batch_size is not None and value.size(0) != batch_size:
        raise ValueError(f"{name} batch size must be {batch_size}, got {value.size(0)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    return value


def _resize_like(value: torch.Tensor, reference: torch.Tensor, mode: str) -> torch.Tensor:
    if tuple(value.shape[-2:]) == tuple(reference.shape[-2:]):
        return value
    if mode == "nearest":
        return F.interpolate(value, size=reference.shape[-2:], mode=mode)
    return F.interpolate(value, size=reference.shape[-2:], mode=mode, align_corners=False)


def _prepare_probability(
    value: Any,
    name: str,
    reference: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = None if reference is None else reference.size(0)
    tensor = _validate_b1hw(value, name, batch_size=batch_size)
    finite = _finite_by_batch(tensor)
    if reference is None:
        tensor = tensor.detach()
    else:
        tensor = tensor.detach().to(device=reference.device, dtype=reference.dtype)
        tensor = _resize_like(tensor, reference, mode="bilinear")
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return tensor, finite.to(device=tensor.device)


def _prepare_aux_map(
    value: Any,
    name: str,
    reference: torch.Tensor,
    *,
    channels: Optional[int] = 1,
    minimum_channels: Optional[int] = None,
    mode: str = "bilinear",
    clamp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    tensor = value
    if tensor.dim() == 3 and channels == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.dim() != 4:
        raise ValueError(f"{name} must be a 4D tensor, got {tuple(tensor.shape)}")
    if tensor.size(0) != reference.size(0):
        raise ValueError(
            f"{name} batch size must be {reference.size(0)}, got {tensor.size(0)}"
        )
    if channels is not None and tensor.size(1) != channels:
        raise ValueError(f"{name} must have {channels} channel(s), got {tensor.size(1)}")
    if minimum_channels is not None and tensor.size(1) < minimum_channels:
        raise ValueError(
            f"{name} must have at least {minimum_channels} channels, got {tensor.size(1)}"
        )
    finite = _finite_by_batch(tensor).to(device=reference.device)
    tensor = tensor.detach().to(device=reference.device, dtype=reference.dtype)
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    tensor = _resize_like(tensor, reference, mode=mode)
    if clamp:
        tensor = tensor.clamp(0.0, 1.0)
    return tensor, finite


def _merge_numeric_mapping(
    defaults: Mapping[str, float],
    override: Optional[Mapping[str, float]],
    name: str,
    *,
    maximum: Optional[float] = None,
) -> Dict[str, float]:
    result = {key: float(value) for key, value in defaults.items()}
    if override is not None:
        if not isinstance(override, Mapping):
            raise TypeError(f"{name} must be a mapping")
        unknown = set(override) - set(defaults)
        if unknown:
            raise KeyError(f"unsupported {name} keys: {sorted(unknown)}")
        for key, value in override.items():
            result[key] = float(value)
    for key, value in result.items():
        if not torch.isfinite(torch.tensor(value)) or value < 0.0:
            raise ValueError(f"{name}[{key!r}] must be finite and non-negative")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name}[{key!r}] must be <= {maximum}")
    return result


def _soft_iou(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (prediction * target).sum(dim=(-2, -1))
    union = (prediction + target - prediction * target).sum(dim=(-2, -1))
    return (intersection / (union + eps)).mean(dim=1).clamp(0.0, 1.0)


@torch.no_grad()
def parse_cbm_evidence(retrieval_aux: Any, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Parse CBM retrieval evidence and align it to a [B,1,H,W] reference."""

    reference, reference_finite = _prepare_probability(reference, "reference")
    aux = retrieval_aux if isinstance(retrieval_aux, Mapping) else {}
    batch_size = reference.size(0)

    present: Dict[str, bool] = {}
    finite_flags: List[torch.Tensor] = [reference_finite]

    def single_map(key: str, *, alias: Optional[str] = None, mode: str = "bilinear"):
        value = aux.get(key)
        if value is None and alias is not None:
            value = aux.get(alias)
        present[key] = value is not None
        if value is None:
            return reference.new_zeros(reference.shape)
        tensor, finite = _prepare_aux_map(value, key, reference, mode=mode)
        finite_flags.append(finite)
        return tensor

    y_value = aux.get("Y_ctx")
    present["Y_ctx"] = y_value is not None
    if y_value is None:
        y_ctx = reference.new_zeros((batch_size, 4, *reference.shape[-2:]))
    else:
        y_ctx, y_finite = _prepare_aux_map(
            y_value,
            "Y_ctx",
            reference,
            channels=None,
            minimum_channels=4,
            clamp=False,
        )
        finite_flags.append(y_finite)

    u_map = single_map("U_map")
    cons_map = single_map("cons_map")
    b3 = single_map("B3", alias="B_query")
    gate3 = single_map("gate3")
    valid_map = single_map("valid_map", mode="nearest")
    prob3 = single_map("prob3")

    all_present = all(present.get(key, False) for key in (
        "Y_ctx", "U_map", "cons_map", "B3", "gate3", "valid_map", "prob3"
    ))
    finite = torch.stack(finite_flags, dim=0).all(dim=0)
    has_valid_token = (valid_map > 0.5).reshape(batch_size, -1).any(dim=1)
    evidence_valid = finite & has_valid_token
    if not all_present:
        evidence_valid = torch.zeros_like(evidence_valid)

    s_fg = (y_ctx[:, 0:1] + y_ctx[:, 1:2]).clamp(0.0, 1.0)
    s_bg = (y_ctx[:, 2:3] + y_ctx[:, 3:4]).clamp(0.0, 1.0)
    s_bd = (y_ctx[:, 1:2] - y_ctx[:, 2:3]).clamp(-1.0, 1.0)

    return {
        "Y_ctx": y_ctx.detach(),
        "U_map": u_map.detach(),
        "cons_map": cons_map.detach(),
        "B3": b3.detach(),
        "B_query": b3.detach(),
        "gate3": gate3.detach(),
        "valid_map": (valid_map > 0.5).to(dtype=reference.dtype).detach(),
        "prob3": prob3.detach(),
        "S_fg": s_fg.detach(),
        "S_bg": s_bg.detach(),
        "S_bd": s_bd.detach(),
        "evidence_valid": evidence_valid.detach(),
    }


def _fit_last_dim(value: torch.Tensor, width: int) -> torch.Tensor:
    if value.size(-1) == width:
        return value
    if value.size(-1) > width:
        return value[..., :width]
    return F.pad(value, (0, width - value.size(-1)))


def _global_query(value: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(value):
        raise TypeError("x3_global_u must be a torch.Tensor")
    if value.dim() == 4:
        query = F.adaptive_avg_pool2d(value.detach(), 1).flatten(1)
    elif value.dim() == 2:
        query = value.detach()
    else:
        raise ValueError(
            f"x3_global_u must have shape [B,C,H,W] or [B,D], got {tuple(value.shape)}"
        )
    if query.size(0) < 1 or query.size(1) < 1:
        raise ValueError("x3_global_u must have non-empty batch and feature dimensions")
    if not query.is_floating_point():
        raise TypeError("x3_global_u must be a floating-point tensor")
    finite = _finite_by_batch(query)
    query = torch.nan_to_num(query, nan=0.0, posinf=1.0, neginf=-1.0)
    return query, finite.to(device=query.device)


def _global_memory(existing_global_memory: Any) -> Tuple[Optional[torch.Tensor], Optional[List[str]]]:
    if existing_global_memory is None:
        return None, None
    if torch.is_tensor(existing_global_memory):
        return existing_global_memory, None
    if isinstance(existing_global_memory, (tuple, list)) and len(existing_global_memory) == 2:
        keys, image_ids = existing_global_memory
        if not torch.is_tensor(keys):
            raise TypeError("existing_global_memory tuple must contain a Tensor as its first item")
        ids = None if image_ids is None else [str(item) for item in image_ids]
        return keys, ids
    getter = getattr(existing_global_memory, "get_image_keys", None)
    if callable(getter):
        result = getter()
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise TypeError("memory.get_image_keys() must return (keys, image_ids)")
        keys, image_ids = result
        if not torch.is_tensor(keys):
            raise TypeError("memory.get_image_keys() must return Tensor keys")
        return keys, [str(item) for item in image_ids]
    raise TypeError(
        "existing_global_memory must be a Tensor, (keys, ids), or expose get_image_keys()"
    )


@torch.no_grad()
def compute_global_type_metadata(
    x3_global_u: torch.Tensor,
    existing_global_memory: Any,
    *,
    tau_match: float = 0.70,
    tau_low: float = 0.55,
) -> Dict[str, Any]:
    """Classify each unlabeled image against labeled global memory."""

    tau_match = float(tau_match)
    tau_low = float(tau_low)
    if not 0.0 <= tau_low <= tau_match <= 1.0:
        raise ValueError("tau_low and tau_match must satisfy 0 <= tau_low <= tau_match <= 1")

    query, query_valid = _global_query(x3_global_u)
    keys, image_ids = _global_memory(existing_global_memory)
    batch_size = query.size(0)

    memory_available = keys is not None and keys.numel() > 0
    if keys is not None:
        if keys.dim() != 2:
            raise ValueError(f"global memory keys must be 2D, got {tuple(keys.shape)}")
        if image_ids is not None and len(image_ids) != keys.size(0):
            raise ValueError("global memory keys and image_ids lengths do not match")
        if keys.numel() > 0 and not bool(torch.isfinite(keys).all()):
            raise ValueError("global memory keys must contain only finite values")

    if not memory_available:
        sim_max = query.new_full((batch_size,), -1.0)
        nearest_index = torch.zeros(batch_size, device=query.device, dtype=torch.long)
        diversity_gain = query.new_zeros((batch_size,))
    else:
        keys = keys.detach().to(device=query.device, dtype=query.dtype)
        query_fit = _fit_last_dim(query, keys.size(1))
        similarities = F.normalize(query_fit, dim=1) @ F.normalize(keys, dim=1).transpose(0, 1)
        sim_max, nearest_index = similarities.max(dim=1)
        diversity_gain = (1.0 - sim_max).clamp(0.0, 1.0)

    sim_max = torch.where(query_valid, sim_max, sim_max.new_full(sim_max.shape, -1.0))
    diversity_gain = torch.where(query_valid, diversity_gain, torch.zeros_like(diversity_gain))

    global_types: List[str] = []
    nearest_ids: List[Optional[str]] = []
    metadata: List[dict] = []
    for index in range(batch_size):
        similarity = float(sim_max[index].item())
        if similarity >= tau_match:
            global_type = "matched"
        elif similarity >= tau_low:
            global_type = "expanded"
        else:
            global_type = "novel_pending"
        nearest_id = None
        if memory_available and image_ids is not None and bool(query_valid[index]):
            nearest_id = image_ids[int(nearest_index[index].item())]
        global_types.append(global_type)
        nearest_ids.append(nearest_id)
        metadata.append(
            {
                "global_type": global_type,
                "novel_activated": False,
                "global_similarity": similarity,
                "nearest_labeled_id": nearest_id,
            }
        )

    return {
        "sim_max": sim_max.detach(),
        "nearest_labeled_id": nearest_ids,
        "global_type": global_types,
        "novel_activated": torch.zeros(batch_size, device=query.device, dtype=torch.bool),
        "diversity_gain": diversity_gain.detach(),
        "memory_available": bool(memory_available),
        "query_valid": query_valid.detach(),
        "metadata": metadata,
    }


def _candidate_masks(sam_aux: Any, reference: torch.Tensor) -> Optional[torch.Tensor]:
    if not isinstance(sam_aux, Mapping):
        return None
    prompt_pack = sam_aux.get("prompt_pack")
    if not isinstance(prompt_pack, Mapping):
        return None
    value = prompt_pack.get("candidate_masks")
    if not torch.is_tensor(value):
        return None
    masks = value.detach()
    if masks.dim() == 3 and masks.size(0) == reference.size(0):
        masks = masks.unsqueeze(1)
    if masks.dim() != 4 or masks.size(0) != reference.size(0) or masks.size(1) < 1:
        return None
    if not bool(torch.isfinite(masks).all()):
        return None
    masks = masks.to(device=reference.device, dtype=reference.dtype)
    return _resize_like(masks, reference, mode="bilinear").clamp(0.0, 1.0)


def _sam_score_map(sam_aux: Any, reference: torch.Tensor) -> Optional[torch.Tensor]:
    if not isinstance(sam_aux, Mapping):
        return None
    value = sam_aux.get("sam_score")
    if not torch.is_tensor(value) or value.size(0) != reference.size(0):
        return None
    if not bool(torch.isfinite(value).all()):
        return None
    score = value.detach().to(device=reference.device, dtype=reference.dtype)
    score = score.reshape(score.size(0), -1).mean(dim=1).clamp(0.0, 1.0)
    return score.reshape(-1, 1, 1, 1).expand_as(reference)


def _sam_stability_map(sam_aux: Any, reference: torch.Tensor) -> Tuple[torch.Tensor, str]:
    if isinstance(sam_aux, Mapping):
        for key in ("R_stability",):
            value = sam_aux.get(key)
            if torch.is_tensor(value):
                try:
                    tensor, finite = _prepare_probability(value, key, reference)
                except (TypeError, ValueError):
                    continue
                if bool(finite.all()):
                    return tensor, key

    masks = _candidate_masks(sam_aux, reference)
    if masks is not None and masks.size(1) > 1:
        stability = 1.0 - masks.float().var(dim=1, keepdim=True, unbiased=False)
        return stability.to(dtype=reference.dtype).clamp(0.0, 1.0), "candidate_variance"

    if isinstance(sam_aux, Mapping):
        value = sam_aux.get("R_sam")
        if torch.is_tensor(value):
            try:
                tensor, finite = _prepare_probability(value, "R_sam", reference)
            except (TypeError, ValueError):
                tensor, finite = None, None
            if tensor is not None and bool(finite.all()):
                return tensor, "R_sam"

    score_map = _sam_score_map(sam_aux, reference)
    if score_map is not None:
        return score_map, "sam_score"
    return torch.zeros_like(reference), "missing"


def _image_prompt_stability(sam_aux: Any, reference: torch.Tensor) -> Tuple[torch.Tensor, str]:
    masks = _candidate_masks(sam_aux, reference)
    if masks is not None and masks.size(1) > 1:
        pair_scores = []
        for left in range(masks.size(1)):
            for right in range(left + 1, masks.size(1)):
                pair_scores.append(_soft_iou(masks[:, left : left + 1], masks[:, right : right + 1]))
        return torch.stack(pair_scores, dim=1).mean(dim=1), "candidate_pairwise_iou"
    stability, source = _sam_stability_map(sam_aux, reference)
    return stability.mean(dim=(1, 2, 3)).clamp(0.0, 1.0), source


@torch.no_grad()
def compute_image_consistency(
    p_raw: torch.Tensor,
    p_ref: torch.Tensor,
    p_sam: torch.Tensor,
    sam_aux: Any,
    retrieval_aux: Any,
    x3_global_u: torch.Tensor,
    existing_global_memory: Any,
    *,
    weights: Optional[Mapping[str, float]] = None,
    tau_image: float = 0.80,
    cbm_logit_scale: float = DEFAULT_CBM_LOGIT_SCALE,
) -> Dict[str, Any]:
    """Compute image-level SV-UME consistency and admission decision."""

    image_weights = _merge_numeric_mapping(DEFAULT_IMAGE_WEIGHTS, weights, "image weights")
    tau_image = float(tau_image)
    if not 0.0 <= tau_image <= 1.0:
        raise ValueError("tau_image must be in [0, 1]")
    cbm_logit_scale = float(cbm_logit_scale)
    if not bool(torch.isfinite(torch.tensor(cbm_logit_scale))) or cbm_logit_scale <= 0.0:
        raise ValueError("cbm_logit_scale must be finite and positive")

    p_ref, ref_finite = _prepare_probability(p_ref, "p_ref")
    p_raw, raw_finite = _prepare_probability(p_raw, "p_raw", p_ref)
    p_sam, sam_finite = _prepare_probability(p_sam, "p_sam", p_ref)
    evidence = parse_cbm_evidence(retrieval_aux, p_ref)
    global_metadata = compute_global_type_metadata(x3_global_u, existing_global_memory)
    if global_metadata["sim_max"].size(0) != p_ref.size(0):
        raise ValueError("x3_global_u batch size must match p_ref")

    teacher_sam_agreement = _soft_iou(p_raw, p_ref)
    fg_support = torch.sigmoid(
        cbm_logit_scale * (evidence["S_fg"] - evidence["S_bg"])
    )
    bg_support = torch.sigmoid(
        cbm_logit_scale * (evidence["S_bg"] - evidence["S_fg"])
    )
    cbm_agreement = (p_ref * fg_support + (1.0 - p_ref) * bg_support)
    cbm_agreement = cbm_agreement * evidence["valid_map"]
    changed = (p_ref - p_raw).abs() > 0.30
    changed_count = changed.to(dtype=p_ref.dtype).sum(dim=(1, 2, 3))
    supported_sum = (changed.to(dtype=p_ref.dtype) * cbm_agreement).sum(dim=(1, 2, 3))
    supported_change = torch.where(
        changed_count > 0,
        supported_sum / changed_count.clamp_min(1.0),
        torch.ones_like(changed_count),
    ).clamp(0.0, 1.0)

    prompt_stability, stability_source = _image_prompt_stability(sam_aux, p_ref)
    area_ratio = p_ref.mean(dim=(1, 2, 3))
    area_reasonable = ((area_ratio > 0.001) & (area_ratio < 0.70)).to(dtype=p_ref.dtype)
    diversity_gain = global_metadata["diversity_gain"].to(device=p_ref.device, dtype=p_ref.dtype)

    sam_area = (p_sam > 0.5).to(dtype=p_ref.dtype).sum(dim=(1, 2, 3))
    teacher_area = (p_raw > 0.5).to(dtype=p_ref.dtype).sum(dim=(1, 2, 3)).clamp_min(1e-6)
    over_seg_penalty = F.relu(sam_area / teacher_area - 1.5).clamp(0.0, 1.0)

    components = {
        "global_teacher_sam_agreement": teacher_sam_agreement,
        "cbm_supported_change_score": supported_change,
        "sam_prompt_stability": prompt_stability,
        "area_reasonable_score": area_reasonable,
        "diversity_gain": diversity_gain,
        "over_seg_penalty": over_seg_penalty,
    }
    score = (
        image_weights["global_teacher_sam_agreement"] * teacher_sam_agreement
        + image_weights["cbm_supported_change_score"] * supported_change
        + image_weights["sam_prompt_stability"] * prompt_stability
        + image_weights["area_reasonable_score"] * area_reasonable
        + image_weights["diversity_gain"] * diversity_gain
        - image_weights["over_seg_penalty"] * over_seg_penalty
    ).clamp(0.0, 1.0)

    input_valid = (
        ref_finite
        & raw_finite
        & sam_finite
        & global_metadata["query_valid"].to(device=p_ref.device)
    )
    evidence_valid = evidence["evidence_valid"] & input_valid
    allow_image = (score > tau_image) & evidence_valid
    return {
        "score": score.detach(),
        "components": {key: value.detach() for key, value in components.items()},
        "allow_image": allow_image.detach(),
        "evidence_valid": evidence_valid.detach(),
        "threshold": tau_image,
        "weights": dict(image_weights),
        "cbm_logit_scale": cbm_logit_scale,
        "stability_source": stability_source,
        "global_metadata": global_metadata,
    }


def _validate_region_pack(region_pack: Any) -> Dict[str, Any]:
    if not isinstance(region_pack, Mapping):
        raise TypeError("region_pack must be a mapping returned by build_sam_refined_regions")
    for key in ("p_ref3", "conf_ref3", "regions", "valid"):
        if key not in region_pack:
            raise KeyError(f"region_pack is missing {key!r}")

    p_ref3, p_ref_finite = _prepare_probability(region_pack["p_ref3"], "region_pack.p_ref3")
    conf_ref3, conf_finite = _prepare_probability(
        region_pack["conf_ref3"], "region_pack.conf_ref3", p_ref3
    )
    valid, valid_finite = _prepare_probability(region_pack["valid"], "region_pack.valid", p_ref3)
    regions_value = region_pack["regions"]
    if not isinstance(regions_value, Mapping):
        raise TypeError("region_pack.regions must be a mapping")

    regions: Dict[str, torch.Tensor] = {}
    region_finite = torch.ones(p_ref3.size(0), device=p_ref3.device, dtype=torch.bool)
    for region in REGION_NAMES:
        if region not in regions_value:
            raise KeyError(f"region_pack.regions is missing {region!r}")
        mask, finite = _prepare_probability(
            regions_value[region], f"region_pack.regions.{region}", p_ref3
        )
        regions[region] = mask
        region_finite &= finite

    region_sum = sum(regions[region] for region in REGION_NAMES)
    if not torch.allclose(region_sum, torch.ones_like(region_sum), atol=1e-5, rtol=0.0):
        raise ValueError("region_pack regions must be mutually exclusive and exhaustive")
    input_valid = p_ref_finite & conf_finite & valid_finite & region_finite
    return {
        "p_ref3": p_ref3,
        "conf_ref3": conf_ref3,
        "valid": valid,
        "regions": regions,
        "input_valid": input_valid,
    }


def _prepare_p3(p3: Any, reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(p3):
        raise TypeError("p3 must be a torch.Tensor")
    if p3.dim() != 4 or p3.size(0) != reference.size(0):
        raise ValueError("p3 must have shape [B, C, H3, W3] with batch matching region_pack")
    if tuple(p3.shape[-2:]) != tuple(reference.shape[-2:]) or p3.size(1) < 1:
        raise ValueError("p3 spatial size must match region_pack and channels must be non-empty")
    if not p3.is_floating_point():
        raise TypeError("p3 must be a floating-point tensor")
    finite = _finite_by_batch(p3).to(device=reference.device)
    p3 = p3.detach().to(device=reference.device, dtype=reference.dtype)
    p3 = torch.nan_to_num(p3, nan=0.0, posinf=1.0, neginf=-1.0)
    return p3, finite


def _region_memory_keys(memory: Any, region: str, reference: torch.Tensor) -> torch.Tensor:
    if memory is None:
        return reference.new_empty((0, 0))
    getter = getattr(memory, "get_region_memory", None)
    if callable(getter):
        result = getter(region)
        if not isinstance(result, (tuple, list)) or len(result) < 1:
            raise TypeError("memory.get_region_memory() must return a tuple beginning with keys")
        keys = result[0]
    else:
        keys_mapping = getattr(memory, "keys", None)
        if keys_mapping is None and isinstance(memory, Mapping):
            keys_mapping = memory
        if not isinstance(keys_mapping, Mapping) or region not in keys_mapping:
            raise TypeError("memory must expose region-indexed keys or get_region_memory()")
        keys = keys_mapping[region]
    if not torch.is_tensor(keys) or keys.dim() != 2:
        raise ValueError(f"memory keys for {region} must be a 2D tensor")
    if keys.numel() > 0 and not bool(torch.isfinite(keys).all()):
        raise ValueError(f"memory keys for {region} must contain only finite values")
    return keys.detach().to(device=reference.device, dtype=reference.dtype)


def _density_map(
    p3: torch.Tensor,
    region_mask: torch.Tensor,
    memory_keys: torch.Tensor,
    density_k: int,
) -> torch.Tensor:
    output = p3.new_zeros((p3.size(0), 1, *p3.shape[-2:]))
    if memory_keys.numel() == 0:
        return output
    query = p3.permute(0, 2, 3, 1).reshape(p3.size(0), -1, p3.size(1))
    query = F.normalize(_fit_last_dim(query, memory_keys.size(1)), dim=-1)
    memory_norm = F.normalize(memory_keys, dim=1)
    flat_mask = region_mask[:, 0].reshape(p3.size(0), -1) > 0.5
    output_flat = output[:, 0].reshape(p3.size(0), -1)
    k = min(max(1, int(density_k)), memory_norm.size(0))
    for batch_index in range(p3.size(0)):
        positions = flat_mask[batch_index].nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        similarities = query[batch_index].index_select(0, positions) @ memory_norm.transpose(0, 1)
        density = similarities.topk(k=k, dim=1).values.clamp(0.0, 1.0).mean(dim=1)
        output_flat[batch_index, positions] = density
    return output


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = (mask > 0.5).to(dtype=value.dtype)
    count = mask_float.sum(dim=(1, 2, 3))
    total = (value * mask_float).sum(dim=(1, 2, 3))
    return torch.where(count > 0, total / count.clamp_min(1.0), torch.zeros_like(total))


def _region_diversity(
    p3: torch.Tensor,
    region_mask: torch.Tensor,
    memory_keys: torch.Tensor,
) -> torch.Tensor:
    result = p3.new_zeros((p3.size(0),))
    if memory_keys.numel() == 0:
        return result
    query = p3.permute(0, 2, 3, 1).reshape(p3.size(0), -1, p3.size(1))
    query = _fit_last_dim(query, memory_keys.size(1))
    memory_norm = F.normalize(memory_keys, dim=1)
    flat_mask = region_mask[:, 0].reshape(p3.size(0), -1) > 0.5
    for batch_index in range(p3.size(0)):
        positions = flat_mask[batch_index].nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        prototype = query[batch_index].index_select(0, positions).mean(dim=0, keepdim=True)
        similarity = (F.normalize(prototype, dim=1) @ memory_norm.transpose(0, 1)).max()
        result[batch_index] = (1.0 - similarity).clamp(0.0, 1.0)
    return result


def _cbm_support_map(
    evidence: Mapping[str, torch.Tensor],
    region: str,
    cbm_logit_scale: float,
) -> torch.Tensor:
    if region == "fg_core":
        support = torch.sigmoid(
            cbm_logit_scale * (evidence["S_fg"] - evidence["S_bg"])
        )
    elif region == "bg_far":
        support = torch.sigmoid(
            cbm_logit_scale * (evidence["S_bg"] - evidence["S_fg"])
        )
    else:
        support = torch.sigmoid(cbm_logit_scale * evidence["S_bd"].abs())
    batch_valid = evidence["evidence_valid"].reshape(-1, 1, 1, 1).to(dtype=support.dtype)
    return support * evidence["valid_map"] * batch_valid


@torch.no_grad()
def compute_region_consistency(
    p_raw: torch.Tensor,
    p_sam: torch.Tensor,
    region_pack: Mapping[str, Any],
    sam_aux: Any,
    retrieval_aux: Any,
    p3: torch.Tensor,
    existing_memory: Any,
    *,
    weights: Optional[Mapping[str, float]] = None,
    thresholds: Optional[Mapping[str, float]] = None,
    density_k: int = 16,
    cbm_logit_scale: float = DEFAULT_CBM_LOGIT_SCALE,
) -> Dict[str, Any]:
    """Compute per-image consistency for each SV-UME pseudo region."""

    if int(density_k) <= 0:
        raise ValueError("density_k must be positive")
    cbm_logit_scale = float(cbm_logit_scale)
    if not bool(torch.isfinite(torch.tensor(cbm_logit_scale))) or cbm_logit_scale <= 0.0:
        raise ValueError("cbm_logit_scale must be finite and positive")
    region_weights = _merge_numeric_mapping(DEFAULT_REGION_WEIGHTS, weights, "region weights")
    region_thresholds = _merge_numeric_mapping(
        DEFAULT_REGION_THRESHOLDS, thresholds, "region thresholds", maximum=1.0
    )
    pack = _validate_region_pack(region_pack)
    reference = pack["p_ref3"]
    p_raw3, raw_finite = _prepare_probability(p_raw, "p_raw", reference)
    p_sam3, sam_finite = _prepare_probability(p_sam, "p_sam", reference)
    p3, p3_finite = _prepare_p3(p3, reference)
    evidence = parse_cbm_evidence(retrieval_aux, reference)
    stability_map, stability_source = _sam_stability_map(sam_aux, reference)

    agreement_map = (1.0 - (p_raw3 - p_sam3).abs()).clamp(0.0, 1.0)
    component_names = tuple(DEFAULT_REGION_WEIGHTS)
    components = {name: {} for name in component_names}
    scores: Dict[str, torch.Tensor] = {}
    allowed: Dict[str, torch.Tensor] = {}
    nonempty: Dict[str, torch.Tensor] = {}
    memory_counts: Dict[str, int] = {}
    batch_valid = (
        pack["input_valid"]
        & raw_finite
        & sam_finite
        & p3_finite
        & evidence["evidence_valid"]
    )

    for region in REGION_NAMES:
        mask = pack["regions"][region]
        keys = _region_memory_keys(existing_memory, region, reference)
        memory_counts[region] = int(keys.size(0))
        density_map = _density_map(p3, mask, keys, int(density_k))
        region_components = {
            "teacher_sam_region_agreement": _masked_mean(agreement_map, mask),
            "cbm_region_agreement": _masked_mean(
                _cbm_support_map(evidence, region, cbm_logit_scale), mask
            ),
            "sam_region_stability": _masked_mean(stability_map, mask),
            "region_density": _masked_mean(density_map, mask),
            "region_diversity": _region_diversity(p3, mask, keys),
        }
        has_region = (mask > 0.5).reshape(mask.size(0), -1).any(dim=1)
        score = sum(
            region_weights[name] * region_components[name]
            for name in component_names
        ).clamp(0.0, 1.0)
        score = torch.where(has_region, score, torch.zeros_like(score))
        allow = (score > region_thresholds[region]) & has_region & batch_valid
        scores[region] = score.detach()
        allowed[region] = allow.detach()
        nonempty[region] = has_region.detach()
        for name in component_names:
            components[name][region] = region_components[name].detach()

    return {
        "score": scores,
        "components": components,
        "allow": allowed,
        "nonempty": nonempty,
        "evidence_valid": batch_valid.detach(),
        "thresholds": dict(region_thresholds),
        "weights": dict(region_weights),
        "memory_counts": memory_counts,
        "cbm_logit_scale": cbm_logit_scale,
        "stability_source": stability_source,
    }


def _local_diversity_map(
    value: Any,
    pack: Mapping[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    reference = pack["p_ref3"]
    if value is None:
        return torch.ones_like(reference), torch.ones(
            reference.size(0), device=reference.device, dtype=torch.bool
        )
    if torch.is_tensor(value):
        return _prepare_probability(value, "r_diversity_local", reference)
    if not isinstance(value, Mapping):
        raise TypeError("r_diversity_local must be a Tensor, region mapping, or None")
    unknown = set(value) - set(REGION_NAMES)
    if unknown:
        raise KeyError(f"unsupported r_diversity_local regions: {sorted(unknown)}")
    output = torch.ones_like(reference)
    finite = torch.ones(reference.size(0), device=reference.device, dtype=torch.bool)
    for region in REGION_NAMES:
        if region not in value:
            continue
        region_value, region_finite = _prepare_probability(
            value[region], f"r_diversity_local.{region}", reference
        )
        mask = pack["regions"][region] > 0.5
        output = torch.where(mask, region_value, output)
        finite &= region_finite
    return output, finite


TOKEN_SCORE_MODES = ("product", "geometric_mean", "weighted_sum")
TOKEN_FACTOR_NAMES = (
    "r_teacher",
    "r_sam",
    "r_cbm",
    "r_context",
    "r_density",
    "r_temporal",
    "r_diversity_local",
)
TOKEN_WEIGHTED_SUM_WEIGHTS = (0.20, 0.20, 0.20, 0.15, 0.15, 0.05, 0.05)


def combine_token_reliability(
    components: Mapping[str, torch.Tensor],
    score_mode: str = "product",
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Combine the seven reliability factors with an explicit scoring mode."""
    mode = str(score_mode).strip().lower()
    if mode not in TOKEN_SCORE_MODES:
        raise ValueError(f"score_mode must be one of {TOKEN_SCORE_MODES}, got {score_mode!r}")
    missing = [name for name in TOKEN_FACTOR_NAMES if name not in components]
    if missing:
        raise KeyError(f"token reliability components are missing: {missing}")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    factors = [components[name].clamp(0.0, 1.0) for name in TOKEN_FACTOR_NAMES]
    reference_shape = tuple(factors[0].shape)
    if any(tuple(value.shape) != reference_shape for value in factors[1:]):
        raise ValueError("token reliability components must have identical shapes")
    if mode == "product":
        score = torch.ones_like(factors[0])
        for value in factors:
            score = score * value
    elif mode == "geometric_mean":
        stacked = torch.stack([value.clamp_min(eps) for value in factors], dim=0)
        score = torch.exp(torch.log(stacked).mean(dim=0))
    else:
        score = torch.zeros_like(factors[0])
        for weight, value in zip(TOKEN_WEIGHTED_SUM_WEIGHTS, factors):
            score = score + float(weight) * value
    return score.clamp(0.0, 1.0)


@torch.no_grad()
def compute_token_reliability(
    p_raw: torch.Tensor,
    region_pack: Mapping[str, Any],
    p3: torch.Tensor,
    retrieval_aux: Any,
    existing_memory: Any,
    *,
    p_ref_previous: Optional[torch.Tensor] = None,
    r_diversity_local: Any = None,
    thresholds: Optional[Mapping[str, float]] = None,
    density_k: int = 16,
    cbm_logit_scale: float = DEFAULT_CBM_LOGIT_SCALE,
    score_mode: str = "product",
    context_floor: float = 0.30,
    non_boundary_context: float = 0.80,
) -> Dict[str, Any]:
    """Compute configurable seven-factor reliability for every p3 token."""

    if int(density_k) <= 0:
        raise ValueError("density_k must be positive")
    cbm_logit_scale = float(cbm_logit_scale)
    if not bool(torch.isfinite(torch.tensor(cbm_logit_scale))) or cbm_logit_scale <= 0.0:
        raise ValueError("cbm_logit_scale must be finite and positive")
    context_floor = float(context_floor)
    non_boundary_context = float(non_boundary_context)
    if not 0.0 <= context_floor <= 1.0:
        raise ValueError("context_floor must be in [0, 1]")
    if not 0.0 <= non_boundary_context <= 1.0:
        raise ValueError("non_boundary_context must be in [0, 1]")
    token_thresholds = _merge_numeric_mapping(
        DEFAULT_TOKEN_THRESHOLDS, thresholds, "token thresholds", maximum=1.0
    )
    pack = _validate_region_pack(region_pack)
    reference = pack["p_ref3"]
    p_raw3, raw_finite = _prepare_probability(p_raw, "p_raw", reference)
    p3, p3_finite = _prepare_p3(p3, reference)
    evidence = parse_cbm_evidence(retrieval_aux, reference)

    r_teacher = (p_raw3 - 0.5).abs().mul(2.0).clamp(0.0, 1.0)
    r_sam = pack["conf_ref3"].clamp(0.0, 1.0)
    r_cbm = torch.zeros_like(reference)
    r_density = torch.zeros_like(reference)
    memory_counts: Dict[str, int] = {}
    for region in REGION_NAMES:
        mask = pack["regions"][region] > 0.5
        region_support = _cbm_support_map(evidence, region, cbm_logit_scale)
        r_cbm = torch.where(mask, region_support, r_cbm)
        keys = _region_memory_keys(existing_memory, region, reference)
        memory_counts[region] = int(keys.size(0))
        region_density = _density_map(p3, pack["regions"][region], keys, int(density_k))
        r_density = torch.where(mask, region_density, r_density)

    boundary_mask = (
        (pack["regions"]["fg_boundary"] > 0.5)
        | (pack["regions"]["bg_near"] > 0.5)
    )
    boundary_context = context_floor + (1.0 - context_floor) * evidence["cons_map"].clamp(0.0, 1.0)
    r_context = torch.where(
        boundary_mask,
        boundary_context,
        torch.full_like(reference, non_boundary_context),
    )

    if p_ref_previous is None:
        r_temporal = torch.ones_like(reference)
        temporal_finite = torch.ones(reference.size(0), device=reference.device, dtype=torch.bool)
        temporal_source = "no_history"
    else:
        previous, temporal_finite = _prepare_probability(
            p_ref_previous, "p_ref_previous", reference
        )
        r_temporal = (1.0 - (reference - previous).abs()).clamp(0.0, 1.0)
        temporal_source = "previous_p_ref"

    r_diversity, diversity_finite = _local_diversity_map(r_diversity_local, pack)
    components = {
        "r_teacher": r_teacher,
        "r_sam": r_sam,
        "r_cbm": r_cbm,
        "r_context": r_context,
        "r_density": r_density,
        "r_temporal": r_temporal,
        "r_diversity_local": r_diversity,
    }
    score = combine_token_reliability(components, score_mode=score_mode)

    batch_valid = (
        pack["input_valid"]
        & raw_finite
        & p3_finite
        & temporal_finite
        & diversity_finite
        & evidence["evidence_valid"]
    )
    structural_valid = pack["valid"] > 0.5
    cbm_valid = evidence["valid_map"] > 0.5
    batch_valid_map = batch_valid.reshape(-1, 1, 1, 1)
    allowed: Dict[str, torch.Tensor] = {}
    for region in REGION_NAMES:
        valid = (
            (pack["regions"][region] > 0.5)
            & structural_valid
            & batch_valid_map
        )
        if region in ("fg_boundary", "bg_near"):
            valid = valid & cbm_valid
        allowed[region] = (valid & (score > token_thresholds[region])).detach()

    return {
        "score": score.detach(),
        "components": {name: value.detach() for name, value in components.items()},
        "allow": allowed,
        "structural_valid": structural_valid.detach(),
        "cbm_valid": cbm_valid.detach(),
        "batch_valid_map": batch_valid_map.detach(),
        "evidence_valid": batch_valid.detach(),
        "thresholds": dict(token_thresholds),
        "score_mode": str(score_mode).strip().lower(),
        "memory_counts": memory_counts,
        "cbm_logit_scale": cbm_logit_scale,
        "temporal_source": temporal_source,
    }


__all__ = [
    "DEFAULT_IMAGE_WEIGHTS",
    "DEFAULT_REGION_WEIGHTS",
    "DEFAULT_REGION_THRESHOLDS",
    "DEFAULT_TOKEN_THRESHOLDS",
    "DEFAULT_CBM_LOGIT_SCALE",
    "TOKEN_SCORE_MODES",
    "TOKEN_FACTOR_NAMES",
    "TOKEN_WEIGHTED_SUM_WEIGHTS",
    "combine_token_reliability",
    "parse_cbm_evidence",
    "compute_global_type_metadata",
    "compute_image_consistency",
    "compute_region_consistency",
    "compute_token_reliability",
]
