from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from CBM.boundary.regions import build_gt_regions
from CBM.core.tensor_ops import js_divergence, normalize_distribution


LOSS_KEYS = (
    "loss_cbm_mem",
    "loss_cbm_bd",
    "loss_cbm_ctx",
    "loss_cbm_aff",
    "loss_cbm_gate_sparse",
    "loss_cbm_gate_boundary",
    "loss_cbm_gate",
    "loss_cbm_total",
    "raw_cbm_L_mem_ce",
    "raw_cbm_L_bd_margin",
    "raw_cbm_L_ctx",
    "raw_cbm_L_aff",
    "raw_cbm_L_gate_sparse",
    "raw_cbm_L_gate_boundary",
)


def compute_boundary_memory_losses(
    aux: Optional[Dict[str, Any]],
    gt: Optional[torch.Tensor],
    config: Any = None,
) -> Dict[str, torch.Tensor]:
    aux = aux or {}
    zero = _zero_like(aux, gt)
    losses = _empty_loss_dict(zero)
    if gt is None or not aux.get("cbm_used", False):
        return losses

    y_map = _as_map(aux.get("Y_map"))
    valid_map = _as_map(aux.get("valid_map"))
    if y_map is None or y_map.dim() != 4 or y_map.size(1) < 4 or valid_map is None:
        return losses

    eps = float(getattr(config, "cbm_loss_eps", 1e-6))
    device = y_map.device
    dtype = y_map.dtype
    target_size = y_map.shape[-2:]

    regions = build_gt_regions(gt.to(device=device), target_size=target_size)
    region_label = regions["region_label"].to(device=device, dtype=torch.long)
    fg_boundary = regions["fg_boundary"].to(device=device, dtype=dtype) > 0.5
    bg_near = regions["bg_near"].to(device=device, dtype=dtype) > 0.5

    valid = _prepare_single_channel(valid_map, y_map, mode="nearest") > 0.5
    pred_boundary = _prepare_pred_boundary(aux, y_map)
    sample_mask = (fg_boundary | bg_near | pred_boundary) & valid

    y_evidence = normalize_distribution(y_map[:, :4], dim=1, eps=eps)
    raw_mem = _memory_ce_loss(y_evidence, region_label, sample_mask, zero, eps)
    raw_bd = _boundary_margin_loss(y_evidence, fg_boundary & valid, bg_near & valid, zero, config)
    raw_ctx = _context_loss(y_evidence, aux.get("Y_ctx"), valid, zero, eps)
    raw_aff = _affinity_loss(aux, gt, sample_mask, valid, y_map, zero, config, eps)
    raw_gate_sparse, raw_gate_boundary = _gate_losses(aux, valid, y_map, zero, eps)

    loss_mem = _weight(config, "cbm_lambda_mem", 0.2) * raw_mem
    loss_bd = _weight(config, "cbm_lambda_bd", 0.2) * raw_bd
    loss_ctx = _weight(config, "cbm_lambda_ctx", 0.05) * raw_ctx
    loss_aff = _weight(config, "cbm_lambda_aff", 0.05) * raw_aff
    loss_gate_sparse = _weight(config, "cbm_lambda_gate_sparse", 0.01) * raw_gate_sparse
    loss_gate_boundary = _weight(config, "cbm_lambda_gate_boundary", 0.05) * raw_gate_boundary
    loss_gate = loss_gate_sparse + loss_gate_boundary
    loss_total = loss_mem + loss_bd + loss_ctx + loss_aff + loss_gate

    losses.update(
        {
            "loss_cbm_mem": _finite(loss_mem),
            "loss_cbm_bd": _finite(loss_bd),
            "loss_cbm_ctx": _finite(loss_ctx),
            "loss_cbm_aff": _finite(loss_aff),
            "loss_cbm_gate_sparse": _finite(loss_gate_sparse),
            "loss_cbm_gate_boundary": _finite(loss_gate_boundary),
            "loss_cbm_gate": _finite(loss_gate),
            "loss_cbm_total": _finite(loss_total),
            "raw_cbm_L_mem_ce": _finite(raw_mem),
            "raw_cbm_L_bd_margin": _finite(raw_bd),
            "raw_cbm_L_ctx": _finite(raw_ctx),
            "raw_cbm_L_aff": _finite(raw_aff),
            "raw_cbm_L_gate_sparse": _finite(raw_gate_sparse),
            "raw_cbm_L_gate_boundary": _finite(raw_gate_boundary),
        }
    )
    return losses


def _memory_ce_loss(
    y_evidence: torch.Tensor,
    region_label: torch.Tensor,
    sample_mask: torch.Tensor,
    zero: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if not sample_mask.any():
        return zero
    target_prob = y_evidence.gather(1, region_label.unsqueeze(1)).clamp_min(eps)
    nll = -target_prob.log()
    return _masked_mean(nll, sample_mask, zero)


def _boundary_margin_loss(
    y_evidence: torch.Tensor,
    fg_mask: torch.Tensor,
    bg_mask: torch.Tensor,
    zero: torch.Tensor,
    config: Any,
) -> torch.Tensor:
    margin = float(getattr(config, "cbm_boundary_margin", 0.2))
    margin_score = y_evidence[:, 1:2] - y_evidence[:, 2:3]
    pieces = []
    if fg_mask.any():
        pieces.append(F.relu(margin - margin_score)[fg_mask])
    if bg_mask.any():
        pieces.append(F.relu(margin + margin_score)[bg_mask])
    if not pieces:
        return zero
    return torch.cat(pieces).mean()


def _context_loss(
    y_evidence: torch.Tensor,
    y_ctx: Any,
    valid: torch.Tensor,
    zero: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    y_ctx = _as_map(y_ctx)
    if y_ctx is None or y_ctx.dim() != 4 or y_ctx.size(1) < 4:
        return zero
    y_ctx = _prepare_map(y_ctx[:, :4], y_evidence, channels=4, mode="bilinear")
    if y_ctx is None or not valid.any():
        return zero
    js = js_divergence(y_evidence, y_ctx, eps=eps).unsqueeze(1)
    return _masked_mean(js, valid, zero)


def _affinity_loss(
    aux: Dict[str, Any],
    gt: torch.Tensor,
    center_mask: torch.Tensor,
    valid: torch.Tensor,
    ref: torch.Tensor,
    zero: torch.Tensor,
    config: Any,
    eps: float,
) -> torch.Tensor:
    if not center_mask.any():
        return zero
    prob = _prediction_probability(aux, ref)
    if prob is None:
        return zero
    prob = prob.clamp(eps, 1.0 - eps)
    gt_bin = _binary_gt(gt, ref)

    kernel_size = int(getattr(config, "cbm_affinity_kernel_size", 3))
    if kernel_size < 1 or kernel_size % 2 == 0:
        kernel_size = 3
    padding = kernel_size // 2

    p_center = prob.flatten(2)
    p_neighbors = F.unfold(prob, kernel_size=kernel_size, padding=padding)
    a_pred = p_center * p_neighbors + (1.0 - p_center) * (1.0 - p_neighbors)

    gt_center = gt_bin.flatten(2)
    gt_neighbors = F.unfold(gt_bin, kernel_size=kernel_size, padding=padding)
    a_gt = (gt_center > 0.5).eq(gt_neighbors > 0.5).to(dtype=ref.dtype)

    center_flat = center_mask.flatten(2)
    valid_neighbors = F.unfold(valid.to(dtype=ref.dtype), kernel_size=kernel_size, padding=padding) > 0.5
    pair_mask = center_flat & valid_neighbors
    if not pair_mask.any():
        return zero

    bce = F.binary_cross_entropy(a_pred.clamp(eps, 1.0 - eps), a_gt, reduction="none")
    return _masked_mean(bce, pair_mask, zero)


def _gate_losses(
    aux: Dict[str, Any],
    valid: torch.Tensor,
    ref: torch.Tensor,
    zero: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate = _as_map(aux.get("gate3"))
    b_query = _as_map(aux.get("B_query"))
    if gate is None or b_query is None:
        return zero, zero
    gate = _prepare_single_channel(gate, ref, mode="bilinear").clamp(0.0, 1.0)
    b_query = _prepare_single_channel(b_query, ref, mode="bilinear").clamp(0.0, 1.0)
    if not valid.any():
        return zero, zero

    sparse = _masked_mean(gate, valid, zero)
    bce = F.binary_cross_entropy(gate.clamp(eps, 1.0 - eps), b_query.detach(), reduction="none")
    boundary = _masked_mean(bce, valid, zero)
    return sparse, boundary


def _prediction_probability(aux: Dict[str, Any], ref: torch.Tensor) -> Optional[torch.Tensor]:
    p_final = _as_map(aux.get("p_final"))
    z_mem3 = _as_map(aux.get("z_mem3"))
    if p_final is not None and (p_final.requires_grad or z_mem3 is None):
        return _prepare_single_channel(p_final, ref, mode="bilinear")
    if z_mem3 is not None:
        return torch.sigmoid(_prepare_single_channel(z_mem3, ref, mode="bilinear"))
    return None


def _binary_gt(gt: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    gt = gt.to(device=ref.device, dtype=ref.dtype)
    if tuple(gt.shape[-2:]) != tuple(ref.shape[-2:]):
        gt = F.interpolate(gt, size=ref.shape[-2:], mode="nearest")
    return (gt >= 0.5).to(dtype=ref.dtype)


def _prepare_pred_boundary(aux: Dict[str, Any], ref: torch.Tensor) -> torch.Tensor:
    boundary = _as_map(aux.get("boundary_mask"))
    if boundary is not None:
        return _prepare_single_channel(boundary, ref, mode="nearest") > 0.5
    b_query = _as_map(aux.get("B_query"))
    if b_query is not None:
        return _prepare_single_channel(b_query, ref, mode="bilinear") > 0.0
    return torch.zeros(ref.size(0), 1, *ref.shape[-2:], device=ref.device, dtype=torch.bool)


def _prepare_map(
    x: torch.Tensor,
    ref: torch.Tensor,
    channels: int,
    mode: str,
) -> Optional[torch.Tensor]:
    if x.dim() != 4 or x.size(0) != ref.size(0) or x.size(1) != channels:
        return None
    x = x.to(device=ref.device, dtype=ref.dtype)
    if tuple(x.shape[-2:]) == tuple(ref.shape[-2:]):
        return x
    if mode == "nearest":
        return F.interpolate(x, size=ref.shape[-2:], mode=mode)
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False)


def _prepare_single_channel(x: torch.Tensor, ref: torch.Tensor, mode: str) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.size(1) != 1:
        raise ValueError(f"Expected single-channel map, got {tuple(x.shape)}")
    x = x.to(device=ref.device, dtype=ref.dtype)
    if tuple(x.shape[-2:]) == tuple(ref.shape[-2:]):
        return x
    if mode == "nearest":
        return F.interpolate(x, size=ref.shape[-2:], mode=mode)
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(values)
    if not mask.any():
        return zero
    return values[mask].mean()


def _as_map(value: Any) -> Optional[torch.Tensor]:
    return value if isinstance(value, torch.Tensor) else None


def _weight(config: Any, name: str, default: float) -> float:
    return float(getattr(config, name, default))


def _zero_like(aux: Dict[str, Any], gt: Optional[torch.Tensor] = None) -> torch.Tensor:
    if isinstance(gt, torch.Tensor):
        return gt.new_zeros(())
    for value in aux.values():
        if isinstance(value, torch.Tensor):
            return value.new_zeros(())
    return torch.zeros(())


def _empty_loss_dict(zero: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {key: zero for key in LOSS_KEYS}


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
