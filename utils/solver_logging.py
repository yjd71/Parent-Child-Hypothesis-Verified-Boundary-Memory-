from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn.functional as F
import wandb
from torch.distributed import get_rank


def log_info(logger, message: str) -> None:
    if logger is None:
        print(message)
        return
    log_fn = getattr(logger, "info", None) or getattr(logger, "key_info", None)
    if log_fn is not None:
        log_fn(message)


def format_loss_info(loss_dict: Mapping[str, Any], title: str, include_cbm_losses: bool = True) -> str:
    info_loss = title
    for loss_name, loss_value in loss_dict.items():
        if not include_cbm_losses and (loss_name.startswith("loss_cbm_") or loss_name.startswith("raw_cbm_")):
            continue
        info_loss += ", {}: {:.3f}".format(loss_name, float(loss_value))
    return info_loss


def add_weighted_unsup_stats(loss_dict: Dict[str, Any], conf_ref, loss_weight, refine_band, boost_map) -> None:
    loss_dict["conf_ref_mean"] = _tensor_or_float_mean(conf_ref, default=0.0)
    loss_dict["conf_ref_min"] = _tensor_min(conf_ref, default=0.0)
    loss_dict["conf_ref_max"] = _tensor_max(conf_ref, default=0.0)
    loss_dict["boundary_boost_mean"] = _tensor_or_float_mean(boost_map - 1.0, default=0.0)
    loss_dict["weighted_unsup_weight_mean"] = _tensor_or_float_mean(loss_weight, default=0.0)
    loss_dict["weighted_unsup_weight_min"] = _tensor_min(loss_weight, default=0.0)
    loss_dict["weighted_unsup_weight_max"] = _tensor_max(loss_weight, default=0.0)
    loss_dict["weighted_unsup_refine_band_mean"] = _tensor_or_float_mean(refine_band, default=0.0)
    loss_dict["loss_weighted_unsup"] = 1.0


def log_training_progress(
    logger,
    loss_dict: Mapping[str, Any],
    title: str,
    wandb_prefix: str,
    epoch: int,
    total_epochs: int,
    batch_idx: int,
    num_batches: int,
    step: int,
    distributed_train: bool = False,
    include_cbm_losses: bool = True,
    progress_label: str = "",
) -> None:
    label = "{} ".format(progress_label) if progress_label else ""
    info_progress = "{}Epoch[{}/{}] Iter[{}/{}].".format(
        label,
        epoch,
        total_epochs,
        batch_idx,
        num_batches,
    )
    info_loss = format_loss_info(loss_dict, title, include_cbm_losses=include_cbm_losses)
    log_info(logger, " ".join((info_progress, info_loss)))
    if _is_rank_zero(distributed_train):
        wandb.log({"{}-{}".format(wandb_prefix, k): v for k, v in loss_dict.items()}, step=step)


def record_cbm_aux(loss_dict: Dict[str, Any], cbm, cbm_stage: int, aux, branch_name: str, logger=None) -> None:
    if cbm is not None:
        loss_dict["cbm_stage"] = float(cbm_stage)
        loss_dict["memory_ready"] = 1.0 if cbm.memory.is_ready() else 0.0
    if not aux:
        return
    loss_dict["gate_mean"] = float(aux.get("gate_mean", 0.0) or 0.0)
    loss_dict["valid_ratio"] = float(aux.get("valid_ratio", 0.0) or 0.0)
    loss_dict["retrieval_uncertainty_mean"] = float(aux.get("u_mean", 0.0) or 0.0)
    loss_dict["memory_tokens"] = float(aux.get("num_memory_tokens", 0) or 0)
    if aux.get("fallback_reason"):
        log_info(logger, "[CBM] {} fallback={}".format(branch_name, aux.get("fallback_reason")))


def record_svb_aux(loss_dict: Dict[str, Any], sam_aux, p_t, p_ref, conf_ref, logger=None) -> None:
    sam_aux = sam_aux or {}
    batch_size = int(p_ref.size(0)) if torch.is_tensor(p_ref) and p_ref.dim() > 0 else 1
    loss_dict["svb_used_sam"] = 1.0 if sam_aux.get("used_sam", False) else 0.0
    loss_dict["svb_cache_hit"] = 1.0 if sam_aux.get("cache_hit", False) else 0.0
    loss_dict["svb_conf_ref_mean"] = _tensor_or_float_mean(conf_ref, default=0.0)
    loss_dict["svb_changed_ratio"] = _changed_ratio(p_ref, p_t)
    loss_dict["svb_refine_band_mean"] = 0.0
    loss_dict["svb_beta_mean"] = 0.0
    loss_dict["svb_R_sam_mean"] = 0.0
    loss_dict["svb_boundary_changed_ratio"] = 0.0
    loss_dict["svb_sam_score_mean"] = _tensor_or_float_mean(sam_aux.get("sam_score"), default=0.0)
    loss_dict["svb_used_conformal"] = 1.0 if bool(sam_aux.get("used_conformal", False)) else 0.0
    loss_dict["svb_lambda_epoch"] = float(sam_aux.get("lambda_epoch", 0.0) or 0.0)

    backend_fallback, backend_fallback_ratio = _backend_fallback_stats(
        sam_aux.get("backend_aux"),
        batch_size=batch_size,
    )
    if sam_aux.get("fallback_reason"):
        backend_fallback = 1.0
        backend_fallback_ratio = max(backend_fallback_ratio, 1.0)
    loss_dict["svb_backend_fallback"] = backend_fallback
    loss_dict["svb_backend_fallback_ratio"] = backend_fallback_ratio

    prompt_stats = _prompt_stats(sam_aux.get("prompt_pack"))
    loss_dict["svb_prompt_empty_box_ratio"] = prompt_stats.get("empty_box_ratio", 0.0)
    loss_dict["svb_prompt_empty_point_ratio"] = prompt_stats.get("empty_point_ratio", 0.0)
    loss_dict["svb_prompt_all_empty"] = prompt_stats.get("all_prompt_empty", 0.0)
    loss_dict["svb_prompt_box_count"] = prompt_stats.get("box_count", 0.0)
    loss_dict["svb_prompt_point_count"] = prompt_stats.get("point_count", 0.0)
    loss_dict["svb_prompt_boundary_point_count"] = prompt_stats.get("boundary_point_count", 0.0)
    loss_dict["svb_prompt_has_mask"] = prompt_stats.get("has_mask", 0.0)

    for expert_name, ratio in _best_expert_ratios(sam_aux.get("selector_aux"), batch_size).items():
        loss_dict["svb_best_expert_{}".format(expert_name)] = ratio

    _record_svb_maps(loss_dict, sam_aux, p_t, p_ref)
    if sam_aux.get("fallback_reason"):
        log_info(logger, "[SVB-PLR] fallback={}".format(sam_aux.get("fallback_reason")))


def log_svb_calibrator_state(logger, svb_plr, prefix: str) -> None:
    calibrator = getattr(svb_plr, "calibrator", None) if svb_plr is not None else None
    if calibrator is None:
        log_info(logger, "{}: unavailable.".format(prefix))
        return
    q_alpha = getattr(calibrator, "q_alpha", None)
    num_pixels = getattr(calibrator, "num_calibration_pixels", None)
    q_value = float(q_alpha.detach().cpu().item()) if torch.is_tensor(q_alpha) else float("nan")
    pixel_count = int(num_pixels.detach().cpu().item()) if torch.is_tensor(num_pixels) else 0
    fitted = bool(calibrator.is_fitted()) if hasattr(calibrator, "is_fitted") else False
    log_info(
        logger,
        "{}: fitted={}, q_alpha={:.6f}, num_calibration_pixels={}.".format(
            prefix,
            fitted,
            q_value,
            pixel_count,
        ),
    )


def _record_svb_maps(loss_dict: Dict[str, Any], sam_aux, p_t, p_ref) -> None:
    refine_band = sam_aux.get("refine_band")
    beta = sam_aux.get("beta")
    r_sam = sam_aux.get("R_sam")
    if torch.is_tensor(refine_band):
        band = refine_band.detach().to(device=p_ref.device, dtype=p_ref.dtype)
        if tuple(band.shape[-2:]) != tuple(p_ref.shape[-2:]):
            band = F.interpolate(band, size=p_ref.shape[-2:], mode="nearest")
        loss_dict["svb_refine_band_mean"] = float(band.mean().item())
        band_mask = band > 0.5
        if band_mask.any():
            changed = ((p_ref.detach() - p_t.detach()).abs() > 0.05).float()
            loss_dict["svb_boundary_changed_ratio"] = float(changed[band_mask.expand_as(changed)].mean().item())
    if torch.is_tensor(beta):
        loss_dict["svb_beta_mean"] = _tensor_or_float_mean(beta, default=0.0)
    if torch.is_tensor(r_sam):
        loss_dict["svb_R_sam_mean"] = _tensor_or_float_mean(r_sam, default=0.0)


def _changed_ratio(p_ref, p_t) -> float:
    if not torch.is_tensor(p_ref) or not torch.is_tensor(p_t):
        return 0.0
    return float(((p_ref.detach() - p_t.detach()).abs() > 0.05).float().mean().item())


def _is_rank_zero(distributed_train: bool) -> bool:
    if not distributed_train:
        return True
    try:
        return get_rank() == 0
    except Exception:
        return False


def _tensor_or_float_mean(value, default=0.0) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().mean().item())
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _tensor_min(value, default=0.0) -> float:
    if not torch.is_tensor(value) or value.numel() == 0:
        return float(default)
    return float(value.detach().float().min().item())


def _tensor_max(value, default=0.0) -> float:
    if not torch.is_tensor(value) or value.numel() == 0:
        return float(default)
    return float(value.detach().float().max().item())


def _backend_fallback_stats(backend_aux, batch_size: int) -> Tuple[float, float]:
    records = []

    def visit(node):
        if not isinstance(node, dict):
            return
        has_fallback_fields = any(key in node for key in ("used_fallback", "fallback_reason", "fallback_samples"))
        if has_fallback_fields:
            used = bool(node.get("used_fallback", False) or node.get("fallback_reason"))
            samples = node.get("fallback_samples")
            if isinstance(samples, (list, tuple)):
                sample_count = len(samples)
                used = used or sample_count > 0
            else:
                sample_count = int(batch_size) if used else 0
            records.append((used, sample_count))
        for value in node.values():
            if isinstance(value, dict):
                visit(value)

    visit(backend_aux)
    if not records:
        return 0.0, 0.0
    any_fallback = any(used for used, _ in records)
    fallback_samples = sum(sample_count for _, sample_count in records)
    denom = max(1, int(batch_size) * len(records))
    return (1.0 if any_fallback else 0.0), float(min(1.0, fallback_samples / denom))


def _prompt_stats(prompt_pack) -> Dict[str, float]:
    if not isinstance(prompt_pack, dict):
        return {}
    stats = prompt_pack.get("prompt_stats")
    if not isinstance(stats, dict):
        return {}
    out = {}
    for key, value in stats.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = 0.0
    return out


def _best_expert_ratios(selector_aux, batch_size: int) -> Dict[str, float]:
    default_names = ("box", "box_point", "mask", "boundary", "default", "fallback")
    counts = {name: 0 for name in default_names}
    if isinstance(selector_aux, dict):
        best_expert = selector_aux.get("best_expert")
        if isinstance(best_expert, (list, tuple)):
            for name in best_expert:
                clean = str(name).replace("-", "_").replace(" ", "_")
                counts[clean] = counts.get(clean, 0) + 1
    denom = max(1, int(batch_size))
    return {name: float(count / denom) for name, count in counts.items()}


__all__ = [
    "add_weighted_unsup_stats",
    "format_loss_info",
    "log_training_progress",
    "log_info",
    "log_svb_calibrator_state",
    "record_cbm_aux",
    "record_svb_aux",
]
