from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn.functional as F
import wandb
from torch.distributed import get_rank


BASIC_TRAIN_LOG_INTERVAL = 20

_CBM_METRIC_KEYS = {
    "cbm_stage",
    "memory_ready",
    "gate_mean",
    "valid_ratio",
    "retrieval_uncertainty_mean",
    "memory_tokens",
}
_SVB_METRIC_KEYS = {
    "conf_ref_mean",
    "conf_ref_min",
    "conf_ref_max",
    "boundary_boost_mean",
    "weighted_unsup_weight_mean",
    "weighted_unsup_weight_min",
    "weighted_unsup_weight_max",
    "weighted_unsup_refine_band_mean",
    "loss_weighted_unsup",
}
_SV_UME_METRIC_KEYS = {
    "loss_ume_evi",
    "loss_source_cons",
    "loss_sv_ume",
}


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


def partition_training_metrics(
    loss_dict: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Split baseline training metrics from module-owned diagnostics."""
    base_metrics: Dict[str, Any] = {}
    module_metrics: Dict[str, Dict[str, Any]] = {
        "CBM": {},
        "SVB-PLR": {},
        "SV-UME": {},
    }
    for name, value in loss_dict.items():
        group = _module_metric_group(str(name))
        if group is None:
            base_metrics[name] = value
        else:
            module_metrics[group][name] = value
    return base_metrics, module_metrics


def should_log_training_progress(batch_idx: Any) -> bool:
    """Return the fixed baseline logging cadence, independent of module config."""
    try:
        return int(batch_idx) % BASIC_TRAIN_LOG_INTERVAL == 0
    except (TypeError, ValueError):
        return False


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
    log_base: bool = True,
    log_modules: bool = False,
) -> None:
    base_metrics, module_metrics = partition_training_metrics(loss_dict)
    label = "{} ".format(progress_label) if progress_label else ""
    info_progress = "{}Epoch[{}/{}] Iter[{}/{}].".format(
        label,
        epoch,
        total_epochs,
        batch_idx,
        num_batches,
    )
    if log_base:
        info_loss = format_loss_info(
            base_metrics,
            title,
            include_cbm_losses=include_cbm_losses,
        )
        log_info(logger, " ".join((info_progress, info_loss)))

    if log_modules:
        module_progress = "Epoch[{}/{}] Iter[{}/{}].".format(
            epoch,
            total_epochs,
            batch_idx,
            num_batches,
        )
        for module_name, metrics in module_metrics.items():
            if not metrics:
                continue
            info_metrics = format_loss_info(metrics, "Metrics")
            log_info(
                logger,
                "[{}] {} {} {}".format(
                    module_name,
                    wandb_prefix,
                    module_progress,
                    info_metrics,
                ),
            )

    if _is_rank_zero(distributed_train):
        wandb_metrics: Dict[str, Any] = {}
        if log_base:
            wandb_metrics.update(base_metrics)
        if log_modules:
            for metrics in module_metrics.values():
                wandb_metrics.update(metrics)
        if wandb_metrics:
            wandb.log(
                {"{}-{}".format(wandb_prefix, k): v for k, v in wandb_metrics.items()},
                step=step,
            )


def record_cbm_aux(
    loss_dict: Dict[str, Any],
    cbm,
    cbm_stage: int,
    aux,
    branch_name: str,
    logger=None,
    log_enabled: bool = True,
) -> None:
    if cbm is not None:
        loss_dict["cbm_stage"] = float(cbm_stage)
        loss_dict["memory_ready"] = 1.0 if cbm.memory.is_ready() else 0.0
    if not aux:
        return
    loss_dict["gate_mean"] = float(aux.get("gate_mean", 0.0) or 0.0)
    loss_dict["valid_ratio"] = float(aux.get("valid_ratio", 0.0) or 0.0)
    loss_dict["retrieval_uncertainty_mean"] = float(aux.get("u_mean", 0.0) or 0.0)
    loss_dict["memory_tokens"] = float(aux.get("num_memory_tokens", 0) or 0)
    if log_enabled and aux.get("fallback_reason"):
        log_info(logger, "[CBM] {} fallback={}".format(branch_name, aux.get("fallback_reason")))


def record_svb_aux(loss_dict: Dict[str, Any], sam_aux, p_t, p_ref, conf_ref, logger=None, log_enabled: bool = True) -> None:
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
    loss_dict["svb_valid_sam_candidate_ratio"] = _valid_candidate_ratio(sam_aux.get("selector_aux"))
    sam_teacher_mae, sam_teacher_exact_ratio = _sam_teacher_stats(sam_aux.get("sam_mask"), p_t)
    loss_dict["svb_sam_teacher_mae"] = sam_teacher_mae
    loss_dict["svb_sam_teacher_exact_ratio"] = sam_teacher_exact_ratio

    backend_fallback, backend_fallback_ratio = _backend_fallback_stats(
        sam_aux.get("backend_aux"),
        batch_size=batch_size,
    )
    if sam_aux.get("fallback_reason"):
        backend_fallback = 1.0
        backend_fallback_ratio = max(backend_fallback_ratio, 1.0)
    loss_dict["svb_backend_fallback"] = backend_fallback
    loss_dict["svb_backend_fallback_ratio"] = backend_fallback_ratio
    embedding_hits, embedding_misses = _embedding_cache_stats(sam_aux.get("backend_aux"))
    embedding_total = embedding_hits + embedding_misses
    loss_dict["svb_embedding_cache_hits"] = float(embedding_hits)
    loss_dict["svb_embedding_cache_misses"] = float(embedding_misses)
    loss_dict["svb_embedding_cache_hit_rate"] = (
        float(embedding_hits / embedding_total) if embedding_total > 0 else 0.0
    )

    prompt_stats = _prompt_stats(sam_aux.get("prompt_pack"))
    loss_dict["svb_prompt_empty_box_ratio"] = prompt_stats.get("empty_box_ratio", 0.0)
    loss_dict["svb_prompt_empty_point_ratio"] = prompt_stats.get("empty_point_ratio", 0.0)
    loss_dict["svb_prompt_all_empty"] = prompt_stats.get("all_prompt_empty", 0.0)
    loss_dict["svb_prompt_box_count"] = prompt_stats.get("box_count", 0.0)
    loss_dict["svb_prompt_point_count"] = prompt_stats.get("point_count", 0.0)
    loss_dict["svb_prompt_boundary_point_count"] = prompt_stats.get("boundary_point_count", 0.0)
    loss_dict["svb_prompt_has_mask"] = prompt_stats.get("has_mask", 0.0)

    _record_svb_maps(loss_dict, sam_aux, p_t, p_ref)
    if log_enabled and sam_aux.get("fallback_reason"):
        log_info(logger, "[SVB-PLR] fallback={}".format(sam_aux.get("fallback_reason")))


def log_svb_calibrator_state(logger, svb_plr, prefix: str, log_enabled: bool = True) -> None:
    if not log_enabled:
        return
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


def _module_metric_group(name: str):
    if name in _CBM_METRIC_KEYS or name.startswith(("loss_cbm_", "raw_cbm_", "cbm_")):
        return "CBM"
    if name in _SVB_METRIC_KEYS or name.startswith("svb_"):
        return "SVB-PLR"
    if name in _SV_UME_METRIC_KEYS or name.startswith(("sv_ume_", "ume_")):
        return "SV-UME"
    return None


def _valid_candidate_ratio(selector_aux) -> float:
    if not isinstance(selector_aux, Mapping):
        return 0.0
    valid = selector_aux.get("valid_candidates")
    if torch.is_tensor(valid) and valid.numel() > 0:
        return float(valid.detach().float().mean().item())
    return _tensor_or_float_mean(selector_aux.get("valid_candidate_ratio"), default=0.0)


def _sam_teacher_stats(sam_mask, teacher_prob) -> Tuple[float, float]:
    if not torch.is_tensor(sam_mask) or not torch.is_tensor(teacher_prob):
        return 0.0, 0.0
    sam = sam_mask.detach().float()
    teacher = teacher_prob.detach().to(device=sam.device, dtype=sam.dtype)
    if sam.dim() == 3:
        sam = sam.unsqueeze(1)
    if teacher.dim() == 3:
        teacher = teacher.unsqueeze(1)
    if sam.dim() != 4 or teacher.dim() != 4 or sam.size(0) != teacher.size(0):
        return 0.0, 0.0
    if sam.size(1) != 1:
        sam = sam[:, :1]
    if teacher.size(1) != 1:
        teacher = teacher[:, :1]
    if tuple(sam.shape[-2:]) != tuple(teacher.shape[-2:]):
        sam = F.interpolate(sam, size=teacher.shape[-2:], mode="bilinear", align_corners=False)
    diff = (sam - teacher).abs()
    mae = float(diff.mean().item())
    exact_ratio = float((diff.flatten(1).amax(dim=1) == 0).float().mean().item())
    return mae, exact_ratio


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


def _embedding_cache_stats(backend_aux) -> Tuple[int, int]:
    hits = 0
    misses = 0

    def visit(node):
        nonlocal hits, misses
        if not isinstance(node, dict):
            return
        if "embedding_cache_hits" in node or "embedding_cache_misses" in node:
            hits += int(node.get("embedding_cache_hits", 0) or 0)
            misses += int(node.get("embedding_cache_misses", 0) or 0)
        for key, value in node.items():
            # embedding_cache_info contains cumulative counters; do not mix
            # them with the current-batch hit rate recorded above.
            if key != "embedding_cache_info" and isinstance(value, dict):
                visit(value)

    visit(backend_aux)
    return hits, misses


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


__all__ = [
    "BASIC_TRAIN_LOG_INTERVAL",
    "add_weighted_unsup_stats",
    "format_loss_info",
    "log_training_progress",
    "log_info",
    "log_svb_calibrator_state",
    "partition_training_metrics",
    "record_cbm_aux",
    "record_svb_aux",
    "should_log_training_progress",
]
