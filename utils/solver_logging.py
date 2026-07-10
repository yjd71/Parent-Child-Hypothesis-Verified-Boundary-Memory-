from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import wandb
from torch.distributed import get_rank


BASIC_TRAIN_LOG_INTERVAL = 20

_PC_HBM_METRIC_KEYS = {
    "L_seg_total",
    "L_parent_ce",
    "L_child_verify",
    "L_geometry",
    "L_gate",
    "L_boundary_aux",
    "L_mix_oracle",
    "L_branch",
    "L_quality",
    "L_usage",
    "L_reg",
    "pi_keep_mean",
    "pi_res_mean",
    "pi_def_mean",
    "pi_sup_mean",
    "gate_pc_mean",
    "C23_mean",
    "route_entropy",
    "parent_entropy",
    "memory_ready",
    "soft_teacher_loss",
    "soft_teacher_bce",
    "soft_teacher_weighted_iou",
    "hard_teacher_loss",
    "loss_u_total",
}


def log_info(logger, message: str) -> None:
    if logger is None:
        print(message)
        return
    log_fn = getattr(logger, "info", None) or getattr(logger, "key_info", None)
    if log_fn is not None:
        log_fn(message)


def format_loss_info(loss_dict: Mapping[str, Any], title: str, include_module_losses: bool = True) -> str:
    info_loss = title
    for loss_name, loss_value in loss_dict.items():
        if not include_module_losses and _module_metric_group(str(loss_name)) is not None:
            continue
        info_loss += ", {}: {:.3f}".format(loss_name, float(loss_value))
    return info_loss


def partition_training_metrics(
    loss_dict: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Split baseline training metrics from module-owned diagnostics."""
    base_metrics: Dict[str, Any] = {}
    module_metrics: Dict[str, Dict[str, Any]] = {
        "PC-HBM": {},
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
    include_module_losses: bool = True,
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
            include_module_losses=include_module_losses,
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


def _is_rank_zero(distributed_train: bool) -> bool:
    if not distributed_train:
        return True
    try:
        return get_rank() == 0
    except Exception:
        return False


def _module_metric_group(name: str):
    if name in _PC_HBM_METRIC_KEYS or name.startswith(("pc_hbm_", "L_", "pi_")):
        return "PC-HBM"
    return None


__all__ = [
    "BASIC_TRAIN_LOG_INTERVAL",
    "format_loss_info",
    "log_training_progress",
    "log_info",
    "partition_training_metrics",
    "should_log_training_progress",
]
