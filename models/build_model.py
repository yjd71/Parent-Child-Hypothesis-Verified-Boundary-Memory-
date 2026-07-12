import os
import torch
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

from utils import Logger
from PC_HBM import build_pc_hbm, pc_hbm_enabled
from config import Config
from .talnet import ModelEMA
from .sinet import SINet_ResNet50
from .sinetv2 import SINet_v2
from .fspnet import FSPNet


def build_model(config: Config) -> torch.nn.Module:
    if config.model_name == 'Default':
        model = ModelEMA(config=config, bb_pretrained=True)
    elif config.model_name == 'SINet':
        model = SINet_ResNet50(config=config)
    elif config.model_name == 'SINetv2':
        model = SINet_v2(config, channel=32, imagenet_pretrained=True)
    elif config.model_name == 'FSPNet':
        model = FSPNet(config)
    else:
        raise NotImplementedError(f"Unsupported model_name: {config.model_name}")
    return model


def _log(logger: Logger, names, message: str) -> None:
    if logger is None:
        print(message)
        return
    for name in names:
        log_fn = getattr(logger, name, None)
        if log_fn is not None:
            log_fn(message)
            return


def _attach_pc_hbm_if_enabled(config: Config, logger: Logger, device: torch.device, model: torch.nn.Module):
    if not pc_hbm_enabled(config):
        return None
    if not isinstance(model, ModelEMA):
        _log(logger, ("warn_info", "warning", "info"), "[!] PC-HBM is enabled but current model does not support set_pc_hbm; PC-HBM skipped.")
        return None
    pc_hbm = build_pc_hbm(config, device=device, logger=logger)
    model.set_pc_hbm(pc_hbm)
    _log(logger, ("key_info", "info"), "[+] PC-HBM engine attached before optimizer/checkpoint loading.")
    return pc_hbm


def _model_state_from_checkpoint(checkpoint):
    return checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint


def _checkpoint_has_pc_hbm_keys(model_state) -> bool:
    return hasattr(model_state, "keys") and any(str(key).startswith("pc_hbm.") for key in model_state.keys())


def _load_shape_matched_state(model: torch.nn.Module, state_dict, logger: Logger = None, prefix: str = "Model checkpoint"):
    model_state = model.state_dict()
    keep = {}
    skip = []
    for key, value in state_dict.items():
        if key in model_state and hasattr(value, "shape") and tuple(value.shape) == tuple(model_state[key].shape):
            keep[key] = value
        else:
            skip.append(key)

    merged_state = dict(model_state)
    merged_state.update(keep)
    load_result = model.load_state_dict(merged_state, strict=False)
    _log(logger, ("key_info", "info"), f"[+] {prefix} shape-matched keys loaded: {len(keep)}")
    if skip:
        _log(logger, ("warn_info", "warning", "info"), f"[!] {prefix} mismatched/unexpected keys skipped: {len(skip)}")
        _log(logger, ("warn_info", "warning", "info"), f"[!] {prefix} first skipped keys: {skip[:10]}")
    return load_result, keep, skip


def _load_model_checkpoint_state(
    model: torch.nn.Module,
    checkpoint,
    config: Config,
    logger: Logger,
    prefix: str = "Model checkpoint",
):
    model_state = _model_state_from_checkpoint(checkpoint)
    strategy = str(getattr(config, "checkpoint_load_strategy", "shape_matched")).strip().lower()
    if strategy in ("shape_matched", "shape-matched", "shape_safe", "shape-safe"):
        load_result, _, _ = _load_shape_matched_state(model, model_state, logger=logger, prefix=prefix)
        return load_result
    if strategy in ("non_strict", "non-strict", "strict_false"):
        load_result = model.load_state_dict(model_state, strict=False)
        _log_incompatible_keys(logger, load_result, prefix)
        return load_result
    if strategy == "strict":
        return model.load_state_dict(model_state, strict=True)
    raise NotImplementedError(f"Unsupported checkpoint_load_strategy: {strategy}")


def _log_incompatible_keys(logger: Logger, load_result, prefix: str) -> None:
    missing = list(getattr(load_result, "missing_keys", []) or [])
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    if missing:
        _log(logger, ("warn_info", "warning", "info"), f"[!] {prefix} missing keys: {len(missing)}")
    if unexpected:
        _log(logger, ("warn_info", "warning", "info"), f"[!] {prefix} unexpected keys: {len(unexpected)}")


def _load_pc_hbm_memory_from_checkpoint(pc_hbm, checkpoint, config: Config, device: torch.device, logger: Logger) -> None:
    if pc_hbm is None or not bool(getattr(config, "pc_hbm_checkpoint_memory", True)):
        return
    if not isinstance(checkpoint, dict) or "pc_hbm_memory" not in checkpoint:
        _log(logger, ("warn_info", "warning", "info"), "[!] Checkpoint has no PC-HBM memory; PC-HBM eval will fallback until memory is rebuilt.")
        return
    try:
        pc_hbm.load_memory_state_dict(checkpoint["pc_hbm_memory"], device=device)
        _log(logger, ("key_info", "info"), f"[+] PC-HBM memory restored from checkpoint: ready={pc_hbm.memory.is_ready()}.")
    except Exception as exc:
        pc_hbm.memory.clear()
        _log(logger, ("warn_info", "warning", "info"), f"[!] Failed to restore PC-HBM memory: {exc}. Fallback to baseline.")


def _build_optimizer(
    config: Config,
    model: torch.nn.Module,
    lr: float = None,
) -> optim.Optimizer:
    optimizer_lr = float(config.lr if lr is None else lr)
    if config.optimizer == 'AdamW':
        return optim.AdamW(
            params=model.parameters(),
            lr=optimizer_lr,
            weight_decay=float(getattr(config, "weight_decay", 1e-2)),
        )
    if config.optimizer == 'Adam':
        return optim.Adam(
            params=model.parameters(),
            lr=optimizer_lr,
            weight_decay=float(getattr(config, "weight_decay", 0.0)),
        )
    raise NotImplementedError(f"Unsupported optimizer: {config.optimizer}")


def _build_lr_scheduler(config: Config, optimizer: optim.Optimizer):
    scheduler_type = str(getattr(config, "scheduler_type", "multistep")).strip().lower()
    if scheduler_type in ("cosine", "cosineannealing", "cosine_annealing"):
        total_epochs = int(getattr(config, "scheduler_t_max", getattr(config, "tot_epochs", 1)))
        warmup_epochs = max(0, int(getattr(config, "scheduler_warmup_epochs", 0)))
        eta_min = float(getattr(config, "scheduler_eta_min", 0.0))
        if warmup_epochs > 0:
            start_factor = float(getattr(config, "scheduler_warmup_start_factor", 0.2))
            start_factor = min(max(start_factor, 1e-8), 1.0)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_epochs - warmup_epochs),
                eta_min=eta_min,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs),
            eta_min=eta_min,
        )

    if scheduler_type in ("multistep", "multi_step", "step"):
        decay_epochs = getattr(config, "lr_decay_epochs", [1e4])
        if not isinstance(decay_epochs, (list, tuple)):
            decay_epochs = [decay_epochs]
        total_epochs = int(getattr(config, "tot_epochs", 1))
        milestones = sorted(
            int(lde) if int(lde) > 0 else total_epochs + int(lde) + 1
            for lde in decay_epochs
        )
        gamma = float(getattr(config, "lr_decay_rate", 0.5))
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )

    raise NotImplementedError(f"Unsupported scheduler_type: {scheduler_type}")


def _canonical_scheduler_type(value) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    compact = normalized.replace("_", "")
    if compact in {"multistep", "step"}:
        return "multistep"
    if compact in {"cosine", "cosineannealing"}:
        return "cosine"
    if compact in {
        "twostagecosine",
        "twostagecosineannealing",
        "twostagecosinerestart",
        "twostagecosinelr",
    }:
        return "two_stage_cosine"
    return normalized


def _checkpoint_scheduler_type(checkpoint) -> str | None:
    if not isinstance(checkpoint, dict):
        return None
    meta = checkpoint.get("lr_schedule_meta")
    if isinstance(meta, dict) and meta.get("scheduler_type") is not None:
        return _canonical_scheduler_type(meta["scheduler_type"])
    state = checkpoint.get("lr_scheduler")
    if isinstance(state, dict) and state.get("scheduler_name") is not None:
        return _canonical_scheduler_type(state["scheduler_name"])
    return None


def _restore_fixed_optimizer_lr(
    optimizer: optim.Optimizer,
    base_lr: float,
    scheduler_state=None,
) -> None:
    group_scales = []
    if isinstance(scheduler_state, dict):
        group_scales = list(scheduler_state.get("group_scales", []))
    for index, group in enumerate(optimizer.param_groups):
        scale = group.get("lr_scale")
        if scale is None and index < len(group_scales):
            scale = group_scales[index]
        scale = float(scale) if scale is not None else 1.0
        group["lr"] = float(base_lr) * scale
        group["initial_lr"] = float(base_lr) * scale


def build_model_optimizers(config: Config, logger: Logger, device: torch.device, resume: str = None) -> any:
    model = build_model(config)
    pc_hbm = _attach_pc_hbm_if_enabled(config, logger, device, model)
    epoch_st = 0
    checkpoint = None

    if resume is not None:
        if os.path.isfile(resume):
            logger.key_info("[+] Loading model checkpoint from '{}'".format(resume))
            checkpoint = torch.load(resume, map_location='cpu')
            _load_model_checkpoint_state(model, checkpoint, config, logger, prefix="Model checkpoint")
            _load_pc_hbm_memory_from_checkpoint(pc_hbm, checkpoint, config, device, logger)
        else:
            logger.warn_info("[!] No checkpoint found at '{}'".format(resume))

    if config.distributed_train:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
    else:
        model = model.to(device)

    if config.compile_model:
        model = torch.compile(model, mode=['default', 'reduce-overhead', 'max-autotune'][0])
    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    optimizer = _build_optimizer(config, model, lr=float(config.lr))

    lr_scheduler = _build_lr_scheduler(config, optimizer)
    logger.freeze_info("Scheduler type: {}".format(str(getattr(config, "scheduler_type", "multistep"))))

    if checkpoint is not None and bool(getattr(config, "resume_training_state", False)):
        configured_scheduler_type = _canonical_scheduler_type(
            getattr(config, "scheduler_type", "multistep")
        )
        checkpoint_scheduler_type = _checkpoint_scheduler_type(checkpoint)
        scheduler_mismatch = (
            checkpoint_scheduler_type is not None
            and checkpoint_scheduler_type != configured_scheduler_type
        )
        optimizer_state_restored = False
        if 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                optimizer_state_restored = True
            except ValueError as exc:
                _log(logger, ("warn_info", "warning", "info"), f"[!] Optimizer state was not restored: {exc}")
        if scheduler_mismatch:
            _restore_fixed_optimizer_lr(
                optimizer,
                float(config.lr),
                checkpoint.get('lr_scheduler'),
            )
            _log(
                logger,
                ("warn_info", "warning", "info"),
                "[!] Checkpoint scheduler type "
                f"{checkpoint_scheduler_type!r} does not match configured type "
                f"{configured_scheduler_type!r}; scheduler state was skipped and "
                f"optimizer LR was restored to {float(config.lr):.3e} while keeping "
                + (
                    "the restored optimizer moments."
                    if optimizer_state_restored
                    else "the freshly initialized optimizer state."
                ),
            )
        elif 'lr_scheduler' in checkpoint:
            try:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            except (ValueError, KeyError, TypeError) as exc:
                _log(
                    logger,
                    ("warn_info", "warning", "info"),
                    "[!] LR scheduler state was not restored: "
                    f"{exc}. Using the freshly configured scheduler.",
                )
        if 'epoch' in checkpoint:
            epoch_st = checkpoint['epoch'] + 1
            logger.key_info("[+] Resume training from epoch {}".format(epoch_st))
    elif checkpoint is not None:
        _log(logger, ("warn_info", "warning", "info"), "[!] Training state resume disabled; optimizer/lr_scheduler/epoch were not restored.")

    logger.freeze_info("Optimizer details: {}".format(str(optimizer)))
    logger.freeze_info("Scheduler details: {}".format(str(lr_scheduler.state_dict())))

    return model, optimizer, lr_scheduler, epoch_st


def build_model_eval(config: Config, logger: Logger, resume: str, device: torch.device = 'cpu') -> torch.nn.Module:
    model = build_model(config=config)
    pc_hbm = _attach_pc_hbm_if_enabled(config, logger, device, model)
    logger.freeze_info("[+] Loading model from {} to evaluate...".format(resume))
    assert os.path.isfile(resume), "[x] target checkpoint not exists!"
    checkpoint = torch.load(resume, map_location='cpu')
    model_state = _model_state_from_checkpoint(checkpoint)
    has_pc_hbm_keys = _checkpoint_has_pc_hbm_keys(model_state)
    if pc_hbm is None and has_pc_hbm_keys:
        _log(logger, ("warn_info", "warning", "info"), "[!] Checkpoint contains PC-HBM parameters but use_pc_hbm/pc_hbm_enable is false; loading baseline weights and ignoring PC-HBM keys.")
    _load_model_checkpoint_state(model, checkpoint, config, logger, prefix="Eval checkpoint")
    if pc_hbm is not None:
        _load_pc_hbm_memory_from_checkpoint(pc_hbm, checkpoint, config, device, logger)
    model = model.to(device)
    return model
