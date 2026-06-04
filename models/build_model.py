import os
import torch
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

from utils import Logger
from CBM import build_cbm_pfi
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


def _attach_cbm_if_enabled(config: Config, logger: Logger, device: torch.device, model: torch.nn.Module):
    if not bool(getattr(config, "cbm_pfi_enable", False)):
        return None
    if not isinstance(model, ModelEMA):
        _log(logger, ("warn_info", "warning", "info"), "[!] CBM-PFI is enabled but current model does not support set_cbm; CBM skipped.")
        return None

    cbm = build_cbm_pfi(config, device=device, logger=logger)
    cbm.initialize_modules(device=device)
    model.set_cbm(cbm)
    _log(logger, ("key_info", "info"), "[+] CBM-PFI engine attached before optimizer/checkpoint loading.")
    return cbm


def _model_state_from_checkpoint(checkpoint):
    return checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint


def _checkpoint_has_cbm_keys(model_state) -> bool:
    return hasattr(model_state, "keys") and any(str(key).startswith("cbm.") for key in model_state.keys())


def _log_incompatible_keys(logger: Logger, load_result, prefix: str) -> None:
    missing = list(getattr(load_result, "missing_keys", []) or [])
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    if missing:
        _log(logger, ("warn_info", "warning", "info"), f"[!] {prefix} missing keys: {len(missing)}")
    if unexpected:
        _log(logger, ("warn_info", "warning", "info"), f"[!] {prefix} unexpected keys: {len(unexpected)}")


def _load_cbm_memory_from_checkpoint(cbm, checkpoint, config: Config, device: torch.device, logger: Logger) -> None:
    if cbm is None or not bool(getattr(config, "cbm_checkpoint_memory", True)):
        return
    if not isinstance(checkpoint, dict) or "cbm_memory" not in checkpoint:
        _log(logger, ("warn_info", "warning", "info"), "[!] Checkpoint has no CBM memory; CBM eval will fallback until memory is rebuilt.")
        return
    try:
        cbm.load_memory_state_dict(checkpoint["cbm_memory"], device=device)
        _log(logger, ("key_info", "info"), f"[+] CBM memory restored from checkpoint: ready={cbm.memory.is_ready()}.")
    except Exception as exc:
        cbm.memory.clear()
        _log(logger, ("warn_info", "warning", "info"), f"[!] Failed to restore CBM memory: {exc}. Fallback to baseline.")


def build_model_optimizers(config: Config, logger: Logger, device: torch.device, resume: str = None) -> any:
    model = build_model(config)
    cbm = _attach_cbm_if_enabled(config, logger, device, model)
    epoch_st = 0
    checkpoint = None

    if resume is not None:
        if os.path.isfile(resume):
            logger.key_info("[+] Loading model checkpoint from '{}'".format(resume))
            checkpoint = torch.load(resume, map_location='cpu')
            load_result = model.load_state_dict(_model_state_from_checkpoint(checkpoint), strict=False)
            _log_incompatible_keys(logger, load_result, "Model checkpoint")
            _load_cbm_memory_from_checkpoint(cbm, checkpoint, config, device, logger)
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

    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0)
    else:
        raise NotImplementedError(f"Unsupported optimizer: {config.optimizer}")

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[lde if lde > 0 else config.tot_epochs + lde + 1 for lde in config.lr_decay_epochs],
        gamma=config.lr_decay_rate,
    )

    if checkpoint is not None:
        if 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError as exc:
                _log(logger, ("warn_info", "warning", "info"), f"[!] Optimizer state was not restored: {exc}")
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if 'epoch' in checkpoint:
            epoch_st = checkpoint['epoch'] + 1
            logger.key_info("[+] Resume training from epoch {}".format(epoch_st))

    logger.freeze_info("Optimizer details: {}".format(str(optimizer)))
    logger.freeze_info("Scheduler details: {}".format(str(lr_scheduler.state_dict())))

    return model, optimizer, lr_scheduler, epoch_st


def build_model_eval(config: Config, logger: Logger, resume: str, device: torch.device = 'cpu') -> torch.nn.Module:
    model = build_model(config=config)
    cbm = _attach_cbm_if_enabled(config, logger, device, model)
    logger.freeze_info("[+] Loading model from {} to evaluate...".format(resume))
    assert os.path.isfile(resume), "[x] target checkpoint not exists!"
    checkpoint = torch.load(resume, map_location='cpu')
    model_state = _model_state_from_checkpoint(checkpoint)
    has_cbm_keys = _checkpoint_has_cbm_keys(model_state)
    if cbm is None and has_cbm_keys:
        _log(logger, ("warn_info", "warning", "info"), "[!] Checkpoint contains CBM parameters but cbm_pfi_enable is false; loading baseline weights and ignoring CBM keys.")
    load_result = model.load_state_dict(model_state, strict=(cbm is None and not has_cbm_keys))
    if cbm is not None or has_cbm_keys:
        _log_incompatible_keys(logger, load_result, "Eval checkpoint")
    if cbm is not None:
        _load_cbm_memory_from_checkpoint(cbm, checkpoint, config, device, logger)
    model = model.to(device)
    return model
