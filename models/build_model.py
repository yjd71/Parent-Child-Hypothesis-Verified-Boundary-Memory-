import os
import torch
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

from utils import Logger
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


def build_model_optimizers(config: Config, logger: Logger, device: torch.device, resume: str = None) -> any:
    model = build_model(config)
    epoch_st = 0
    checkpoint = None

    if resume is not None:
        if os.path.isfile(resume):
            logger.key_info("[+] Loading model checkpoint from '{}'".format(resume))
            checkpoint = torch.load(resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
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
            optimizer.load_state_dict(checkpoint['optimizer'])
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
    logger.freeze_info("[+] Loading model from {} to evaluate...".format(resume))
    assert os.path.isfile(resume), "[x] target checkpoint not exists!"
    state_dict = torch.load(resume, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    return model
