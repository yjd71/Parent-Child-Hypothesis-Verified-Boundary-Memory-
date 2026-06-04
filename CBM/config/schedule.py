from __future__ import annotations

from typing import Any, Optional


def cbm_stage_epoch(config: Any, epoch: Optional[int]) -> Optional[int]:
    if epoch is None:
        return None
    return int(epoch) + int(getattr(config, "cbm_stage_epoch_offset", 1))


def cbm_stage_id(config: Any, epoch: Optional[int]) -> int:
    stage_epoch = cbm_stage_epoch(config, epoch)
    if stage_epoch is None:
        return 0
    if stage_epoch <= int(getattr(config, "cbm_stage1_end", 5)):
        return 1
    if stage_epoch <= int(getattr(config, "cbm_stage2_end", 15)):
        return 2
    return 3


def cbm_stage_name(config: Any, epoch: Optional[int]) -> str:
    stage = cbm_stage_id(config, epoch)
    return {0: "unknown", 1: "baseline_warmup", 2: "labeled_cbm", 3: "labeled_unlabeled_cbm"}[stage]


def cbm_should_rebuild_memory(config: Any, epoch: Optional[int]) -> bool:
    if not bool(getattr(config, "cbm_pfi_enable", False)):
        return False
    return cbm_stage_id(config, epoch) in (2, 3)


def cbm_unlabeled_enabled(config: Any, epoch: Optional[int]) -> bool:
    if not bool(getattr(config, "cbm_pfi_enable", False)):
        return False
    stage_epoch = cbm_stage_epoch(config, epoch)
    if stage_epoch is None:
        return False
    return stage_epoch >= int(getattr(config, "cbm_unlabeled_start_epoch", 16))


def cbm_enabled_for_epoch(config: Any, epoch: Optional[int], memory_ready: bool) -> bool:
    if not bool(getattr(config, "cbm_pfi_enable", False)):
        return False
    if not bool(memory_ready):
        return False
    if epoch is None:
        return True
    stage_epoch = cbm_stage_epoch(config, epoch)
    start_epoch = int(getattr(config, "cbm_start_epoch", 0))
    stage_start_epoch = int(getattr(config, "cbm_stage1_end", 5)) + 1
    return stage_epoch >= max(start_epoch, stage_start_epoch)
