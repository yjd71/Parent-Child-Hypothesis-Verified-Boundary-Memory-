"""Epoch-indexed two-stage cosine learning-rate scheduler."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch


class TwoStageCosineLR:
    """Apply two independent cosine schedules without replacing the optimizer.

    Stage 1 covers ``[0, split_epoch)`` and stage 2 covers
    ``[split_epoch, total_epochs)``. Call :meth:`step_epoch` before training an
    epoch so the optimizer uses the LR associated with that exact epoch index.
    """

    scheduler_name = "TwoStageCosineLR"

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_epochs: int,
        split_epoch: int,
        stage1_initial_lr: float,
        stage1_min_lr: float,
        stage2_initial_lr: float,
        stage2_min_lr: float,
    ) -> None:
        total_epochs = int(total_epochs)
        split_epoch = int(split_epoch)
        if total_epochs <= 1:
            raise ValueError(f"total_epochs must be greater than 1, got {total_epochs}")
        if not 0 < split_epoch < total_epochs:
            raise ValueError(
                "split_epoch must satisfy 0 < split_epoch < total_epochs, "
                f"got split_epoch={split_epoch}, total_epochs={total_epochs}"
            )

        values = {
            "stage1_initial_lr": float(stage1_initial_lr),
            "stage1_min_lr": float(stage1_min_lr),
            "stage2_initial_lr": float(stage2_initial_lr),
            "stage2_min_lr": float(stage2_min_lr),
        }
        for name, value in values.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative value, got {value}")
        if values["stage1_min_lr"] > values["stage1_initial_lr"]:
            raise ValueError("stage1_min_lr must not exceed stage1_initial_lr")
        if values["stage2_min_lr"] > values["stage2_initial_lr"]:
            raise ValueError("stage2_min_lr must not exceed stage2_initial_lr")
        if not optimizer.param_groups:
            raise ValueError("optimizer must contain at least one parameter group")

        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.split_epoch = split_epoch
        self.stage1_initial_lr = values["stage1_initial_lr"]
        self.stage1_min_lr = values["stage1_min_lr"]
        self.stage2_initial_lr = values["stage2_initial_lr"]
        self.stage2_min_lr = values["stage2_min_lr"]
        self.group_scales = self._infer_group_scales()
        self.last_epoch = -1
        self._last_lr = [float(group["lr"]) for group in optimizer.param_groups]

    def _infer_group_scales(self) -> List[float]:
        scales = []
        for index, group in enumerate(self.optimizer.param_groups):
            if "lr_scale" in group:
                scale = float(group["lr_scale"])
            elif self.stage1_initial_lr > 0.0:
                scale = float(group["lr"]) / self.stage1_initial_lr
            elif float(group["lr"]) == 0.0:
                scale = 1.0
            else:
                raise ValueError(
                    "Cannot infer lr_scale for parameter group "
                    f"{index} when stage1_initial_lr is zero"
                )
            if not math.isfinite(scale) or scale < 0.0:
                raise ValueError(f"lr_scale for parameter group {index} is invalid: {scale}")
            scales.append(scale)
        return scales

    @staticmethod
    def _cosine_value(
        initial_lr: float,
        min_lr: float,
        local_epoch: int,
        stage_length: int,
    ) -> float:
        if stage_length <= 1:
            return initial_lr
        progress = local_epoch / float(stage_length - 1)
        return min_lr + 0.5 * (initial_lr - min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def stage_for_epoch(self, epoch: int) -> int:
        epoch = int(epoch)
        if not 0 <= epoch < self.total_epochs:
            raise ValueError(
                f"epoch must satisfy 0 <= epoch < {self.total_epochs}, got {epoch}"
            )
        return 1 if epoch < self.split_epoch else 2

    def lr_for_epoch(self, epoch: int) -> float:
        epoch = int(epoch)
        if self.stage_for_epoch(epoch) == 1:
            return self._cosine_value(
                self.stage1_initial_lr,
                self.stage1_min_lr,
                local_epoch=epoch,
                stage_length=self.split_epoch,
            )
        return self._cosine_value(
            self.stage2_initial_lr,
            self.stage2_min_lr,
            local_epoch=epoch - self.split_epoch,
            stage_length=self.total_epochs - self.split_epoch,
        )

    def step_epoch(self, epoch: int) -> List[float]:
        epoch = int(epoch)
        if len(self.optimizer.param_groups) != len(self.group_scales):
            raise ValueError("optimizer parameter-group count changed after scheduler creation")
        base_lr = self.lr_for_epoch(epoch)
        lrs = []
        for group, scale in zip(self.optimizer.param_groups, self.group_scales):
            lr = base_lr * scale
            group["lr"] = lr
            lrs.append(lr)
        self.last_epoch = epoch
        self._last_lr = lrs
        return list(lrs)

    def get_last_lr(self) -> List[float]:
        return list(self._last_lr)

    @property
    def current_stage(self) -> int:
        return 1 if self.last_epoch < 0 else self.stage_for_epoch(self.last_epoch)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "scheduler_name": self.scheduler_name,
            "total_epochs": self.total_epochs,
            "split_epoch": self.split_epoch,
            "stage1_initial_lr": self.stage1_initial_lr,
            "stage1_min_lr": self.stage1_min_lr,
            "stage2_initial_lr": self.stage2_initial_lr,
            "stage2_min_lr": self.stage2_min_lr,
            "last_epoch": self.last_epoch,
            "last_lr": list(self._last_lr),
            "group_scales": list(self.group_scales),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        required_keys = {
            "scheduler_name",
            "total_epochs",
            "split_epoch",
            "stage1_initial_lr",
            "stage1_min_lr",
            "stage2_initial_lr",
            "stage2_min_lr",
            "last_epoch",
            "last_lr",
            "group_scales",
        }
        missing = sorted(required_keys.difference(state_dict))
        if missing:
            raise KeyError(f"TwoStageCosineLR state is missing keys: {missing}")
        if state_dict["scheduler_name"] != self.scheduler_name:
            raise ValueError(
                "Incompatible scheduler state: expected "
                f"{self.scheduler_name}, got {state_dict['scheduler_name']}"
            )

        expected = {
            "total_epochs": self.total_epochs,
            "split_epoch": self.split_epoch,
            "stage1_initial_lr": self.stage1_initial_lr,
            "stage1_min_lr": self.stage1_min_lr,
            "stage2_initial_lr": self.stage2_initial_lr,
            "stage2_min_lr": self.stage2_min_lr,
        }
        for name, expected_value in expected.items():
            loaded_value = state_dict[name]
            if loaded_value != expected_value:
                raise ValueError(
                    f"Scheduler state/config mismatch for {name}: "
                    f"checkpoint={loaded_value}, current={expected_value}"
                )

        group_scales = [float(value) for value in state_dict["group_scales"]]
        last_lr = [float(value) for value in state_dict["last_lr"]]
        group_count = len(self.optimizer.param_groups)
        if len(group_scales) != group_count or len(last_lr) != group_count:
            raise ValueError(
                "Scheduler parameter-group state does not match the optimizer: "
                f"checkpoint={len(group_scales)}, optimizer={group_count}"
            )

        last_epoch = int(state_dict["last_epoch"])
        if not -1 <= last_epoch < self.total_epochs:
            raise ValueError(
                f"last_epoch must be in [-1, {self.total_epochs - 1}], got {last_epoch}"
            )

        self.group_scales = group_scales
        self.last_epoch = -1
        if last_epoch >= 0:
            restored_lrs = self.step_epoch(last_epoch)
            if any(
                not math.isclose(actual, saved, rel_tol=1e-12, abs_tol=1e-15)
                for actual, saved in zip(restored_lrs, last_lr)
            ):
                raise ValueError("Scheduler last_lr is inconsistent with its saved epoch")
        else:
            for group, lr in zip(self.optimizer.param_groups, last_lr):
                group["lr"] = lr
            self._last_lr = last_lr
