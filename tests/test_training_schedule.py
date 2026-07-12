import os
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "data", "logs"), exist_ok=True)

from engine.solver import SemiSupervisedTrainer, training_epoch_range
from models.build_model import (
    _build_lr_scheduler,
    _checkpoint_scheduler_type,
    _restore_fixed_optimizer_lr,
)


def test_training_epoch_range_is_strictly_zero_based_and_end_exclusive():
    epochs = list(training_epoch_range(0, 40))
    assert epochs == list(range(40))
    assert epochs[0] == 0
    assert epochs[-1] == 39
    assert len(epochs) == 40


def test_multistep_milestone_outside_run_keeps_lr_constant():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    config = SimpleNamespace(
        scheduler_type="multistep",
        lr_decay_epochs=[10000],
        lr_decay_rate=0.5,
        tot_epochs=40,
    )
    scheduler = _build_lr_scheduler(config, optimizer)

    used_lrs = []
    for _ in training_epoch_range(0, 40):
        used_lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad()
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()

    assert used_lrs == pytest.approx([1e-4] * 40)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_existing_cosine_builder_still_creates():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    config = SimpleNamespace(
        scheduler_type="cosine",
        scheduler_t_max=40,
        scheduler_warmup_epochs=0,
        scheduler_eta_min=1e-7,
        tot_epochs=40,
    )
    scheduler = _build_lr_scheduler(config, optimizer)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_unlabeled_boundary_checkpoints_remain_enabled():
    trainer = SemiSupervisedTrainer.__new__(SemiSupervisedTrainer)
    trainer.config = SimpleNamespace(
        save_step=1,
        save_last=12,
        save_stage_boundary=True,
        sup_only_train_epoch=20,
    )

    assert trainer._should_save_checkpoint(19, 40)
    assert trainer._should_save_checkpoint(20, 40)
    assert not trainer._should_save_checkpoint(18, 40)
    assert trainer._should_save_checkpoint(28, 40)


def test_checkpoint_lr_metadata_is_scheduler_generic():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    config = SimpleNamespace(
        scheduler_type="multistep",
        lr_decay_epochs=[10000],
        lr_decay_rate=0.5,
        tot_epochs=40,
    )
    scheduler = _build_lr_scheduler(config, optimizer)
    trainer = SemiSupervisedTrainer.__new__(SemiSupervisedTrainer)
    trainer.config = config
    trainer.model_optimizer = optimizer
    trainer.model_lr_scheduler = scheduler

    metadata = trainer._lr_schedule_meta(40)
    assert metadata == {
        "scheduler_type": "multistep",
        "tot_epochs": 40,
        "current_lr": [1e-4],
        "scheduler_epoch": scheduler.last_epoch,
    }
    assert "active_stage" not in metadata


def test_legacy_scheduler_mismatch_restores_lr_and_keeps_adam_moments():
    source_parameter = torch.nn.Parameter(torch.tensor(2.0))
    source_optimizer = torch.optim.Adam([source_parameter], lr=1e-4)
    source_optimizer.zero_grad()
    source_parameter.square().backward()
    source_optimizer.step()
    source_optimizer.param_groups[0]["lr"] = 1e-7
    optimizer_state = source_optimizer.state_dict()
    source_moments = next(iter(source_optimizer.state.values()))

    legacy_scheduler_state = {
        "scheduler_name": "TwoStageCosineLR",
        "group_scales": [1.0],
        "last_epoch": 19,
    }
    checkpoint = {
        "optimizer": optimizer_state,
        "lr_scheduler": legacy_scheduler_state,
        "lr_schedule_meta": {"scheduler_type": "two_stage_cosine"},
    }
    assert _checkpoint_scheduler_type(checkpoint) == "two_stage_cosine"

    target_parameter = torch.nn.Parameter(torch.tensor(2.0))
    target_optimizer = torch.optim.Adam([target_parameter], lr=1e-4)
    target_optimizer.load_state_dict(checkpoint["optimizer"])
    _restore_fixed_optimizer_lr(
        target_optimizer,
        1e-4,
        checkpoint["lr_scheduler"],
    )
    target_moments = next(iter(target_optimizer.state.values()))

    assert target_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert torch.equal(target_moments["step"], source_moments["step"])
    assert torch.equal(target_moments["exp_avg"], source_moments["exp_avg"])
    assert torch.equal(target_moments["exp_avg_sq"], source_moments["exp_avg_sq"])
    assert not hasattr(SemiSupervisedTrainer, "_step_lr_before_epoch")
