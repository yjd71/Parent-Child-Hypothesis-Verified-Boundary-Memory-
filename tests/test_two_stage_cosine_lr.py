import importlib
import logging
import os
import sys
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_original_file_handler = logging.FileHandler
logging.FileHandler = lambda *args, **kwargs: logging.NullHandler()
try:
    from engine.solver import SemiSupervisedTrainer, training_epoch_range
    from models.build_model import _build_lr_scheduler
finally:
    logging.FileHandler = _original_file_handler

from utils.two_stage_cosine_lr import TwoStageCosineLR


def _new_optimizer_and_scheduler(param_groups=None):
    if param_groups is None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        param_groups = [parameter]
    optimizer = torch.optim.Adam(param_groups, lr=1e-4)
    scheduler = TwoStageCosineLR(
        optimizer,
        total_epochs=40,
        split_epoch=20,
        stage1_initial_lr=1e-4,
        stage1_min_lr=1e-7,
        stage2_initial_lr=1e-4,
        stage2_min_lr=1e-7,
    )
    return optimizer, scheduler


def _two_stage_config(**overrides):
    values = {
        "optimizer": "Adam",
        "lr": 1e-4,
        "scheduler_type": "two_stage_cosine",
        "tot_epochs": 40,
        "sup_only_train_epoch": 20,
        "unlabeled_start_epoch": 20,
        "stage1_initial_lr": 1e-4,
        "stage1_min_lr": 1e-7,
        "stage2_initial_lr": 1e-4,
        "stage2_min_lr": 1e-7,
        "preserve_optimizer_state_across_stages": True,
        "reset_optimizer_at_stage2": False,
        "require_lr_stage_match_unlabeled_stage": True,
        "save_stage_boundary": True,
        "save_step": 1,
        "save_last": 12,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _RecordingLogger:
    def __init__(self):
        self.messages = []

    def _record(self, message):
        self.messages.append(str(message))

    key_info = _record
    warn_info = _record
    freeze_info = _record
    success_info = _record
    info = _record


def test_two_stage_cosine_endpoints_stages_and_monotonicity():
    _, scheduler = _new_optimizer_and_scheduler()
    lrs = [scheduler.lr_for_epoch(epoch) for epoch in range(40)]

    assert lrs[0] == pytest.approx(1e-4)
    assert lrs[19] == pytest.approx(1e-7)
    assert lrs[20] == pytest.approx(1e-4)
    assert lrs[39] == pytest.approx(1e-7)
    assert all(left >= right for left, right in zip(lrs[:19], lrs[1:20]))
    assert all(left >= right for left, right in zip(lrs[20:39], lrs[21:40]))
    assert lrs[20] > lrs[19]

    assert scheduler.stage_for_epoch(0) == 1
    assert scheduler.stage_for_epoch(19) == 1
    assert scheduler.stage_for_epoch(20) == 2
    assert scheduler.stage_for_epoch(39) == 2


@pytest.mark.parametrize("checkpoint_epoch", [10, 19, 20, 31])
def test_scheduler_checkpoint_resume_matches_uninterrupted_trace(checkpoint_epoch):
    source_optimizer, source_scheduler = _new_optimizer_and_scheduler()
    for epoch in range(checkpoint_epoch + 1):
        source_scheduler.step_epoch(epoch)

    checkpoint = {
        "optimizer": source_optimizer.state_dict(),
        "lr_scheduler": source_scheduler.state_dict(),
    }
    restored_optimizer, restored_scheduler = _new_optimizer_and_scheduler()
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    assert restored_scheduler.last_epoch == checkpoint_epoch
    assert restored_scheduler.get_last_lr() == pytest.approx(
        source_scheduler.get_last_lr()
    )
    for epoch in range(checkpoint_epoch + 1, 40):
        source_scheduler.step_epoch(epoch)
        restored_scheduler.step_epoch(epoch)
        assert restored_scheduler.get_last_lr() == pytest.approx(
            source_scheduler.get_last_lr()
        )


def test_stage_restart_keeps_optimizer_identity_and_adam_moments():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer, scheduler = _new_optimizer_and_scheduler([parameter])

    for epoch in (18, 19):
        scheduler.step_epoch(epoch)
        optimizer.zero_grad()
        parameter.square().backward()
        optimizer.step()

    optimizer_id = id(optimizer)
    state = optimizer.state[parameter]
    step_before = state["step"].clone()
    exp_avg_before = state["exp_avg"].clone()
    exp_avg_sq_before = state["exp_avg_sq"].clone()

    scheduler.step_epoch(20)

    assert id(optimizer) == optimizer_id
    assert optimizer.state[parameter] is state
    torch.testing.assert_close(state["step"], step_before)
    torch.testing.assert_close(state["exp_avg"], exp_avg_before)
    torch.testing.assert_close(state["exp_avg_sq"], exp_avg_sq_before)

    optimizer.zero_grad()
    parameter.square().backward()
    optimizer.step()
    assert state["step"].item() == pytest.approx(step_before.item() + 1)
    assert torch.count_nonzero(state["exp_avg"]).item() > 0
    assert torch.count_nonzero(state["exp_avg_sq"]).item() > 0


def test_multiple_param_groups_preserve_lr_scale():
    parameter_a = torch.nn.Parameter(torch.tensor(1.0))
    parameter_b = torch.nn.Parameter(torch.tensor(2.0))
    groups = [
        {"params": [parameter_a], "lr": 1e-4, "lr_scale": 1.0},
        {"params": [parameter_b], "lr": 5e-5, "lr_scale": 0.5},
    ]
    optimizer, scheduler = _new_optimizer_and_scheduler(groups)

    assert scheduler.step_epoch(20) == pytest.approx([1e-4, 5e-5])
    assert scheduler.step_epoch(39) == pytest.approx([1e-7, 5e-8])
    assert optimizer.param_groups[0]["lr_scale"] == pytest.approx(1.0)
    assert optimizer.param_groups[1]["lr_scale"] == pytest.approx(0.5)


def test_training_epoch_range_is_strictly_zero_based_and_end_exclusive():
    visited = list(training_epoch_range(0, 40))

    assert visited == list(range(40))
    assert len(visited) == 40
    assert visited[0] == 0
    assert visited[-1] == 39
    assert 40 not in visited


def test_existing_multistep_and_cosine_builders_still_create():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    multistep_optimizer = torch.optim.Adam([parameter], lr=1e-4)
    multistep = _build_lr_scheduler(
        SimpleNamespace(
            scheduler_type="multistep",
            tot_epochs=40,
            lr_decay_epochs=[23, 27],
            lr_decay_rate=0.2,
        ),
        multistep_optimizer,
    )
    assert isinstance(multistep, torch.optim.lr_scheduler.MultiStepLR)

    cosine_parameter = torch.nn.Parameter(torch.tensor(1.0))
    cosine_optimizer = torch.optim.Adam([cosine_parameter], lr=1e-4)
    cosine = _build_lr_scheduler(
        SimpleNamespace(
            scheduler_type="cosine",
            tot_epochs=40,
            scheduler_warmup_epochs=0,
            scheduler_eta_min=1e-7,
        ),
        cosine_optimizer,
    )
    assert isinstance(cosine, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_lr_stage_must_match_unlabeled_stage_in_strict_mode():
    optimizer, _ = _new_optimizer_and_scheduler()
    config = _two_stage_config(unlabeled_start_epoch=21)

    with pytest.raises(ValueError, match="sup_only_train_epoch=20"):
        _build_lr_scheduler(config, optimizer)


def test_epoch_start_logging_and_stage_boundary_checkpoint_policy():
    optimizer, scheduler = _new_optimizer_and_scheduler()
    trainer = object.__new__(SemiSupervisedTrainer)
    trainer.config = _two_stage_config()
    trainer.model_optimizer = optimizer
    trainer.model_lr_scheduler = scheduler
    trainer.logger = _RecordingLogger()

    for epoch in (0, 19, 20, 39):
        trainer._step_lr_before_epoch(epoch)

    log_text = "\n".join(trainer.logger.messages)
    assert "LR before epoch: epoch=0, stage=1, lr=1.000e-04" in log_text
    assert "LR before epoch: epoch=19, stage=1, lr=1.000e-07" in log_text
    assert "Two-stage cosine restart: epoch=20" in log_text
    assert "previous_lr=1.000e-07" in log_text
    assert "restart_lr=1.000e-04" in log_text
    assert "optimizer_state_preserved=True" in log_text
    assert "LR before epoch: epoch=39, stage=2, lr=1.000e-07" in log_text

    assert trainer._should_save_checkpoint(19, 40)
    assert trainer._should_save_checkpoint(20, 40)
    assert not trainer._should_save_checkpoint(27, 40)
    assert trainer._should_save_checkpoint(28, 40)

    meta = trainer._lr_schedule_meta(40)
    assert meta == {
        "scheduler_type": "two_stage_cosine",
        "tot_epochs": 40,
        "sup_only_train_epoch": 20,
        "active_stage": 2,
        "current_lr": pytest.approx([1e-7]),
    }


def test_optional_stage2_reset_replaces_optimizer_only_at_boundary():
    model = torch.nn.Linear(1, 1)
    config = _two_stage_config(
        preserve_optimizer_state_across_stages=False,
        reset_optimizer_at_stage2=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = _build_lr_scheduler(config, optimizer)
    scheduler.step_epoch(19)

    trainer = object.__new__(SemiSupervisedTrainer)
    trainer.config = config
    trainer.model = model
    trainer.model_optimizer = optimizer
    trainer.model_lr_scheduler = scheduler
    trainer.logger = _RecordingLogger()
    optimizer_id = id(optimizer)

    trainer._step_lr_before_epoch(20)

    assert id(trainer.model_optimizer) != optimizer_id
    assert trainer.model_lr_scheduler.current_stage == 2
    assert trainer.model_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert any(
        "optimizer_state_preserved=False" in message
        for message in trainer.logger.messages
    )


def test_legacy_scheduler_state_warns_and_reconstructs_from_checkpoint_epoch(
    monkeypatch,
    tmp_path,
):
    build_model_module = importlib.import_module("models.build_model")
    source_model = torch.nn.Linear(1, 1)
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=1e-4)
    legacy_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        source_optimizer,
        milestones=[10],
        gamma=0.1,
    )
    checkpoint_path = tmp_path / "legacy_scheduler.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "lr_scheduler": legacy_scheduler.state_dict(),
            "epoch": 19,
        },
        checkpoint_path,
    )

    monkeypatch.setattr(
        build_model_module,
        "build_model",
        lambda config: torch.nn.Linear(1, 1),
    )
    monkeypatch.setattr(
        build_model_module,
        "_attach_pc_hbm_if_enabled",
        lambda config, logger, device, model: None,
    )
    config = _two_stage_config(
        distributed_train=False,
        compile_model=False,
        precisionHigh=False,
        resume_training_state=True,
        checkpoint_load_strategy="strict",
    )
    logger = _RecordingLogger()

    _, _, scheduler, epoch_st = build_model_module.build_model_optimizers(
        config,
        logger,
        torch.device("cpu"),
        resume=str(checkpoint_path),
    )

    assert epoch_st == 20
    assert scheduler.last_epoch == 19
    assert scheduler.get_last_lr() == pytest.approx([1e-7])
    assert any(
        "LR scheduler state was not restored" in message
        for message in logger.messages
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"split_epoch": 0}, "0 < split_epoch < total_epochs"),
        ({"split_epoch": 40}, "0 < split_epoch < total_epochs"),
        ({"stage1_min_lr": 2e-4}, "stage1_min_lr"),
        ({"stage2_min_lr": 2e-4}, "stage2_min_lr"),
    ],
)
def test_scheduler_validates_boundaries_and_minimum_lr(kwargs, message):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    values = {
        "total_epochs": 40,
        "split_epoch": 20,
        "stage1_initial_lr": 1e-4,
        "stage1_min_lr": 1e-7,
        "stage2_initial_lr": 1e-4,
        "stage2_min_lr": 1e-7,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        TwoStageCosineLR(optimizer, **values)
