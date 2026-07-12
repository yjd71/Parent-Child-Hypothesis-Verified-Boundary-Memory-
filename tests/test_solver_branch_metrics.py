import os
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "data", "logs"), exist_ok=True)

import engine.solver as solver_module
from engine.solver import SemiSupervisedTrainer, optimizer_grad_l2_norm


def _meter_trainer():
    trainer = SemiSupervisedTrainer.__new__(SemiSupervisedTrainer)
    trainer.config = SimpleNamespace(distributed_train=False)
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    trainer.model_optimizer = torch.optim.Adam([parameter], lr=1e-4)
    trainer._reset_epoch_meters()
    return trainer


def test_optimizer_grad_l2_norm_uses_all_parameter_groups():
    first = torch.nn.Parameter(torch.tensor(0.0))
    second = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam(
        [{"params": [first]}, {"params": [second], "lr": 5e-5}],
        lr=1e-4,
    )
    first.grad = torch.tensor(3.0)
    second.grad = torch.tensor(4.0)
    assert optimizer_grad_l2_norm(optimizer) == pytest.approx(5.0)


def test_branch_epoch_meters_are_separate_and_reset_each_epoch():
    trainer = _meter_trainer()
    trainer._record_branch_update("Sup", 1.0, 2, 3.0)
    trainer._record_branch_update("Sup", 3.0, 1, 5.0)
    trainer._record_branch_update("Unsup", 10.0, 4, 2.0)
    trainer.unsup_full_forward_meter.update(1.0)
    trainer.unsup_full_forward_meter.update(0.0)

    summary = trainer._build_epoch_summary(enable_unsup=True)
    assert summary["sup_loss"] == pytest.approx(5.0 / 3.0)
    assert summary["sup_grad_norm"] == pytest.approx(4.0)
    assert summary["unsup_loss"] == pytest.approx(10.0)
    assert summary["unsup_grad_norm"] == pytest.approx(2.0)
    assert summary["unsup_full_forward_ratio"] == pytest.approx(0.5)

    trainer._reset_epoch_meters()
    reset_summary = trainer._build_epoch_summary(enable_unsup=False)
    assert reset_summary["sup_loss"] is None
    assert "unsup_loss" not in reset_summary


def test_full_student_forward_runs_every_four_unsupervised_batches():
    trainer = SemiSupervisedTrainer.__new__(SemiSupervisedTrainer)
    trainer.config = SimpleNamespace(pc_hbm_unsup_full_forward_interval=4)
    assert [trainer._use_full_unsup_student(i) for i in range(10)] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
    ]

    trainer.config.pc_hbm_unsup_full_forward_interval = -1
    with pytest.raises(ValueError, match="must be non-negative"):
        trainer._use_full_unsup_student(0)


def test_train_batch_returns_branch_local_metrics_and_gradient_norms():
    trainer = SemiSupervisedTrainer.__new__(SemiSupervisedTrainer)
    trainer.config = SimpleNamespace(
        out_ref=False,
        distributed_train=False,
        use_pc_hbm=False,
        log_branch_grad_norms=True,
    )
    trainer.device = torch.device("cpu")
    trainer.pc_hbm = None
    trainer.model = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
    with torch.no_grad():
        trainer.model.weight.fill_(1.0)
    trainer.model_optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-4)
    trainer.pix_loss = lambda prediction, target: (prediction - target).square().mean()
    trainer._reset_epoch_meters()
    batch = (torch.ones(2, 1, 2, 2), torch.zeros(2, 1, 2, 2))

    sup_metrics = trainer._train_batch(batch, branch_name="Sup")
    sup_metrics["sup_only_sentinel"] = 1.0
    unsup_metrics = trainer._train_batch(batch, branch_name="Unsup")

    assert "sup_only_sentinel" not in unsup_metrics
    assert sup_metrics["grad_norm"] > 0
    assert unsup_metrics["grad_norm"] > 0
    assert trainer.epoch_meters["Sup"]["loss"].count == 2
    assert trainer.epoch_meters["Unsup"]["loss"].count == 2


def test_epoch_summary_logs_unsupervised_as_disabled(monkeypatch):
    class RecordingLogger:
        def __init__(self):
            self.messages = []

        def key_info(self, message):
            self.messages.append(message)

    wandb_calls = []
    monkeypatch.setattr(
        solver_module.wandb,
        "log",
        lambda metrics, step: wandb_calls.append((metrics, step)),
    )
    trainer = SemiSupervisedTrainer.__new__(SemiSupervisedTrainer)
    trainer.config = SimpleNamespace(distributed_train=False)
    trainer.logger = RecordingLogger()
    trainer.global_step = 7
    summary = {
        "sup_loss": 1.25,
        "sup_grad_norm": 3.5,
        "unsup_enabled": False,
    }

    trainer._log_epoch_summary(3, summary)

    assert "Unsup=disabled" in trainer.logger.messages[-1]
    assert wandb_calls == [
        (
            {"Epoch/Sup-loss": 1.25, "Epoch/Sup-grad_norm": 3.5},
            7,
        )
    ]
