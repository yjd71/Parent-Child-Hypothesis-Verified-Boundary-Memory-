import importlib.util
import sys
import types

import torch
import torch.nn as nn

from CBM import build_cbm_pfi
from CBM.config.defaults import apply_cbm_defaults
from CBM.config.schedule import cbm_enabled_for_epoch, cbm_should_rebuild_memory, cbm_stage_id, cbm_unlabeled_enabled
from CBM.memory.bank import DenseBoundaryMemory
from CBM.memory.builder import LabeledMemoryBuilder


class Config:
    cbm_pfi_enable = True
    cbm_print_diagnostics = False
    lateral_channels_in_collection = [16, 8, 4, 2]


def _config():
    return apply_cbm_defaults(Config())


def _square_gt(batch_size=2, height=16, width=16):
    gt = torch.zeros(batch_size, 1, height, width)
    gt[:, :, 4:12, 4:12] = 1.0
    return gt


def test_cbm_schedule_uses_epoch_plus_one_mapping():
    config = _config()

    assert [cbm_stage_id(config, epoch) for epoch in range(0, 5)] == [1, 1, 1, 1, 1]
    assert [cbm_stage_id(config, epoch) for epoch in range(5, 15)] == [2] * 10
    assert [cbm_stage_id(config, epoch) for epoch in range(15, 30)] == [3] * 15

    assert not cbm_should_rebuild_memory(config, 4)
    assert cbm_should_rebuild_memory(config, 5)
    assert not cbm_unlabeled_enabled(config, 14)
    assert cbm_unlabeled_enabled(config, 15)
    assert not cbm_enabled_for_epoch(config, 4, memory_ready=True)
    assert cbm_enabled_for_epoch(config, 5, memory_ready=True)


class FakeMemoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def extract_cbm_memory_features(self, inputs, ema=True):
        del ema
        bsz = inputs.size(0)
        x3 = torch.ones(bsz, 8, 2, 2, device=inputs.device) * self.weight
        p3 = torch.ones(bsz, 4, 4, 4, device=inputs.device) * self.weight
        return {"x3": x3, "p3": p3}


def test_labeled_memory_builder_rebuilds_memory_from_loader():
    memory = DenseBoundaryMemory(
        sample_per_image={"fg_core": 2, "fg_boundary": 2, "bg_near": 2, "bg_far": 2},
        max_sizes={"fg_core": 16, "fg_boundary": 16, "bg_near": 16, "bg_far": 16},
        print_diagnostics=False,
    )
    builder = LabeledMemoryBuilder(memory)
    model = FakeMemoryModel()
    loader = [
        (
            torch.randn(2, 3, 16, 16),
            _square_gt(2),
            ["img-a", "img-b"],
            torch.tensor([0, 1]),
        )
    ]

    builder.prepare_epoch(model, loader, epoch=5)

    assert memory.is_ready()
    image_keys, image_ids = memory.get_image_keys()
    assert image_keys.shape == (2, 128)
    assert image_ids == ["img-a", "img-b"]


class FailingMemoryModel(FakeMemoryModel):
    def extract_cbm_memory_features(self, inputs, ema=True):
        raise RuntimeError("feature extraction failed")


def test_cbm_engine_memory_build_failure_falls_back_to_empty_memory():
    config = _config()
    cbm = build_cbm_pfi(config, device=torch.device("cpu"), logger=None)
    loader = [(torch.randn(1, 3, 16, 16), _square_gt(1), ["bad"], torch.tensor([0]))]

    cbm.prepare_epoch(FailingMemoryModel(), loader, epoch=5)

    assert not cbm.memory.is_ready()
    assert cbm.state.memory_build_failed
    assert "feature extraction failed" in cbm.state.memory_build_error
    assert not cbm.enabled_for_epoch(5)


def _load_solver_with_stubs(monkeypatch):
    wandb = types.ModuleType("wandb")
    wandb.log = lambda *args, **kwargs: None
    wandb.run = types.SimpleNamespace(step=0)
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    data = types.ModuleType("data")
    data.prepare_dataloader = lambda *args, **kwargs: None
    data.prepare_labeled_memory_dataloader = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "data", data)

    utils = types.ModuleType("utils")

    class AverageMeter:
        def __init__(self):
            self.avg = 0.0

        def update(self, value, n=1):
            del n
            self.avg = float(value)

    utils.AverageMeter = AverageMeter
    utils.retry_if_cuda_oom = lambda fn: fn
    monkeypatch.setitem(sys.modules, "utils", utils)

    engine_pkg = types.ModuleType("engine")
    engine_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "engine", engine_pkg)

    loss_mod = types.ModuleType("engine.loss")

    class PixLoss(nn.Module):
        def __init__(self, config):
            super().__init__()

        def forward(self, scaled_preds, gt):
            return ((scaled_preds[-1] - gt) ** 2).mean()

    loss_mod.PixLoss = PixLoss
    monkeypatch.setitem(sys.modules, "engine.loss", loss_mod)

    evaluator_mod = types.ModuleType("engine.evaluator")
    evaluator_mod.Evaluator = object
    monkeypatch.setitem(sys.modules, "engine.evaluator", evaluator_mod)

    spec = importlib.util.spec_from_file_location("engine.solver", "engine/solver.py")
    solver = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "engine.solver", solver)
    spec.loader.exec_module(solver)
    return solver


class TrainerConfig:
    out_ref = False
    distributed_train = False
    sup_only_train_epoch = 15
    cbm_pfi_enable = True
    cbm_unsup_loss_alpha = 0.1
    cbm_vis_enable = False
    cbm_vis_interval = 1
    cbm_vis_max_images = 1
    cbm_vis_labeled_only = True
    cbm_vis_dir = None


class FakeTrainerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.cbm = None
        self.last_use_memory = None

    def forward(self, inputs, use_memory=None, return_aux=False, ema=False):
        del ema
        self.last_use_memory = use_memory
        pred = inputs[:, :1] * self.scale
        bsz, _, height, width = pred.shape
        aux = {
            "cbm_used": bool(use_memory),
            "gate_mean": 0.25,
            "valid_ratio": 0.5,
            "u_mean": 0.125,
            "num_memory_tokens": 7,
            "prob3": torch.sigmoid(pred),
            "B_query": torch.sigmoid(pred),
            "Y_map": torch.rand(bsz, 8, height, width, device=pred.device),
            "U_map": torch.rand(bsz, 1, height, width, device=pred.device),
            "cons_map": torch.rand(bsz, 1, height, width, device=pred.device),
            "gate3": torch.sigmoid(pred),
            "p_main": torch.sigmoid(pred),
            "p_final": torch.sigmoid(pred + 0.1),
        }
        if return_aux:
            return [pred], aux
        return [pred]


class FakeMemory:
    def __init__(self, ready=True):
        self.ready = ready

    def is_ready(self):
        return self.ready

    def diagnostic_string(self):
        return "[CBM] fake memory"


class FakeCBM:
    def __init__(self, ready=True):
        self.memory = FakeMemory(ready)
        self.state = types.SimpleNamespace(loss_dict={})

    def compute_losses(self, aux, gt):
        del aux
        loss = gt.mean() * 0.0 + 0.1
        self.state.loss_dict = {"loss_cbm_total": float(loss.detach().item())}
        return loss

    def enabled_for_epoch(self, epoch=None):
        return self.memory.is_ready()


def test_trainer_rebuilds_cbm_from_memory_labeled_loader(monkeypatch):
    solver = _load_solver_with_stubs(monkeypatch)
    trainer = solver.SemiSupervisedTrainer((None, {}), TrainerConfig(), torch.device("cpu"), logger=None)
    trainer.model = object()
    trainer.labeled_dataloader = object()
    trainer.memory_labeled_dataloader = object()

    calls = []

    class PreparingCBM(FakeCBM):
        def prepare_epoch(self, model, loader, epoch):
            calls.append((model, loader, epoch))

    trainer.cbm = PreparingCBM(ready=True)
    monkeypatch.setattr(solver, "cbm_should_rebuild_memory", lambda config, epoch: True)
    monkeypatch.setattr(solver, "cbm_stage_id", lambda config, epoch: 2)
    monkeypatch.setattr(solver, "cbm_stage_epoch", lambda config, epoch: epoch)
    monkeypatch.setattr(solver, "cbm_stage_name", lambda config, epoch: "memory")

    trainer._prepare_cbm_epoch(5)

    assert calls == [(trainer.model, trainer.memory_labeled_dataloader, 5)]


def test_trainer_train_batch_merges_cbm_loss_and_diagnostics(monkeypatch):
    solver = _load_solver_with_stubs(monkeypatch)
    trainer = solver.SemiSupervisedTrainer((None, {}), TrainerConfig(), torch.device("cpu"), logger=None)
    model = FakeTrainerModel()
    cbm = FakeCBM(ready=True)
    model.cbm = cbm
    trainer.model = model
    trainer.cbm = cbm
    trainer.model_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.loss_dict = {}
    batch = (torch.ones(2, 3, 4, 4), torch.zeros(2, 1, 4, 4))

    trainer._train_batch(batch, use_memory=True, enable_cbm_loss=True, branch_name="Sup")

    assert model.last_use_memory is True
    assert abs(trainer.loss_dict["loss_cbm_total"] - 0.1) < 1e-6
    assert trainer.loss_dict["gate_mean"] == 0.25
    assert trainer.loss_dict["valid_ratio"] == 0.5
    assert trainer.loss_dict["retrieval_uncertainty_mean"] == 0.125


def test_trainer_train_batch_saves_cbm_visualizations_without_breaking_backward(monkeypatch, tmp_path):
    solver = _load_solver_with_stubs(monkeypatch)

    class VisualConfig(TrainerConfig):
        cbm_vis_enable = True
        cbm_vis_dir = str(tmp_path)

    trainer = solver.SemiSupervisedTrainer((None, {}), VisualConfig(), torch.device("cpu"), logger=None)
    model = FakeTrainerModel()
    cbm = FakeCBM(ready=True)
    model.cbm = cbm
    trainer.model = model
    trainer.cbm = cbm
    trainer.model_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.loss_dict = {}
    trainer.current_epoch = 2
    batch = (torch.ones(1, 3, 4, 4), torch.zeros(1, 1, 4, 4), ["vis-img"])

    trainer._train_batch(batch, use_memory=True, enable_cbm_loss=True, branch_name="Sup")

    files = list(tmp_path.glob("*.png"))
    assert len(files) == 11
    assert any("epoch002_iter000000_Sup_vis-img_p_final.png" in str(path) for path in files)
    assert model.scale.grad is not None


def test_trainer_unsup_log_hides_stale_cbm_loss_fields(monkeypatch):
    solver = _load_solver_with_stubs(monkeypatch)
    trainer = solver.SemiSupervisedTrainer((None, {}), TrainerConfig(), torch.device("cpu"), logger=None)
    trainer.loss_dict = {
        "loss_pix": 1.122,
        "loss_gdt": 1.386,
        "loss_cbm_mem": 1.125,
        "loss_cbm_total": 1.292,
        "raw_cbm_L_mem_ce": 5.627,
        "cbm_stage": 3.0,
        "memory_ready": 1.0,
        "gate_mean": 0.062,
        "valid_ratio": 0.138,
        "retrieval_uncertainty_mean": 0.079,
        "memory_tokens": 282.0,
    }

    sup_info = trainer._format_loss_info("Semi-Supervised Training Losses")
    unsup_info = trainer._format_loss_info("Unsueprvised Training Losses", include_cbm_losses=False)

    assert "loss_cbm_mem: 1.125" in sup_info
    assert "loss_cbm_total: 1.292" in sup_info
    assert "raw_cbm_L_mem_ce: 5.627" in sup_info

    assert "loss_cbm_mem" not in unsup_info
    assert "loss_cbm_total" not in unsup_info
    assert "raw_cbm_L_mem_ce" not in unsup_info
    assert "loss_pix: 1.122" in unsup_info
    assert "loss_gdt: 1.386" in unsup_info
    assert "cbm_stage: 3.000" in unsup_info
    assert "memory_ready: 1.000" in unsup_info
    assert "gate_mean: 0.062" in unsup_info
    assert "valid_ratio: 0.138" in unsup_info
    assert "retrieval_uncertainty_mean: 0.079" in unsup_info
    assert "memory_tokens: 282.000" in unsup_info
