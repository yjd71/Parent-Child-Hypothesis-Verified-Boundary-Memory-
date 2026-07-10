from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PC_HBM.core.pc_config import PC_HBM_DEFAULTS
import PC_HBM.training.pc_losses as pc_losses
from PC_HBM.training.pc_losses import (
    _build_hard_teacher_target,
    _build_hard_teacher_valid_mask,
    _hard_teacher_ramp_factor,
    _rsbl_hard_structure_loss,
    _teacher_edge_weight,
    compute_pc_hbm_unlabeled_loss,
)
from utils.solver_logging import partition_training_metrics


def _config(
    *,
    lambda_u: float = 1.0,
    hard_weight: float = 1.0,
    threshold: float = 0.5,
    use_hard_teacher_loss: bool = True,
    use_soft_teacher_weighted_iou: bool = True,
    soft_teacher_weighted_iou_weight: float = 0.25,
    foreground_threshold: float = 0.7,
    background_threshold: float = 0.3,
    confidence_threshold: float = 0.25,
    hard_rampup_epochs: int = 3,
    unlabeled_start_epoch: int = 16,
) -> SimpleNamespace:
    return SimpleNamespace(
        lambda_u=lambda_u,
        use_hard_teacher_loss=use_hard_teacher_loss,
        hard_teacher_loss_weight=hard_weight,
        hard_teacher_threshold=threshold,
        hard_teacher_foreground_threshold=foreground_threshold,
        hard_teacher_background_threshold=background_threshold,
        hard_teacher_confidence_threshold=confidence_threshold,
        hard_teacher_rampup_epochs=hard_rampup_epochs,
        unlabeled_start_epoch=unlabeled_start_epoch,
        use_soft_teacher_weighted_iou=use_soft_teacher_weighted_iou,
        soft_teacher_weighted_iou_weight=soft_teacher_weighted_iou_weight,
        pc_hbm_unsup_final_consistency_weight=0.0,
    )


def _reference_rsbl_structure_loss(
    logits: torch.Tensor,
    hard_target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones_like(hard_target)
    valid = (hard_target * valid_mask).sum(dim=(1, 2, 3)) > 0
    if not torch.any(valid):
        return logits.sum() * 0.0
    logits = logits[valid]
    hard_target = hard_target[valid].to(dtype=logits.dtype)
    valid_mask = valid_mask[valid].to(dtype=logits.dtype)
    weight = valid_mask * (1.0 + 5.0 * torch.abs(
        F.avg_pool2d(
            hard_target,
            kernel_size=31,
            stride=1,
            padding=15,
            count_include_pad=False,
        ) - hard_target
    ))
    weighted_bce = F.binary_cross_entropy_with_logits(logits, hard_target, reduction="none")
    weighted_bce = (weight * weighted_bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3)).clamp_min(1.0)
    probability = torch.sigmoid(logits)
    intersection = ((probability * hard_target) * weight).sum(dim=(2, 3))
    union = ((probability + hard_target) * weight).sum(dim=(2, 3))
    weighted_iou = 1.0 - (intersection + 1.0) / (union - intersection + 1.0)
    return (weighted_bce + weighted_iou).mean()


def _reference_confidence_aware_soft_weighted_iou(
    logits: torch.Tensor,
    soft_target: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    sample_confidence = confidence.mean(dim=(1, 2, 3))
    valid = sample_confidence > 0
    if not torch.any(valid):
        return logits.sum() * 0.0
    logits = logits[valid]
    soft_target = soft_target[valid]
    confidence = confidence[valid]
    sample_confidence = sample_confidence[valid]
    edge_weight = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(
            soft_target,
            kernel_size=31,
            stride=1,
            padding=15,
            count_include_pad=False,
        ) - soft_target
    )
    weight = confidence * edge_weight
    probability = torch.sigmoid(logits)
    intersection = ((probability * soft_target) * weight).sum(dim=(2, 3))
    union = (
        (probability.square() + soft_target.square() - probability * soft_target) * weight
    ).sum(dim=(2, 3))
    per_sample = (1.0 - (intersection + 1.0) / (union + 1.0)).mean(dim=1)
    return (per_sample * sample_confidence).sum() / sample_confidence.sum()


def test_rsbl_structure_loss_matches_reference():
    logits = torch.linspace(-2.0, 2.0, 33 * 33, dtype=torch.float64).reshape(1, 1, 33, 33)
    hard_target = torch.zeros_like(logits)
    hard_target[:, :, 8:25, 9:24] = 1.0

    actual = _rsbl_hard_structure_loss(logits, hard_target)
    expected = _reference_rsbl_structure_loss(logits, hard_target)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_hard_target_uses_strict_gt_threshold():
    pseudo = torch.tensor([0.49, 0.50, 0.51], dtype=torch.float64).reshape(1, 1, 1, 3)
    actual = _build_hard_teacher_target(pseudo, threshold=0.5, dtype=torch.float64)
    expected = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).reshape(1, 1, 1, 3)

    assert torch.equal(actual, expected)


def test_hard_valid_mask_uses_double_threshold_and_confidence_boundary():
    pseudo = torch.tensor([0.29, 0.30, 0.31, 0.69, 0.70, 0.71], dtype=torch.float64).reshape(1, 1, 1, 6)
    confidence = torch.tensor([0.25, 0.24, 0.90, 0.90, 0.25, 0.24], dtype=torch.float64).reshape(1, 1, 1, 6)

    actual = _build_hard_teacher_valid_mask(
        pseudo,
        confidence,
        background_threshold=0.3,
        foreground_threshold=0.7,
        confidence_threshold=0.25,
        dtype=torch.float64,
    )
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64).reshape(1, 1, 1, 6)

    assert torch.equal(actual, expected)


def test_teacher_edge_weight_has_no_constant_target_border_artifact():
    for value in (0.0, 0.2, 1.0):
        target = torch.full((1, 1, 33, 33), value, dtype=torch.float64)
        assert torch.allclose(_teacher_edge_weight(target), torch.ones_like(target), atol=1e-12, rtol=0.0)


def test_hard_loss_empty_batch_is_differentiable_zero():
    logits = torch.randn(2, 1, 33, 33, requires_grad=True)
    hard_target = torch.zeros_like(logits)

    loss = _rsbl_hard_structure_loss(logits, hard_target)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0


def test_hard_loss_filters_empty_samples_without_batch_dilution():
    torch.manual_seed(7)
    logits = torch.randn(2, 1, 33, 33, dtype=torch.float64, requires_grad=True)
    hard_target = torch.zeros_like(logits)
    hard_target[1, :, 10:23, 11:22] = 1.0

    actual = _rsbl_hard_structure_loss(logits, hard_target)
    expected = _reference_rsbl_structure_loss(logits.detach()[1:], hard_target[1:])
    actual.backward()

    assert torch.allclose(actual.detach(), expected, atol=1e-10, rtol=1e-10)
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0]).item() == 0
    assert torch.count_nonzero(logits.grad[1]).item() > 0


def test_masked_hard_loss_matches_reference_and_ignored_pixels_have_no_gradient():
    torch.manual_seed(13)
    logits = torch.randn(1, 1, 33, 33, dtype=torch.float64, requires_grad=True)
    hard_target = torch.zeros_like(logits)
    hard_target[:, :, 10:23, 11:22] = 1.0
    valid_mask = torch.zeros_like(logits)
    valid_mask[:, :, 12:21, 13:20] = 1.0
    valid_mask[:, :, :6, :6] = 1.0

    actual = _rsbl_hard_structure_loss(logits, hard_target, valid_mask)
    expected = _reference_rsbl_structure_loss(logits.detach(), hard_target, valid_mask)
    actual.backward()

    ignored = valid_mask == 0
    reliable_foreground = (valid_mask > 0) & (hard_target > 0)
    reliable_background = (valid_mask > 0) & (hard_target == 0)
    assert torch.allclose(actual.detach(), expected, atol=1e-10, rtol=1e-10)
    assert torch.count_nonzero(logits.grad[ignored]).item() == 0
    assert torch.count_nonzero(logits.grad[reliable_foreground]).item() > 0
    assert torch.count_nonzero(logits.grad[reliable_background]).item() > 0


def test_soft_teacher_loss_keeps_fractional_targets_and_total_formula():
    logits = torch.linspace(-1.5, 1.5, 33 * 33, dtype=torch.float64).reshape(1, 1, 33, 33)
    logits.requires_grad_(True)
    pseudo = torch.full_like(logits, 0.2)
    pseudo[:, :, 16:, :] = 0.8
    confidence = torch.ones_like(pseudo)
    config = _config(lambda_u=0.25, hard_weight=1.0, use_soft_teacher_weighted_iou=False)

    total, log = compute_pc_hbm_unlabeled_loss(
        {"z_nomix": logits, "z_main": logits, "forward_mode": "student_core", "mixture_skipped": True},
        pseudo,
        confidence,
        config,
    )

    hard_target = (pseudo > 0.5).to(dtype=logits.dtype)
    expected_soft = F.binary_cross_entropy_with_logits(logits.detach(), pseudo)
    thresholded_bce = F.binary_cross_entropy_with_logits(logits.detach(), hard_target)
    expected_hard = _reference_rsbl_structure_loss(logits.detach(), hard_target)
    expected_total = 0.25 * expected_soft + expected_hard

    assert not torch.allclose(expected_soft, thresholded_bce)
    assert torch.allclose(log["L_u"], expected_soft)
    assert torch.allclose(log["soft_teacher_loss"], expected_soft)
    assert torch.allclose(log["soft_teacher_bce"], expected_soft)
    assert log["soft_teacher_weighted_iou"].item() == 0.0
    assert torch.allclose(log["hard_teacher_loss"], expected_hard)
    assert torch.allclose(log["hard_teacher_weighted_loss"], expected_hard)
    assert log["hard_teacher_effective_weight"].item() == 1.0
    assert torch.allclose(log["loss_u_total"], expected_total)
    assert torch.allclose(total.detach(), expected_total)


def test_soft_weighted_iou_uses_continuous_targets_confidence_and_lambda_u():
    logits = torch.linspace(-1.2, 1.4, 33 * 33, dtype=torch.float64).reshape(1, 1, 33, 33)
    logits.requires_grad_(True)
    pseudo = torch.linspace(0.1, 0.9, 33 * 33, dtype=torch.float64).reshape(1, 1, 33, 33)
    confidence = torch.linspace(0.2, 1.0, 33 * 33, dtype=torch.float64).reshape(1, 1, 33, 33)
    config = _config(
        lambda_u=0.5,
        hard_weight=0.0,
        use_soft_teacher_weighted_iou=True,
        soft_teacher_weighted_iou_weight=0.25,
    )

    total, log = compute_pc_hbm_unlabeled_loss(
        {"z_nomix": logits, "forward_mode": "student_core", "mixture_skipped": True},
        pseudo,
        confidence,
        config,
    )

    expected_bce = (
        F.binary_cross_entropy_with_logits(logits.detach(), pseudo, reduction="none") * confidence
    ).sum() / confidence.sum()
    expected_iou = _reference_confidence_aware_soft_weighted_iou(
        logits.detach(), pseudo, confidence
    )
    expected_soft = expected_bce + 0.25 * expected_iou
    expected_total = 0.5 * expected_soft

    assert torch.allclose(log["soft_teacher_bce"], expected_bce, atol=1e-10, rtol=1e-10)
    assert torch.allclose(log["soft_teacher_weighted_iou"], expected_iou, atol=1e-10, rtol=1e-10)
    assert torch.allclose(log["soft_teacher_loss"], expected_soft, atol=1e-10, rtol=1e-10)
    assert torch.allclose(log["loss_u_total"], expected_total, atol=1e-10, rtol=1e-10)
    assert torch.allclose(total.detach(), expected_total, atol=1e-10, rtol=1e-10)


def test_probability_preserving_soft_iou_is_zero_with_zero_gradient_at_teacher_probability():
    soft_target = torch.linspace(0.05, 0.95, 33 * 33, dtype=torch.float64).reshape(1, 1, 33, 33)
    confidence = torch.linspace(0.2, 1.0, 33 * 33, dtype=torch.float64).reshape(1, 1, 33, 33)
    logits = torch.logit(soft_target).requires_grad_(True)

    loss = pc_losses._confidence_aware_soft_weighted_iou_loss(logits, soft_target, confidence)
    loss.backward()

    assert torch.allclose(loss.detach(), torch.zeros_like(loss), atol=1e-12, rtol=0.0)
    assert logits.grad is not None
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad), atol=1e-12, rtol=0.0)


def test_soft_iou_sample_confidence_aggregation_ignores_zero_confidence_samples():
    logits = torch.linspace(-1.0, 1.0, 33 * 33, dtype=torch.float64).reshape(1, 1, 33, 33)
    soft_target = torch.linspace(0.1, 0.9, 33 * 33, dtype=torch.float64).reshape(1, 1, 33, 33)
    confidence = torch.full_like(soft_target, 0.6)

    single = pc_losses._confidence_aware_soft_weighted_iou_loss(logits, soft_target, confidence)
    batched = pc_losses._confidence_aware_soft_weighted_iou_loss(
        torch.cat((logits, logits + 0.5), dim=0),
        torch.cat((soft_target, soft_target), dim=0),
        torch.cat((confidence, torch.zeros_like(confidence)), dim=0),
    )

    assert torch.allclose(batched, single, atol=1e-12, rtol=1e-12)

    second_logits = logits + 0.5
    second_confidence = torch.full_like(confidence, 0.2)
    weighted_batch = pc_losses._confidence_aware_soft_weighted_iou_loss(
        torch.cat((logits, second_logits), dim=0),
        torch.cat((soft_target, soft_target), dim=0),
        torch.cat((confidence, second_confidence), dim=0),
    )
    expected = _reference_confidence_aware_soft_weighted_iou(
        torch.cat((logits, second_logits), dim=0),
        torch.cat((soft_target, soft_target), dim=0),
        torch.cat((confidence, second_confidence), dim=0),
    )
    assert torch.allclose(weighted_batch, expected, atol=1e-12, rtol=1e-12)


def test_soft_iou_all_zero_confidence_is_differentiable_zero():
    logits = torch.randn(2, 1, 33, 33, requires_grad=True)
    soft_target = torch.rand_like(logits)
    confidence = torch.zeros_like(logits)

    loss = pc_losses._confidence_aware_soft_weighted_iou_loss(logits, soft_target, confidence)
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0


def test_teacher_targets_detached_and_student_gradients_routed():
    torch.manual_seed(11)
    z_nomix = torch.randn(1, 1, 33, 33, requires_grad=True)
    z_main = torch.randn(1, 1, 33, 33, requires_grad=True)
    z_final = torch.randn(1, 1, 33, 33, requires_grad=True)
    pseudo = torch.linspace(0.1, 0.9, 33 * 33).reshape(1, 1, 33, 33).requires_grad_(True)
    confidence = torch.full_like(pseudo, 0.75, requires_grad=True)

    total, _ = compute_pc_hbm_unlabeled_loss(
        {
            "z_nomix": z_nomix,
            "z_main": z_main,
            "z_final": z_final,
            "forward_mode": "student_core",
            "mixture_skipped": True,
        },
        pseudo,
        confidence,
        _config(),
    )
    total.backward()

    assert z_nomix.grad is not None
    assert torch.count_nonzero(z_nomix.grad).item() > 0
    assert z_main.grad is None
    assert z_final.grad is None
    assert pseudo.grad is None
    assert confidence.grad is None


def test_hard_teacher_linear_ramp_and_total_formula():
    config = _config(lambda_u=0.0, hard_weight=1.0, unlabeled_start_epoch=16, hard_rampup_epochs=3)
    assert _hard_teacher_ramp_factor(config, 15) == 0.0
    assert _hard_teacher_ramp_factor(config, 16) == pytest.approx(1.0 / 3.0)
    assert _hard_teacher_ramp_factor(config, 17) == pytest.approx(2.0 / 3.0)
    assert _hard_teacher_ramp_factor(config, 18) == 1.0
    assert _hard_teacher_ramp_factor(config, 30) == 1.0
    assert _hard_teacher_ramp_factor(config, None) == 1.0
    assert _hard_teacher_ramp_factor(_config(hard_rampup_epochs=0), 16) == 1.0

    z_student = torch.zeros(1, 1, 33, 33, dtype=torch.float64, requires_grad=True)
    pseudo = torch.zeros_like(z_student)
    pseudo[:, :, 10:23, 11:22] = 0.9
    confidence = torch.ones_like(z_student)
    total, log = compute_pc_hbm_unlabeled_loss(
        {"z_nomix": z_student, "forward_mode": "student_core", "mixture_skipped": True},
        pseudo,
        confidence,
        config,
        epoch=16,
    )

    assert log["hard_teacher_ramp_factor"].item() == pytest.approx(1.0 / 3.0)
    assert log["hard_teacher_effective_weight"].item() == pytest.approx(1.0 / 3.0)
    assert torch.allclose(log["hard_teacher_weighted_loss"], log["hard_teacher_loss"] / 3.0)
    assert torch.allclose(total.detach(), log["hard_teacher_weighted_loss"])


def test_invalid_hard_teacher_threshold_order_raises():
    z_student = torch.zeros(1, 1, 33, 33, requires_grad=True)
    with pytest.raises(ValueError, match="0 <= background < hard < foreground <= 1"):
        compute_pc_hbm_unlabeled_loss(
            {"z_nomix": z_student, "forward_mode": "student_core", "mixture_skipped": True},
            torch.full_like(z_student, 0.9),
            torch.ones_like(z_student),
            _config(background_threshold=0.6),
        )


def test_disabled_hard_teacher_loss_skips_target_and_loss_computation():
    original_target_builder = pc_losses._build_hard_teacher_target
    original_valid_mask_builder = pc_losses._build_hard_teacher_valid_mask
    original_structure_loss = pc_losses._rsbl_hard_structure_loss

    def _unexpected_call(*args, **kwargs):
        raise AssertionError("hard teacher branch must be skipped when disabled")

    pc_losses._build_hard_teacher_target = _unexpected_call
    pc_losses._build_hard_teacher_valid_mask = _unexpected_call
    pc_losses._rsbl_hard_structure_loss = _unexpected_call
    try:
        for config, epoch in (
            (_config(use_hard_teacher_loss=False), None),
            (_config(hard_weight=0.0), None),
            (_config(unlabeled_start_epoch=16), 15),
        ):
            z_student = torch.randn(1, 1, 33, 33, requires_grad=True)
            pseudo = torch.full_like(z_student, 0.9)
            confidence = torch.zeros_like(z_student)
            total, log = compute_pc_hbm_unlabeled_loss(
                {"z_nomix": z_student, "forward_mode": "student_core", "mixture_skipped": True},
                pseudo,
                confidence,
                config,
                epoch=epoch,
            )
            total.backward()
            assert total.item() == 0.0
            assert log["hard_teacher_loss"].item() == 0.0
            assert log["hard_teacher_effective_weight"].item() == 0.0
            assert log["loss_u_total"].item() == 0.0
            assert z_student.grad is not None
            assert torch.count_nonzero(z_student.grad).item() == 0
    finally:
        pc_losses._build_hard_teacher_target = original_target_builder
        pc_losses._build_hard_teacher_valid_mask = original_valid_mask_builder
        pc_losses._rsbl_hard_structure_loss = original_structure_loss


def test_disabled_soft_weighted_iou_skips_helper_and_keeps_only_soft_bce():
    original_soft_iou = pc_losses._confidence_aware_soft_weighted_iou_loss

    def _unexpected_call(*args, **kwargs):
        raise AssertionError("soft weighted-IoU helper must be skipped when disabled")

    pc_losses._confidence_aware_soft_weighted_iou_loss = _unexpected_call
    try:
        for config in (
            _config(hard_weight=0.0, use_soft_teacher_weighted_iou=False),
            _config(hard_weight=0.0, soft_teacher_weighted_iou_weight=0.0),
        ):
            z_student = torch.randn(1, 1, 33, 33, requires_grad=True)
            pseudo = torch.full_like(z_student, 0.7)
            confidence = torch.full_like(z_student, 0.6)
            total, log = compute_pc_hbm_unlabeled_loss(
                {"z_nomix": z_student, "forward_mode": "student_core", "mixture_skipped": True},
                pseudo,
                confidence,
                config,
            )
            expected_bce = F.binary_cross_entropy_with_logits(z_student.detach(), pseudo)
            total.backward()

            assert torch.allclose(total.detach(), expected_bce)
            assert torch.allclose(log["soft_teacher_bce"], expected_bce)
            assert log["soft_teacher_weighted_iou"].item() == 0.0
            assert z_student.grad is not None
            assert torch.count_nonzero(z_student.grad).item() > 0
    finally:
        pc_losses._confidence_aware_soft_weighted_iou_loss = original_soft_iou


def test_pc_hbm_config_sources_use_requested_teacher_loss_values():
    expected = {
        "lambda_u": 1.0,
        "use_hard_teacher_loss": True,
        "hard_teacher_loss_weight": 1.0,
        "hard_teacher_threshold": 0.5,
        "hard_teacher_foreground_threshold": 0.7,
        "hard_teacher_background_threshold": 0.3,
        "hard_teacher_confidence_threshold": 0.25,
        "hard_teacher_rampup_epochs": 3,
        "use_soft_teacher_weighted_iou": True,
        "soft_teacher_weighted_iou_weight": 0.25,
    }
    base = runpy.run_path(str(ROOT / "config" / "base" / "pc_hbm.py"))["PC_HBM_DEFAULTS"]
    for key, value in expected.items():
        assert PC_HBM_DEFAULTS[key] == value
        assert base[key] == value

    for relative_path in ("config/runs/run.py", "config/runs/finetune_27_cbm.py"):
        run_config = runpy.run_path(str(ROOT / relative_path))
        for key, value in expected.items():
            assert run_config[key] == value


def test_solver_passes_current_epoch_to_unlabeled_teacher_loss():
    solver_source = (ROOT / "engine" / "solver.py").read_text(encoding="utf-8")
    call_start = solver_source.index("loss_u, log = compute_pc_hbm_unlabeled_loss(")
    call_block = solver_source[call_start : call_start + 320]
    assert 'epoch=getattr(self, "current_epoch", None)' in call_block


def test_teacher_loss_metrics_are_grouped_with_pc_hbm_logs():
    metrics = {
        "soft_teacher_loss": 1.0,
        "soft_teacher_bce": 0.7,
        "soft_teacher_weighted_iou": 0.3,
        "soft_teacher_iou_valid_sample_ratio": 0.75,
        "hard_teacher_loss": 2.0,
        "hard_teacher_ramp_factor": 1.0 / 3.0,
        "hard_teacher_effective_weight": 1.0 / 3.0,
        "hard_teacher_weighted_loss": 2.0 / 3.0,
        "hard_teacher_valid_pixel_ratio": 0.4,
        "hard_teacher_valid_sample_ratio": 0.5,
        "loss_u_total": 5.0,
    }
    base, modules = partition_training_metrics(metrics)

    assert base == {}
    assert modules["PC-HBM"] == metrics


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("PC-HBM teacher loss tests passed.")
