import os
import sys
from types import SimpleNamespace

import torch
import pytest
from PC_HBM.memory import PCHBMMemory

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PC_HBM.training.pc_losses import (
    _boundary_aux,
    compute_pc_hbm_unlabeled_loss,
    structure_aware_confidence,
)
from PC_HBM.training.pc_supervision import build_need_correction_map


def _logit(prob: float) -> float:
    p = torch.tensor(prob)
    return torch.logit(p).item()


def test_structure_confidence_decreases_when_mix_disagrees_with_main():
    p_final = torch.full((1, 1, 4, 4), 0.9)
    aligned = {
        "p_final": p_final,
        "z_main": torch.full((1, 1, 4, 4), _logit(0.9)),
    }
    disagree = {
        "p_final": p_final,
        "z_main": torch.full((1, 1, 4, 4), _logit(0.1)),
    }

    conf_aligned = structure_aware_confidence(aligned)
    conf_disagree = structure_aware_confidence(disagree)

    assert conf_aligned.mean() > conf_disagree.mean()
    assert conf_disagree.mean() < conf_aligned.mean() * 0.4


def test_structure_confidence_uses_normalized_route_entropy():
    p_final = torch.full((1, 1, 4, 4), 0.9)
    base = {
        "p_final": p_final,
        "z_main": torch.full((1, 1, 4, 4), _logit(0.9)),
    }
    no_route_penalty = structure_aware_confidence(
        {**base, "pc_hbm": {"route_entropy_norm": torch.zeros(1)}}
    )
    uniform_route = structure_aware_confidence(
        {
            **base,
            "pc_hbm": {
                "route_entropy": torch.tensor([32.0]).log(),
                "route_entropy_norm": torch.ones(1),
            },
        }
    )
    legacy_raw_only = structure_aware_confidence(
        {**base, "pc_hbm": {"route_entropy": torch.tensor([32.0]).log()}}
    )

    assert torch.allclose(uniform_route, no_route_penalty * 0.75, atol=1e-6)
    assert torch.allclose(legacy_raw_only, uniform_route, atol=1e-6)
    assert uniform_route.mean().item() == pytest.approx(0.6, abs=1e-6)
    assert torch.isfinite(uniform_route).all()
    assert uniform_route.min() >= 0
    assert uniform_route.max() <= 1


def test_normalized_route_confidence_reactivates_hard_teacher_mask():
    pseudo = torch.full((1, 1, 4, 4), 0.9)
    confidence = structure_aware_confidence(
        {
            "p_final": pseudo,
            "z_main": torch.full_like(pseudo, _logit(0.9)),
            "pc_hbm": {"route_entropy_norm": torch.ones(1)},
        }
    )
    student = torch.zeros_like(pseudo, requires_grad=True)
    config = type(
        "Cfg",
        (),
        {
            "lambda_u": 0.5,
            "use_hard_teacher_loss": True,
            "hard_teacher_loss_weight": 0.25,
            "hard_teacher_threshold": 0.5,
            "hard_teacher_foreground_threshold": 0.7,
            "hard_teacher_background_threshold": 0.3,
            "hard_teacher_confidence_threshold": 0.25,
            "use_soft_teacher_weighted_iou": False,
            "pc_hbm_unsup_final_consistency_weight": 0.0,
        },
    )()

    loss, log = compute_pc_hbm_unlabeled_loss(
        {
            "z_main": student,
            "z_nomix": student,
            "mixture_skipped": True,
            "forward_mode": "student_core",
        },
        pseudo,
        confidence,
        config,
    )

    assert confidence.mean().item() == pytest.approx(0.6, abs=1e-6)
    assert log["pseudo_conf_valid_pixel_ratio"].item() == pytest.approx(1.0)
    assert log["hard_teacher_valid_pixel_ratio"].item() == pytest.approx(1.0)
    assert log["hard_teacher_valid_sample_ratio"].item() == pytest.approx(1.0)
    assert log["hard_teacher_weighted_loss"].item() > 0
    assert loss.item() > 0


def test_route_entropy_reports_raw_and_topk_normalized_values():
    memory = PCHBMMemory(4, 3, 2, config=SimpleNamespace())
    memory.route = {
        "route_embed": torch.eye(4)[:3],
        "img_ids": ["img0", "img1", "img2"],
    }
    memory.is_ready = lambda: True
    query = torch.ones(1, 4)

    routed = memory.route_query(query, top_img_k=3)
    assert torch.allclose(
        routed["route_entropy"],
        torch.tensor([3.0]).log(),
        atol=1e-6,
    )
    assert torch.allclose(routed["route_entropy_norm"], torch.ones(1), atol=1e-6)

    routed_single = memory.route_query(query, top_img_k=1)
    assert routed_single["route_entropy"].item() == pytest.approx(0.0)
    assert routed_single["route_entropy_norm"].item() == pytest.approx(0.0)

    empty = PCHBMMemory(4, 3, 2, config=SimpleNamespace())
    routed_empty = empty.route_query(query, top_img_k=3)
    assert routed_empty["route_entropy"].item() == pytest.approx(0.0)
    assert routed_empty["route_entropy_norm"].item() == pytest.approx(0.0)


def test_full_student_routes_confidence_weighted_final_consistency_gradient():
    pseudo = torch.full((1, 1, 2, 2), 0.8)
    confidence = torch.tensor([[[[1.0, 0.5], [0.25, 0.0]]]])
    config = type(
        "Cfg",
        (),
        {
            "pc_hbm_unsup_final_consistency_weight": 0.05,
            "lambda_u": 0.5,
            "hard_teacher_loss_weight": 0.0,
            "use_soft_teacher_weighted_iou": False,
        },
    )()

    z_main = torch.zeros_like(pseudo, requires_grad=True)
    z_final = torch.zeros_like(pseudo, requires_grad=True)
    loss, log = compute_pc_hbm_unlabeled_loss(
        {
            "z_main": z_main,
            "z_nomix": z_main,
            "z_final": z_final,
            "mixture_skipped": False,
            "forward_mode": "full",
        },
        pseudo,
        confidence,
        config,
    )
    loss.backward()

    expected_final = torch.nn.functional.binary_cross_entropy_with_logits(
        z_final.detach(),
        pseudo,
        reduction="none",
    )
    expected_final = (expected_final * confidence).sum() / confidence.sum()
    assert torch.allclose(log["final_consistency_loss"], expected_final)
    assert torch.allclose(
        log["final_consistency_weighted_loss"],
        expected_final * 0.05,
    )
    assert z_final.grad is not None
    assert z_final.grad.abs().sum() > 0


def test_boundary_aux_supervises_g2_refined_with_need_correction():
    gt = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    z_main = torch.tensor([[[[8.0, 8.0], [-8.0, 0.0]]]])
    need = build_need_correction_map(z_main, gt, (2, 2), threshold=0.25)

    low = _boundary_aux({}, {"G2_refined_map": need.clamp(0.01, 0.99)}, {}, gt, z_main)
    high = _boundary_aux({}, {"G2_refined_map": (1.0 - need).clamp(0.01, 0.99)}, {}, gt, z_main)

    assert low < high
    assert low.item() < 0.05
