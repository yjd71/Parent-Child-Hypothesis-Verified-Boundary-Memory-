import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PC_HBM.training.pc_losses import _child_verify, _gate_loss, _geometry_loss, _parent_ce, compute_pc_hbm_labeled_loss
from PC_HBM.training.pc_supervision import (
    REGION_BG_NEAR,
    REGION_FG_BOUNDARY,
    build_geometry_target,
    build_region_label_map,
    gather_by_boundary_indices,
)


def _gt() -> torch.Tensor:
    gt = torch.zeros(1, 1, 16, 16)
    gt[:, :, 6:10, 6:10] = 1.0
    return gt


def _boundary_for_region(gt: torch.Tensor, region_id: int, size=(8, 8)):
    labels = build_region_label_map(gt, size)
    match = (labels[0] == int(region_id)).nonzero(as_tuple=False)
    assert match.numel() > 0
    yx = match[0]
    flat = yx[0] * size[1] + yx[1]
    return {
        "batch_ids": torch.tensor([0], dtype=torch.long),
        "flat_indices": flat.reshape(1).long(),
    }


def test_parent_ce_uses_query_gt_region_not_memory_label():
    gt = _gt()
    boundary = _boundary_for_region(gt, REGION_FG_BOUNDARY)
    correct = torch.full((1, 4), 1e-4)
    correct[0, REGION_FG_BOUNDARY] = 0.9997
    wrong = torch.full((1, 4), 1e-4)
    wrong[0, REGION_BG_NEAR] = 0.9997
    memory_values = torch.zeros(1, 3, 8)
    memory_values[..., REGION_BG_NEAR] = 1.0
    pc_base = {"B3": torch.zeros(1, 1, 8, 8), "boundary_indices3": boundary, "top_parent_values": memory_values}

    low = _parent_ce({**pc_base, "P3_group": correct}, gt)
    high = _parent_ce({**pc_base, "P3_group": wrong}, gt)

    assert low.item() < 0.01
    assert high.item() > 5.0


def test_child_verify_target_is_gt_region_consistency():
    gt = _gt()
    boundary = _boundary_for_region(gt, REGION_FG_BOUNDARY)
    pc_base = {
        "B3": torch.zeros(1, 1, 8, 8),
        "boundary_indices3": boundary,
        "top_parent_region_ids": torch.tensor([[REGION_FG_BOUNDARY, REGION_BG_NEAR, 0]], dtype=torch.long),
    }

    low = _child_verify({**pc_base, "S_child": torch.tensor([[8.0, -8.0, -8.0]])}, gt)
    high = _child_verify({**pc_base, "S_child": torch.tensor([[-8.0, 8.0, 8.0]])}, gt)

    assert low.item() < 0.01
    assert high.item() > 8.0


def test_geometry_loss_uses_gt_sdf_normal_and_offset_with_gradients():
    gt = _gt()
    boundary = _boundary_for_region(gt, REGION_FG_BOUNDARY)
    geo = build_geometry_target(gt, (8, 8))
    sdf = gather_by_boundary_indices(geo["sdf"], boundary).view(-1)
    normal = gather_by_boundary_indices(geo["normal"], boundary)
    offset = gather_by_boundary_indices(geo["offset"], boundary)
    g_attn = torch.zeros(1, 6)
    g_attn[:, 0] = sdf
    g_attn[:, 1:3] = normal
    g_attn = g_attn.detach().requires_grad_(True)
    o_pc = offset.detach().clone().requires_grad_(True)

    loss = _geometry_loss(
        {
            "B3": torch.zeros(1, 1, 8, 8),
            "boundary_indices3": boundary,
            "G_attn": g_attn,
            "O_pc_token": o_pc,
        },
        gt,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert g_attn.grad is not None
    assert o_pc.grad is not None


def test_gate_loss_targets_need_correction_times_low_c23():
    gt = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    z_main = torch.tensor([[[[8.0, 8.0], [-8.0, 0.0]]]])
    boundary = {
        "batch_ids": torch.tensor([0, 0, 0], dtype=torch.long),
        "flat_indices": torch.tensor([0, 1, 2], dtype=torch.long),
    }
    pc_base = {
        "B3": torch.zeros(1, 1, 2, 2),
        "boundary_indices3": boundary,
        "C23_token": torch.tensor([[0.0], [0.0], [0.8]]),
    }

    low = _gate_loss({**pc_base, "gate_pc_token": torch.tensor([[0.01], [0.99], [0.20]])}, {"z_main": z_main}, gt)
    high = _gate_loss({**pc_base, "gate_pc_token": torch.tensor([[0.99], [0.01], [0.99]])}, {"z_main": z_main}, gt)

    assert low.item() < 0.25
    assert high.item() > 2.0
    assert high > low


def test_labeled_loss_exposes_rewritten_memory_terms():
    gt = _gt()
    boundary = _boundary_for_region(gt, REGION_FG_BOUNDARY)
    outputs = [torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 16, 16)]
    pc = {
        "B3": torch.zeros(1, 1, 8, 8),
        "boundary_indices3": boundary,
        "P3_group": torch.tensor([[1e-4, 0.9997, 1e-4, 1e-4]]),
        "S_child": torch.tensor([[8.0]]),
        "top_parent_region_ids": torch.tensor([[REGION_FG_BOUNDARY]], dtype=torch.long),
        "G_attn": torch.zeros(1, 6),
        "O_pc_token": torch.zeros(1, 2),
        "gate_pc_token": torch.tensor([[0.1]]),
        "C23_token": torch.tensor([[0.5]]),
    }
    loss, log = compute_pc_hbm_labeled_loss(outputs, {"z_main": outputs[-1], "pc_hbm": pc}, gt, SimpleNamespace())

    assert torch.isfinite(loss)
    for key in ("L_parent_ce", "L_child_verify", "L_geometry", "L_gate"):
        assert key in log
        assert torch.isfinite(log[key])
