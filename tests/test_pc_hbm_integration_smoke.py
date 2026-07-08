import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PC_HBM.core import PCHBMEngine
from PC_HBM.debug.diagnostics import collect_pc_hbm_diagnostics
from PC_HBM.training.pc_supervision import REGION_BG_NEAR, REGION_FG_BOUNDARY, build_region_label_map
from scripts.pc_hbm_sanity_train_one_epoch import StubTalNet, cfg, make_batch


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


def test_collect_diagnostics_uses_gt_region_ids_and_need_correction():
    gt = torch.zeros(1, 1, 16, 16)
    gt[:, :, 6:10, 6:10] = 1.0
    boundary = _boundary_for_region(gt, REGION_FG_BOUNDARY)
    aux = {
        "z_main": torch.full((1, 1, 16, 16), -8.0),
        "z_final": torch.full((1, 1, 16, 16), -8.0),
        "p_final": torch.full((1, 1, 16, 16), 0.1),
        "pc_hbm": {
            "B3": torch.zeros(1, 1, 8, 8),
            "boundary_indices3": boundary,
            "top_parent_region_ids": torch.tensor([[REGION_FG_BOUNDARY, REGION_BG_NEAR]], dtype=torch.long),
            "S_child": torch.tensor([[8.0, -8.0]]),
            "G_attn": torch.zeros(1, 6),
            "O_pc_token": torch.zeros(1, 2),
            "gate_pc_map": torch.ones(1, 1, 8, 8),
            "C23_map": torch.zeros(1, 1, 8, 8),
        },
        "mixture": {},
    }

    diag = collect_pc_hbm_diagnostics(aux, gt)

    assert diag["parent_top1_region_acc"].item() == 1.0
    assert diag["parent_topk_region_acc"].item() == 1.0
    assert torch.isfinite(diag["child_verify_auc"])
    assert torch.isfinite(diag["gate_pc_on_error"])


def test_pc_hbm_memory_rebuild_and_full_path_smoke_cpu():
    config = cfg()
    model = StubTalNet()
    engine = PCHBMEngine(config, device=torch.device("cpu"))
    img, gt = make_batch(torch.device("cpu"))

    engine.prepare_epoch(model, [(img, gt, ["tiny0"])], epoch=6)
    assert engine.memory.is_ready()
    values = engine.memory.parent["p3_values"]
    assert torch.all(values[:, 4] + values[:, 5] == 1)

    outputs, aux = engine.forward_talnet(model, img, memory=engine.memory, use_memory=True, epoch=11)
    assert outputs[-1].shape == (1, 1, 640, 640)
    assert aux["pc_hbm_used"] is True
    assert aux["pc_hbm"]["top_parent_region_ids"].shape == aux["pc_hbm"]["S_child"].shape
