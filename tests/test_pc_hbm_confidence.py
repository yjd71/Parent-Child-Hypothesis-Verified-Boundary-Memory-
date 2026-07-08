import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PC_HBM.training.pc_losses import _boundary_aux, structure_aware_confidence
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


def test_boundary_aux_supervises_g2_refined_with_need_correction():
    gt = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    z_main = torch.tensor([[[[8.0, 8.0], [-8.0, 0.0]]]])
    need = build_need_correction_map(z_main, gt, (2, 2), threshold=0.25)

    low = _boundary_aux({}, {"G2_refined_map": need.clamp(0.01, 0.99)}, {}, gt, z_main)
    high = _boundary_aux({}, {"G2_refined_map": (1.0 - need).clamp(0.01, 0.99)}, {}, gt, z_main)

    assert low < high
    assert low.item() < 0.05
