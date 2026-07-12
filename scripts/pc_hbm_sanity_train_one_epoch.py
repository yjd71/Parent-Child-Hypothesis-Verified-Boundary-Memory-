from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PC_HBM.core import PCHBMEngine, apply_pc_hbm_defaults
from PC_HBM.training.pc_losses import compute_pc_hbm_labeled_loss, compute_pc_hbm_unlabeled_loss, structure_aware_confidence


def cfg():
    return apply_pc_hbm_defaults(
        SimpleNamespace(
            use_pc_hbm=True,
            pc_hbm_enable=True,
            backbone="swin_v1_l",
            lateral_channels_in_collection=[3072, 1536, 768, 384],
            cbm_memory_dim=512,
            cbm_value_dim=8,
            geometry_dim=6,
            parent_topk=16,
            cbm_top_img_k=1,
            p2_boundary_top_ratio=0.005,
            p1_boundary_top_ratio=0.001,
            mixture_init_bias=[1.0, -0.5, -0.5, -0.5],
        )
    )


class StubDecoder:
    def forward_to_p3(self, features):
        x, x1, x2, x3, x4 = features
        bsz = x.size(0)
        p3 = torch.randn(bsz, 768, 40, 40, device=x.device)
        m3 = torch.randn(bsz, 1, 40, 40, device=x.device)
        state = {
            "x": x,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "m4": torch.randn(bsz, 1, 20, 20, device=x.device),
            "m3": m3,
        }
        return state, p3, m3

    def forward_p2_from_p3(self, state, p3):
        bsz = p3.size(0)
        p2 = torch.randn(bsz, 384, 80, 80, device=p3.device)
        m2 = torch.randn(bsz, 1, 80, 80, device=p3.device)
        state2 = dict(state)
        state2.update({"p3": p3, "p2": p2, "m2": m2})
        return state2, p2, m2

    def forward_p1_from_p2(self, state2, p2):
        bsz = p2.size(0)
        p1 = torch.randn(bsz, 192, 160, 160, device=p2.device)
        z = torch.randn(bsz, 1, 640, 640, device=p2.device)
        return [state2["m4"], state2["m3"], state2["m2"], z], p1, z


class StubTalNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.decoder = StubDecoder()

    def _build_decoder_features(self, x):
        bsz = x.size(0)
        x1 = torch.randn(bsz, 384, 160, 160, device=x.device)
        x2 = torch.randn(bsz, 768, 80, 80, device=x.device)
        x3 = torch.randn(bsz, 1536, 40, 40, device=x.device)
        x4 = torch.randn(bsz, 3072, 20, 20, device=x.device)
        return x1, x2, x3, x4, [x, x1, x2, x3, x4]

    @torch.no_grad()
    def forward_return_pc_hbm_features(self, x, ema=True):
        del ema
        x1, x2, x3, x4, features = self._build_decoder_features(x)
        state, p3, m3 = self.decoder.forward_to_p3(features)
        _, p2, m2 = self.decoder.forward_p2_from_p3(state, p3)
        return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "p3": p3, "p2": p2, "m3": m3, "m2": m2}


def make_batch(device):
    img = torch.randn(1, 3, 640, 640, device=device)
    gt = torch.zeros(1, 1, 640, 640, device=device)
    gt[:, :, 220:420, 240:430] = 1.0
    return img, gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    config = cfg()
    model = StubTalNet().to(device)
    engine = PCHBMEngine(config, device=device)
    img, gt = make_batch(device)

    engine.prepare_epoch(model, [(img, gt, ["tiny0"])], epoch=6)
    assert engine.memory.is_ready()

    outputs, aux = engine.forward_talnet(model, img, memory=engine.memory, use_memory=True, epoch=11)
    loss, log = compute_pc_hbm_labeled_loss(outputs, aux, gt, config)
    assert torch.isfinite(loss)
    for key in ("L_parent_ce", "L_child_verify", "L_geometry", "L_gate"):
        assert key in log
        assert torch.isfinite(log[key])
    pc = aux["pc_hbm"]
    assert pc["P3_group"].shape[-1] == 4
    assert pc["top_parent_region_ids"].shape == pc["S_child"].shape
    assert torch.isfinite(pc["route_entropy_norm"]).all()
    assert pc["route_entropy_norm"].min() >= 0
    assert pc["route_entropy_norm"].max() <= 1

    with torch.no_grad():
        _, teacher_aux = engine.forward_talnet(model, img, memory=engine.memory, use_memory=True, epoch=16)
        pseudo = teacher_aux["p_final"]
        confidence = structure_aware_confidence(teacher_aux)
    _, student_aux = engine.forward_talnet(model, img, memory=engine.memory, use_memory=True, epoch=16)
    loss_u, log_u = compute_pc_hbm_unlabeled_loss(student_aux, pseudo, confidence, config)
    assert torch.isfinite(loss_u)
    assert "L_u" in log_u
    assert "final_consistency_loss" in log_u
    assert torch.isfinite(log_u["final_consistency_loss"])
    print(
        "pc_hbm_sanity ok "
        f"loss={float(loss.detach().cpu()):.4f} "
        f"L_parent_ce={float(log['L_parent_ce'].cpu()):.4f} "
        f"L_child_verify={float(log['L_child_verify'].cpu()):.4f} "
        f"L_geometry={float(log['L_geometry'].cpu()):.4f} "
        f"L_gate={float(log['L_gate'].cpu()):.4f} "
        f"L_u={float(log_u['L_u'].cpu()):.4f}"
    )


if __name__ == "__main__":
    main()
