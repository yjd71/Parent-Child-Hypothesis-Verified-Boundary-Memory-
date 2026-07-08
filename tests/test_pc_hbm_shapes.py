import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PC_HBM.common import boundary_features_from_logits
from PC_HBM.core import PCHBMEngine, apply_pc_hbm_defaults
from PC_HBM.fusion import P3GatedResidual, PCHCA
from PC_HBM.memory import PCHBMMemory
from PC_HBM.refinement import (
    AdaptiveMixtureHead,
    BoundaryQueryHead1,
    BoundaryQueryHead2,
    BoundaryQueryHead3,
    P1PixelRefinementAttention,
    P2BoundaryRetargetAttention,
)
from PC_HBM.retrieval import ChildLocalEncoder, ChildQueryBuilder, ChildVerifierV2, ParentRetriever


def cfg():
    c = SimpleNamespace(
        use_pc_hbm=True,
        pc_hbm_enable=True,
        backbone="swin_v1_l",
        lateral_channels_in_collection=[3072, 1536, 768, 384],
        cbm_memory_dim=512,
        cbm_value_dim=8,
        geometry_dim=6,
        parent_topk=64,
        cbm_top_img_k=2,
        p2_boundary_top_ratio=0.02,
        p1_boundary_top_ratio=0.005,
        mixture_init_bias=[1.0, -0.5, -0.5, -0.5],
    )
    return apply_pc_hbm_defaults(c)


def make_memory(n_parent=128, n_img=2):
    mem = PCHBMMemory(512, 8, 6)
    route = torch.randn(n_img, 512)
    mem.append_route(
        x3_global=route,
        x3_boundary=route,
        x3_uncertain=route,
        x3_bg_near=route,
        x3_environment=route,
        route_embed=torch.nn.functional.normalize(route, dim=1),
        img_ids=[f"img{i}" for i in range(n_img)],
    )
    child_keys = torch.randn(n_parent, 512)
    child_geo = torch.randn(n_parent, 6)
    child_meta = [{"image_id": f"img{i % n_img}", "region": "fg_boundary"} for i in range(n_parent)]
    child_ptr = mem.append_child(child_keys, child_geo, child_meta)
    values = torch.zeros(n_parent, 8)
    values[:, 1] = 1.0
    values[:, 5] = 1.0
    values[:, 7] = 1.0
    parent_meta = [{"image_id": f"img{i % n_img}", "region": "fg_boundary", "region_id": 1} for i in range(n_parent)]
    mem.append_parent(torch.randn(n_parent, 512), values, torch.randn(n_parent, 6), child_ptr, parent_meta)
    mem.finalize()
    return mem


def test_config_import():
    c = cfg()
    assert c.use_pc_hbm is True
    assert c.cbm_memory_dim == 512
    assert c.parent_topk == 64


def test_memory_state_dict_roundtrip():
    mem = make_memory()
    assert mem.is_ready()
    state = mem.state_dict()
    loaded = PCHBMMemory(512, 8, 6)
    loaded.load_state_dict(state)
    assert loaded.is_ready()
    assert loaded.parent["p3_keys"].shape[1] == 512


def test_child_local_encoder_shape():
    enc = ChildLocalEncoder(384, 512)
    out = enc(torch.randn(7, 384, 5, 5))
    assert out.shape == (7, 512)


def test_boundary_query_heads_shape():
    for head, x in [
        (BoundaryQueryHead3(), torch.randn(2, 5, 40, 40)),
        (BoundaryQueryHead2(), torch.randn(2, 8, 80, 80)),
        (BoundaryQueryHead1(), torch.randn(2, 8, 160, 160)),
    ]:
        score, idx = head(x)
        assert score.shape[:2] == (2, 1)
        assert idx["batch_ids"].numel() == idx["flat_indices"].numel()


def test_parent_child_hca_shapes():
    mem = make_memory()
    p3 = torch.randn(2, 768, 40, 40)
    logits3 = torch.randn(2, 1, 40, 40)
    score, idx = BoundaryQueryHead3(max_tokens=16)(boundary_features_from_logits(logits3))
    retriever = ParentRetriever(768, 512, topk=64)
    subbank = mem.get_parent_subbank([["img0", "img1"], ["img0", "img1"]], device=p3.device, dtype=p3.dtype)
    parent = retriever(p3, idx["batch_ids"], idx["flat_indices"], subbank)
    assert parent["top_parent_keys"].shape == (idx["batch_ids"].numel(), 64, 512)
    p2_pre = torch.randn(2, 384, 80, 80)
    child_query = ChildQueryBuilder(384, 512)(p2_pre, idx["batch_ids"], idx["flat_indices"], (40, 40))
    child_bank = mem.get_child_by_ptr(parent["top_child_ptrs"], device=p3.device, dtype=p3.dtype)
    verifier = ChildVerifierV2(512, 8, 6)
    child = verifier(child_query["q_child"], child_query["G2_query"], parent, child_bank)
    m = idx["batch_ids"].numel()
    assert child["S_child"].shape == (m, 64)
    assert child["S_geo"].shape == (m, 64)
    assert child["prior_bias"].shape == (m, 64)
    assert child["C23_token"].shape == (m, 1)
    hca = PCHCA(512, 8, 64)
    q_new, attn = hca(torch.randn(m, 512), torch.randn(m, 64, 512), child["prior_bias"], torch.randn(m, 512))
    assert q_new.shape == (m, 512)
    assert attn.shape == (m, 64)


def test_p3_gated_residual_shape():
    p3 = torch.randn(2, 768, 40, 40)
    batch = torch.tensor([0, 1, 1])
    flat = torch.tensor([0, 10, 20])
    mod = P3GatedResidual(512, 768)
    out, delta = mod(p3, batch, flat, torch.randn(3, 512), torch.ones(3, 1), torch.ones(3, 1))
    assert out.shape == (2, 768, 40, 40)
    assert delta.shape == (3, 768)


def pc_maps(b=2):
    return {
        "Z3_map": torch.randn(b, 512, 40, 40),
        "E_attn_map": torch.randn(b, 8, 40, 40),
        "G_attn_map": torch.randn(b, 6, 40, 40),
        "M_pc_map": torch.rand(b, 1, 40, 40),
        "gate_pc_map": torch.rand(b, 1, 40, 40),
        "C23_map": torch.rand(b, 1, 40, 40),
        "O_pc_map": torch.randn(b, 2, 40, 40),
        "valid3_map": torch.ones(b, 1, 40, 40),
    }


def test_p2_bra_shapes():
    mod = P2BoundaryRetargetAttention(384, top_ratio=0.02)
    out = mod(torch.randn(2, 384, 80, 80), torch.rand(2, 1, 80, 80), pc_maps())
    assert out["p2_refined"].shape == (2, 384, 80, 80)
    assert out["F2_ref_map"].shape == (2, 512, 80, 80)
    assert out["B2_refined_map"].shape == (2, 1, 80, 80)
    assert out["O2_refined_map"].shape == (2, 2, 80, 80)


def test_p1_pra_shapes():
    p2_aux = P2BoundaryRetargetAttention(384, top_ratio=0.02)(torch.randn(2, 384, 80, 80), torch.rand(2, 1, 80, 80), pc_maps())
    mod = P1PixelRefinementAttention(192, top_ratio=0.005)
    out = mod(torch.randn(2, 192, 160, 160), torch.randn(2, 1, 640, 640), p2_aux)
    assert out["G1_map"].shape == (2, 1, 160, 160)
    assert out["R1_map"].shape == (2, 1, 160, 160)
    assert out["R_sup_map"].shape == (2, 1, 160, 160)
    assert out["O1_map"].shape == (2, 2, 160, 160)


def test_adaptive_mixture_shapes():
    p1_aux = {
        "B1": torch.rand(2, 1, 160, 160),
        "G1_map": torch.rand(2, 1, 160, 160),
        "R1_map": torch.randn(2, 1, 160, 160),
        "O1_map": torch.randn(2, 2, 160, 160),
        "R_sup_map": torch.rand(2, 1, 160, 160),
        "valid1_map": torch.ones(2, 1, 160, 160),
    }
    out = AdaptiveMixtureHead()(torch.randn(2, 1, 640, 640), p1_aux, pc_maps())
    assert out["pi"].shape == (2, 4, 640, 640)
    assert out["z_final"].shape == (2, 1, 640, 640)
    assert out["p_final"].shape == (2, 1, 640, 640)


class StubDecoder:
    def forward_to_p3(self, features):
        x, x1, x2, x3, x4 = features
        b = x.size(0)
        p3 = torch.randn(b, 768, 40, 40, device=x.device)
        m3 = torch.randn(b, 1, 40, 40, device=x.device)
        state = {"x": x, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "m4": torch.randn(b, 1, 20, 20, device=x.device), "m3": m3, "gdt_gt": None, "outs": [], "outs_gdt_pred": [], "outs_gdt_label": []}
        return state, p3, m3

    def forward_p2_from_p3(self, state, p3):
        b = p3.size(0)
        p2 = torch.randn(b, 384, 80, 80, device=p3.device)
        m2 = torch.randn(b, 1, 80, 80, device=p3.device)
        state2 = dict(state)
        state2.update({"p3": p3, "p2": p2, "m2": m2})
        return state2, p2, m2

    def forward_p1_from_p2(self, state2, p2):
        b = p2.size(0)
        p1 = torch.randn(b, 192, 160, 160, device=p2.device)
        z = torch.randn(b, 1, 640, 640, device=p2.device)
        outs = [state2["m4"], state2["m3"], state2["m2"], z]
        return outs, p1, z


class StubTalNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = cfg()
        self.decoder = StubDecoder()

    def _build_decoder_features(self, x):
        b = x.size(0)
        x1 = torch.randn(b, 384, 160, 160, device=x.device)
        x2 = torch.randn(b, 768, 80, 80, device=x.device)
        x3 = torch.randn(b, 1536, 40, 40, device=x.device)
        x4 = torch.randn(b, 3072, 20, 20, device=x.device)
        return x1, x2, x3, x4, [x, x1, x2, x3, x4]

    def forward_pc_hbm(self, x, memory=None, use_memory=True, return_all_logits=True, epoch=None, engine=None):
        return engine.forward_talnet(self, x, memory=memory, use_memory=use_memory, return_all_logits=return_all_logits, epoch=epoch)


def test_forward_pc_hbm_fallback_and_full_path():
    c = cfg()
    engine = PCHBMEngine(c)
    tal = StubTalNet()
    x = torch.randn(1, 3, 640, 640)
    outputs, aux = tal.forward_pc_hbm(x, memory=PCHBMMemory(512, 8, 6), engine=engine)
    assert aux["fallback_reason"] == "memory_not_ready"
    assert aux["p_final"].shape == (1, 1, 640, 640)
    mem = make_memory(n_parent=128, n_img=2)
    outputs, aux = tal.forward_pc_hbm(x, memory=mem, engine=engine, epoch=None)
    assert outputs[-1].shape == (1, 1, 640, 640)
    assert aux["features"]["x3"].shape == (1, 1536, 40, 40)
    assert aux["p3"].shape == (1, 768, 40, 40)
    assert aux["p2_pre"].shape == (1, 384, 80, 80)
    assert aux["p2"].shape == (1, 384, 80, 80)
    assert aux["p1"].shape == (1, 192, 160, 160)
    pc = aux["pc_hbm"]
    assert pc["top_parent_keys"].shape[-2:] == (64, 512)
    assert pc["K_child_top"].shape[-2:] == (64, 512)
    assert pc["H_tokens"].shape[-2:] == (64, 512)
    assert pc["q3_new"].shape[-1] == 512
    assert pc["M_pc_map"].shape == (1, 1, 40, 40)
    assert pc["C23_map"].shape == (1, 1, 40, 40)
    assert pc["gate_pc_map"].shape == (1, 1, 40, 40)
    assert pc["Z3_map"].shape == (1, 512, 40, 40)
    assert aux["p2_bra"]["F2_ref_map"].shape == (1, 512, 80, 80)
    assert aux["p2_bra"]["B2_refined_map"].shape == (1, 1, 80, 80)
    assert aux["p2_bra"]["O2_refined_map"].shape == (1, 2, 80, 80)
    assert aux["p1_pra"]["G1_map"].shape == (1, 1, 160, 160)
    assert aux["p1_pra"]["R1_map"].shape == (1, 1, 160, 160)
    assert aux["p1_pra"]["O1_map"].shape == (1, 2, 160, 160)
    assert aux["p1_pra"]["R_sup_map"].shape == (1, 1, 160, 160)
    assert aux["mixture"]["pi"].shape == (1, 4, 640, 640)
    assert aux["z_final"].shape == (1, 1, 640, 640)
    assert aux["p_final"].shape == (1, 1, 640, 640)


if __name__ == "__main__":
    try:
        import pytest

        raise SystemExit(pytest.main([__file__, "-q"]))
    except ModuleNotFoundError:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
        print("PC-HBM shape tests passed.")
