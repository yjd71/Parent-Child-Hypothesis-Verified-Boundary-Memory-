import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PC_HBM.memory import parent_values_from_region
from PC_HBM.fusion.pc_token_decoder import PCTokenDecoder
from PC_HBM.retrieval import ParentRetriever
from PC_HBM.retrieval.structured_prior_bias_net import StructuredPriorBiasNet
from PC_HBM.training.pc_supervision import build_region_label_map, gather_by_boundary_indices


def test_parent_values_follow_scheme_fg_bg_indices():
    fg_v = parent_values_from_region("fg_boundary", sdf=torch.zeros(3), reliability=torch.ones(3))
    bg_v = parent_values_from_region("bg_near", sdf=torch.zeros(3), reliability=torch.ones(3))

    assert torch.all(fg_v[:, 1] == 1)
    assert torch.all(fg_v[:, 4] == 1)
    assert torch.all(fg_v[:, 5] == 0)

    assert torch.all(bg_v[:, 2] == 1)
    assert torch.all(bg_v[:, 4] == 0)
    assert torch.all(bg_v[:, 5] == 1)


def test_parent_retriever_uses_scheme_fg_bg_indices_for_scores():
    retriever = ParentRetriever(p3_ch=1, dim=1, topk=2, tau=0.01)
    with torch.no_grad():
        retriever.proj_parent_q.weight.fill_(1.0)

    values = torch.zeros(2, 8)
    values[0, 1] = 1.0
    values[0, 4] = 1.0
    values[1, 2] = 1.0
    values[1, 5] = 1.0
    subbank = {
        "p3_keys": torch.tensor([[1.0], [-1.0]]),
        "p3_values": values,
        "p3_geometry": torch.zeros(2, 6),
        "child_ptr": torch.arange(2),
        "parent_meta": [{"region_id": 1}, {"region_id": 2}],
    }

    out = retriever(
        torch.ones(1, 1, 1, 1),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        subbank,
    )

    assert out["S_fg_parent"].item() > 0.99
    assert out["S_bg_parent"].item() < 0.01
    assert out["M_parent"].item() > 0.99
    assert out["top_parent_region_ids"].shape == (1, 2)
    assert out["top_parent_region_ids"][0, 0].item() == 1


def test_structured_prior_bias_net_reads_fg_from_v4_and_bg_from_v5():
    prior = StructuredPriorBiasNet(value_dim=8, geometry_dim=6)
    values = torch.zeros(1, 2, 8)
    values[0, 0, 1] = 1.0
    values[0, 0, 4] = 1.0
    values[0, 1, 2] = 1.0
    values[0, 1, 5] = 1.0

    out = prior(
        values,
        parent_geo=torch.zeros(1, 2, 6),
        child_geo=torch.zeros(1, 2, 6),
        s_child=torch.full((1, 2), 10.0),
        s_geo=torch.full((1, 2), 10.0),
    )

    assert out[0, 0] > out[0, 1]


def test_pc_token_decoder_m_pc_is_evidence_margin_when_residual_zero():
    decoder = PCTokenDecoder(dim=4)
    q3_new = torch.zeros(2, 4)
    attn = torch.tensor([[2.0, 1.0, 1.0], [1.0, 3.0, 0.0]])
    values = torch.zeros(2, 3, 8)
    values[0, 0, 0] = 1.0
    values[0, 1, 1] = 1.0
    values[0, 2, 2] = 1.0
    values[1, 0, 2] = 1.0
    values[1, 1, 3] = 1.0
    values[1, 2, 0] = 1.0
    parent_ret = {
        "top_parent_values": values,
        "top_parent_geo": torch.zeros(2, 3, 6),
    }
    child_ver = {"G2_child_top": torch.zeros(2, 3, 6)}

    out = decoder(q3_new, attn, parent_ret, child_ver)
    weights = attn / attn.sum(dim=1, keepdim=True)
    e_attn = (weights.unsqueeze(-1) * values).sum(dim=1)
    expected = e_attn[:, 0] + e_attn[:, 1] - e_attn[:, 2] - e_attn[:, 3]

    torch.testing.assert_close(out["M_pc_evidence"].squeeze(1), expected)
    torch.testing.assert_close(out["M_pc_token"].squeeze(1), expected.clamp(-1.0, 1.0))
    assert torch.all(out["M_pc_residual"] == 0)


def test_supervision_gather_returns_current_query_region_labels():
    gt = torch.zeros(1, 1, 16, 16)
    gt[:, :, 6:10, 6:10] = 1.0
    labels = build_region_label_map(gt, (8, 8))
    boundary = {
        "batch_ids": torch.tensor([0, 0], dtype=torch.long),
        "flat_indices": torch.tensor([27, 0], dtype=torch.long),
    }

    gathered = gather_by_boundary_indices(labels, boundary)

    assert gathered.shape == (2,)
    assert gathered[0].item() in (0, 1)
    assert gathered[1].item() in (2, 3)
