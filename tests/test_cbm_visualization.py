import torch
from pathlib import Path

from CBM.config.defaults import apply_cbm_defaults
from CBM.diagnostics.visualization import save_pfi_binary_visualizations_v42


class Config:
    ckpt_dir = "."
    cbm_vis_enable = True
    cbm_vis_interval = 1
    cbm_vis_max_images = 1
    cbm_vis_labeled_only = True
    cbm_vis_dir = None


def _config(tmp_path):
    config = apply_cbm_defaults(Config())
    config.cbm_vis_dir = str(tmp_path)
    return config


def _make_aux(batch_size=2, height=8, width=8):
    p_main = torch.rand(batch_size, 1, height * 2, width * 2)
    p_final = torch.rand(batch_size, 1, height * 2, width * 2)
    return {
        "cbm_used": True,
        "prob3": torch.rand(batch_size, 1, height, width),
        "B_query": torch.rand(batch_size, 1, height, width),
        "Y_map": torch.rand(batch_size, 8, height, width),
        "U_map": torch.rand(batch_size, 1, height, width),
        "cons_map": torch.rand(batch_size, 1, height, width),
        "gate3": torch.rand(batch_size, 1, height, width),
        "p_main": p_main,
        "p_final": p_final,
    }


def test_save_pfi_binary_visualizations_v42_writes_expected_maps(tmp_path):
    config = _config(tmp_path)
    batch = (torch.randn(2, 3, 16, 16), torch.zeros(2, 1, 16, 16), ["img-a", "img-b"])

    paths = save_pfi_binary_visualizations_v42(
        aux=_make_aux(),
        batch=batch,
        epoch=3,
        iteration=20,
        config=config,
        branch_name="Sup",
    )

    assert len(paths) == 11
    assert all(path.endswith(".png") for path in paths)
    assert all("epoch003_iter000020_Sup_img-a" in path for path in paths)
    assert all(Path(path).exists() for path in paths)
    assert {path.split("_img-a_")[-1].replace(".png", "") for path in paths} == {
        "m3_prob",
        "B_query",
        "fg_boundary_score",
        "bg_near_score",
        "M_bd",
        "U_map",
        "cons_map",
        "gate3",
        "p_main",
        "p_final",
        "p_final_minus_p_main",
    }


def test_save_pfi_binary_visualizations_v42_noops_when_disabled_or_not_due(tmp_path):
    config = _config(tmp_path)
    config.cbm_vis_enable = False
    batch = (torch.randn(1, 3, 16, 16), torch.zeros(1, 1, 16, 16), ["img"])
    assert save_pfi_binary_visualizations_v42(_make_aux(1), batch, 1, 0, config) == []

    config.cbm_vis_enable = True
    config.cbm_vis_interval = 10
    assert save_pfi_binary_visualizations_v42(_make_aux(1), batch, 1, 3, config) == []
    assert list(tmp_path.glob("*.png")) == []


def test_save_pfi_binary_visualizations_v42_noops_for_missing_required_aux(tmp_path):
    config = _config(tmp_path)
    batch = (torch.randn(1, 3, 16, 16), torch.zeros(1, 1, 16, 16), ["img"])
    aux = _make_aux(1)
    aux["p_final"] = None

    assert save_pfi_binary_visualizations_v42(aux, batch, 1, 0, config) == []
    aux = _make_aux(1)
    aux["cbm_used"] = False
    assert save_pfi_binary_visualizations_v42(aux, batch, 1, 0, config) == []
    assert list(tmp_path.glob("*.png")) == []
