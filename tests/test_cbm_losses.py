import torch

from CBM.config.defaults import apply_cbm_defaults
from CBM.losses.total import compute_cbm_losses


class Config:
    cbm_pfi_enable = True
    cbm_print_diagnostics = False


def _config():
    return apply_cbm_defaults(Config())


def _square_gt(batch_size=2, height=8, width=8):
    gt = torch.zeros(batch_size, 1, height, width)
    gt[:, :, 2:6, 2:6] = 1.0
    return gt


def _make_aux(batch_size=2, height=8, width=8, requires_grad=False, include_p_final=True, valid=True):
    sources = {}

    y_src = torch.randn(batch_size, 8, height, width)
    y_ctx_src = torch.randn(batch_size, 8, height, width)
    gate_src = torch.randn(batch_size, 1, height, width)
    z_mem3 = torch.randn(batch_size, 1, height, width)
    if requires_grad:
        y_src.requires_grad_()
        y_ctx_src.requires_grad_()
        gate_src.requires_grad_()
        z_mem3.requires_grad_()
    sources["Y_src"] = y_src
    sources["Y_ctx_src"] = y_ctx_src
    sources["gate_src"] = gate_src
    sources["z_mem3"] = z_mem3

    valid_map = torch.ones(batch_size, 1, height, width) if valid else torch.zeros(batch_size, 1, height, width)
    boundary_mask = torch.zeros(batch_size, 1, height, width, dtype=torch.bool)
    boundary_mask[:, :, 1:7, 1:7] = True
    b_query = boundary_mask.float()

    aux = {
        "cbm_used": True,
        "Y_map": torch.sigmoid(y_src),
        "Y_ctx": torch.sigmoid(y_ctx_src),
        "R_map": torch.randn(batch_size, 128, height, width),
        "R_ctx": torch.randn(batch_size, 128, height, width),
        "U_map": torch.rand(batch_size, 1, height, width),
        "valid_map": valid_map,
        "cons_map": torch.rand(batch_size, 1, height, width),
        "B_query": b_query,
        "boundary_mask": boundary_mask,
        "gate3": torch.sigmoid(gate_src),
        "z_mem3": z_mem3,
        "prob3": torch.rand(batch_size, 1, height, width),
    }
    if include_p_final:
        p_src = torch.randn(batch_size, 1, height * 2, width * 2)
        if requires_grad:
            p_src.requires_grad_()
        sources["p_src"] = p_src
        aux["p_final"] = torch.sigmoid(p_src)
    return aux, sources


def _assert_zero_losses(losses):
    for value in losses.values():
        assert value.shape == ()
        assert torch.isfinite(value)
        assert value.item() == 0.0


def test_cbm_losses_fallbacks_return_complete_zero_dict():
    gt = _square_gt()
    required_keys = {
        "loss_cbm_mem",
        "loss_cbm_bd",
        "loss_cbm_ctx",
        "loss_cbm_aff",
        "loss_cbm_gate_sparse",
        "loss_cbm_gate_boundary",
        "loss_cbm_gate",
        "loss_cbm_total",
    }

    for aux, target in [({}, gt), ({"cbm_used": False}, gt), (_make_aux()[0], None)]:
        total, losses = compute_cbm_losses(aux, target, _config())
        assert total.shape == ()
        assert required_keys.issubset(losses.keys())
        _assert_zero_losses(losses)


def test_cbm_losses_zero_valid_returns_zero_without_nan():
    aux, _ = _make_aux(valid=False)
    total, losses = compute_cbm_losses(aux, _square_gt(), _config())

    assert total.item() == 0.0
    _assert_zero_losses(losses)


def test_cbm_losses_finite_and_total_matches_weighted_terms():
    aux, _ = _make_aux()
    total, losses = compute_cbm_losses(aux, _square_gt(), _config())

    for value in losses.values():
        assert value.shape == ()
        assert torch.isfinite(value)
        assert value.item() >= 0.0

    expected = (
        losses["loss_cbm_mem"]
        + losses["loss_cbm_bd"]
        + losses["loss_cbm_ctx"]
        + losses["loss_cbm_aff"]
        + losses["loss_cbm_gate_sparse"]
        + losses["loss_cbm_gate_boundary"]
    )
    assert torch.allclose(total, expected)
    assert torch.allclose(losses["loss_cbm_total"], expected)
    assert torch.allclose(losses["loss_cbm_gate"], losses["loss_cbm_gate_sparse"] + losses["loss_cbm_gate_boundary"])


def test_cbm_losses_backward_with_valid_tokens():
    aux, sources = _make_aux(requires_grad=True)
    total, losses = compute_cbm_losses(aux, _square_gt(), _config())

    assert total.requires_grad
    assert losses["loss_cbm_mem"].item() > 0.0
    assert losses["loss_cbm_aff"].item() > 0.0
    total.backward()

    for name in ("Y_src", "Y_ctx_src", "gate_src", "p_src"):
        grad = sources[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name


def test_cbm_affinity_loss_falls_back_to_z_mem3_without_p_final():
    aux, sources = _make_aux(requires_grad=True, include_p_final=False)
    total, losses = compute_cbm_losses(aux, _square_gt(), _config())

    assert losses["loss_cbm_aff"].item() > 0.0
    total.backward()
    assert sources["z_mem3"].grad is not None
    assert torch.isfinite(sources["z_mem3"].grad).all()
