from .total import compute_cbm_losses


def gate_loss(aux=None, gt=None):
    _, losses = compute_cbm_losses(aux, gt)
    return losses["loss_cbm_gate"]
