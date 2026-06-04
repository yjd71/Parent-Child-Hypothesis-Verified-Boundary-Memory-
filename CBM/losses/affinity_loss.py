from .total import compute_cbm_losses


def affinity_loss(aux=None, gt=None):
    _, losses = compute_cbm_losses(aux, gt)
    return losses["loss_cbm_aff"]
