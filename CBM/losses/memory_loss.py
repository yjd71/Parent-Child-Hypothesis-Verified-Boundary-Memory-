from .total import compute_cbm_losses


def memory_loss(aux=None, gt=None):
    _, losses = compute_cbm_losses(aux, gt)
    return losses["loss_cbm_mem"]
