from __future__ import annotations


def prepare_epoch(cbm, model, labeled_loader, epoch):
    if cbm is not None:
        cbm.prepare_epoch(model, labeled_loader, epoch)


def merge_cbm_loss(base_loss, cbm, aux, gt):
    if cbm is None:
        return base_loss
    return base_loss + cbm.compute_losses(aux, gt)
