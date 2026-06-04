from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from CBM.losses.boundary_memory_losses import compute_boundary_memory_losses


def compute_cbm_losses(
    aux: Dict[str, Any] | None,
    gt: torch.Tensor | None = None,
    config: Any = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    loss_dict = compute_boundary_memory_losses(aux, gt, config=config)
    return loss_dict["loss_cbm_total"], loss_dict
