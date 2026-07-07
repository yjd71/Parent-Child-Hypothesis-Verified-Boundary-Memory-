"""Branch oracle targets for PC-HBM adaptive mixture supervision."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def branch_errors(branch_logits: Dict[str, torch.Tensor], gt: torch.Tensor) -> torch.Tensor:
    """Return per-branch pixel errors ``[B,4,H,W]``."""

    target = gt
    if target.shape[-2:] != branch_logits["z_keep"].shape[-2:]:
        target = F.interpolate(target.float(), size=branch_logits["z_keep"].shape[-2:], mode="nearest")
    branches = [branch_logits[name] for name in ("z_keep", "z_res", "z_def", "z_sup")]
    errors = []
    for z in branches:
        bce = F.binary_cross_entropy_with_logits(z, target, reduction="none")
        abs_err = (torch.sigmoid(z) - target).abs()
        errors.append(bce + abs_err)
    return torch.cat(errors, dim=1)


def oracle_distribution(branch_logits: Dict[str, torch.Tensor], gt: torch.Tensor, tau: float = 0.5) -> Dict[str, torch.Tensor]:
    """Build soft oracle mixture targets and improvement mask."""

    err = branch_errors(branch_logits, gt)
    target_mix = torch.softmax(-err / max(float(tau), 1e-6), dim=1)
    keep = err[:, 0:1]
    best = err.min(dim=1, keepdim=True).values
    improvement = keep - best
    b_pix = branch_logits.get("B_pix", torch.ones_like(keep))
    oracle_mask = (improvement > 0.03).float() * b_pix
    return {"pixel_error": err, "target_mix": target_mix, "oracle_mask": oracle_mask, "improvement": improvement}
