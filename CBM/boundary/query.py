from __future__ import annotations

from typing import Tuple

import torch

from .morphology import as_4d_mask, dilate, erode, gradient_magnitude


def build_pred_boundary(
    prob: torch.Tensor,
    kernel: int = 3,
    alpha_unc: float = 0.5,
    alpha_grad: float = 0.5,
    theta: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build PLAN_V4.2 predicted boundary query map from a probability map."""
    prob = as_4d_mask(prob, "prob").float().clamp(0.0, 1.0)
    p_bin = (prob >= 0.5).to(dtype=prob.dtype)

    b_morph = (dilate(p_bin, kernel=kernel) - erode(p_bin, kernel=kernel)).clamp(0.0, 1.0)
    b_unc = (4.0 * prob * (1.0 - prob)).clamp(0.0, 1.0)
    b_grad = gradient_magnitude(prob)

    b_query = b_morph + float(alpha_unc) * b_unc + float(alpha_grad) * b_grad
    denom = b_query.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    b_query = (b_query / denom).clamp(0.0, 1.0)
    boundary_mask = b_query > float(theta)
    return b_query, boundary_mask
