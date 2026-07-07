"""Scatter PC-HBM token evidence back to p3 maps."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from ..common.utils import scatter_tokens


def pc_scatter(
    batch_size: int,
    height: int,
    width: int,
    batch_ids: torch.Tensor,
    flat_indices: torch.Tensor,
    token_aux: Dict[str, torch.Tensor],
    gate_pc_token: torch.Tensor,
    c23_token: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Scatter token tensors into p3 BCHW maps."""

    device = gate_pc_token.device
    dtype = gate_pc_token.dtype
    shape1 = (batch_size, 1, height, width)
    return {
        "M_pc_map": scatter_tokens(shape1, batch_ids, flat_indices, token_aux["M_pc_token"], reduce="replace"),
        "O_pc_map": scatter_tokens((batch_size, 2, height, width), batch_ids, flat_indices, token_aux["O_pc_token"], reduce="replace"),
        "gate_pc_map": scatter_tokens(shape1, batch_ids, flat_indices, gate_pc_token, reduce="replace"),
        "C23_map": scatter_tokens(shape1, batch_ids, flat_indices, c23_token, reduce="replace"),
        "Z3_map": scatter_tokens((batch_size, token_aux["Z3_token"].size(1), height, width), batch_ids, flat_indices, token_aux["Z3_token"], reduce="replace"),
        "E_attn_map": scatter_tokens((batch_size, token_aux["E_attn"].size(1), height, width), batch_ids, flat_indices, token_aux["E_attn"], reduce="replace"),
        "G_attn_map": scatter_tokens((batch_size, token_aux["G_attn"].size(1), height, width), batch_ids, flat_indices, token_aux["G_attn"], reduce="replace"),
        "valid3_map": scatter_tokens(shape1, batch_ids, flat_indices, torch.ones_like(gate_pc_token), reduce="replace"),
    }
