from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def build_fallback_aux(reason: str, p3: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    aux: Dict[str, Any] = {
        "cbm_used": False,
        "fallback_reason": reason,
        "top_img_ids": [],
        "img_scores": None,
        "num_memory_tokens": 0,
        "num_valid_boundary_tokens": 0,
        "valid_ratio": 0.0,
        "B3_mean": 0.0,
        "gate_mean": 0.0,
        "cons_mean": 0.0,
        "u_mean": 0.0,
        "p_final": None,
        "p_main": None,
        "B_query": None,
        "boundary_mask": None,
        "z_mem3": None,
        "gate3": None,
        "Y_map": None,
        "Y_ctx": None,
        "R_map": None,
        "R_ctx": None,
        "U_map": None,
        "valid_map": None,
        "cons_map": None,
        "prob3": None,
    }
    if p3 is not None:
        aux["p3_shape"] = tuple(p3.shape)
    return aux


def build_used_aux(
    *,
    top_img_ids,
    img_scores: torch.Tensor,
    K_mem: torch.Tensor,
    B_query: torch.Tensor,
    boundary_mask: torch.Tensor,
    gate3: torch.Tensor,
    z_mem3: torch.Tensor,
    Y_map: torch.Tensor,
    Y_ctx: torch.Tensor,
    R_map: torch.Tensor,
    R_ctx: torch.Tensor,
    cons_map: torch.Tensor,
    U_map: torch.Tensor,
    valid_map: torch.Tensor,
    prob3: torch.Tensor,
    meta,
    p_final: Optional[torch.Tensor] = None,
    p_main: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    valid_float = valid_map.to(dtype=B_query.dtype)
    return {
        "cbm_used": True,
        "fallback_reason": None,
        "top_img_ids": top_img_ids,
        "img_scores": img_scores.detach(),
        "num_memory_tokens": int(K_mem.size(0)),
        "num_valid_boundary_tokens": int(valid_float.sum().detach().item()),
        "valid_ratio": float(valid_float.mean().detach().item()),
        "B3_mean": float(B_query.detach().mean().item()),
        "gate_mean": float(gate3.detach().mean().item()),
        "cons_mean": float(cons_map.detach().mean().item()),
        "u_mean": float(U_map.detach().mean().item()),
        "p_final": None if p_final is None else p_final.detach(),
        "p_main": None if p_main is None else p_main.detach(),
        "B_query": B_query,
        "boundary_mask": boundary_mask,
        "z_mem3": z_mem3,
        "gate3": gate3,
        "Y_map": Y_map,
        "Y_ctx": Y_ctx,
        "R_map": R_map,
        "R_ctx": R_ctx,
        "U_map": U_map,
        "valid_map": valid_map,
        "cons_map": cons_map,
        "prob3": prob3,
        "memory_meta": meta,
    }
