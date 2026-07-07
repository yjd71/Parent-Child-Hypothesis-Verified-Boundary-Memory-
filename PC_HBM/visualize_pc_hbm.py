"""Visualization helpers for PC-HBM diagnostics."""

from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn.functional as F


def save_pc_hbm_visualizations(aux: Dict, out_dir: str, prefix: str = "pc_hbm") -> list[str]:
    """Save key PC-HBM maps as grayscale PNG files.

    The function is intentionally optional and callable from debugging scripts;
    training does not save these maps by default.
    """

    os.makedirs(out_dir, exist_ok=True)
    try:
        from torchvision.utils import save_image
    except Exception:
        return []
    saved = []
    pc = aux.get("pc_hbm", {}) or {}
    p2 = aux.get("p2_bra", {}) or {}
    p1 = aux.get("p1_pra", {}) or {}
    mix = aux.get("mixture", {}) or {}
    maps = {
        "B3": pc.get("B3"),
        "B2": p2.get("B2"),
        "B1": p1.get("B1"),
        "C23_map": pc.get("C23_map"),
        "gate_pc_map": pc.get("gate_pc_map"),
        "M_pc_map": pc.get("M_pc_map"),
        "O_pc_norm": _norm(pc.get("O_pc_map")),
        "p3_corr_delta_norm": _p3_delta(aux),
        "G_pix": mix.get("G_pix"),
        "R_pix": mix.get("R_pix"),
        "O_pix_norm": _norm(mix.get("O_pix")),
        "R_sup": mix.get("R_sup"),
        "Mask_corr": mix.get("Mask_corr"),
        "pi_keep": _channel(mix.get("pi"), 0),
        "pi_res": _channel(mix.get("pi"), 1),
        "pi_def": _channel(mix.get("pi"), 2),
        "pi_sup": _channel(mix.get("pi"), 3),
        "z_main": torch.sigmoid(aux.get("z_main")) if torch.is_tensor(aux.get("z_main")) else None,
        "z_res": torch.sigmoid(mix.get("z_res")) if torch.is_tensor(mix.get("z_res")) else None,
        "z_def": torch.sigmoid(mix.get("z_def")) if torch.is_tensor(mix.get("z_def")) else None,
        "z_sup": torch.sigmoid(mix.get("z_sup")) if torch.is_tensor(mix.get("z_sup")) else None,
        "z_final": torch.sigmoid(aux.get("z_final")) if torch.is_tensor(aux.get("z_final")) else None,
        "teacher_pseudo": aux.get("teacher_pseudo"),
    }
    for name, tensor in maps.items():
        if not torch.is_tensor(tensor):
            continue
        path = os.path.join(out_dir, f"{prefix}_{name}.png")
        img = _normalize(tensor.detach().float().cpu())
        save_image(img[: min(4, img.size(0))], path)
        saved.append(path)
    return saved


def _channel(tensor, idx):
    if not torch.is_tensor(tensor) or tensor.size(1) <= idx:
        return None
    return tensor[:, idx : idx + 1]


def _norm(tensor):
    if not torch.is_tensor(tensor):
        return None
    return torch.linalg.vector_norm(tensor, dim=1, keepdim=True)


def _p3_delta(aux):
    p3 = aux.get("p3")
    p3_corr = aux.get("p3_corr")
    if not torch.is_tensor(p3) or not torch.is_tensor(p3_corr):
        return None
    return torch.linalg.vector_norm(p3_corr - p3, dim=1, keepdim=True)


def _normalize(tensor):
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.size(1) != 1:
        tensor = tensor[:, :1]
    t_min = tensor.amin(dim=(-2, -1), keepdim=True)
    t_max = tensor.amax(dim=(-2, -1), keepdim=True)
    return (tensor - t_min) / (t_max - t_min).clamp_min(1e-6)
