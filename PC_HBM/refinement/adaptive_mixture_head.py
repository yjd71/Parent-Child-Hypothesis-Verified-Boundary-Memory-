"""Final adaptive keep/residual/deformation/suppress mixture."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boundary_deformation import deform_logits
from ..common.utils import gradient_strength


class AdaptiveMixtureHead(nn.Module):
    """Pixel-level adaptive mixture over keep/residual/deformation/suppress logits."""

    def __init__(
        self,
        r_max: float = 2.0,
        max_offset: float = 3.0,
        mask_corr_epsilon: float = 0.10,
        init_bias: list[float] | tuple[float, ...] = (1.0, -0.5, -0.5, -0.5),
        use_branch_quality: bool = True,
        use_branch_dropout: bool = True,
    ) -> None:
        super().__init__()
        self.r_max = float(r_max)
        self.max_offset = float(max_offset)
        self.mask_corr_epsilon = float(mask_corr_epsilon)
        self.use_branch_dropout = bool(use_branch_dropout)
        self.mix_head = nn.Sequential(
            nn.Conv2d(14, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 4, kernel_size=1),
        )
        with torch.no_grad():
            self.mix_head[-1].bias.copy_(torch.tensor(init_bias, dtype=self.mix_head[-1].bias.dtype))
        self.quality_head = nn.Sequential(
            nn.Conv2d(14, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 4, kernel_size=1),
        ) if use_branch_quality else None

    def forward(
        self,
        z_main: torch.Tensor,
        p1_aux: Dict[str, torch.Tensor],
        pc_maps: Dict[str, torch.Tensor],
        epoch: int | None = None,
        temperature: float = 1.0,
        eps_floor: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        size = z_main.shape[-2:]
        p_main = torch.sigmoid(z_main)
        B_pix = F.interpolate(p1_aux["B1"], size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        G_pix = F.interpolate(p1_aux["G1_map"], size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        valid_pix = F.interpolate(p1_aux["valid1_map"], size=size, mode="nearest").clamp(0.0, 1.0)
        R_pix = torch.tanh(F.interpolate(p1_aux["R1_map"], size=size, mode="bilinear", align_corners=False)) * self.r_max
        O_pix = torch.tanh(F.interpolate(p1_aux["O1_map"], size=size, mode="bilinear", align_corners=False)) * self.max_offset
        R_sup = F.softplus(F.interpolate(p1_aux["R_sup_map"], size=size, mode="bilinear", align_corners=False))
        Mask_corr = valid_pix * G_pix * (self.mask_corr_epsilon + (1.0 - self.mask_corr_epsilon) * B_pix)
        z_keep = z_main
        z_res = z_main + Mask_corr * R_pix
        z_def = deform_logits(z_main, O_pix, Mask_corr)
        z_sup = z_main - Mask_corr * R_sup
        uncertainty = 4.0 * p_main * (1.0 - p_main)
        grad = gradient_strength(p_main)
        C23_up = F.interpolate(pc_maps["C23_map"], size=size, mode="bilinear", align_corners=False)
        M_pc_up = F.interpolate(pc_maps["M_pc_map"], size=size, mode="bilinear", align_corners=False)
        O_mag = torch.linalg.vector_norm(O_pix, dim=1, keepdim=True) / max(self.max_offset, 1e-6)
        context = torch.cat([p_main, B_pix, G_pix, Mask_corr, uncertainty, grad, C23_up, M_pc_up, O_pix, R_pix, R_sup, valid_pix, O_mag], dim=1)
        mix_logits = self.mix_head(context)
        if self.training and self.use_branch_dropout:
            drop = torch.rand(mix_logits.size(0), 4, 1, 1, device=mix_logits.device, dtype=mix_logits.dtype)
            mix_logits = mix_logits.masked_fill(drop < 0.02, -1.0e4)
        pi = torch.softmax(mix_logits / max(float(temperature), 1e-6), dim=1)
        if eps_floor > 0:
            pi = (1.0 - 4.0 * eps_floor) * pi + eps_floor
            pi = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-6)
        branches = torch.cat([z_keep, z_res, z_def, z_sup], dim=1)
        z_final = (pi * branches).sum(dim=1, keepdim=True)
        p_final = torch.sigmoid(z_final)
        quality = self.quality_head(context) if self.quality_head is not None else torch.zeros_like(pi)
        return {
            "z_keep": z_keep,
            "z_res": z_res,
            "z_def": z_def,
            "z_sup": z_sup,
            "z_warp": z_def,
            "pi": pi,
            "mix_logits": mix_logits,
            "pred_gain": quality,
            "branch_quality": quality,
            "B_pix": B_pix,
            "G_pix": G_pix,
            "Mask_corr": Mask_corr,
            "R_pix": R_pix,
            "O_pix": O_pix,
            "R_sup": R_sup,
            "valid_pix": valid_pix,
            "z_final": z_final,
            "p_final": p_final,
        }
