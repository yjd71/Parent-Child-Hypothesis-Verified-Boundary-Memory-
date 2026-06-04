from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from CBM.correction.gates import make_gate_head


class BoundaryCorrectionHead(nn.Module):
    """Boundary-aware p3 correction and memory-logit generation."""

    def __init__(
        self,
        p3_channels: int,
        memory_dim: int = 128,
        value_dim: int = 8,
        lambda_feat: float = 0.1,
    ) -> None:
        super().__init__()
        if p3_channels <= 0:
            raise ValueError(f"p3_channels must be positive, got {p3_channels}")
        if memory_dim <= 0:
            raise ValueError(f"memory_dim must be positive, got {memory_dim}")
        if value_dim < 4:
            raise ValueError(f"value_dim must be at least 4, got {value_dim}")

        self.p3_channels = int(p3_channels)
        self.memory_dim = int(memory_dim)
        self.value_dim = int(value_dim)
        self.lambda_feat = float(lambda_feat)

        self.gate_head = make_gate_head(self.value_dim)
        self.r_back = nn.Conv2d(self.memory_dim, self.p3_channels, kernel_size=1, bias=False)

        self.logit_tau_fg_bg = nn.Parameter(torch.tensor(1.0))
        self.logit_tau_bd = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        p3: torch.Tensor,
        m3: torch.Tensor,
        B_query: torch.Tensor,
        Y_map: torch.Tensor,
        Y_ctx: torch.Tensor,
        R_ctx: torch.Tensor,
        U_map: torch.Tensor,
        cons_map: torch.Tensor,
        valid_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if p3.dim() != 4 or p3.size(1) != self.p3_channels:
            raise ValueError(f"p3 must have shape [B, {self.p3_channels}, H, W], got {tuple(p3.shape)}")

        Y_map = self._prepare_context_map(Y_map, "Y_map", p3, self.value_dim)
        Y_ctx = self._prepare_context_map(Y_ctx, "Y_ctx", p3, self.value_dim)
        R_ctx = self._prepare_context_map(R_ctx, "R_ctx", p3, self.memory_dim)

        prob3 = torch.sigmoid(self._prepare_single_channel(m3, "m3", p3, mode="bilinear")).clamp_(0.0, 1.0)
        B_query = self._prepare_single_channel(B_query, "B_query", p3, mode="bilinear").clamp_(0.0, 1.0)
        U_map = self._prepare_single_channel(U_map, "U_map", p3, mode="bilinear").clamp_(0.0, 1.0)
        cons_map = self._prepare_single_channel(cons_map, "cons_map", p3, mode="bilinear").clamp_(0.0, 1.0)
        valid_map = self._prepare_single_channel(valid_map, "valid_map", p3, mode="nearest") > 0.5
        valid_float = valid_map.to(dtype=p3.dtype)

        gate_in = torch.cat(
            [
                Y_map,
                Y_ctx,
                Y_map - Y_ctx,
                U_map,
                cons_map,
                B_query,
                prob3,
            ],
            dim=1,
        )
        gate3 = torch.sigmoid(self.gate_head(gate_in)) * cons_map.detach() * valid_float

        r_ctx_back = self.r_back(R_ctx)
        p3_corr = p3 + self.lambda_feat * gate3 * B_query * (r_ctx_back - p3)

        fg_evidence = Y_ctx[:, 0:1] + Y_ctx[:, 1:2]
        bg_evidence = Y_ctx[:, 2:3] + Y_ctx[:, 3:4]
        boundary_evidence = Y_ctx[:, 1:2] - Y_ctx[:, 2:3]
        z_mem3 = (
            self.logit_tau_fg_bg.to(dtype=p3.dtype) * (fg_evidence - bg_evidence)
            + self.logit_tau_bd.to(dtype=p3.dtype) * boundary_evidence
            + self.logit_bias.to(dtype=p3.dtype)
        )
        z_mem3 = z_mem3 * valid_float
        return p3_corr, z_mem3, gate3

    def _prepare_context_map(
        self,
        x: torch.Tensor,
        name: str,
        ref: torch.Tensor,
        channels: int,
    ) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"{name} must have shape [B, {channels}, H, W], got {tuple(x.shape)}")
        if x.size(0) != ref.size(0) or x.size(1) != channels or tuple(x.shape[-2:]) != tuple(ref.shape[-2:]):
            raise ValueError(
                f"{name} must match shape [B, {channels}, H, W] with p3 batch/spatial, "
                f"got {tuple(x.shape)} for p3 {tuple(ref.shape)}"
            )
        return x.to(device=ref.device, dtype=ref.dtype)

    def _prepare_single_channel(
        self,
        x: torch.Tensor,
        name: str,
        ref: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"{name} must have shape [B, 1, H, W] or [B, H, W], got {tuple(x.shape)}")
        if x.size(0) != ref.size(0):
            raise ValueError(f"{name} batch size must match p3, got {x.size(0)} and {ref.size(0)}")

        x = x.to(device=ref.device, dtype=ref.dtype)
        if tuple(x.shape[-2:]) == tuple(ref.shape[-2:]):
            return x
        if mode == "nearest":
            return F.interpolate(x, size=ref.shape[-2:], mode=mode)
        return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False)
