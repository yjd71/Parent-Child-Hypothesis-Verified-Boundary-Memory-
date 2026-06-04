from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from CBM.core.tensor_ops import entropy_uncertainty
from CBM.retrieval.scatter import scatter_tokens


class PointwiseBoundaryRetriever(nn.Module):
    """Point-wise dense memory retrieval for predicted p3 boundary tokens."""

    def __init__(
        self,
        p3_channels: int,
        memory_dim: int = 128,
        value_dim: int = 8,
        topk_token: int = 16,
        tau: float = 0.07,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if p3_channels <= 0:
            raise ValueError(f"p3_channels must be positive, got {p3_channels}")
        if memory_dim <= 0:
            raise ValueError(f"memory_dim must be positive, got {memory_dim}")
        if value_dim <= 0:
            raise ValueError(f"value_dim must be positive, got {value_dim}")
        self.memory_dim = int(memory_dim)
        self.value_dim = int(value_dim)
        self.topk_token = int(topk_token)
        self.tau = float(tau)
        self.eps = float(eps)
        self.q_proj = nn.Conv2d(int(p3_channels), self.memory_dim, kernel_size=1, bias=False)

    def forward(
        self,
        p3: torch.Tensor,
        B_query: Optional[torch.Tensor] = None,
        boundary_mask: Optional[torch.Tensor] = None,
        K_mem: Optional[torch.Tensor] = None,
        V_mem: Optional[torch.Tensor] = None,
        topk_token: Optional[int] = None,
        tau: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if p3.dim() != 4:
            raise ValueError(f"p3 must have shape [B, C, H, W], got {tuple(p3.shape)}")

        k_req = self.topk_token if topk_token is None else int(topk_token)
        tau_value = self.tau if tau is None else float(tau)
        if k_req <= 0 or K_mem is None or V_mem is None:
            return self._empty_outputs(p3)

        self._validate_memory(K_mem, V_mem)
        if K_mem.size(0) == 0:
            return self._empty_outputs(p3)

        mask = self._prepare_boundary_mask(p3, B_query, boundary_mask)
        if mask is None or not mask.any():
            return self._empty_outputs(p3)

        K_mem = K_mem.to(device=p3.device, dtype=p3.dtype)
        V_mem = V_mem.to(device=p3.device, dtype=p3.dtype)
        K_norm = F.normalize(K_mem, dim=1, eps=self.eps)

        q_map = F.normalize(self.q_proj(p3), dim=1, eps=self.eps)
        bsz, _, height, width = q_map.shape
        q_flat = q_map.permute(0, 2, 3, 1).reshape(bsz, height * width, self.memory_dim)
        mask_flat = mask[:, 0].reshape(bsz, height * width)
        boundary_positions = mask_flat.nonzero(as_tuple=False)
        if boundary_positions.numel() == 0:
            return self._empty_outputs(p3)

        batch_indices = boundary_positions[:, 0]
        spatial_indices = boundary_positions[:, 1]
        q_bd = q_flat[batch_indices, spatial_indices]

        k = min(k_req, K_norm.size(0))
        if k <= 0:
            return self._empty_outputs(p3)
        sim = q_bd @ K_norm.transpose(0, 1)
        topv, topi = sim.topk(k=k, dim=1)
        attn = F.softmax(topv / max(tau_value, self.eps), dim=1)

        v_top = V_mem[topi]
        k_top = K_norm[topi]
        y = (attn.unsqueeze(-1) * v_top).sum(dim=1)
        r = (attn.unsqueeze(-1) * k_top).sum(dim=1)
        u = entropy_uncertainty(y[:, :4], eps=self.eps).unsqueeze(1)

        outputs = self._empty_outputs(p3)
        outputs["Y_map"] = scatter_tokens(outputs["Y_map"], batch_indices, spatial_indices, y, height, width)
        outputs["R_map"] = scatter_tokens(outputs["R_map"], batch_indices, spatial_indices, r, height, width)
        outputs["U_map"] = scatter_tokens(outputs["U_map"], batch_indices, spatial_indices, u, height, width)
        outputs["valid_map"] = scatter_tokens(
            outputs["valid_map"],
            batch_indices,
            spatial_indices,
            torch.ones_like(u),
            height,
            width,
        )
        return outputs

    def _empty_outputs(self, p3: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, _, height, width = p3.shape
        return {
            "Y_map": torch.zeros(bsz, self.value_dim, height, width, device=p3.device, dtype=p3.dtype),
            "R_map": torch.zeros(bsz, self.memory_dim, height, width, device=p3.device, dtype=p3.dtype),
            "U_map": torch.zeros(bsz, 1, height, width, device=p3.device, dtype=p3.dtype),
            "valid_map": torch.zeros(bsz, 1, height, width, device=p3.device, dtype=p3.dtype),
        }

    def _validate_memory(self, K_mem: torch.Tensor, V_mem: torch.Tensor) -> None:
        if K_mem.dim() != 2 or K_mem.size(1) != self.memory_dim:
            raise ValueError(f"K_mem must have shape [N, {self.memory_dim}], got {tuple(K_mem.shape)}")
        if V_mem.dim() != 2 or V_mem.size(1) != self.value_dim:
            raise ValueError(f"V_mem must have shape [N, {self.value_dim}], got {tuple(V_mem.shape)}")
        if K_mem.size(0) != V_mem.size(0):
            raise ValueError(f"K_mem and V_mem must have the same N, got {K_mem.size(0)} and {V_mem.size(0)}")

    def _prepare_boundary_mask(
        self,
        p3: torch.Tensor,
        B_query: Optional[torch.Tensor],
        boundary_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        target_size = p3.shape[-2:]
        if boundary_mask is None:
            if B_query is None:
                return None
            query = self._as_4d_single_channel(B_query, "B_query").to(device=p3.device, dtype=p3.dtype)
            if tuple(query.shape[-2:]) != tuple(target_size):
                query = F.interpolate(query, size=target_size, mode="bilinear", align_corners=False)
            return query > 0

        mask = self._as_4d_single_channel(boundary_mask, "boundary_mask").to(device=p3.device)
        if tuple(mask.shape[-2:]) != tuple(target_size):
            mask = F.interpolate(mask.float(), size=target_size, mode="nearest")
        return mask.bool()

    def _as_4d_single_channel(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"{name} must have shape [B, 1, H, W] or [B, H, W], got {tuple(x.shape)}")
        return x
