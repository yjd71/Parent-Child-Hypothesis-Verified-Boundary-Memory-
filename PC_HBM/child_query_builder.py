"""Build p2_pre child queries from p3 boundary token coordinates."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .child_local_encoder import ChildLocalEncoder
from .utils import gather_tokens, scale_flat_indices


class ChildQueryBuilder(nn.Module):
    """Crop p2_pre patches and encode ``q_child [M,512]`` plus geometry query."""

    def __init__(self, p2_ch: int, dim: int = 512, window: int = 5) -> None:
        super().__init__()
        self.window = int(window)
        self.encoder = ChildLocalEncoder(p2_ch, dim=dim, window=window)

    def forward(
        self,
        p2_pre: torch.Tensor,
        batch_ids3: torch.Tensor,
        flat_indices3: torch.Tensor,
        p3_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        p2_hw = tuple(int(v) for v in p2_pre.shape[-2:])
        flat2 = scale_flat_indices(flat_indices3, p3_hw, p2_hw)
        patches = self._crop_patches(p2_pre, batch_ids3, flat2)
        q_child = self.encoder(patches)
        g2_query = self._geometry_from_indices(flat2, p2_hw, p2_pre.device, p2_pre.dtype)
        return {"q_child": q_child, "G2_query": g2_query, "child_patches": patches, "flat_indices2_from_p3": flat2}

    def _crop_patches(self, p2: torch.Tensor, batch_ids: torch.Tensor, flat2: torch.Tensor) -> torch.Tensor:
        if flat2.numel() == 0:
            return p2.new_empty(0, p2.size(1), self.window, self.window)
        pad = self.window // 2
        padded = F.pad(p2, (pad, pad, pad, pad), mode="replicate")
        h, w = p2.shape[-2:]
        y = torch.div(flat2.long(), w, rounding_mode="floor") + pad
        x = flat2.long().remainder(w) + pad
        patches = []
        for idx in range(flat2.numel()):
            b = int(batch_ids[idx])
            yy = int(y[idx])
            xx = int(x[idx])
            patches.append(padded[b : b + 1, :, yy - pad : yy + pad + 1, xx - pad : xx + pad + 1])
        return torch.cat(patches, dim=0)

    def _geometry_from_indices(self, flat2: torch.Tensor, hw: Tuple[int, int], device, dtype) -> torch.Tensor:
        if flat2.numel() == 0:
            return torch.empty(0, 6, device=device, dtype=dtype)
        h, w = int(hw[0]), int(hw[1])
        y = torch.div(flat2.long(), w, rounding_mode="floor").to(device=device, dtype=dtype)
        x = flat2.long().remainder(w).to(device=device, dtype=dtype)
        yy = y / max(h - 1, 1) * 2.0 - 1.0
        xx = x / max(w - 1, 1) * 2.0 - 1.0
        zeros = torch.zeros_like(xx)
        ones = torch.ones_like(xx)
        return torch.stack([zeros, xx, yy, xx, yy, ones], dim=1)
