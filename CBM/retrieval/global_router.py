from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalMemoryRouter(nn.Module):
    """Image-level global routing from x3 features to labeled image memory."""

    def __init__(
        self,
        x3_channels: int,
        memory_dim: int = 128,
        top_img_k: int = 8,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if x3_channels <= 0:
            raise ValueError(f"x3_channels must be positive, got {x3_channels}")
        if memory_dim <= 0:
            raise ValueError(f"memory_dim must be positive, got {memory_dim}")
        self.memory_dim = int(memory_dim)
        self.top_img_k = int(top_img_k)
        self.eps = float(eps)
        self.proj = nn.Conv2d(int(x3_channels), self.memory_dim, kernel_size=1, bias=False)

    def forward(
        self,
        x3: torch.Tensor,
        memory,
        top_img_k: Optional[int] = None,
    ) -> Tuple[List[List[str]], torch.Tensor]:
        if x3.dim() != 4:
            raise ValueError(f"x3 must have shape [B, C, H, W], got {tuple(x3.shape)}")

        q = self._encode_query(x3)
        requested_k = self.top_img_k if top_img_k is None else int(top_img_k)
        if requested_k <= 0 or memory is None or not memory.is_ready():
            return self._empty_result(x3.size(0), q)

        image_keys, image_ids = memory.get_image_keys(device=x3.device, dtype=q.dtype)
        n_img = min(image_keys.size(0), len(image_ids))
        if n_img <= 0:
            return self._empty_result(x3.size(0), q)

        image_keys = image_keys[:n_img]
        image_ids = list(image_ids[:n_img])
        if image_keys.size(1) != self.memory_dim:
            image_keys = self._fit_memory_dim(image_keys)

        k = min(requested_k, n_img)
        image_keys = F.normalize(image_keys, dim=1, eps=self.eps)
        sim = q @ image_keys.transpose(0, 1)
        img_scores, top_indices = sim.topk(k=k, dim=1)
        top_img_ids = self._indices_to_img_ids(top_indices, image_ids)
        return top_img_ids, img_scores

    def _encode_query(self, x3: torch.Tensor) -> torch.Tensor:
        q = self.proj(x3)
        q = F.adaptive_avg_pool2d(q, output_size=1).flatten(1)
        return F.normalize(q, dim=1, eps=self.eps)

    def _empty_result(self, batch_size: int, q: torch.Tensor) -> Tuple[List[List[str]], torch.Tensor]:
        return [[] for _ in range(batch_size)], torch.empty(
            batch_size,
            0,
            device=q.device,
            dtype=q.dtype,
        )

    def _fit_memory_dim(self, image_keys: torch.Tensor) -> torch.Tensor:
        if image_keys.size(1) > self.memory_dim:
            return image_keys[:, : self.memory_dim]
        pad = self.memory_dim - image_keys.size(1)
        return F.pad(image_keys, (0, pad))

    def _indices_to_img_ids(self, top_indices: torch.Tensor, image_ids: Iterable[str]) -> List[List[str]]:
        image_ids = list(image_ids)
        return [
            [str(image_ids[int(idx)]) for idx in row]
            for row in top_indices.detach().cpu().tolist()
        ]
