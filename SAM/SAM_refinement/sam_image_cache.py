from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Callable, Optional, Tuple

import numpy as np
import torch


class SAMImageEmbeddingCache:
    """Lightweight LRU cache for SAM image embeddings.

    The cache stores detached CPU tensors keyed by a stable hash of the image
    content plus an optional backend/model tag. It is intentionally small and
    in-memory: the goal is to avoid recomputing image encoder outputs when the
    exact same image is revisited within a run.
    """

    def __init__(self, cfg=None, backend_tag: str = "", model_tag: str = "", enabled: Optional[bool] = None, max_items: Optional[int] = None) -> None:
        self.enabled = bool(getattr(cfg, "use_sam_cache", True)) if enabled is None else bool(enabled)
        if max_items is None:
            max_items = getattr(cfg, "sam_image_embedding_cache_size", 128)
        self.max_items = max(1, int(max_items))
        self.backend_tag = str(backend_tag)
        self.model_tag = str(model_tag)
        self._entries: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get_or_compute(
        self,
        image,
        compute_fn: Callable[[], torch.Tensor],
        device=None,
        dtype=None,
        extra_tag: str = "",
    ) -> Tuple[torch.Tensor, bool]:
        """Return a cached embedding if available, otherwise compute and store it."""
        if not self.enabled:
            value = self._ensure_tensor(compute_fn())
            return self._move(value, device=device, dtype=dtype), False

        key = self.make_key(image, extra_tag=extra_tag)
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            self._entries.move_to_end(key)
            return self._move(cached, device=device, dtype=dtype), True

        self.misses += 1
        value = self._ensure_tensor(compute_fn())
        self._entries[key] = value.detach().cpu()
        self._entries.move_to_end(key)
        self._trim()
        return self._move(value, device=device, dtype=dtype), False

    def make_key(self, image, extra_tag: str = "") -> str:
        array = self._as_contiguous_array(image)
        digest = hashlib.sha1(array.tobytes()).hexdigest()
        return "|".join(
            (
                self.backend_tag,
                self.model_tag,
                str(extra_tag),
                str(array.dtype),
                str(array.shape),
                digest,
            )
        )

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def cache_info(self) -> dict:
        return {
            "enabled": self.enabled,
            "size": len(self._entries),
            "max_items": self.max_items,
            "hits": self.hits,
            "misses": self.misses,
        }

    def _trim(self) -> None:
        while len(self._entries) > self.max_items:
            self._entries.popitem(last=False)

    @staticmethod
    def _ensure_tensor(value) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError("SAM image embedding cache only supports torch.Tensor values")
        return value.detach()

    @staticmethod
    def _move(value: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
        out = value
        if device is not None or dtype is not None:
            out = out.to(device=device if device is not None else out.device, dtype=dtype if dtype is not None else out.dtype)
        return out

    @staticmethod
    def _as_contiguous_array(image) -> np.ndarray:
        if torch.is_tensor(image):
            return image.detach().cpu().contiguous().numpy()
        array = np.asarray(image)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return array


__all__ = ["SAMImageEmbeddingCache"]
