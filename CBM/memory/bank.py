from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from CBM.boundary.regions import build_gt_regions
from CBM.memory.labels import DEFAULT_MAX_SIZES, DEFAULT_SAMPLE_PER_IMAGE, REGION_NAMES, REGION_TO_ID
from CBM.memory.reliability import prepare_reliability
from CBM.memory.sampler import sample_indices


class DenseBoundaryMemory:
    """Labeled-only dense feature-label memory for PLAN_V4.2 CBM-PFI."""

    def __init__(
        self,
        mem_dim: int = 128,
        value_dim: int = 8,
        regions: Sequence[str] = REGION_NAMES,
        sample_per_image: Optional[Dict[str, int]] = None,
        max_sizes: Optional[Dict[str, int]] = None,
    ) -> None:
        if value_dim != 8:
            raise ValueError("DenseBoundaryMemory currently uses fixed value_dim=8")
        self.mem_dim = int(mem_dim)
        self.value_dim = int(value_dim)
        self.regions = tuple(regions)
        self.sample_per_image = sample_per_image or dict(DEFAULT_SAMPLE_PER_IMAGE)
        self.max_sizes = max_sizes or dict(DEFAULT_MAX_SIZES)
        self.clear()

    def clear(self) -> None:
        self.image_keys_list: List[torch.Tensor] = []
        self.image_ids: List[str] = []
        self.keys_list: Dict[str, List[torch.Tensor]] = {region: [] for region in self.regions}
        self.values_list: Dict[str, List[torch.Tensor]] = {region: [] for region in self.regions}
        self.meta_list: Dict[str, List[dict]] = {region: [] for region in self.regions}
        self.image_keys = torch.empty(0, self.mem_dim)
        self.keys = {region: torch.empty(0, self.mem_dim) for region in self.regions}
        self.values = {region: torch.empty(0, self.value_dim) for region in self.regions}
        self.meta = {region: [] for region in self.regions}
        self._finalized = False

    @torch.no_grad()
    def append_batch(
        self,
        x3: torch.Tensor,
        p3: torch.Tensor,
        gt: torch.Tensor,
        img_ids: Iterable[object],
        reliability: Optional[torch.Tensor] = None,
    ) -> None:
        if x3.dim() != 4 or p3.dim() != 4:
            raise ValueError(f"x3 and p3 must be 4D tensors, got {tuple(x3.shape)} and {tuple(p3.shape)}")
        if x3.size(0) != p3.size(0):
            raise ValueError("x3 and p3 batch sizes must match")

        batch_size = p3.size(0)
        img_ids = [str(img_id) for img_id in img_ids]
        if len(img_ids) != batch_size:
            raise ValueError(f"img_ids length must match batch size {batch_size}, got {len(img_ids)}")

        device = p3.device
        key_dtype = p3.dtype
        image_keys = self._fit_mem_dim(F.adaptive_avg_pool2d(x3.detach(), 1).flatten(1).to(device=device, dtype=key_dtype))
        self.image_keys_list.append(image_keys.cpu())
        self.image_ids.extend(img_ids)

        regions = build_gt_regions(gt.to(device=device), target_size=p3.shape[-2:])
        p3_tokens = self._fit_mem_dim(p3.detach().flatten(2).transpose(1, 2))
        rel_map = prepare_reliability(reliability, p3, key_dtype)

        for batch_idx, img_id in enumerate(img_ids):
            for region in self.regions:
                region_mask = regions[region][batch_idx, 0].bool().flatten()
                token_indices = region_mask.nonzero(as_tuple=False).flatten()
                if token_indices.numel() == 0:
                    continue
                token_indices = sample_indices(token_indices, self.sample_per_image.get(region, token_indices.numel()))

                token_keys = p3_tokens[batch_idx, token_indices].to(dtype=key_dtype)
                values = self._build_values_for_tokens(regions, region, batch_idx, token_indices, rel_map)
                metas = self._build_meta_for_tokens(img_id, region, token_indices, p3.shape[-2:], values)

                self.keys_list[region].append(token_keys.cpu())
                self.values_list[region].append(values.cpu())
                self.meta_list[region].extend(metas)
        self._finalized = False

    def finalize(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        dtype = dtype or self._infer_dtype()
        device = torch.device("cpu") if device is None else torch.device(device)

        self.image_keys = self._cat_or_empty(self.image_keys_list, self.mem_dim, device, dtype)
        self.keys = {}
        self.values = {}
        self.meta = {}
        for region in self.regions:
            region_keys = self._cat_or_empty(self.keys_list[region], self.mem_dim, device, dtype)
            region_values = self._cat_or_empty(self.values_list[region], self.value_dim, device, dtype)
            region_meta = list(self.meta_list[region])
            max_size = int(self.max_sizes.get(region, region_keys.size(0)))
            if region_keys.size(0) > max_size:
                keep = torch.randperm(region_keys.size(0), device=region_keys.device)[:max_size].sort().values
                region_keys = region_keys.index_select(0, keep)
                region_values = region_values.index_select(0, keep)
                keep_cpu = keep.cpu().tolist()
                region_meta = [region_meta[idx] for idx in keep_cpu]
            self.keys[region] = region_keys
            self.values[region] = region_values
            self.meta[region] = region_meta

        self._finalized = True
    def is_ready(self) -> bool:
        return self._finalized and sum(self.keys[region].size(0) for region in self.regions) > 0

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "mem_dim": self.mem_dim,
            "value_dim": self.value_dim,
            "regions": list(self.regions),
            "sample_per_image": dict(self.sample_per_image),
            "max_sizes": dict(self.max_sizes),
            "image_keys": self.image_keys.detach().cpu(),
            "image_ids": list(self.image_ids),
            "keys": {region: self.keys[region].detach().cpu() for region in self.regions},
            "values": {region: self.values[region].detach().cpu() for region in self.regions},
            "meta": {region: list(self.meta[region]) for region in self.regions},
            "finalized": bool(self._finalized),
        }

    def load_state_dict(
        self,
        state: Optional[Dict[str, Any]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.clear()
        if not state:
            return

        mem_dim = int(state.get("mem_dim", self.mem_dim))
        value_dim = int(state.get("value_dim", self.value_dim))
        if mem_dim != self.mem_dim:
            raise ValueError(f"Memory state mem_dim={mem_dim} does not match current mem_dim={self.mem_dim}")
        if value_dim != self.value_dim:
            raise ValueError(f"Memory state value_dim={value_dim} does not match current value_dim={self.value_dim}")

        device = torch.device("cpu") if device is None else torch.device(device)
        dtype = dtype or self._infer_state_dtype(state)

        self.image_keys = self._load_2d_state_tensor(state.get("image_keys"), self.mem_dim, device, dtype)
        self.image_ids = [str(item) for item in state.get("image_ids", [])]

        raw_keys = state.get("keys", {}) or {}
        raw_values = state.get("values", {}) or {}
        raw_meta = state.get("meta", {}) or {}
        for region in self.regions:
            self.keys[region] = self._load_2d_state_tensor(raw_keys.get(region), self.mem_dim, device, dtype)
            self.values[region] = self._load_2d_state_tensor(raw_values.get(region), self.value_dim, device, dtype)
            if self.keys[region].size(0) != self.values[region].size(0):
                raise ValueError(
                    f"Memory state region {region} has mismatched keys/values: "
                    f"{self.keys[region].size(0)} vs {self.values[region].size(0)}"
                )
            meta_region = raw_meta.get(region, [])
            self.meta[region] = list(meta_region) if isinstance(meta_region, list) else []

        self._finalized = bool(state.get("finalized", True))

    def get_image_keys(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        target = self.image_keys
        if device is not None or dtype is not None:
            target = target.to(device=device or target.device, dtype=dtype or target.dtype)
        return target, list(self.image_ids)

    def get_sub_memory(
        self,
        top_img_ids: Optional[Iterable[object]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
        selected = None if top_img_ids is None else self._normalize_img_id_selection(top_img_ids)
        key_chunks = []
        value_chunks = []
        meta_out: List[dict] = []

        for region in self.regions:
            keys = self.keys[region]
            values = self.values[region]
            metas = self.meta[region]
            if keys.numel() == 0:
                continue
            if selected is None:
                keep = torch.arange(keys.size(0), device=keys.device)
            else:
                keep_list = [idx for idx, item in enumerate(metas) if item["image_id"] in selected]
                if not keep_list:
                    continue
                keep = torch.tensor(keep_list, device=keys.device, dtype=torch.long)
            key_chunks.append(keys.index_select(0, keep))
            value_chunks.append(values.index_select(0, keep))
            meta_out.extend([metas[idx] for idx in keep.cpu().tolist()])

        out_device = torch.device("cpu") if device is None else torch.device(device)
        out_dtype = dtype or self._infer_dtype()
        if not key_chunks:
            return (
                torch.empty(0, self.mem_dim, device=out_device, dtype=out_dtype),
                torch.empty(0, self.value_dim, device=out_device, dtype=out_dtype),
                [],
            )
        keys_out = torch.cat(key_chunks, dim=0).to(device=out_device, dtype=out_dtype)
        values_out = torch.cat(value_chunks, dim=0).to(device=out_device, dtype=out_dtype)
        return keys_out, values_out, meta_out

    def diagnostic_string(self) -> str:
        token_counts = {region: int(self.keys[region].size(0)) for region in self.regions}
        total_tokens = sum(token_counts.values())
        device = self.image_keys.device if isinstance(self.image_keys, torch.Tensor) else torch.device("cpu")
        dtype = self.image_keys.dtype if isinstance(self.image_keys, torch.Tensor) else torch.float32
        region_text = ", ".join(f"{region}={count}" for region, count in token_counts.items())
        return (
            "[CBM] DenseBoundaryMemory "
            f"images={len(self.image_ids)}, tokens={total_tokens}, {region_text}, device={device}, dtype={dtype}"
        )

    def _fit_mem_dim(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == self.mem_dim:
            return x
        if x.size(-1) > self.mem_dim:
            return x[..., : self.mem_dim]
        pad = self.mem_dim - x.size(-1)
        return F.pad(x, (0, pad))

    def _build_values_for_tokens(
        self,
        regions: Dict[str, torch.Tensor],
        region: str,
        batch_idx: int,
        token_indices: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        value_dtype = reliability.dtype
        device = reliability.device
        onehot = torch.zeros(token_indices.numel(), 4, device=device, dtype=value_dtype)
        onehot[:, REGION_TO_ID[region]] = 1.0
        is_fg = 1.0 if region in ("fg_core", "fg_boundary") else 0.0
        fg_bg = torch.empty(token_indices.numel(), 2, device=device, dtype=value_dtype)
        fg_bg[:, 0] = 1.0 - is_fg
        fg_bg[:, 1] = is_fg
        sdf = regions["sdf_approx"][batch_idx, 0].flatten().index_select(0, token_indices).unsqueeze(1).to(dtype=value_dtype)
        rel = reliability[batch_idx, 0].flatten().index_select(0, token_indices).unsqueeze(1)
        return torch.cat([onehot, fg_bg, sdf, rel], dim=1)

    def _build_meta_for_tokens(
        self,
        img_id: str,
        region: str,
        token_indices: torch.Tensor,
        spatial_size: Tuple[int, int],
        values: torch.Tensor,
    ) -> List[dict]:
        height, width = spatial_size
        metas = []
        values_cpu = values.detach().cpu()
        for local_idx, flat_idx in enumerate(token_indices.detach().cpu().tolist()):
            h = int(flat_idx // width)
            w = int(flat_idx % width)
            metas.append(
                {
                    "image_id": img_id,
                    "coord": (h, w),
                    "flat_index": int(flat_idx),
                    "region": region,
                    "region_id": REGION_TO_ID[region],
                    "sdf": float(values_cpu[local_idx, 6]),
                    "reliability": float(values_cpu[local_idx, 7]),
                    "height": int(height),
                    "width": int(width),
                }
            )
        return metas

    def _cat_or_empty(
        self,
        chunks: List[torch.Tensor],
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not chunks:
            return torch.empty(0, width, device=device, dtype=dtype)
        return torch.cat(chunks, dim=0).to(device=device, dtype=dtype)

    def _infer_dtype(self) -> torch.dtype:
        for chunks in [self.image_keys_list, *self.keys_list.values()]:
            if chunks:
                return chunks[0].dtype
        if isinstance(self.image_keys, torch.Tensor) and self.image_keys.numel() > 0:
            return self.image_keys.dtype
        return torch.float32

    def _infer_state_dtype(self, state: Dict[str, Any]) -> torch.dtype:
        image_keys = state.get("image_keys")
        if isinstance(image_keys, torch.Tensor) and image_keys.numel() > 0:
            return image_keys.dtype
        for tensor_map_name in ("keys", "values"):
            tensor_map = state.get(tensor_map_name, {}) or {}
            for value in tensor_map.values():
                if isinstance(value, torch.Tensor) and value.numel() > 0:
                    return value.dtype
        return torch.float32

    def _load_2d_state_tensor(
        self,
        tensor: Any,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if tensor is None:
            return torch.empty(0, width, device=device, dtype=dtype)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Memory state entries must be tensors, got {type(tensor).__name__}")
        if tensor.numel() == 0:
            return torch.empty(0, width, device=device, dtype=dtype)
        if tensor.dim() != 2 or tensor.size(1) != width:
            raise ValueError(f"Memory state tensor must have shape [N, {width}], got {tuple(tensor.shape)}")
        return tensor.detach().to(device=device, dtype=dtype).contiguous()

    def _flatten_img_ids(self, top_img_ids: Iterable[object]) -> List[object]:
        if isinstance(top_img_ids, torch.Tensor):
            return top_img_ids.detach().cpu().reshape(-1).tolist()
        flat = []
        for item in top_img_ids:
            if isinstance(item, (list, tuple, set)):
                flat.extend(item)
            elif isinstance(item, torch.Tensor):
                flat.extend(item.detach().cpu().reshape(-1).tolist())
            else:
                flat.append(item)
        return flat

    def _normalize_img_id_selection(self, top_img_ids: Iterable[object]) -> set[str]:
        selected = set()
        for item in self._flatten_img_ids(top_img_ids):
            if isinstance(item, (int, float)) and int(item) == item:
                index = int(item)
                if 0 <= index < len(self.image_ids):
                    selected.add(str(self.image_ids[index]))
            selected.add(str(item))
        return selected
