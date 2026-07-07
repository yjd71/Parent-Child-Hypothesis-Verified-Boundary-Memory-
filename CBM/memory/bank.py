from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from CBM.boundary.regions import build_gt_regions
from CBM.config.labeled_memory import LabeledMemorySelectionConfig
from CBM.memory.labels import DEFAULT_MAX_SIZES, DEFAULT_SAMPLE_PER_IMAGE, REGION_NAMES, REGION_TO_ID
from CBM.memory.reliability import prepare_reliability


@dataclass
class CandidateChunk:
    image_id: str
    region: str
    keys: torch.Tensor
    values: torch.Tensor
    coords: torch.Tensor
    flat_indices: torch.Tensor
    component_ids: torch.Tensor
    grid_ids: torch.Tensor
    reliability: torch.Tensor
    uids: Tuple[str, ...]
    height: int
    width: int

    @property
    def size(self) -> int:
        return int(self.keys.size(0))


def _default_selection_config(
    sample_per_image: Dict[str, int], max_sizes: Dict[str, int]
) -> LabeledMemorySelectionConfig:
    return LabeledMemorySelectionConfig(
        profile_name="legacy",
        split=None,
        sample_per_image=dict(sample_per_image),
        max_sizes=dict(max_sizes),
        top_img_k=8,
        grid_size=4,
        min_tokens_per_component={"fg_core": 2, "fg_boundary": 4, "bg_near": 4, "bg_far": 2},
        min_spatial_dist={"fg_core": 2, "fg_boundary": 2, "bg_near": 2, "bg_far": 3},
        grid_quota_ratio={"fg_core": 0.35, "fg_boundary": 0.20, "bg_near": 0.20, "bg_far": 0.15},
        max_feature_sim=0.98,
        relaxed_min_spatial_dist=1.0,
        relaxed_max_feature_sim=0.995,
        allow_underfill=True,
        use_component_quota=True,
        use_grid_quota=True,
        use_spatial_diversity=True,
        use_feature_diversity=True,
        relax_diversity_if_underfilled=True,
        global_fill_max_per_image=None,
    )


class DenseBoundaryMemory:
    """Labeled-only dense feature-label memory for PLAN_V4.2 CBM-PFI."""

    def __init__(
        self,
        mem_dim: int = 128,
        value_dim: int = 8,
        regions: Sequence[str] = REGION_NAMES,
        sample_per_image: Optional[Dict[str, int]] = None,
        max_sizes: Optional[Dict[str, int]] = None,
        selection_config: Optional[LabeledMemorySelectionConfig] = None,
    ) -> None:
        if value_dim != 8:
            raise ValueError("DenseBoundaryMemory currently uses fixed value_dim=8")
        self.mem_dim = int(mem_dim)
        self.value_dim = int(value_dim)
        self.regions = tuple(regions)
        self.compat_meta: Dict[str, Any] = {}
        requested_sample = sample_per_image or dict(DEFAULT_SAMPLE_PER_IMAGE)
        requested_sizes = max_sizes or dict(DEFAULT_MAX_SIZES)
        self.selection_config = selection_config or _default_selection_config(requested_sample, requested_sizes)
        self.sample_per_image = dict(self.selection_config.sample_per_image)
        self.max_sizes = dict(self.selection_config.max_sizes)
        if sample_per_image is not None:
            self.sample_per_image.update(sample_per_image)
        if max_sizes is not None:
            self.max_sizes.update(max_sizes)
        self.clear()

    def clear(self) -> None:
        self.image_keys_list: List[torch.Tensor] = []
        self.image_ids: List[str] = []
        self.candidate_pool: Dict[str, List[CandidateChunk]] = {region: [] for region in self.regions}
        # Legacy collection attributes remain available for callers that inspect them.
        self.keys_list: Dict[str, List[torch.Tensor]] = {region: [] for region in self.regions}
        self.values_list: Dict[str, List[torch.Tensor]] = {region: [] for region in self.regions}
        self.meta_list: Dict[str, List[dict]] = {region: [] for region in self.regions}
        self.image_keys = torch.empty(0, self.mem_dim)
        self.keys = {region: torch.empty(0, self.mem_dim) for region in self.regions}
        self.values = {region: torch.empty(0, self.value_dim) for region in self.regions}
        self.meta = {region: [] for region in self.regions}
        self.build_info: Dict[str, Any] = {}
        self._distribution_stats: Dict[str, Dict[str, Any]] = {}
        self._diversity_stats: Dict[str, Dict[str, Any]] = {}
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
        duplicate_ids = set(self.image_ids).intersection(img_ids)
        duplicate_ids.update(image_id for image_id, count in Counter(img_ids).items() if count > 1)
        if duplicate_ids:
            raise ValueError(f"DenseBoundaryMemory requires unique image IDs, got duplicates: {sorted(duplicate_ids)[:10]}")

        device = p3.device
        key_dtype = p3.dtype
        image_keys = self._fit_mem_dim(
            F.adaptive_avg_pool2d(x3.detach(), 1).flatten(1).to(device=device, dtype=key_dtype)
        )
        self.image_keys_list.append(image_keys.cpu())
        self.image_ids.extend(img_ids)

        regions = build_gt_regions(gt.to(device=device), target_size=p3.shape[-2:])
        p3_tokens = self._fit_mem_dim(p3.detach().flatten(2).transpose(1, 2))
        rel_map = prepare_reliability(reliability, p3, key_dtype)
        height, width = int(p3.size(2)), int(p3.size(3))

        for batch_idx, image_id in enumerate(img_ids):
            for region in self.regions:
                region_mask = regions[region][batch_idx, 0].bool()
                flat_indices = region_mask.flatten().nonzero(as_tuple=False).flatten()
                if flat_indices.numel() == 0:
                    continue

                component_map = self._connected_components(region_mask)
                coords = torch.stack(
                    [torch.div(flat_indices, width, rounding_mode="floor"), flat_indices.remainder(width)], dim=1
                )
                component_ids = component_map.flatten().index_select(0, flat_indices.cpu()).to(torch.long)
                grid_ids = self._grid_ids(coords, height, width)
                token_reliability = rel_map[batch_idx, 0].flatten().index_select(0, flat_indices)
                keep_local = self._presample_candidates(
                    flat_indices=flat_indices,
                    component_ids=component_ids.to(device=flat_indices.device),
                    grid_ids=grid_ids.to(device=flat_indices.device),
                    reliability=token_reliability,
                    sample_count=int(self.sample_per_image.get(region, flat_indices.numel())),
                )
                flat_indices = flat_indices.index_select(0, keep_local)
                coords = coords.index_select(0, keep_local).cpu().to(torch.long)
                component_ids = component_ids.index_select(0, keep_local.cpu()).cpu()
                grid_ids = grid_ids.index_select(0, keep_local.cpu()).cpu()
                token_reliability = token_reliability.index_select(0, keep_local).detach().cpu().float()
                token_keys = p3_tokens[batch_idx].index_select(0, flat_indices).detach().cpu()
                token_values = self._build_values_for_tokens(
                    regions, region, batch_idx, flat_indices, rel_map
                ).detach().cpu()
                flat_cpu = flat_indices.detach().cpu().to(torch.long)
                uids = tuple(f"{image_id}:{region}:{int(index)}" for index in flat_cpu.tolist())
                chunk = CandidateChunk(
                    image_id=image_id,
                    region=region,
                    keys=token_keys,
                    values=token_values,
                    coords=coords,
                    flat_indices=flat_cpu,
                    component_ids=component_ids,
                    grid_ids=grid_ids,
                    reliability=token_reliability,
                    uids=uids,
                    height=height,
                    width=width,
                )
                self.candidate_pool[region].append(chunk)
                self.keys_list[region].append(chunk.keys)
                self.values_list[region].append(chunk.values)
        self._finalized = False

    def finalize(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        dtype = dtype or self._infer_dtype()
        device = torch.device("cpu") if device is None else torch.device(device)
        self._finalize_image_keys(device=device, dtype=dtype)
        self.keys = {}
        self.values = {}
        self.meta = {}
        self._distribution_stats = {}
        self._diversity_stats = {}

        for region in self.regions:
            keys, values, metas, distribution, diversity = self._finalize_region(
                self.candidate_pool[region], region, int(self.max_sizes.get(region, 0))
            )
            self.keys[region] = keys.to(device=device, dtype=dtype)
            self.values[region] = values.to(device=device, dtype=dtype)
            self.meta[region] = metas
            self._distribution_stats[region] = distribution
            self._diversity_stats[region] = diversity

        self.build_info = {
            "profile": self.selection_config.profile_name,
            "split": self.selection_config.split,
            "candidate_tokens": {
                region: sum(chunk.size for chunk in self.candidate_pool[region]) for region in self.regions
            },
            "selected_tokens": {region: int(self.keys[region].size(0)) for region in self.regions},
        }
        self._finalized = True

    def is_ready(self) -> bool:
        return self._finalized and sum(self.keys[region].size(0) for region in self.regions) > 0

    def set_compat_meta(self, compat_meta: Optional[Dict[str, Any]]) -> None:
        self.compat_meta = dict(compat_meta or {})

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "format_version": 2,
            "compat_meta": dict(self.compat_meta),
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
            "build_info": dict(self.build_info),
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
        self.compat_meta = dict(state.get("compat_meta", {}) or {})
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
        self.build_info = dict(state.get("build_info", {}) or {})
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
                keep_list = [idx for idx, item in enumerate(metas) if item.get("image_id") in selected]
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
        return (
            torch.cat(key_chunks, dim=0).to(device=out_device, dtype=out_dtype),
            torch.cat(value_chunks, dim=0).to(device=out_device, dtype=out_dtype),
            meta_out,
        )

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

    def distribution_log_lines(self, split: Optional[float] = None) -> List[str]:
        split = self.selection_config.split if split is None else split
        lines = []
        for region in self.regions:
            stats = self._distribution_stats.get(region, {})
            lines.append(
                "[CBM_MEM_DIST] split={} region={} total={} unique_images={} min={} max={} mean={:.2f}".format(
                    split,
                    region,
                    stats.get("total", 0),
                    stats.get("unique_images", 0),
                    stats.get("min", 0),
                    stats.get("max", 0),
                    stats.get("mean", 0.0),
                )
            )
            lines.append(
                f"[CBM_MEM_DIST] split={split} region={region} "
                f"top10_img_token_counts={stats.get('top10_img_token_counts', [])}"
            )
        return lines

    def diversity_log_lines(self, split: Optional[float] = None) -> List[str]:
        split = self.selection_config.split if split is None else split
        lines = []
        for region in self.regions:
            stats = self._diversity_stats.get(region, {})
            lines.append(
                "[CBM_DIVERSITY] split={} region={} selected={} unique_images={} "
                "avg_num_components={:.2f} avg_used_components={:.2f} avg_num_grids={:.2f} "
                "avg_used_grids={:.2f} avg_max_grid_ratio={:.3f} "
                "avg_pairwise_dist={:.3f} avg_pairwise_feat_sim={:.3f}".format(
                    split,
                    region,
                    stats.get("selected", 0),
                    stats.get("unique_images", 0),
                    stats.get("avg_num_components", 0.0),
                    stats.get("avg_used_components", 0.0),
                    stats.get("avg_num_grids", 0.0),
                    stats.get("avg_used_grids", 0.0),
                    stats.get("avg_max_grid_ratio", 0.0),
                    stats.get("avg_pairwise_dist", 0.0),
                    stats.get("avg_pairwise_feat_sim", 0.0),
                )
            )
        return lines

    def _finalize_image_keys(self, device: torch.device, dtype: torch.dtype) -> None:
        keys = self._cat_or_empty(self.image_keys_list, self.mem_dim, torch.device("cpu"), dtype)
        if keys.size(0) != len(self.image_ids):
            raise ValueError("image key count does not match image id count")
        order = sorted(range(len(self.image_ids)), key=lambda index: self.image_ids[index])
        if order:
            keep = torch.tensor(order, dtype=torch.long)
            keys = keys.index_select(0, keep)
            self.image_ids = [self.image_ids[index] for index in order]
        self.image_keys = keys.to(device=device, dtype=dtype)

    def _finalize_region(
        self, chunks: List[CandidateChunk], region: str, max_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[dict], Dict[str, Any], Dict[str, Any]]:
        if not chunks or max_size <= 0:
            empty_keys = torch.empty(0, self.mem_dim)
            empty_values = torch.empty(0, self.value_dim)
            return empty_keys, empty_values, [], self._distribution([]), self._diversity([], [], region)

        chunks = sorted(chunks, key=lambda chunk: chunk.image_id)
        if len(chunks) > max_size:
            chunks = sorted(
                chunks,
                key=lambda chunk: (-float(chunk.reliability.max().item()), chunk.image_id),
            )[:max_size]
            chunks.sort(key=lambda chunk: chunk.image_id)
        per_image_quota = max(1, max_size // max(len(chunks), 1))
        selected: Dict[str, List[int]] = {}
        fill_queues: Dict[str, List[int]] = {}
        for chunk in chunks:
            base_k = min(per_image_quota, chunk.size)
            base = self._select_single_image_region(chunk, base_k, region)
            selected[chunk.image_id] = base
            fill_queues[chunk.image_id] = self._build_fill_queue(chunk, base, region)

        selected_total = sum(len(indices) for indices in selected.values())
        remaining = max(0, max_size - selected_total)
        extra_counts = Counter()
        hard_caps = self.selection_config.global_fill_max_per_image or {}
        chunk_by_id = {chunk.image_id: chunk for chunk in chunks}
        while remaining > 0:
            active = []
            for image_id, queue in fill_queues.items():
                hard_cap = int(hard_caps.get(region, 2**31 - 1))
                if queue and extra_counts[image_id] < hard_cap:
                    next_idx = queue[0]
                    chunk = chunk_by_id[image_id]
                    active.append((-float(chunk.reliability[next_idx]), chunk.uids[next_idx], image_id))
            if not active:
                break
            active.sort()
            for _, _, image_id in active:
                if remaining <= 0:
                    break
                selected[image_id].append(fill_queues[image_id].pop(0))
                extra_counts[image_id] += 1
                remaining -= 1

        if remaining > 0 and not self.selection_config.allow_underfill:
            raw_queues = {}
            for chunk in chunks:
                chosen = set(selected[chunk.image_id])
                raw_queues[chunk.image_id] = [idx for idx in self._ordered_indices(chunk) if idx not in chosen]
            while remaining > 0 and any(raw_queues.values()):
                for image_id in sorted(raw_queues):
                    if remaining <= 0:
                        break
                    if raw_queues[image_id]:
                        selected[image_id].append(raw_queues[image_id].pop(0))
                        remaining -= 1

        key_chunks: List[torch.Tensor] = []
        value_chunks: List[torch.Tensor] = []
        metas: List[dict] = []
        selected_records: List[Tuple[CandidateChunk, List[int]]] = []
        for chunk in chunks:
            indices = selected[chunk.image_id]
            if not indices:
                continue
            indices = sorted(indices, key=lambda idx: chunk.uids[idx])
            keep = torch.tensor(indices, dtype=torch.long)
            key_chunks.append(chunk.keys.index_select(0, keep))
            value_chunks.append(chunk.values.index_select(0, keep))
            selected_records.append((chunk, indices))
            for idx in indices:
                metas.append(self._meta_from_candidate(chunk, idx))

        keys = self._cat_or_empty(key_chunks, self.mem_dim, torch.device("cpu"), self._infer_dtype())
        values = self._cat_or_empty(value_chunks, self.value_dim, torch.device("cpu"), self._infer_dtype())
        distribution = self._distribution(metas)
        diversity = self._diversity(chunks, selected_records, region)
        return keys, values, metas, distribution, diversity

    def _select_single_image_region(self, chunk: CandidateChunk, k: int, region: str) -> List[int]:
        if k <= 0 or chunk.size == 0:
            return []
        normalized = F.normalize(chunk.keys.float(), dim=1, eps=1e-6)
        selected: List[int] = []
        grid_counts: Counter = Counter()
        max_per_grid = max(1, int(math.ceil(k * self.selection_config.grid_quota_ratio[region])))
        quotas = self._component_quotas(chunk, k, region)
        for component_id in sorted(quotas):
            candidates = [
                idx for idx in self._ordered_indices(chunk) if int(chunk.component_ids[idx]) == component_id
            ]
            self._greedy_select(
                chunk,
                candidates,
                quotas[component_id],
                selected,
                grid_counts,
                max_per_grid,
                float(self.selection_config.min_spatial_dist[region]),
                float(self.selection_config.max_feature_sim),
                normalized,
                ignore_grid=not self.selection_config.use_grid_quota,
            )
        if len(selected) < k:
            leftovers = [idx for idx in self._ordered_indices(chunk) if idx not in set(selected)]
            self._greedy_select(
                chunk,
                leftovers,
                k - len(selected),
                selected,
                grid_counts,
                max_per_grid,
                float(self.selection_config.min_spatial_dist[region]),
                float(self.selection_config.max_feature_sim),
                normalized,
                ignore_grid=not self.selection_config.use_grid_quota,
            )
        if len(selected) < k and self.selection_config.relax_diversity_if_underfilled:
            leftovers = [idx for idx in self._ordered_indices(chunk) if idx not in set(selected)]
            self._greedy_select(
                chunk,
                leftovers,
                k - len(selected),
                selected,
                grid_counts,
                max_per_grid,
                float(self.selection_config.relaxed_min_spatial_dist),
                float(self.selection_config.relaxed_max_feature_sim),
                normalized,
                ignore_grid=True,
            )
        if len(selected) < k and not self.selection_config.allow_underfill:
            for idx in self._ordered_indices(chunk):
                if idx not in selected:
                    selected.append(idx)
                    if len(selected) >= k:
                        break
        return selected[:k]

    def _build_fill_queue(self, chunk: CandidateChunk, selected: List[int], region: str) -> List[int]:
        normalized = F.normalize(chunk.keys.float(), dim=1, eps=1e-6)
        queue: List[int] = []
        working = list(selected)
        grid_counts = Counter(int(chunk.grid_ids[idx]) for idx in working)
        leftovers = [idx for idx in self._ordered_indices(chunk) if idx not in set(working)]
        self._greedy_select(
            chunk,
            leftovers,
            len(leftovers),
            working,
            grid_counts,
            chunk.size,
            float(self.selection_config.min_spatial_dist[region]),
            float(self.selection_config.max_feature_sim),
            normalized,
            ignore_grid=True,
            added_out=queue,
        )
        if self.selection_config.relax_diversity_if_underfilled:
            used = set(working)
            leftovers = [idx for idx in self._ordered_indices(chunk) if idx not in used]
            self._greedy_select(
                chunk,
                leftovers,
                len(leftovers),
                working,
                grid_counts,
                chunk.size,
                float(self.selection_config.relaxed_min_spatial_dist),
                float(self.selection_config.relaxed_max_feature_sim),
                normalized,
                ignore_grid=True,
                added_out=queue,
            )
        return queue

    def _greedy_select(
        self,
        chunk: CandidateChunk,
        candidates: List[int],
        count: int,
        selected: List[int],
        grid_counts: Counter,
        max_per_grid: int,
        min_spatial_dist: float,
        max_feature_sim: float,
        normalized_keys: torch.Tensor,
        ignore_grid: bool,
        added_out: Optional[List[int]] = None,
    ) -> None:
        added = 0
        for idx in candidates:
            if added >= count:
                break
            grid_id = int(chunk.grid_ids[idx])
            if not ignore_grid and grid_counts[grid_id] >= max_per_grid:
                continue
            if not self._passes_diversity(
                chunk, idx, selected, min_spatial_dist, max_feature_sim, normalized_keys
            ):
                continue
            selected.append(idx)
            grid_counts[grid_id] += 1
            if added_out is not None:
                added_out.append(idx)
            added += 1

    def _passes_diversity(
        self,
        chunk: CandidateChunk,
        idx: int,
        selected: List[int],
        min_spatial_dist: float,
        max_feature_sim: float,
        normalized_keys: torch.Tensor,
    ) -> bool:
        if not selected:
            return True
        keep = torch.tensor(selected, dtype=torch.long)
        if self.selection_config.use_spatial_diversity and min_spatial_dist > 0:
            delta = chunk.coords.index_select(0, keep).float() - chunk.coords[idx].float()
            if torch.linalg.vector_norm(delta, dim=1).min().item() < min_spatial_dist:
                return False
        if self.selection_config.use_feature_diversity:
            sims = normalized_keys[idx].view(1, -1) @ normalized_keys.index_select(0, keep).transpose(0, 1)
            if sims.max().item() > max_feature_sim:
                return False
        return True

    def _component_quotas(self, chunk: CandidateChunk, k: int, region: str) -> Dict[int, int]:
        component_counts = Counter(int(value) for value in chunk.component_ids.tolist())
        if not self.selection_config.use_component_quota:
            return {component: min(count, k) for component, count in component_counts.items()}
        components = sorted(component_counts)
        if k < len(components):
            ranked = sorted(
                components,
                key=lambda component: (
                    -max(
                        float(chunk.reliability[idx])
                        for idx in range(chunk.size)
                        if int(chunk.component_ids[idx]) == component
                    ),
                    component,
                ),
            )
            return {component: 1 for component in ranked[:k]}
        minimum = int(self.selection_config.min_tokens_per_component[region])
        quotas = {component: min(component_counts[component], minimum) for component in components}
        if sum(quotas.values()) > k:
            quotas = {component: 1 for component in components}
        remaining = k - sum(quotas.values())
        while remaining > 0:
            eligible = [component for component in components if quotas[component] < component_counts[component]]
            if not eligible:
                break
            total_area = sum(component_counts[component] for component in eligible)
            raw = {
                component: remaining * component_counts[component] / max(total_area, 1)
                for component in eligible
            }
            additions = {
                component: min(
                    component_counts[component] - quotas[component],
                    int(math.floor(raw[component])),
                )
                for component in eligible
            }
            added = sum(additions.values())
            for component, amount in additions.items():
                quotas[component] += amount
            remaining -= added
            if remaining <= 0:
                break
            ranked = sorted(
                eligible,
                key=lambda component: (-(raw[component] - math.floor(raw[component])), component),
            )
            for component in ranked:
                if remaining <= 0:
                    break
                if quotas[component] < component_counts[component]:
                    quotas[component] += 1
                    remaining -= 1
                    added += 1
            if added == 0:
                break
        return quotas

    def _ordered_indices(self, chunk: CandidateChunk) -> List[int]:
        return sorted(range(chunk.size), key=lambda idx: (-float(chunk.reliability[idx]), chunk.uids[idx]))

    def _presample_candidates(
        self,
        flat_indices: torch.Tensor,
        component_ids: torch.Tensor,
        grid_ids: torch.Tensor,
        reliability: torch.Tensor,
        sample_count: int,
    ) -> torch.Tensor:
        count = int(flat_indices.numel())
        if sample_count <= 0 or count <= sample_count:
            return torch.arange(count, device=flat_indices.device)
        groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        flat_cpu = flat_indices.detach().cpu().tolist()
        comp_cpu = component_ids.detach().cpu().tolist()
        grid_cpu = grid_ids.detach().cpu().tolist()
        rel_cpu = reliability.detach().cpu().tolist()
        for idx in range(count):
            groups[(int(comp_cpu[idx]), int(grid_cpu[idx]))].append(idx)
        for key in groups:
            groups[key].sort(key=lambda idx: (-float(rel_cpu[idx]), int(flat_cpu[idx])))
        selected: List[int] = []
        offsets = {key: 0 for key in groups}
        group_keys = sorted(groups)
        while len(selected) < sample_count:
            progressed = False
            for key in group_keys:
                offset = offsets[key]
                if offset < len(groups[key]):
                    selected.append(groups[key][offset])
                    offsets[key] += 1
                    progressed = True
                    if len(selected) >= sample_count:
                        break
            if not progressed:
                break
        return torch.tensor(selected, device=flat_indices.device, dtype=torch.long)

    def _connected_components(self, region_mask: torch.Tensor) -> torch.Tensor:
        mask = region_mask.detach().cpu().numpy().astype(np.uint8, copy=False)
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        return torch.from_numpy(labels.astype(np.int64, copy=False))

    def _grid_ids(self, coords: torch.Tensor, height: int, width: int) -> torch.Tensor:
        grid_size = max(1, int(self.selection_config.grid_size))
        cell_h = max(1, int(math.ceil(height / grid_size)))
        cell_w = max(1, int(math.ceil(width / grid_size)))
        grid_y = torch.div(coords[:, 0], cell_h, rounding_mode="floor").clamp(max=grid_size - 1)
        grid_x = torch.div(coords[:, 1], cell_w, rounding_mode="floor").clamp(max=grid_size - 1)
        return (grid_y * grid_size + grid_x).cpu().to(torch.long)

    def _meta_from_candidate(self, chunk: CandidateChunk, idx: int) -> dict:
        return {
            "uid": chunk.uids[idx],
            "image_id": chunk.image_id,
            "coord": tuple(int(value) for value in chunk.coords[idx].tolist()),
            "flat_index": int(chunk.flat_indices[idx]),
            "region": chunk.region,
            "region_id": REGION_TO_ID[chunk.region],
            "component_id": int(chunk.component_ids[idx]),
            "grid_id": int(chunk.grid_ids[idx]),
            "sdf": float(chunk.values[idx, 6]),
            "reliability": float(chunk.reliability[idx]),
            "height": int(chunk.height),
            "width": int(chunk.width),
        }

    def _distribution(self, metas: List[dict]) -> Dict[str, Any]:
        counts = Counter(item["image_id"] for item in metas)
        values = list(counts.values())
        return {
            "total": len(metas),
            "unique_images": len(counts),
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "mean": float(sum(values) / len(values)) if values else 0.0,
            "top10_img_token_counts": sorted(values, reverse=True)[:10],
        }

    def _diversity(
        self,
        chunks: List[CandidateChunk],
        selected_records: List[Tuple[CandidateChunk, List[int]]],
        region: str,
    ) -> Dict[str, Any]:
        del region
        chunk_by_id = {chunk.image_id: chunk for chunk in chunks}
        records = []
        for chunk, indices in selected_records:
            if not indices:
                continue
            keep = torch.tensor(indices, dtype=torch.long)
            components = set(int(value) for value in chunk.component_ids.tolist())
            used_components = set(int(value) for value in chunk.component_ids.index_select(0, keep).tolist())
            grids = set(int(value) for value in chunk.grid_ids.tolist())
            selected_grids = [int(value) for value in chunk.grid_ids.index_select(0, keep).tolist()]
            used_grids = set(selected_grids)
            grid_counts = Counter(selected_grids)
            max_grid_ratio = max(grid_counts.values()) / max(len(indices), 1)
            coords = chunk.coords.index_select(0, keep).float()
            features = F.normalize(chunk.keys.index_select(0, keep).float(), dim=1, eps=1e-6)
            if len(indices) > 1:
                upper = torch.triu_indices(len(indices), len(indices), offset=1)
                pair_dist = torch.cdist(coords, coords)[upper[0], upper[1]].mean().item()
                pair_sim = (features @ features.transpose(0, 1))[upper[0], upper[1]].mean().item()
            else:
                pair_dist = 0.0
                pair_sim = 1.0
            records.append(
                {
                    "num_components": len(components),
                    "used_components": len(used_components),
                    "num_grids": len(grids),
                    "used_grids": len(used_grids),
                    "max_grid_ratio": max_grid_ratio,
                    "pair_dist": pair_dist,
                    "pair_sim": pair_sim,
                }
            )
        count = len(records)
        mean = lambda name: sum(record[name] for record in records) / count if count else 0.0
        return {
            "selected": sum(len(indices) for _, indices in selected_records),
            "unique_images": count,
            "avg_num_components": mean("num_components"),
            "avg_used_components": mean("used_components"),
            "avg_num_grids": mean("num_grids"),
            "avg_used_grids": mean("used_grids"),
            "avg_max_grid_ratio": mean("max_grid_ratio"),
            "avg_pairwise_dist": mean("pair_dist"),
            "avg_pairwise_feat_sim": mean("pair_sim"),
        }

    def _fit_mem_dim(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == self.mem_dim:
            return x
        if x.size(-1) > self.mem_dim:
            return x[..., : self.mem_dim]
        return F.pad(x, (0, self.mem_dim - x.size(-1)))

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
        sdf = regions["sdf_approx"][batch_idx, 0].flatten().index_select(0, token_indices).unsqueeze(1)
        sdf = sdf.to(dtype=value_dtype)
        rel = reliability[batch_idx, 0].flatten().index_select(0, token_indices).unsqueeze(1)
        return torch.cat([onehot, fg_bg, sdf, rel], dim=1)

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
        for chunks in self.candidate_pool.values():
            if chunks:
                return chunks[0].keys.dtype
        if self.image_keys_list:
            return self.image_keys_list[0].dtype
        if isinstance(self.image_keys, torch.Tensor) and self.image_keys.numel() > 0:
            return self.image_keys.dtype
        return torch.float32

    def _infer_state_dtype(self, state: Dict[str, Any]) -> torch.dtype:
        image_keys = state.get("image_keys")
        if isinstance(image_keys, torch.Tensor) and image_keys.numel() > 0:
            return image_keys.dtype
        for tensor_map_name in ("keys", "values"):
            for value in (state.get(tensor_map_name, {}) or {}).values():
                if isinstance(value, torch.Tensor) and value.numel() > 0:
                    return value.dtype
        return torch.float32

    def _load_2d_state_tensor(
        self, tensor: Any, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if tensor is None or (isinstance(tensor, torch.Tensor) and tensor.numel() == 0):
            return torch.empty(0, width, device=device, dtype=dtype)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Memory state entries must be tensors, got {type(tensor).__name__}")
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


__all__ = ["CandidateChunk", "DenseBoundaryMemory"]
