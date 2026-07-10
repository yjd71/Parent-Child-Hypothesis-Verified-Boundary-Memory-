"""Labelled-only parent-child boundary memory for PC-HBM.

The memory stores CPU tensors by default and moves selected subbanks to the
forward device/dtype on demand.  It never creates synthetic contents in
production; callers must append labelled features before ``is_ready`` becomes
true.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F

from ..common.utils import EPS, REGION_TO_ID, normalize


class PCHBMMemory:
    """Container for route, parent, and child PC-HBM memory tensors."""

    def __init__(self, memory_dim: int = 512, value_dim: int = 8, geometry_dim: int = 6, config: Any | None = None) -> None:
        self.memory_dim = int(memory_dim)
        self.value_dim = int(value_dim)
        self.geometry_dim = int(geometry_dim)
        self.config = config
        self.parent_img_to_indices: Dict[str, torch.Tensor] = {}
        self.route_img_to_index: Dict[str, int] = {}
        self._gpu_cache: Dict[tuple, torch.Tensor] = {}
        self._cache_signature = None
        self._cache_version = 0
        self.clear()

    def clear(self) -> None:
        self._invalidate_cache()
        self.route_lists: Dict[str, List[torch.Tensor]] = {
            "x3_global": [],
            "x3_boundary": [],
            "x3_uncertain": [],
            "x3_bg_near": [],
            "x3_environment": [],
            "route_embed": [],
        }
        self.route_img_ids: List[str] = []
        self.parent_key_list: List[torch.Tensor] = []
        self.parent_value_list: List[torch.Tensor] = []
        self.parent_geo_list: List[torch.Tensor] = []
        self.parent_child_ptr_list: List[torch.Tensor] = []
        self.parent_meta: List[dict] = []
        self.child_key_list: List[torch.Tensor] = []
        self.child_geo_list: List[torch.Tensor] = []
        self.child_meta: List[dict] = []
        self.route: Dict[str, Any] = {}
        self.parent: Dict[str, Any] = {}
        self.child: Dict[str, Any] = {}
        self.compat_meta: Dict[str, Any] = {}
        self.parent_img_to_indices = {}
        self.route_img_to_index = {}
        self._finalized = False

    def append_route(
        self,
        *,
        x3_global: torch.Tensor,
        x3_boundary: torch.Tensor,
        x3_uncertain: torch.Tensor,
        x3_bg_near: torch.Tensor,
        x3_environment: torch.Tensor,
        route_embed: torch.Tensor,
        img_ids: Sequence[object],
    ) -> None:
        for name, tensor in {
            "x3_global": x3_global,
            "x3_boundary": x3_boundary,
            "x3_uncertain": x3_uncertain,
            "x3_bg_near": x3_bg_near,
            "x3_environment": x3_environment,
            "route_embed": route_embed,
        }.items():
            self._check_2d(tensor, self.memory_dim, name)
            self.route_lists[name].append(tensor.detach().cpu().float())
        self.route_img_ids.extend(str(item) for item in img_ids)
        self._finalized = False
        self._invalidate_cache()

    def append_parent(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        geometry: torch.Tensor,
        child_ptr: torch.Tensor,
        meta: Sequence[dict],
    ) -> None:
        self._check_2d(keys, self.memory_dim, "parent keys")
        self._check_2d(values, self.value_dim, "parent values")
        self._check_2d(geometry, self.geometry_dim, "parent geometry")
        if keys.size(0) != values.size(0) or keys.size(0) != geometry.size(0):
            raise ValueError("parent keys/values/geometry must have the same length")
        if child_ptr.numel() != keys.size(0):
            raise ValueError("child_ptr length must match parent key count")
        if len(meta) != keys.size(0):
            raise ValueError("parent meta length must match parent key count")
        self.parent_key_list.append(keys.detach().cpu().float())
        self.parent_value_list.append(values.detach().cpu().float())
        self.parent_geo_list.append(geometry.detach().cpu().float())
        self.parent_child_ptr_list.append(child_ptr.detach().cpu().long().view(-1))
        self.parent_meta.extend(dict(item) for item in meta)
        self._finalized = False
        self._invalidate_cache()

    def append_child(self, keys: torch.Tensor, geometry: torch.Tensor, meta: Sequence[dict]) -> torch.Tensor:
        self._check_2d(keys, self.memory_dim, "child keys")
        self._check_2d(geometry, self.geometry_dim, "child geometry")
        if keys.size(0) != geometry.size(0) or len(meta) != keys.size(0):
            raise ValueError("child keys/geometry/meta lengths must match")
        start = self.num_children_pending()
        self.child_key_list.append(keys.detach().cpu().float())
        self.child_geo_list.append(geometry.detach().cpu().float())
        self.child_meta.extend(dict(item) for item in meta)
        self._finalized = False
        self._invalidate_cache()
        return torch.arange(start, start + keys.size(0), dtype=torch.long)

    def finalize(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        target_device = torch.device("cpu") if device is None else torch.device(device)
        target_dtype = torch.float32
        self.route = {
            name: self._cat(items, self.memory_dim).to(device=target_device, dtype=target_dtype)
            for name, items in self.route_lists.items()
        }
        self.route["img_ids"] = list(self.route_img_ids)
        self.parent = {
            "p3_keys": self._cat(self.parent_key_list, self.memory_dim).to(device=target_device, dtype=target_dtype),
            "p3_values": self._cat(self.parent_value_list, self.value_dim).to(device=target_device, dtype=target_dtype),
            "p3_geometry": self._cat(self.parent_geo_list, self.geometry_dim).to(device=target_device, dtype=target_dtype),
            "child_ptr": self._cat_long(self.parent_child_ptr_list).to(device=target_device),
            "parent_meta": list(self.parent_meta),
        }
        self.child = {
            "p2_child_keys": self._cat(self.child_key_list, self.memory_dim).to(device=target_device, dtype=target_dtype),
            "p2_child_geo": self._cat(self.child_geo_list, self.geometry_dim).to(device=target_device, dtype=target_dtype),
            "child_meta": list(self.child_meta),
        }
        self._finalized = True
        self._build_parent_img_index()
        self._build_route_img_index()
        self._invalidate_cache()

    def is_ready(self) -> bool:
        return (
            self._finalized
            and self.route.get("route_embed", torch.empty(0, self.memory_dim)).size(0) > 0
            and self.parent.get("p3_keys", torch.empty(0, self.memory_dim)).size(0) > 0
            and self.child.get("p2_child_keys", torch.empty(0, self.memory_dim)).size(0) > 0
        )

    def route_query(self, q_route: torch.Tensor, top_img_k: int) -> Dict[str, Any]:
        """Route ``[B,512]`` query to top labelled image IDs."""

        if not self.is_ready():
            return {"top_img_ids": [[] for _ in range(q_route.size(0))], "top_img_scores": q_route.new_empty(q_route.size(0), 0), "route_entropy": q_route.new_zeros(q_route.size(0))}
        original_device = q_route.device
        use_cache = self._cache_enabled(original_device, "route")
        if use_cache:
            try:
                keys = self._get_cached_tensor("route", "route_embed", original_device)
                q = q_route.float()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                self.clear_gpu_cache()
                keys = self.route["route_embed"].float()
                q = q_route.float().to(device=keys.device, non_blocking=True)
        else:
            keys = self.route["route_embed"].float()
            if keys.device != original_device and not (original_device.type == "cuda" and self._cache_would_exceed_free_memory(original_device)):
                keys = keys.to(device=original_device, dtype=torch.float32, non_blocking=True)
            q = q_route.float().to(device=keys.device, non_blocking=True)
        img_ids = list(self.route.get("img_ids", []))
        k = min(max(1, int(top_img_k)), keys.size(0))
        sim = normalize(q, dim=-1) @ normalize(keys, dim=-1).transpose(0, 1)
        scores, indices = torch.topk(sim, k=k, dim=1)
        probs = torch.softmax(scores, dim=1)
        entropy = -(probs * probs.clamp_min(EPS).log()).sum(dim=1)
        top_ids = [[img_ids[int(idx)] for idx in row] for row in indices.detach().cpu().tolist()]
        if scores.device != original_device:
            scores = scores.to(device=original_device, non_blocking=True)
            entropy = entropy.to(device=original_device, non_blocking=True)
        return {"top_img_ids": top_ids, "top_img_scores": scores, "route_entropy": entropy}

    def get_parent_subbank(self, top_img_ids: Iterable[Iterable[object]] | Iterable[object] | None, device=None, dtype=None) -> Dict[str, Any]:
        """Return parent entries whose metadata image_id is in selected top images."""

        device = torch.device("cpu") if device is None else torch.device(device)
        dtype = torch.float32
        if not self.is_ready():
            return self._empty_parent(device, dtype)
        selected = self._flatten_img_ids(top_img_ids)
        if selected:
            index_tensors = [self.parent_img_to_indices[image_id] for image_id in selected if image_id in self.parent_img_to_indices]
            if not index_tensors:
                return self._empty_parent(device, dtype)
            keep = torch.unique(torch.cat(index_tensors, dim=0), sorted=True)
        else:
            keep = torch.arange(int(self.parent["p3_keys"].size(0)), dtype=torch.long, device=self.parent["p3_keys"].device)
        if keep.numel() == 0:
            return self._empty_parent(device, dtype)
        keep_list = keep.detach().cpu().tolist()
        use_parent_cache = self._cache_enabled(device, "parent")
        if use_parent_cache:
            try:
                p3_keys = self._get_cached_tensor("parent", "p3_keys", device)
                p3_values = self._get_cached_tensor("parent", "p3_values", device)
                p3_geometry = self._get_cached_tensor("parent", "p3_geometry", device)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                self.clear_gpu_cache()
                p3_keys = self.parent["p3_keys"].float()
                p3_values = self.parent["p3_values"].float()
                p3_geometry = self.parent["p3_geometry"].float()
        else:
            p3_keys = self.parent["p3_keys"].float()
            p3_values = self.parent["p3_values"].float()
            p3_geometry = self.parent["p3_geometry"].float()
        child_ptr = self.parent["child_ptr"]
        if p3_keys.device != p3_values.device or p3_keys.device != p3_geometry.device:
            p3_values = p3_values.to(device=p3_keys.device, dtype=torch.float32, non_blocking=True)
            p3_geometry = p3_geometry.to(device=p3_keys.device, dtype=torch.float32, non_blocking=True)
        keep_dev = keep.to(device=p3_keys.device, non_blocking=True)
        child_ptr = child_ptr.to(device=p3_keys.device, non_blocking=True)
        keys = p3_keys.index_select(0, keep_dev)
        values = p3_values.index_select(0, keep_dev)
        geometry = p3_geometry.index_select(0, keep_dev)
        child = child_ptr.index_select(0, keep_dev)
        if keys.device != device:
            keys = keys.to(device=device, dtype=torch.float32, non_blocking=True)
            values = values.to(device=device, dtype=torch.float32, non_blocking=True)
            geometry = geometry.to(device=device, dtype=torch.float32, non_blocking=True)
            child = child.to(device=device, non_blocking=True)
        return {
            "p3_keys": keys,
            "p3_values": values,
            "p3_geometry": geometry,
            "child_ptr": child,
            "parent_meta": [self.parent["parent_meta"][idx] for idx in keep_list],
        }

    def get_child_by_ptr(self, top_child_ptrs: torch.Tensor, device=None, dtype=None) -> Dict[str, Any]:
        """Gather child keys/geometry for ``[M,K]`` child pointers."""

        device = top_child_ptrs.device if device is None else torch.device(device)
        dtype = torch.float32
        if self._cache_enabled(device, "child"):
            try:
                child_keys = self._get_cached_tensor("child", "p2_child_keys", device)
                child_geo = self._get_cached_tensor("child", "p2_child_geo", device)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                self.clear_gpu_cache()
                child_keys = self.child.get("p2_child_keys", torch.empty(0, self.memory_dim)).float()
                child_geo = self.child.get("p2_child_geo", torch.empty(0, self.geometry_dim)).float()
        else:
            child_keys = self.child.get("p2_child_keys", torch.empty(0, self.memory_dim)).float()
            child_geo = self.child.get("p2_child_geo", torch.empty(0, self.geometry_dim)).float()
        if child_keys.numel() == 0 or top_child_ptrs.numel() == 0:
            shape = tuple(top_child_ptrs.shape)
            return {
                "p2_child_keys": torch.empty(*shape, self.memory_dim, device=device, dtype=dtype),
                "p2_child_geo": torch.empty(*shape, self.geometry_dim, device=device, dtype=dtype),
            }
        ptr = top_child_ptrs.detach().to(device=child_keys.device).long().clamp(0, child_keys.size(0) - 1)
        flat = ptr.reshape(-1)
        keys = child_keys.index_select(0, flat).reshape(*ptr.shape, self.memory_dim)
        geo = child_geo.index_select(0, flat).reshape(*ptr.shape, self.geometry_dim)
        if keys.device != device:
            keys = keys.to(device=device, dtype=dtype, non_blocking=True)
            geo = geo.to(device=device, dtype=dtype, non_blocking=True)
        return {"p2_child_keys": keys, "p2_child_geo": geo}

    def state_dict(self) -> Dict[str, Any]:
        return self.to_state_dict()

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "format_version": 1,
            "compat_meta": dict(self.compat_meta),
            "memory_dim": self.memory_dim,
            "value_dim": self.value_dim,
            "geometry_dim": self.geometry_dim,
            "route": {key: value.detach().cpu() if isinstance(value, torch.Tensor) else list(value) for key, value in self.route.items()},
            "parent": {key: value.detach().cpu() if isinstance(value, torch.Tensor) else list(value) for key, value in self.parent.items()},
            "child": {key: value.detach().cpu() if isinstance(value, torch.Tensor) else list(value) for key, value in self.child.items()},
            "finalized": bool(self._finalized),
        }

    def load_state_dict(self, state: Dict[str, Any] | None, device=None, dtype=None) -> None:
        self.clear()
        if not state:
            return
        self.compat_meta = dict(state.get("compat_meta", {}) or {})
        self.memory_dim = int(state.get("memory_dim", self.memory_dim))
        self.value_dim = int(state.get("value_dim", self.value_dim))
        self.geometry_dim = int(state.get("geometry_dim", self.geometry_dim))
        device = torch.device("cpu") if device is None else torch.device(device)
        dtype = torch.float32
        raw_route = state.get("route", {}) or {}
        raw_parent = state.get("parent", {}) or {}
        raw_child = state.get("child", {}) or {}
        self.route = {
            "x3_global": self._state_tensor(raw_route.get("x3_global"), self.memory_dim, device, dtype),
            "x3_boundary": self._state_tensor(raw_route.get("x3_boundary"), self.memory_dim, device, dtype),
            "x3_uncertain": self._state_tensor(raw_route.get("x3_uncertain"), self.memory_dim, device, dtype),
            "x3_bg_near": self._state_tensor(raw_route.get("x3_bg_near"), self.memory_dim, device, dtype),
            "x3_environment": self._state_tensor(raw_route.get("x3_environment"), self.memory_dim, device, dtype),
            "route_embed": self._state_tensor(raw_route.get("route_embed"), self.memory_dim, device, dtype),
            "img_ids": list(raw_route.get("img_ids", [])),
        }
        self.parent = {
            "p3_keys": self._state_tensor(raw_parent.get("p3_keys"), self.memory_dim, device, dtype),
            "p3_values": self._state_tensor(raw_parent.get("p3_values"), self.value_dim, device, dtype),
            "p3_geometry": self._state_tensor(raw_parent.get("p3_geometry"), self.geometry_dim, device, dtype),
            "child_ptr": self._state_long(raw_parent.get("child_ptr"), device),
            "parent_meta": list(raw_parent.get("parent_meta", [])),
        }
        self.child = {
            "p2_child_keys": self._state_tensor(raw_child.get("p2_child_keys"), self.memory_dim, device, dtype),
            "p2_child_geo": self._state_tensor(raw_child.get("p2_child_geo"), self.geometry_dim, device, dtype),
            "child_meta": list(raw_child.get("child_meta", [])),
        }
        self._finalized = bool(state.get("finalized", True))
        self._build_parent_img_index()
        self._build_route_img_index()
        self._invalidate_cache()

    def diagnostic_string(self) -> str:
        n_img = int(self.route.get("route_embed", torch.empty(0, self.memory_dim)).size(0))
        n_parent = int(self.parent.get("p3_keys", torch.empty(0, self.memory_dim)).size(0))
        n_child = int(self.child.get("p2_child_keys", torch.empty(0, self.memory_dim)).size(0))
        return f"[PC-HBM] memory images={n_img}, parent={n_parent}, child={n_child}, ready={self.is_ready()}"

    def num_children_pending(self) -> int:
        return sum(int(item.size(0)) for item in self.child_key_list)

    def clear_gpu_cache(self) -> None:
        self._gpu_cache.clear()

    def _invalidate_cache(self) -> None:
        if not hasattr(self, "_gpu_cache"):
            self._gpu_cache = {}
        self._gpu_cache.clear()
        self._cache_version = int(getattr(self, "_cache_version", 0)) + 1
        self._cache_signature = self._cache_version

    def _build_parent_img_index(self) -> None:
        mapping: Dict[str, List[int]] = {}
        for idx, meta in enumerate(self.parent.get("parent_meta", [])):
            image_id = str(meta.get("image_id"))
            mapping.setdefault(image_id, []).append(idx)
        index_device = self.parent.get("p3_keys", torch.empty(0)).device
        self.parent_img_to_indices = {
            image_id: torch.tensor(indices, dtype=torch.long, device=index_device)
            for image_id, indices in mapping.items()
        }

    def _build_route_img_index(self) -> None:
        self.route_img_to_index = {str(image_id): idx for idx, image_id in enumerate(self.route.get("img_ids", []))}

    def _cache_enabled(self, device: torch.device, group: str) -> bool:
        device = torch.device(device)
        if device.type != "cuda":
            return False
        if not bool(getattr(self.config, "pc_hbm_memory_gpu_cache", True)):
            return False
        cache_device = str(getattr(self.config, "pc_hbm_memory_cache_device", "cuda")).lower()
        if cache_device == "cpu":
            return False
        group_key = {
            "route": "pc_hbm_memory_cache_route",
            "parent": "pc_hbm_memory_cache_parent",
            "child": "pc_hbm_memory_cache_child",
        }.get(group)
        if group_key and not bool(getattr(self.config, group_key, True)):
            return False
        return not self._cache_would_exceed_free_memory(device)

    def _cache_would_exceed_free_memory(self, device: torch.device) -> bool:
        if torch.device(device).type != "cuda":
            return False
        min_free_gb = float(getattr(self.config, "pc_hbm_memory_cache_min_free_gb", 1.5))
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except Exception:
            return False
        return free_bytes < min_free_gb * (1024 ** 3)

    def _safe_cached_tensor(self, group: str, name: str, device: torch.device) -> torch.Tensor:
        try:
            return self._get_cached_tensor(group, name, device)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self.clear_gpu_cache()
                return getattr(self, group)[name].float()
            raise

    def _get_cached_tensor(self, group: str, name: str, device: torch.device) -> torch.Tensor:
        device = torch.device(device)
        cache_dtype = torch.float32
        key = (group, name, str(device), str(cache_dtype), self._cache_signature)
        cached = self._gpu_cache.get(key)
        if cached is not None:
            return cached
        tensor = getattr(self, group)[name]
        cached = tensor.to(device=device, dtype=cache_dtype, non_blocking=True)
        self._gpu_cache[key] = cached
        return cached

    def _flatten_img_ids(self, top_img_ids) -> set[str]:
        if top_img_ids is None:
            return set()
        if isinstance(top_img_ids, torch.Tensor):
            ids = top_img_ids.detach().cpu().reshape(-1).tolist()
        else:
            ids = []
            for item in top_img_ids:
                if isinstance(item, (list, tuple, set)):
                    ids.extend(item)
                elif isinstance(item, torch.Tensor):
                    ids.extend(item.detach().cpu().reshape(-1).tolist())
                else:
                    ids.append(item)
        route_ids = list(self.route.get("img_ids", self.route_img_ids))
        out = set()
        for item in ids:
            if isinstance(item, (int, float)) and int(item) == item and 0 <= int(item) < len(route_ids):
                out.add(str(route_ids[int(item)]))
            out.add(str(item))
        return out

    def _empty_parent(self, device, dtype) -> Dict[str, Any]:
        return {
            "p3_keys": torch.empty(0, self.memory_dim, device=device, dtype=dtype),
            "p3_values": torch.empty(0, self.value_dim, device=device, dtype=dtype),
            "p3_geometry": torch.empty(0, self.geometry_dim, device=device, dtype=dtype),
            "child_ptr": torch.empty(0, device=device, dtype=torch.long),
            "parent_meta": [],
        }

    def _check_2d(self, tensor: torch.Tensor, width: int, name: str) -> None:
        if tensor.dim() != 2 or tensor.size(1) != width:
            raise ValueError(f"{name} must be [N,{width}], got {tuple(tensor.shape)}")

    def _cat(self, items: List[torch.Tensor], width: int) -> torch.Tensor:
        if not items:
            return torch.empty(0, width)
        return torch.cat(items, dim=0)

    def _cat_long(self, items: List[torch.Tensor]) -> torch.Tensor:
        if not items:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(items, dim=0).long()

    def _state_tensor(self, value, width: int, device, dtype) -> torch.Tensor:
        if value is None:
            return torch.empty(0, width, device=device, dtype=dtype)
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.numel() == 0:
            return torch.empty(0, width, device=device, dtype=dtype)
        return tensor.detach().reshape(-1, width).to(device=device, dtype=dtype).contiguous()

    def _state_long(self, value, device) -> torch.Tensor:
        if value is None:
            return torch.empty(0, device=device, dtype=torch.long)
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return tensor.detach().reshape(-1).to(device=device, dtype=torch.long).contiguous()


def parent_values_from_region(region: str, sdf: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
    """Build PC parent value ``[N,8]`` from region name, sdf and reliability."""

    n = int(sdf.numel())
    value = torch.zeros(n, 8, device=sdf.device, dtype=sdf.dtype)
    region_id = int(REGION_TO_ID[region])
    value[:, region_id] = 1.0
    is_fg = 1.0 if region in ("fg_core", "fg_boundary") else 0.0
    value[:, 4] = is_fg
    value[:, 5] = 1.0 - is_fg
    value[:, 6] = sdf.reshape(-1)
    value[:, 7] = reliability.reshape(-1)
    return value
