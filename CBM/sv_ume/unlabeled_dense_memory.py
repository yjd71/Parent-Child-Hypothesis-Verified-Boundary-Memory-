from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from CBM.memory.labels import (
    DEFAULT_SAMPLE_PER_IMAGE,
    REGION_NAMES,
    REGION_TO_ID,
    VALUE_LAYOUT,
)


MEMORY_STATE_VERSION = 2
_REQUIRED_NEW_META_FIELDS = (
    "image_id",
    "coord",
    "region",
    "epoch_added",
    "step_added",
    "global_type",
)


@dataclass(frozen=True)
class UnlabeledMemoryToken:
    """One verified teacher-p3 token and its image-level routing key."""

    key: torch.Tensor
    value: torch.Tensor
    global_key: torch.Tensor
    meta: Mapping[str, Any]
    reliability: float
    diversity: float = 0.0
    global_meta: Optional[Mapping[str, Any]] = None


class UnlabeledDenseBoundaryMemory:
    """Frozen snapshot memory built from verified SAM-refined pseudo labels."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.mem_dim = int(getattr(cfg, "cbm_memory_dim", 128))
        self.value_dim = int(getattr(cfg, "cbm_value_dim", len(VALUE_LAYOUT)))
        if self.mem_dim <= 0:
            raise ValueError(f"cbm_memory_dim must be positive, got {self.mem_dim}")
        if self.value_dim != len(VALUE_LAYOUT):
            raise ValueError(
                f"UnlabeledDenseBoundaryMemory requires value_dim={len(VALUE_LAYOUT)}, "
                f"got {self.value_dim}"
            )

        self.regions = tuple(REGION_NAMES)
        self.value_layout = tuple(VALUE_LAYOUT)
        self.lambda_diversity = float(getattr(cfg, "lambda_diversity", 0.2))
        self.sample_per_image = self._validated_region_mapping(
            getattr(cfg, "sample_per_image_unlabeled", DEFAULT_SAMPLE_PER_IMAGE),
            "sample_per_image_unlabeled",
            cast=int,
            minimum=0,
        )
        self.region_capacity_ratio = self._validated_region_mapping(
            getattr(cfg, "region_capacity_ratio", {region: 1.0 for region in self.regions}),
            "region_capacity_ratio",
            cast=float,
            minimum=0.0,
            maximum=1.0,
        )
        self.use_ema_refresh = bool(getattr(cfg, "use_unlabeled_memory_ema_refresh", False))
        self.ema_momentum = float(getattr(cfg, "unlabeled_memory_momentum", 0.99))
        if not 0.0 <= self.ema_momentum <= 1.0:
            raise ValueError("unlabeled_memory_momentum must be in [0, 1]")

        self._frozen = False
        self._reset_storage()

    def add_global_key(self, x3_global: torch.Tensor, meta: Mapping[str, Any]) -> None:
        self._assert_mutable("add_global_key")
        self._assert_not_finalized("add_global_key")
        global_meta = self._normalize_global_meta(meta)
        image_id = global_meta["image_id"]
        if any(item["image_id"] == image_id for item in self.global_meta):
            raise ValueError(f"duplicate global key for image_id={image_id!r}")
        key = self._as_vector(x3_global, self.mem_dim, "x3_global")
        self.global_keys.append(key.detach().cpu().clone())
        self.global_meta.append(global_meta)

    def add_region_tokens(self, region: str, tokens: Sequence[UnlabeledMemoryToken]) -> None:
        self._assert_mutable("add_region_tokens")
        self._assert_not_finalized("add_region_tokens")
        self._validate_region(region)
        if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
            raise TypeError("tokens must be a sequence of UnlabeledMemoryToken")

        capacity = self._region_capacities[region] if self._capacity_configured else None
        if capacity is not None and len(self.keys[region]) + len(tokens) > capacity:
            raise ValueError(
                f"adding {len(tokens)} tokens would exceed {region} capacity {capacity}"
            )

        for token in tokens:
            normalized = self._normalize_token(region, token)
            self.keys[region].append(normalized.key.detach().cpu().clone())
            self.values[region].append(normalized.value.detach().cpu().clone())
            token_meta = dict(normalized.meta)
            token_meta["reliability"] = float(normalized.reliability)
            token_meta["diversity"] = float(normalized.diversity)
            token_meta["selection_score"] = self._selection_score(normalized)
            self.meta[region].append(token_meta)

    def build_from_candidates(
        self,
        candidate_pool: Mapping[str, Sequence[UnlabeledMemoryToken]],
        labeled_memory,
        previous_memory: Optional["UnlabeledDenseBoundaryMemory"] = None,
        device=None,
        dtype=None,
    ) -> "UnlabeledDenseBoundaryMemory":
        """Build a new capacity-constrained snapshot from verified candidates."""
        self._assert_mutable("build_from_candidates")
        if not isinstance(candidate_pool, Mapping):
            raise TypeError("candidate_pool must be a region-indexed mapping")
        unknown_regions = set(candidate_pool) - set(self.regions)
        if unknown_regions:
            raise KeyError(f"unknown candidate regions: {sorted(unknown_regions)}")

        self.clear()
        self._region_capacities = self._capacities_from_labeled(labeled_memory)
        self._capacity_configured = True
        previous_index = self._previous_key_index(previous_memory) if self.use_ema_refresh else {}
        selected_global: Dict[str, Tuple[float, UnlabeledMemoryToken]] = {}

        for region in self.regions:
            raw_tokens = candidate_pool.get(region, ())
            if isinstance(raw_tokens, (str, bytes)) or not isinstance(raw_tokens, Sequence):
                raise TypeError(f"candidate_pool[{region!r}] must be a sequence")
            normalized_tokens = [self._normalize_token(region, token) for token in raw_tokens]
            ranked_tokens = sorted(
                normalized_tokens,
                key=self._selection_score,
                reverse=True,
            )

            per_image_limit = self.sample_per_image[region]
            region_capacity = self._region_capacities[region]
            per_image_counts: Counter[str] = Counter()
            selected: List[UnlabeledMemoryToken] = []
            for token in ranked_tokens:
                if len(selected) >= region_capacity:
                    break
                image_id = str(token.meta["image_id"])
                if per_image_counts[image_id] >= per_image_limit:
                    continue
                token = self._apply_ema(region, token, previous_index)
                selected.append(token)
                per_image_counts[image_id] += 1

                score = self._selection_score(token)
                current = selected_global.get(image_id)
                if current is None or score > current[0]:
                    selected_global[image_id] = (score, token)

            self.add_region_tokens(region, selected)

        for _, token in selected_global.values():
            global_meta = dict(token.global_meta or {})
            global_meta.setdefault("image_id", token.meta["image_id"])
            if "global_type" in token.meta:
                global_meta.setdefault("global_type", token.meta["global_type"])
            global_meta.setdefault("reliability", float(token.reliability))
            global_meta.setdefault("diversity", float(token.diversity))
            self.add_global_key(token.global_key, global_meta)

        return self.finalize(device=device, dtype=dtype)

    def finalize(self, device=None, dtype=None) -> "UnlabeledDenseBoundaryMemory":
        self._assert_mutable("finalize")
        if self._finalized:
            return self
        target_device = torch.device("cpu") if device is None else torch.device(device)
        target_dtype = dtype or self._infer_dtype()

        self.global_keys = self._stack_or_empty(
            self.global_keys,
            self.mem_dim,
            target_device,
            target_dtype,
        )
        for region in self.regions:
            self.keys[region] = self._stack_or_empty(
                self.keys[region],
                self.mem_dim,
                target_device,
                target_dtype,
            )
            self.values[region] = self._stack_or_empty(
                self.values[region],
                self.value_dim,
                target_device,
                target_dtype,
            )
            self._validate_region_alignment(region)
            if self._capacity_configured and self.keys[region].size(0) > self._region_capacities[region]:
                raise ValueError(
                    f"{region} has {self.keys[region].size(0)} tokens, "
                    f"capacity is {self._region_capacities[region]}"
                )
        if self.global_keys.size(0) != len(self.global_meta):
            raise ValueError("global_keys and global_meta lengths do not match")
        self._refresh_checkpoint_metadata()
        self._finalized = True
        return self

    def is_ready(self) -> bool:
        if not self._finalized or not torch.is_tensor(self.global_keys):
            return False
        dense_count = sum(int(self.keys[region].size(0)) for region in self.regions)
        return self.global_keys.size(0) > 0 and dense_count > 0

    def get_image_keys(self, device=None, dtype=None) -> Tuple[torch.Tensor, List[str]]:
        global_keys = self._query_global_keys()
        global_keys = global_keys.to(
            device=device or global_keys.device,
            dtype=dtype or global_keys.dtype,
        )
        return global_keys, [str(item["image_id"]) for item in self.global_meta]

    def get_region_memory(self, region: str, device=None, dtype=None):
        self._validate_region(region)
        keys, values = self._query_region_tensors(region)
        keys = keys.to(device=device or keys.device, dtype=dtype or keys.dtype)
        values = values.to(device=device or values.device, dtype=dtype or values.dtype)
        return keys, values, copy.deepcopy(self.meta[region])

    def get_sub_memory(
        self,
        top_img_ids: Optional[Iterable[object]] = None,
        regions: Optional[Iterable[str]] = None,
        device=None,
        dtype=None,
    ):
        selected_ids = None if top_img_ids is None else self._normalize_image_ids(top_img_ids)
        selected_regions = self.regions if regions is None else tuple(regions)
        for region in selected_regions:
            self._validate_region(region)

        key_chunks: List[torch.Tensor] = []
        value_chunks: List[torch.Tensor] = []
        meta_out: List[dict] = []
        for region in selected_regions:
            keys, values = self._query_region_tensors(region)
            region_meta = self.meta[region]
            if keys.size(0) == 0:
                continue
            if selected_ids is None:
                indices = list(range(keys.size(0)))
            else:
                indices = [
                    idx
                    for idx, item in enumerate(region_meta)
                    if str(item.get("image_id")) in selected_ids
                ]
            if not indices:
                continue
            index = torch.tensor(indices, device=keys.device, dtype=torch.long)
            key_chunks.append(keys.index_select(0, index))
            value_chunks.append(values.index_select(0, index))
            meta_out.extend(copy.deepcopy(region_meta[idx]) for idx in indices)

        target_device = torch.device("cpu") if device is None else torch.device(device)
        target_dtype = dtype or self._infer_dtype()
        if not key_chunks:
            return (
                torch.empty(0, self.mem_dim, device=target_device, dtype=target_dtype),
                torch.empty(0, self.value_dim, device=target_device, dtype=target_dtype),
                [],
            )
        return (
            torch.cat(key_chunks, dim=0).to(device=target_device, dtype=target_dtype),
            torch.cat(value_chunks, dim=0).to(device=target_device, dtype=target_dtype),
            meta_out,
        )

    def stats(self) -> Dict[str, Any]:
        region_counts = {
            region: self._region_count(region)
            for region in self.regions
        }
        reliabilities = [
            float(item.get("reliability", 0.0))
            for region in self.regions
            for item in self.meta[region]
        ]
        global_types = Counter(
            str(item["global_type"])
            for region in self.regions
            for item in self.meta[region]
            if item.get("global_type") is not None
        )
        return {
            "ready": self.is_ready(),
            "finalized": bool(self._finalized),
            "frozen": bool(self._frozen),
            "num_global_keys": self._global_count(),
            "region_counts": region_counts,
            "region_capacities": dict(self._region_capacities),
            "mean_reliability": (
                sum(reliabilities) / len(reliabilities) if reliabilities else 0.0
            ),
            "global_type_counts": dict(global_types),
        }

    def freeze(self) -> "UnlabeledDenseBoundaryMemory":
        if self._frozen:
            return self
        if not self._finalized:
            self.finalize()
        self.global_keys = self.global_keys.detach().clone()
        self.keys = {
            region: self.keys[region].detach().clone()
            for region in self.regions
        }
        self.values = {
            region: self.values[region].detach().clone()
            for region in self.regions
        }
        self.global_meta = copy.deepcopy(self.global_meta)
        self.meta = copy.deepcopy(self.meta)
        self.temporal_pseudo_label_cache = copy.deepcopy(self.temporal_pseudo_label_cache)
        self.global_type_metadata = copy.deepcopy(self.global_type_metadata)
        self._frozen = True
        return self

    def clear(self) -> "UnlabeledDenseBoundaryMemory":
        self._assert_mutable("clear")
        self._reset_storage()
        return self

    def state_dict(self) -> Dict[str, Any]:
        if not self._finalized:
            raise RuntimeError("finalize memory before calling state_dict")
        self._refresh_checkpoint_metadata()
        return {
            "version": MEMORY_STATE_VERSION,
            "mem_dim": self.mem_dim,
            "value_dim": self.value_dim,
            "regions": list(self.regions),
            "value_layout": list(self.value_layout),
            "global_keys": self.global_keys.detach().cpu().clone(),
            "global_meta": copy.deepcopy(self.global_meta),
            "keys": {
                region: self.keys[region].detach().cpu().clone()
                for region in self.regions
            },
            "values": {
                region: self.values[region].detach().cpu().clone()
                for region in self.regions
            },
            "meta": copy.deepcopy(self.meta),
            "temporal_pseudo_label_cache": copy.deepcopy(
                self.temporal_pseudo_label_cache
            ),
            "global_type_metadata": copy.deepcopy(self.global_type_metadata),
            "region_capacities": dict(self._region_capacities),
            "capacity_configured": bool(self._capacity_configured),
            "finalized": True,
            "frozen": bool(self._frozen),
        }

    def load_state_dict(
        self,
        state,
        device=None,
        dtype=None,
    ) -> "UnlabeledDenseBoundaryMemory":
        self._assert_mutable("load_state_dict")
        if not isinstance(state, Mapping):
            raise TypeError("memory state must be a mapping")
        version = int(state.get("version", 1))
        if version not in (1, MEMORY_STATE_VERSION):
            raise ValueError(f"unsupported memory state version: {state.get('version')}")
        if int(state.get("mem_dim", self.mem_dim)) != self.mem_dim:
            raise ValueError("state mem_dim does not match current configuration")
        if int(state.get("value_dim", self.value_dim)) != self.value_dim:
            raise ValueError("state value_dim does not match current configuration")
        if tuple(state.get("regions", self.regions)) != self.regions:
            raise ValueError("state regions do not match REGION_NAMES")
        if tuple(state.get("value_layout", self.value_layout)) != self.value_layout:
            raise ValueError("state value_layout does not match VALUE_LAYOUT")
        if not bool(state.get("finalized", True)):
            raise ValueError("only finalized unlabeled memory states can be loaded")

        self._reset_storage()
        target_device = torch.device("cpu") if device is None else torch.device(device)
        self.global_keys = self._load_matrix(
            state.get("global_keys"),
            self.mem_dim,
            "global_keys",
            device=target_device,
            dtype=dtype,
        )
        raw_global_meta = state.get("global_meta", [])
        if not isinstance(raw_global_meta, list):
            raise TypeError("global_meta state must be a list")
        self.global_meta = [self._normalize_global_meta(item) for item in raw_global_meta]
        if self.global_keys.size(0) != len(self.global_meta):
            raise ValueError("global_keys and global_meta lengths do not match")

        raw_keys = state.get("keys", {})
        raw_values = state.get("values", {})
        raw_meta = state.get("meta", {})
        if not all(isinstance(item, Mapping) for item in (raw_keys, raw_values, raw_meta)):
            raise TypeError("keys, values, and meta state entries must be mappings")
        for region in self.regions:
            keys = self._load_matrix(
                raw_keys.get(region),
                self.mem_dim,
                f"keys[{region}]",
                device=target_device,
                dtype=dtype,
            )
            values = self._load_matrix(
                raw_values.get(region),
                self.value_dim,
                f"values[{region}]",
                device=target_device,
                dtype=dtype,
            )
            region_meta = raw_meta.get(region, [])
            if not isinstance(region_meta, list):
                raise TypeError(f"meta[{region}] state must be a list")
            normalized_meta = [
                self._normalize_stored_meta(region, item)
                for item in region_meta
            ]
            keys, values, normalized_meta = self._deduplicate_loaded_region(
                region,
                keys,
                values,
                normalized_meta,
            )
            self.keys[region] = keys
            self.values[region] = values
            self.meta[region] = normalized_meta
            self._validate_region_alignment(region)

        raw_temporal_cache = state.get("temporal_pseudo_label_cache")
        self.temporal_pseudo_label_cache = (
            self._normalize_temporal_cache(raw_temporal_cache)
            if raw_temporal_cache is not None
            else self._build_temporal_cache()
        )
        raw_global_types = state.get("global_type_metadata")
        self.global_type_metadata = (
            self._normalize_global_type_metadata(raw_global_types)
            if raw_global_types is not None
            else self._build_global_type_metadata()
        )

        raw_capacities = state.get("region_capacities", {})
        if not isinstance(raw_capacities, Mapping):
            raise TypeError("region_capacities state must be a mapping")
        self._region_capacities = {
            region: int(raw_capacities.get(region, 0))
            for region in self.regions
        }
        self._capacity_configured = bool(state.get("capacity_configured", False))
        if self._capacity_configured:
            for region in self.regions:
                if self.keys[region].size(0) > self._region_capacities[region]:
                    raise ValueError(f"loaded {region} memory exceeds its stored capacity")
        self._finalized = True
        if bool(state.get("frozen", False)):
            self.freeze()
        return self

    def _normalize_token(self, region: str, token: UnlabeledMemoryToken) -> UnlabeledMemoryToken:
        if not isinstance(token, UnlabeledMemoryToken):
            raise TypeError("candidate entries must be UnlabeledMemoryToken instances")
        key = self._as_vector(token.key, self.mem_dim, "token.key")
        value = self._as_vector(token.value, self.value_dim, "token.value")
        global_key = self._as_vector(token.global_key, self.mem_dim, "token.global_key")
        reliability = float(token.reliability)
        diversity = float(token.diversity)
        if not math.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
            raise ValueError("token reliability must be finite and in [0, 1]")
        if not math.isfinite(diversity) or not 0.0 <= diversity <= 1.0:
            raise ValueError("token diversity must be finite and in [0, 1]")

        meta = self._normalize_stored_meta(region, token.meta, require_new_fields=True)
        self._validate_value_layout(region, value, reliability)
        global_meta = None
        if token.global_meta is not None:
            global_meta = self._normalize_global_meta(token.global_meta)
            if global_meta["image_id"] != meta["image_id"]:
                raise ValueError("token global_meta.image_id must match token meta.image_id")
        return replace(
            token,
            key=key,
            value=value,
            global_key=global_key,
            meta=meta,
            reliability=reliability,
            diversity=diversity,
            global_meta=global_meta,
        )

    def _normalize_stored_meta(
        self,
        region: str,
        meta: Mapping[str, Any],
        require_new_fields: bool = False,
    ) -> dict:
        if not isinstance(meta, Mapping):
            raise TypeError("token meta must be a mapping")
        item = dict(meta)
        if require_new_fields:
            missing_new = [name for name in _REQUIRED_NEW_META_FIELDS if name not in item]
            if missing_new:
                raise KeyError(f"token meta is missing checkpoint fields: {missing_new}")
        missing = [name for name in ("image_id", "coord", "region") if name not in item]
        if missing:
            raise KeyError(f"token meta is missing required fields: {missing}")
        item["image_id"] = str(item["image_id"])
        if str(item["region"]) != region:
            raise ValueError(f"token meta region {item['region']!r} does not match {region!r}")
        item["region"] = region
        coord = item["coord"]
        if not isinstance(coord, Sequence) or isinstance(coord, (str, bytes)) or len(coord) != 2:
            raise ValueError("token meta coord must be a two-element sequence")
        item["coord"] = (int(coord[0]), int(coord[1]))
        global_type = item.get("global_type")
        if global_type is not None and str(global_type) not in {
            "matched",
            "expanded",
            "novel_pending",
        }:
            raise ValueError(f"unsupported global_type: {global_type!r}")
        if global_type is not None:
            item["global_type"] = str(global_type)
        if item.get("global_type") == "novel_pending" and not bool(item.get("novel_activated", False)):
            raise ValueError("inactive novel_pending token cannot enter active memory")
        return item

    @staticmethod
    def _normalize_global_meta(meta: Mapping[str, Any]) -> dict:
        if not isinstance(meta, Mapping):
            raise TypeError("global meta must be a mapping")
        item = dict(meta)
        if "image_id" not in item:
            raise KeyError("global meta is missing image_id")
        item["image_id"] = str(item["image_id"])
        return item

    def _validate_value_layout(self, region: str, value: torch.Tensor, reliability: float) -> None:
        expected_onehot = torch.zeros(4, device=value.device, dtype=value.dtype)
        expected_onehot[REGION_TO_ID[region]] = 1.0
        if not torch.allclose(value[:4], expected_onehot, atol=1.0e-5, rtol=0.0):
            raise ValueError(f"token value region one-hot does not match {region}")
        is_fg = region in ("fg_core", "fg_boundary")
        expected_fg_bg = value.new_tensor([0.0, 1.0] if is_fg else [1.0, 0.0])
        if not torch.allclose(value[4:6], expected_fg_bg, atol=1.0e-5, rtol=0.0):
            raise ValueError(f"token value bg/fg fields do not match {region}")
        sdf = float(value[6].detach().item())
        stored_reliability = float(value[7].detach().item())
        if not -1.0 <= sdf <= 1.0:
            raise ValueError("token SDF value must be in [-1, 1]")
        if not 0.0 <= stored_reliability <= 1.0:
            raise ValueError("token value reliability must be in [0, 1]")
        if not math.isclose(stored_reliability, reliability, abs_tol=1.0e-5):
            raise ValueError("token reliability does not match value layout reliability")

    def _apply_ema(
        self,
        region: str,
        token: UnlabeledMemoryToken,
        previous_index: Mapping[Tuple[str, Tuple[int, int], str], torch.Tensor],
    ) -> UnlabeledMemoryToken:
        if not self.use_ema_refresh:
            return token
        identity = (str(token.meta["image_id"]), tuple(token.meta["coord"]), region)
        old_key = previous_index.get(identity)
        if old_key is None:
            return token
        old_key = old_key.to(device=token.key.device, dtype=token.key.dtype)
        blended = self.ema_momentum * old_key + (1.0 - self.ema_momentum) * token.key
        blended = F.normalize(blended.unsqueeze(0), dim=1).squeeze(0)
        meta = dict(token.meta)
        meta["ema_refreshed"] = True
        return replace(token, key=blended, meta=meta)

    def _previous_key_index(self, previous_memory) -> Dict[Tuple[str, Tuple[int, int], str], torch.Tensor]:
        if previous_memory is None:
            return {}
        if not isinstance(previous_memory, UnlabeledDenseBoundaryMemory):
            raise TypeError("previous_memory must be UnlabeledDenseBoundaryMemory")
        if not previous_memory._finalized:
            raise ValueError("previous_memory must be finalized before EMA refresh")
        index = {}
        for region in self.regions:
            for idx, item in enumerate(previous_memory.meta[region]):
                identity = (str(item["image_id"]), tuple(item["coord"]), region)
                index[identity] = previous_memory.keys[region][idx].detach()
        return index

    def _capacities_from_labeled(self, labeled_memory) -> Dict[str, int]:
        labeled_keys = getattr(labeled_memory, "keys", None)
        if not isinstance(labeled_keys, Mapping):
            raise TypeError("labeled_memory must expose a region-indexed keys mapping")
        capacities = {}
        for region in self.regions:
            if region not in labeled_keys:
                raise KeyError(f"labeled_memory.keys is missing region {region}")
            count = self._entry_count(labeled_keys[region])
            capacities[region] = int(math.floor(count * self.region_capacity_ratio[region]))
        return capacities

    def _selection_score(self, token: UnlabeledMemoryToken) -> float:
        return float(token.reliability) * (
            1.0 + self.lambda_diversity * float(token.diversity)
        )

    def _validate_region_alignment(self, region: str) -> None:
        if self.keys[region].size(0) != self.values[region].size(0):
            raise ValueError(f"{region} keys/values lengths do not match")
        if self.keys[region].size(0) != len(self.meta[region]):
            raise ValueError(f"{region} tensors/meta lengths do not match")

    def _refresh_checkpoint_metadata(self) -> None:
        self.temporal_pseudo_label_cache = self._build_temporal_cache()
        self.global_type_metadata = self._build_global_type_metadata()

    def _build_temporal_cache(self) -> Dict[str, List[dict]]:
        grouped: Dict[str, List[dict]] = {}
        best = {}
        order = []
        for region in self.regions:
            for item in self.meta[region]:
                if "p_ref_value" not in item:
                    continue
                p_ref_value = float(item["p_ref_value"])
                if not math.isfinite(p_ref_value):
                    continue
                identity = self._checkpoint_identity(item, region)
                record = {
                    "image_id": identity[0],
                    "coord": identity[1],
                    "region": region,
                    "epoch_added": item.get("epoch_added"),
                    "p_ref_value": max(0.0, min(1.0, p_ref_value)),
                }
                if "conf_ref_value" in item:
                    confidence = float(item["conf_ref_value"])
                    if math.isfinite(confidence):
                        record["conf_ref_value"] = max(0.0, min(1.0, confidence))
                rank = self._metadata_rank(item)
                if identity not in best:
                    order.append(identity)
                if identity not in best or rank > best[identity][0]:
                    best[identity] = (rank, record)
        for identity in order:
            record = best[identity][1]
            grouped.setdefault(record["image_id"], []).append(record)
        return grouped

    def _build_global_type_metadata(self) -> List[dict]:
        result: List[dict] = []
        seen = set()
        for raw_item in self.global_meta:
            item = dict(raw_item)
            if item.get("global_type") is None:
                continue
            identity = (str(item["image_id"]), item.get("epoch_added"))
            if identity in seen:
                continue
            seen.add(identity)
            result.append(copy.deepcopy(item))
        return result

    def _normalize_temporal_cache(self, value) -> Dict[str, List[dict]]:
        if not isinstance(value, Mapping):
            raise TypeError("temporal_pseudo_label_cache must be a mapping")
        normalized: Dict[str, List[dict]] = {}
        for image_id, entries in value.items():
            if not isinstance(entries, list):
                raise TypeError("temporal cache entries must be list[dict]")
            image_key = str(image_id)
            normalized_entries: List[dict] = []
            for raw in entries:
                if not isinstance(raw, Mapping):
                    raise TypeError("temporal cache entries must be list[dict]")
                item = dict(raw)
                item["image_id"] = str(item.get("image_id", image_key))
                coord = item.get("coord")
                if not isinstance(coord, Sequence) or isinstance(coord, (str, bytes)) or len(coord) != 2:
                    raise ValueError("temporal cache coord must have two elements")
                item["coord"] = (int(coord[0]), int(coord[1]))
                region = str(item.get("region"))
                self._validate_region(region)
                item["region"] = region
                p_ref_value = float(item["p_ref_value"])
                if not math.isfinite(p_ref_value):
                    raise ValueError("temporal cache p_ref_value must be finite")
                item["p_ref_value"] = max(0.0, min(1.0, p_ref_value))
                if item.get("conf_ref_value") is not None:
                    confidence = float(item["conf_ref_value"])
                    if not math.isfinite(confidence):
                        raise ValueError("temporal cache conf_ref_value must be finite")
                    item["conf_ref_value"] = max(0.0, min(1.0, confidence))
                normalized_entries.append(item)
            normalized[image_key] = normalized_entries
        return normalized

    @staticmethod
    def _normalize_global_type_metadata(value) -> List[dict]:
        if not isinstance(value, list):
            raise TypeError("global_type_metadata must be list[dict]")
        result = []
        for raw in value:
            if not isinstance(raw, Mapping):
                raise TypeError("global_type_metadata must be list[dict]")
            item = dict(raw)
            if "image_id" not in item:
                raise KeyError("global type metadata is missing image_id")
            item["image_id"] = str(item["image_id"])
            global_type = str(item.get("global_type"))
            if global_type not in {"matched", "expanded", "novel_pending"}:
                raise ValueError(f"unsupported global_type: {global_type!r}")
            item["global_type"] = global_type
            result.append(item)
        return result

    def _deduplicate_loaded_region(
        self,
        region: str,
        keys: torch.Tensor,
        values: torch.Tensor,
        meta: List[dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
        if keys.size(0) != len(meta) or values.size(0) != len(meta):
            raise ValueError(f"loaded {region} memory tensors/meta are not aligned")
        best = {}
        order = []
        for index, item in enumerate(meta):
            identity = self._checkpoint_identity(item, region)
            reliability = float(item.get("reliability", values[index, 7].detach().item()))
            step = int(item.get("step_added", -1))
            rank = (reliability if math.isfinite(reliability) else float("-inf"), step)
            if identity not in best:
                order.append(identity)
            if identity not in best or rank > best[identity][0]:
                best[identity] = (rank, index)
        keep = [best[identity][1] for identity in order]
        if len(keep) == len(meta):
            return keys, values, meta
        key_index = torch.tensor(keep, device=keys.device, dtype=torch.long)
        value_index = key_index.to(device=values.device)
        return (
            keys.index_select(0, key_index),
            values.index_select(0, value_index),
            [meta[item_index] for item_index in keep],
        )

    @staticmethod
    def _checkpoint_identity(item: Mapping[str, Any], region: str):
        coord = item["coord"]
        return (
            str(item["image_id"]),
            (int(coord[0]), int(coord[1])),
            region,
            item.get("epoch_added"),
        )

    @staticmethod
    def _metadata_rank(item: Mapping[str, Any]) -> Tuple[float, int]:
        reliability = float(item.get("reliability", item.get("r_token", 0.0)))
        if not math.isfinite(reliability):
            reliability = float("-inf")
        return reliability, int(item.get("step_added", -1))

    def _query_global_keys(self) -> torch.Tensor:
        if self._finalized:
            return self.global_keys
        if self.global_keys:
            raise RuntimeError("finalize memory before querying non-empty global keys")
        return torch.empty(0, self.mem_dim)

    def _query_region_tensors(self, region: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._finalized:
            return self.keys[region], self.values[region]
        if self.keys[region] or self.values[region]:
            raise RuntimeError("finalize memory before querying non-empty region memory")
        return torch.empty(0, self.mem_dim), torch.empty(0, self.value_dim)

    def _reset_storage(self) -> None:
        self.global_keys: Any = []
        self.global_meta: List[dict] = []
        self.keys: Dict[str, Any] = {region: [] for region in self.regions}
        self.values: Dict[str, Any] = {region: [] for region in self.regions}
        self.meta: Dict[str, List[dict]] = {region: [] for region in self.regions}
        self.temporal_pseudo_label_cache: Dict[str, List[dict]] = {}
        self.global_type_metadata: List[dict] = []
        self._region_capacities = {region: 0 for region in self.regions}
        self._capacity_configured = False
        self._finalized = False
        self._frozen = False

    def _infer_dtype(self) -> torch.dtype:
        if torch.is_tensor(self.global_keys) and self.global_keys.numel() > 0:
            return self.global_keys.dtype
        if isinstance(self.global_keys, list) and self.global_keys:
            return self.global_keys[0].dtype
        for region in self.regions:
            value = self.keys[region]
            if torch.is_tensor(value) and value.numel() > 0:
                return value.dtype
            if isinstance(value, list) and value:
                return value[0].dtype
        return torch.float32

    @staticmethod
    def _stack_or_empty(items, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(items):
            matrix = items
        elif items:
            matrix = torch.stack(list(items), dim=0)
        else:
            matrix = torch.empty(0, width)
        if matrix.dim() != 2 or matrix.size(1) != width:
            raise ValueError(f"expected matrix [N, {width}], got {tuple(matrix.shape)}")
        return matrix.detach().to(device=device, dtype=dtype).clone()

    @staticmethod
    def _load_matrix(
        value,
        width: int,
        name: str,
        *,
        device: torch.device,
        dtype=None,
    ) -> torch.Tensor:
        if value is None:
            return torch.empty(
                0,
                width,
                device=device,
                dtype=dtype or torch.float32,
            )
        if not torch.is_tensor(value):
            raise TypeError(f"{name} state must be a tensor")
        matrix = value.detach()
        if matrix.dim() != 2 or matrix.size(1) != width:
            raise ValueError(f"{name} must have shape [N, {width}], got {tuple(matrix.shape)}")
        return matrix.to(device=device, dtype=dtype or matrix.dtype).clone()

    @staticmethod
    def _as_vector(value: torch.Tensor, width: int, name: str) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        vector = value
        if vector.dim() == 2 and vector.size(0) == 1:
            vector = vector[0]
        if vector.dim() != 1 or vector.numel() != width:
            raise ValueError(f"{name} must have shape [{width}], got {tuple(value.shape)}")
        if not torch.isfinite(vector).all():
            raise ValueError(f"{name} contains non-finite values")
        return vector

    @staticmethod
    def _entry_count(value) -> int:
        if torch.is_tensor(value):
            return int(value.size(0)) if value.dim() > 0 else int(value.numel())
        try:
            return int(len(value))
        except TypeError as exc:
            raise TypeError("memory entries must be sized") from exc

    def _global_count(self) -> int:
        return self._entry_count(self.global_keys)

    def _region_count(self, region: str) -> int:
        return self._entry_count(self.keys[region])

    def _assert_mutable(self, operation: str) -> None:
        if self._frozen:
            raise RuntimeError(f"cannot {operation}: unlabeled memory is frozen")

    def _assert_not_finalized(self, operation: str) -> None:
        if self._finalized:
            raise RuntimeError(f"cannot {operation}: unlabeled memory is finalized")

    def _validate_region(self, region: str) -> None:
        if region not in self.regions:
            raise KeyError(f"unknown memory region: {region!r}")

    @staticmethod
    def _normalize_image_ids(values: Iterable[object]) -> set[str]:
        selected: set[str] = set()

        def visit(value) -> None:
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
            else:
                selected.add(str(value))

        visit(values)
        return selected

    def _validated_region_mapping(
        self,
        value,
        name: str,
        cast,
        minimum,
        maximum=None,
    ) -> dict:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
        out = {}
        for region in self.regions:
            if region not in value:
                raise KeyError(f"{name} is missing region {region}")
            current = cast(value[region])
            if current < minimum or (maximum is not None and current > maximum):
                interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
                raise ValueError(f"{name}[{region!r}] must be {interval}")
            out[region] = current
        return out


__all__ = ["UnlabeledMemoryToken", "UnlabeledDenseBoundaryMemory"]
