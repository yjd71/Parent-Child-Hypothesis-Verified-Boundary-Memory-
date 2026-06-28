from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from CBM.memory.labels import DEFAULT_SAMPLE_PER_IMAGE, REGION_NAMES
from CBM.sv_ume.unlabeled_dense_memory import UnlabeledMemoryToken


DEFAULT_LAMBDA_DIVERSITY = 0.2
DEFAULT_SPATIAL_NMS_DISTANCE = 2.0
DEFAULT_FEATURE_DUP_SIM_THRESHOLD = 0.95


@dataclass(frozen=True)
class _PreparedCandidate:
    token: UnlabeledMemoryToken
    input_index: int
    image_id: str
    coord: Tuple[int, int]
    global_type: str
    normalized_key: torch.Tensor
    normalized_global_key: torch.Tensor
    d_img: float = 0.0
    d_region: float = 0.0
    diversity_score: float = 0.0
    selection_score: float = 0.0


class UMEDiversitySampler:
    """Select capacity-safe, spatially and semantically diverse SV-UME tokens."""

    def __init__(self, cfg, logger=None) -> None:
        self.cfg = cfg
        self.logger = logger
        self.regions = tuple(REGION_NAMES)
        self.use_diversity_selection = bool(
            getattr(cfg, "use_diversity_selection", True)
        )
        self.lambda_diversity = float(
            getattr(cfg, "lambda_diversity", DEFAULT_LAMBDA_DIVERSITY)
        )
        self.spatial_nms_distance = float(
            getattr(cfg, "spatial_nms_distance", DEFAULT_SPATIAL_NMS_DISTANCE)
        )
        self.feature_dup_sim_threshold = float(
            getattr(
                cfg,
                "feature_dup_sim_threshold",
                DEFAULT_FEATURE_DUP_SIM_THRESHOLD,
            )
        )
        self.sample_per_image = self._region_config(
            getattr(cfg, "sample_per_image_unlabeled", DEFAULT_SAMPLE_PER_IMAGE),
            "sample_per_image_unlabeled",
            cast=int,
        )
        self.region_capacity_ratio = self._region_config(
            getattr(
                cfg,
                "region_capacity_ratio",
                {region: 1.0 for region in self.regions},
            ),
            "region_capacity_ratio",
            cast=float,
        )

        if not math.isfinite(self.lambda_diversity) or self.lambda_diversity < 0.0:
            raise ValueError("lambda_diversity must be finite and non-negative")
        if not math.isfinite(self.spatial_nms_distance) or self.spatial_nms_distance < 0.0:
            raise ValueError("spatial_nms_distance must be finite and non-negative")
        if (
            not math.isfinite(self.feature_dup_sim_threshold)
            or not -1.0 <= self.feature_dup_sim_threshold <= 1.0
        ):
            raise ValueError("feature_dup_sim_threshold must be in [-1, 1]")
        for region in self.regions:
            if self.sample_per_image[region] < 0:
                raise ValueError(
                    f"sample_per_image_unlabeled[{region!r}] must be non-negative"
                )
            ratio = self.region_capacity_ratio[region]
            if not math.isfinite(ratio) or ratio < 0.0:
                raise ValueError(
                    f"region_capacity_ratio[{region!r}] must be finite and non-negative"
                )

        self.last_result: Optional[Dict[str, Any]] = None

    @torch.no_grad()
    def select(
        self,
        *,
        candidate_pool,
        labeled_memory,
        prev_unlabeled_memory=None,
    ) -> Dict[str, Any]:
        if not isinstance(candidate_pool, Mapping):
            raise TypeError("candidate_pool must be a region-indexed mapping")
        unknown_regions = set(candidate_pool) - set(self.regions)
        if unknown_regions:
            raise KeyError(f"unknown candidate regions: {sorted(unknown_regions)}")

        labeled_global = self._memory_global_keys(labeled_memory, "labeled_memory")
        previous_global = self._memory_global_keys(
            prev_unlabeled_memory,
            "prev_unlabeled_memory",
            allow_none=True,
        )
        global_banks = tuple(
            bank for bank in (labeled_global, previous_global) if bank.size(0) > 0
        )

        selected_tokens: Dict[str, List[UnlabeledMemoryToken]] = {
            region: [] for region in self.regions
        }
        region_stats: Dict[str, Dict[str, Any]] = {}
        total_rejected: Counter[str] = Counter()

        for region in self.regions:
            labeled_keys = self._memory_region_keys(
                labeled_memory,
                region,
                "labeled_memory",
            )
            previous_keys = self._memory_region_keys(
                prev_unlabeled_memory,
                region,
                "prev_unlabeled_memory",
                allow_none=True,
            )
            capacity = min(
                int(labeled_keys.size(0)),
                int(
                    math.floor(
                        labeled_keys.size(0) * self.region_capacity_ratio[region]
                    )
                ),
            )

            raw_tokens = candidate_pool.get(region, ())
            if isinstance(raw_tokens, (str, bytes)) or not isinstance(
                raw_tokens, Sequence
            ):
                raise TypeError(f"candidate_pool[{region!r}] must be a sequence")

            rejected: Counter[str] = Counter()
            input_types: Counter[str] = Counter()
            eligible: List[_PreparedCandidate] = []
            for input_index, token in enumerate(raw_tokens):
                prepared = self._prepare_candidate(token, region, input_index)
                input_types[prepared.global_type] += 1
                if (
                    prepared.global_type == "novel_pending"
                    and not bool(prepared.token.meta.get("novel_activated", False))
                ):
                    rejected["deferred_novel_pending"] += 1
                    continue
                eligible.append(prepared)

            region_prototypes = tuple(
                prototype
                for prototype in (
                    self._region_prototype(labeled_keys),
                    self._region_prototype(previous_keys),
                )
                if prototype is not None
            )
            ranked = self._score_region_candidates(
                eligible,
                global_banks,
                region_prototypes,
            )
            selected, selection_rejected, selected_stats = self._select_region(
                ranked,
                capacity=capacity,
                per_image_limit=self.sample_per_image[region],
            )
            rejected.update(selection_rejected)
            selected_tokens[region] = [item.token for item in selected]
            total_rejected.update(rejected)

            region_stats[region] = self._region_stats(
                input_count=len(raw_tokens),
                eligible=ranked,
                selected=selected,
                capacity=capacity,
                per_image_limit=self.sample_per_image[region],
                rejected=rejected,
                input_types=input_types,
                selected_stats=selected_stats,
            )

        result = {
            "selected_tokens": selected_tokens,
            "stats": {
                "diversity_enabled": self.use_diversity_selection,
                "lambda_diversity": self.lambda_diversity,
                "spatial_nms_distance": self.spatial_nms_distance,
                "feature_dup_sim_threshold": self.feature_dup_sim_threshold,
                "input_tokens": sum(
                    item["input_tokens"] for item in region_stats.values()
                ),
                "eligible_tokens": sum(
                    item["eligible_tokens"] for item in region_stats.values()
                ),
                "selected_tokens": sum(
                    item["selected_tokens"] for item in region_stats.values()
                ),
                "rejected": dict(total_rejected),
                "regions": region_stats,
            },
        }
        self.last_result = result
        self._log_summary(result["stats"])
        return result

    def _score_region_candidates(
        self,
        candidates: Sequence[_PreparedCandidate],
        global_banks: Sequence[torch.Tensor],
        region_prototypes: Sequence[torch.Tensor],
    ) -> List[_PreparedCandidate]:
        grouped: Dict[str, List[_PreparedCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.image_id].append(candidate)

        scored: List[_PreparedCandidate] = []
        for image_candidates in grouped.values():
            global_query = image_candidates[0].normalized_global_key
            for candidate in image_candidates[1:]:
                if not torch.allclose(
                    candidate.normalized_global_key,
                    global_query,
                    atol=1.0e-5,
                    rtol=1.0e-4,
                ):
                    raise ValueError(
                        "tokens from the same image must share one x3 global key"
                    )

            d_img = self._diversity_from_banks(global_query, global_banks)
            region_query = self._normalized_mean(
                [candidate.normalized_key for candidate in image_candidates]
            )
            d_region = self._diversity_from_prototypes(
                region_query,
                region_prototypes,
            )
            raw_diversity = self._clamp01(0.5 * (d_img + d_region))
            applied_diversity = (
                raw_diversity if self.use_diversity_selection else 0.0
            )

            for candidate in image_candidates:
                selection_score = float(candidate.token.reliability) * (
                    1.0 + self.lambda_diversity * applied_diversity
                )
                token = self._copy_with_diversity(
                    candidate.token,
                    d_img=d_img,
                    d_region=d_region,
                    raw_diversity=raw_diversity,
                    applied_diversity=applied_diversity,
                    selection_score=selection_score,
                )
                scored.append(
                    replace(
                        candidate,
                        token=token,
                        d_img=d_img,
                        d_region=d_region,
                        diversity_score=applied_diversity,
                        selection_score=selection_score,
                    )
                )

        return sorted(
            scored,
            key=lambda item: (-item.selection_score, item.input_index),
        )

    def _select_region(
        self,
        ranked: Sequence[_PreparedCandidate],
        *,
        capacity: int,
        per_image_limit: int,
    ):
        selected: List[_PreparedCandidate] = []
        per_image_counts: Counter[str] = Counter()
        rejected: Counter[str] = Counter()
        min_spatial_distance: Optional[float] = None
        max_feature_similarity: Optional[float] = None
        min_distance_sq = self.spatial_nms_distance * self.spatial_nms_distance

        for position, candidate in enumerate(ranked):
            if len(selected) >= capacity:
                rejected["capacity"] += len(ranked) - position
                break
            if per_image_counts[candidate.image_id] >= per_image_limit:
                rejected["per_image_limit"] += 1
                continue

            if self.use_diversity_selection:
                same_image_distances = [
                    self._distance_sq(candidate.coord, item.coord)
                    for item in selected
                    if item.image_id == candidate.image_id
                ]
                if same_image_distances:
                    nearest_sq = min(same_image_distances)
                    if nearest_sq <= min_distance_sq:
                        rejected["spatial_nms"] += 1
                        continue
                else:
                    nearest_sq = None

                similarities = [
                    float(torch.dot(candidate.normalized_key, item.normalized_key).item())
                    for item in selected
                ]
                if similarities:
                    nearest_similarity = max(similarities)
                    if nearest_similarity >= self.feature_dup_sim_threshold:
                        rejected["feature_duplicate"] += 1
                        continue
                else:
                    nearest_similarity = None

                if nearest_sq is not None:
                    nearest_distance = math.sqrt(nearest_sq)
                    min_spatial_distance = (
                        nearest_distance
                        if min_spatial_distance is None
                        else min(min_spatial_distance, nearest_distance)
                    )
                if nearest_similarity is not None:
                    max_feature_similarity = (
                        nearest_similarity
                        if max_feature_similarity is None
                        else max(max_feature_similarity, nearest_similarity)
                    )

            selected.append(candidate)
            per_image_counts[candidate.image_id] += 1

        return selected, rejected, {
            "selected_per_image": dict(per_image_counts),
            "min_same_image_spatial_distance": min_spatial_distance,
            "max_selected_feature_similarity": max_feature_similarity,
        }

    def _prepare_candidate(
        self,
        token,
        region: str,
        input_index: int,
    ) -> _PreparedCandidate:
        if not isinstance(token, UnlabeledMemoryToken):
            raise TypeError(
                f"candidate_pool[{region!r}][{input_index}] must be "
                "UnlabeledMemoryToken"
            )
        if not isinstance(token.meta, Mapping):
            raise TypeError("candidate token meta must be a mapping")
        missing = [
            name
            for name in ("image_id", "coord", "region", "global_type")
            if name not in token.meta
        ]
        if missing:
            raise KeyError(f"candidate token meta is missing fields: {missing}")
        if str(token.meta["region"]) != region:
            raise ValueError(
                f"candidate region {token.meta['region']!r} does not match {region!r}"
            )

        coord_value = token.meta["coord"]
        if (
            not isinstance(coord_value, Sequence)
            or isinstance(coord_value, (str, bytes))
            or len(coord_value) != 2
        ):
            raise ValueError("candidate coord must be a two-element sequence")
        coord = (int(coord_value[0]), int(coord_value[1]))
        image_id = str(token.meta["image_id"])
        global_type = str(token.meta["global_type"])
        if global_type not in {"matched", "expanded", "novel_pending"}:
            raise ValueError(f"unsupported global_type: {global_type!r}")

        reliability = float(token.reliability)
        if not math.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
            raise ValueError("candidate reliability must be finite and in [0, 1]")
        key = self._candidate_vector(token.key, "candidate.key")
        global_key = self._candidate_vector(token.global_key, "candidate.global_key")
        if not isinstance(token.value, torch.Tensor) or token.value.ndim != 1:
            raise ValueError("candidate.value must be a one-dimensional tensor")
        if not torch.isfinite(token.value).all():
            raise ValueError("candidate.value contains non-finite values")

        return _PreparedCandidate(
            token=token,
            input_index=input_index,
            image_id=image_id,
            coord=coord,
            global_type=global_type,
            normalized_key=self._normalize_vector(key),
            normalized_global_key=self._normalize_vector(global_key),
        )

    def _copy_with_diversity(
        self,
        token: UnlabeledMemoryToken,
        *,
        d_img: float,
        d_region: float,
        raw_diversity: float,
        applied_diversity: float,
        selection_score: float,
    ) -> UnlabeledMemoryToken:
        meta = self._detach_metadata(token.meta)
        meta.update(
            {
                "D_img": float(d_img),
                "D_region": float(d_region),
                "raw_diversity_score": float(raw_diversity),
                "diversity_score": float(applied_diversity),
                "selection_score": float(selection_score),
                "diversity_selection_enabled": self.use_diversity_selection,
            }
        )
        global_meta = (
            None
            if token.global_meta is None
            else self._detach_metadata(token.global_meta)
        )
        return replace(
            token,
            key=token.key.detach().cpu().clone(),
            value=token.value.detach().cpu().clone(),
            global_key=token.global_key.detach().cpu().clone(),
            meta=meta,
            reliability=float(token.reliability),
            diversity=float(applied_diversity),
            global_meta=global_meta,
        )

    def _region_stats(
        self,
        *,
        input_count: int,
        eligible: Sequence[_PreparedCandidate],
        selected: Sequence[_PreparedCandidate],
        capacity: int,
        per_image_limit: int,
        rejected: Counter,
        input_types: Counter,
        selected_stats: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "input_tokens": input_count,
            "eligible_tokens": len(eligible),
            "selected_tokens": len(selected),
            "capacity": capacity,
            "per_image_limit": per_image_limit,
            "rejected": dict(rejected),
            "input_global_type_counts": dict(input_types),
            "selected_global_type_counts": dict(
                Counter(item.global_type for item in selected)
            ),
            "eligible_mean_D_img": self._mean([item.d_img for item in eligible]),
            "eligible_mean_D_region": self._mean(
                [item.d_region for item in eligible]
            ),
            "eligible_mean_diversity_score": self._mean(
                [item.diversity_score for item in eligible]
            ),
            "selected_mean_diversity_score": self._mean(
                [item.diversity_score for item in selected]
            ),
            "selected_mean_selection_score": self._mean(
                [item.selection_score for item in selected]
            ),
            **dict(selected_stats),
        }

    @staticmethod
    def _memory_global_keys(
        memory,
        name: str,
        allow_none: bool = False,
    ) -> torch.Tensor:
        if memory is None:
            if allow_none:
                return torch.empty(0, 0, dtype=torch.float32)
            raise TypeError(f"{name} must not be None")
        if hasattr(memory, "get_image_keys"):
            result = memory.get_image_keys()
            if not isinstance(result, tuple) or len(result) < 1:
                raise TypeError(f"{name}.get_image_keys() must return a tuple")
            raw = result[0]
        elif hasattr(memory, "image_keys"):
            raw = memory.image_keys
        elif hasattr(memory, "global_keys"):
            raw = memory.global_keys
        else:
            raise TypeError(f"{name} must expose image/global keys")
        width = int(getattr(memory, "mem_dim", 0))
        return UMEDiversitySampler._as_matrix(raw, f"{name} global keys", width)

    @staticmethod
    def _memory_region_keys(
        memory,
        region: str,
        name: str,
        allow_none: bool = False,
    ) -> torch.Tensor:
        if memory is None:
            if allow_none:
                return torch.empty(0, 0, dtype=torch.float32)
            raise TypeError(f"{name} must not be None")
        keys = getattr(memory, "keys", None)
        if isinstance(keys, Mapping):
            if region not in keys:
                raise KeyError(f"{name}.keys is missing region {region!r}")
            raw = keys[region]
        elif hasattr(memory, "get_region_memory"):
            result = memory.get_region_memory(region)
            if not isinstance(result, tuple) or len(result) < 1:
                raise TypeError(f"{name}.get_region_memory() must return a tuple")
            raw = result[0]
        else:
            raise TypeError(f"{name} must expose a region-indexed keys mapping")
        width = int(getattr(memory, "mem_dim", 0))
        return UMEDiversitySampler._as_matrix(
            raw,
            f"{name}.keys[{region!r}]",
            width,
        )

    @staticmethod
    def _as_matrix(value, name: str, empty_width: int = 0) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            matrix = value.detach()
            if matrix.ndim != 2:
                raise ValueError(f"{name} must be a two-dimensional tensor")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            parts = []
            for part in value:
                if not isinstance(part, torch.Tensor):
                    raise TypeError(f"{name} sequence entries must be tensors")
                current = part.detach()
                if current.ndim == 1:
                    current = current.unsqueeze(0)
                if current.ndim != 2:
                    raise ValueError(f"{name} entries must be one- or two-dimensional")
                parts.append(current)
            matrix = (
                torch.cat(parts, dim=0)
                if parts
                else torch.empty(0, empty_width, dtype=torch.float32)
            )
        else:
            raise TypeError(f"{name} must be a tensor or tensor sequence")
        if not torch.isfinite(matrix).all():
            raise ValueError(f"{name} contains non-finite values")
        return matrix.to(device="cpu", dtype=torch.float32)

    @staticmethod
    def _candidate_vector(value, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional tensor")
        if value.numel() == 0:
            raise ValueError(f"{name} must not be empty")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        return value.detach().to(device="cpu", dtype=torch.float32)

    @staticmethod
    def _normalize_vector(value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value.unsqueeze(0), dim=1, eps=1.0e-12).squeeze(0)

    @staticmethod
    def _normalized_mean(values: Sequence[torch.Tensor]) -> torch.Tensor:
        if not values:
            raise ValueError("cannot compute a prototype from an empty sequence")
        widths = {int(value.numel()) for value in values}
        if len(widths) != 1:
            raise ValueError("candidate key dimensions do not match")
        return UMEDiversitySampler._normalize_vector(torch.stack(list(values)).mean(dim=0))

    @staticmethod
    def _region_prototype(keys: torch.Tensor) -> Optional[torch.Tensor]:
        if keys.size(0) == 0:
            return None
        return UMEDiversitySampler._normalize_vector(keys.mean(dim=0))

    @staticmethod
    def _diversity_from_banks(
        query: torch.Tensor,
        banks: Sequence[torch.Tensor],
    ) -> float:
        similarities = []
        for bank in banks:
            if bank.size(1) != query.numel():
                raise ValueError("candidate global key dimension does not match memory")
            normalized = F.normalize(bank, dim=1, eps=1.0e-12)
            similarities.append(float(torch.mv(normalized, query).max().item()))
        if not similarities:
            return 1.0
        return UMEDiversitySampler._clamp01(1.0 - max(similarities))

    @staticmethod
    def _diversity_from_prototypes(
        query: torch.Tensor,
        prototypes: Sequence[torch.Tensor],
    ) -> float:
        similarities = []
        for prototype in prototypes:
            if prototype.numel() != query.numel():
                raise ValueError("candidate region key dimension does not match memory")
            similarities.append(float(torch.dot(prototype, query).item()))
        if not similarities:
            return 1.0
        return UMEDiversitySampler._clamp01(1.0 - max(similarities))

    def _region_config(self, value, name: str, cast) -> dict:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a region-indexed mapping")
        unknown = set(value) - set(self.regions)
        if unknown:
            raise KeyError(f"{name} has unknown regions: {sorted(unknown)}")
        missing = [region for region in self.regions if region not in value]
        if missing:
            raise KeyError(f"{name} is missing regions: {missing}")
        return {region: cast(value[region]) for region in self.regions}

    def _log_summary(self, stats: Mapping[str, Any]) -> None:
        if self.logger is None:
            return
        message = (
            "[SV-UME] diversity selection "
            f"input={stats['input_tokens']} eligible={stats['eligible_tokens']} "
            f"selected={stats['selected_tokens']}"
        )
        if hasattr(self.logger, "info"):
            self.logger.info(message)
        elif callable(self.logger):
            self.logger(message)

    @staticmethod
    def _detach_metadata(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, Mapping):
            return {
                key: UMEDiversitySampler._detach_metadata(item)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(UMEDiversitySampler._detach_metadata(item) for item in value)
        if isinstance(value, list):
            return [UMEDiversitySampler._detach_metadata(item) for item in value]
        return value

    @staticmethod
    def _distance_sq(left: Tuple[int, int], right: Tuple[int, int]) -> float:
        return float((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _clamp01(value: float) -> float:
        return min(max(float(value), 0.0), 1.0)


__all__ = [
    "UMEDiversitySampler",
    "DEFAULT_LAMBDA_DIVERSITY",
    "DEFAULT_SPATIAL_NMS_DISTANCE",
    "DEFAULT_FEATURE_DUP_SIM_THRESHOLD",
]
