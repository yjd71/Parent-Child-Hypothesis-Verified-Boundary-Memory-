from __future__ import annotations

import copy
import inspect
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, Iterator, List, Optional

import torch

from CBM.memory.labels import REGION_NAMES
from CBM.sv_ume.schedules import (
    can_use_lagged_memory,
    expected_unlabeled_source_epoch,
    should_build_after_epoch,
    sv_ume_enabled,
)
from utils.log_control import log_enabled


LOGGER = logging.getLogger(__name__)
STATE_VERSION = 2
LEGACY_STATE_VERSION = 1


class SVUMEZeroCandidatesError(RuntimeError):
    """Raised when a complete collection pass admits no memory candidates."""


class SVUMEManager:
    """Manage lagged SAM-refined unlabeled memory across epoch boundaries.

    Candidate-builder contract::

        build_batch(
            *, teacher, sam_refiner, batch, labeled_memory,
            memory_for_retrieval, epoch, device
        ) -> Mapping[str, Sequence[candidate]]

    Memory-builder contract::

        build_memory(*, candidate_pool, labeled_memory, region_capacities, epoch)
        freeze_memory(memory)
        memory_state_dict(memory)
        load_memory_state_dict(state)

    This class deliberately does not write candidates into the memory used by
    the current training epoch. ``U_next`` becomes visible only after
    :meth:`step_epoch` promotes it to a frozen ``U_prev``.
    """

    _CANDIDATE_BUILDER_METHODS = ("build_batch",)
    _MEMORY_BUILDER_METHODS = (
        "build_memory",
        "freeze_memory",
        "memory_state_dict",
        "load_memory_state_dict",
    )

    def __init__(
        self,
        cfg,
        candidate_builder=None,
        memory_builder=None,
        visualizer=None,
        logger=None,
    ) -> None:
        self.cfg = cfg
        self.candidate_builder = candidate_builder
        self.memory_builder = memory_builder
        self.visualizer = visualizer
        self.logger = logger
        self.enabled = sv_ume_enabled(cfg)

        self.U_prev = None
        self.U_next = None
        self.candidate_pool = self._empty_candidate_pool()
        self._candidate_index = {region: {} for region in REGION_NAMES}
        self.epoch_stats: Dict[str, Any] = self._new_epoch_stats(
            epoch=None,
            status="initialized" if self.enabled else "disabled",
        )
        self._candidate_epoch: Optional[int] = None
        self._u_prev_epoch: Optional[int] = None
        self._u_next_epoch: Optional[int] = None
        self.last_used_u_prev_epoch: Optional[int] = None
        self.temporal_pseudo_label_cache: Dict[str, List[dict]] = {}
        self.global_type_metadata: List[dict] = []
        self.loaded_config_snapshot: Dict[str, Any] = {}

        if self.enabled:
            self._validate_enabled_config()
            self._validate_dependency(
                self.candidate_builder,
                self._CANDIDATE_BUILDER_METHODS,
                "candidate_builder",
            )
            self._validate_dependency(
                self.memory_builder,
                self._MEMORY_BUILDER_METHODS,
                "memory_builder",
            )

    @property
    def u_prev_epoch(self) -> Optional[int]:
        return self._u_prev_epoch

    @property
    def u_next_epoch(self) -> Optional[int]:
        return self._u_next_epoch

    def get_unlabeled_memory_for_epoch(self, epoch: int):
        """Return only a ready frozen memory from exactly ``epoch - 1``."""
        if not self.enabled:
            return None

        current_epoch = self._normalize_epoch(epoch)
        usable = (
            can_use_lagged_memory(self.cfg, current_epoch, self._u_prev_epoch)
            and self._memory_ready(self.U_prev)
        )
        memory = self.U_prev if usable else None
        source_epoch = self._u_prev_epoch if self.U_prev is not None else None
        expected_epoch = expected_unlabeled_source_epoch(self.cfg, current_epoch)
        self.last_used_u_prev_epoch = source_epoch if usable else None
        self.epoch_stats.update(
            {
                "query_epoch": current_epoch,
                "u_prev_status": "ready" if usable else "none",
                "u_prev_epoch": source_epoch,
            }
        )
        self._info(
            f"[SV-UME][schedule] current_epoch={current_epoch} "
            f"start_epoch={int(getattr(self.cfg, 'sv_ume_start_epoch', 21))} "
            f"expected_U_prev_epoch={expected_epoch} actual_U_prev_epoch={source_epoch} "
            f"can_use={usable} "
            f"should_build={should_build_after_epoch(self.cfg, current_epoch)} "
            f"build_after_epoch={bool(getattr(self.cfg, 'build_unlabeled_memory_after_epoch', True))} "
            f"current_epoch_use={bool(getattr(self.cfg, 'use_unlabeled_memory_during_current_epoch', False))}"
        )
        return memory

    def clear_candidate_pool(self) -> None:
        self.candidate_pool = self._empty_candidate_pool()
        self._candidate_index = {region: {} for region in REGION_NAMES}

    @torch.no_grad()
    def collect_candidates_after_epoch(
        self,
        teacher,
        sam_refiner,
        unlabeled_loader: Iterable[Any],
        labeled_memory,
        memory_for_retrieval,
        epoch: int,
        device,
    ) -> Dict[str, Any]:
        """Collect an isolated candidate pool after training epoch ``t``."""
        current_epoch = self._normalize_epoch(epoch)
        if not self.enabled or not should_build_after_epoch(self.cfg, current_epoch):
            return self.epoch_stats

        if teacher is None:
            raise ValueError("teacher is required when SV-UME is enabled")
        if bool(getattr(self.cfg, "sv_ume_require_svb_plr", True)) and sam_refiner is None:
            raise ValueError("sam_refiner is required by sv_ume_require_svb_plr=True")
        if unlabeled_loader is None:
            raise ValueError("unlabeled_loader is required when SV-UME is enabled")
        if labeled_memory is None:
            raise ValueError("labeled_memory is required when SV-UME is enabled")

        self.clear_candidate_pool()
        self.U_next = None
        self._u_next_epoch = None
        self._candidate_epoch = current_epoch
        self.epoch_stats = self._new_epoch_stats(current_epoch, "collecting")

        try:
            for batch in unlabeled_loader:
                self.epoch_stats["batches_seen"] += 1
                batch_candidates = self.candidate_builder.build_batch(
                    teacher=teacher,
                    sam_refiner=sam_refiner,
                    batch=batch,
                    labeled_memory=labeled_memory,
                    memory_for_retrieval=memory_for_retrieval,
                    epoch=current_epoch,
                    device=device,
                )
                self._append_batch_candidates(batch_candidates)
                self._accumulate_candidate_diagnostics(
                    getattr(self.candidate_builder, "last_result", None)
                )
        except Exception as exc:
            self.clear_candidate_pool()
            self.U_next = None
            self._candidate_epoch = None
            self._u_next_epoch = None
            self.epoch_stats["status"] = "collect_error"
            self.epoch_stats["error"] = str(exc)
            self.epoch_stats["candidate_counts"] = self._candidate_counts()
            self._error(f"[SV-UME] candidate collection failed at epoch {current_epoch}: {exc}")
            raise

        self.epoch_stats["status"] = "candidates_collected"
        self.epoch_stats["candidate_counts"] = self._candidate_counts()
        self._info(
            f"[SV-UME] epoch={current_epoch} candidates="
            f"{self.epoch_stats['candidate_counts']}"
        )
        self._info(f"[SV-UME][reject] epoch={current_epoch} {self.epoch_stats['rejected']}")
        self._info(
            f"[SV-UME][stats] epoch={current_epoch} "
            f"region_pixel_counts={self.epoch_stats['region_pixel_counts']} "
            f"region_image_counts={self.epoch_stats['region_image_counts']} "
            f"image_score={self.epoch_stats['image_score']} "
            f"image_threshold={self.epoch_stats['image_threshold']} "
            f"image_valid={self.epoch_stats['image_evidence_valid_count']} "
            f"image_above={self.epoch_stats['image_above_threshold_count']} "
            f"image_allowed={self.epoch_stats['image_allowed_count']} "
            f"image_quantiles={self.epoch_stats['image_score_quantiles']} "
            f"image_components={self.epoch_stats['image_component_quantiles']} "
            f"region_score_mean={self.epoch_stats['region_score_mean']} "
            f"token_score={self.epoch_stats['token_score_quantiles']} "
            f"cbm_valid_ratio={self.epoch_stats['cbm_valid_ratio']} "
            f"global_type_counts={self.epoch_stats['global_type_counts']}"
        )
        if sum(self.epoch_stats["candidate_counts"].values()) == 0:
            message = self._zero_candidate_message(current_epoch)
            self.epoch_stats["status"] = "zero_candidates"
            self.epoch_stats["error"] = message
            self._error(message)
            raise SVUMEZeroCandidatesError(message)
        return self.epoch_stats

    def build_next_memory(self, labeled_memory, epoch: int):
        """Build ``U_t`` without exposing it to the current epoch."""
        current_epoch = self._normalize_epoch(epoch)
        if not self.enabled or not should_build_after_epoch(self.cfg, current_epoch):
            return None
        if self._candidate_epoch != current_epoch:
            raise RuntimeError(
                f"candidate pool belongs to epoch {self._candidate_epoch}, "
                f"cannot build U_{current_epoch}"
            )
        if sum(self._candidate_counts().values()) == 0:
            raise SVUMEZeroCandidatesError(self._zero_candidate_message(current_epoch))
        if labeled_memory is None:
            raise ValueError("labeled_memory is required to enforce U:L capacity")

        capacities = self._region_capacities(labeled_memory)
        self.epoch_stats["region_capacities"] = dict(capacities)
        try:
            memory = self.memory_builder.build_memory(
                candidate_pool={region: list(self.candidate_pool[region]) for region in REGION_NAMES},
                labeled_memory=labeled_memory,
                region_capacities=dict(capacities),
                epoch=current_epoch,
            )
            self._validate_memory_object(memory)
            memory_counts = self._memory_region_counts(memory)
            self._validate_capacity(memory_counts, capacities)
            self._validate_active_novel_entries(memory)
        except Exception as exc:
            self.U_next = None
            self._u_next_epoch = None
            self.epoch_stats["status"] = "build_error"
            self.epoch_stats["error"] = str(exc)
            self._error(f"[SV-UME] U_{current_epoch} build failed: {exc}")
            raise

        self.U_next = memory
        self._u_next_epoch = current_epoch
        self.epoch_stats["status"] = "memory_built"
        self.epoch_stats["memory_counts"] = memory_counts
        self._info(f"[SV-UME] built U_{current_epoch} counts={memory_counts}")
        return self.U_next

    def step_epoch(self):
        """Freeze and promote ``U_next``; never reuse a stale prior memory."""
        if not self.enabled:
            return None

        next_epoch = self._u_next_epoch
        next_memory = self.U_next
        self.U_next = None
        self._u_next_epoch = None
        self._candidate_epoch = None
        self.clear_candidate_pool()

        if next_memory is None or next_epoch is None or not self._memory_ready(next_memory):
            self.U_prev = None
            self._u_prev_epoch = None
            self.temporal_pseudo_label_cache = {}
            self.global_type_metadata = []
            self.epoch_stats["status"] = "no_ready_next_memory"
            self.epoch_stats["u_prev_epoch"] = None
            self._info("[SV-UME] no ready U_next; next epoch will use labeled-only memory")
            return None

        frozen_memory = self.memory_builder.freeze_memory(next_memory)
        self._validate_memory_object(frozen_memory)
        self._assert_memory_frozen(frozen_memory)
        if not self._memory_ready(frozen_memory):
            self.U_prev = None
            self._u_prev_epoch = None
            self.epoch_stats["status"] = "frozen_memory_not_ready"
            self.epoch_stats["u_prev_epoch"] = None
            return None

        self.U_prev = frozen_memory
        self._u_prev_epoch = int(next_epoch)
        self._sync_memory_metadata(frozen_memory)
        self.epoch_stats["status"] = "memory_promoted"
        self.epoch_stats["u_prev_epoch"] = self._u_prev_epoch
        self._info(f"[SV-UME] promoted frozen U_{self._u_prev_epoch} to U_prev")
        return self.U_prev

    def state_dict(self) -> Dict[str, Any]:
        """Serialize lagged memory and optional in-progress epoch state."""
        if self.U_prev is not None:
            self._sync_memory_metadata(self.U_prev)
        state: Dict[str, Any] = {
            "version": STATE_VERSION,
            "enabled": bool(self.enabled),
            "u_prev_epoch": self._u_prev_epoch,
            "u_next_epoch": self._u_next_epoch,
            "candidate_epoch": self._candidate_epoch,
            "last_used_u_prev_epoch": self.last_used_u_prev_epoch,
            "u_prev_state": None,
            "u_next_state": None,
            "candidate_pool_state": None,
            "temporal_pseudo_label_cache": copy.deepcopy(
                self.temporal_pseudo_label_cache
            ),
            "global_type_metadata": copy.deepcopy(self.global_type_metadata),
            "epoch_stats": copy.deepcopy(self.epoch_stats),
            "config_snapshot": self._config_snapshot(self.cfg),
        }
        if self.enabled and self.U_prev is not None:
            state["u_prev_state"] = self._memory_state_dict(self.U_prev)
        if self.enabled and self.U_next is not None:
            state["u_next_state"] = self._memory_state_dict(self.U_next)
        if bool(getattr(self.cfg, "sv_ume_checkpoint_candidate_pool", False)):
            state["candidate_pool_state"] = self._serialize_candidate_pool()
        return state

    def load_state_dict(
        self,
        state,
        device=None,
        dtype=None,
    ) -> "SVUMEManager":
        """Restore v1/v2 manager state on the requested runtime device."""
        self._reset_checkpoint_state()

        if not self.enabled or not state:
            return self
        if not isinstance(state, Mapping):
            raise TypeError("SVUMEManager state must be a mapping")
        version = int(state.get("version", LEGACY_STATE_VERSION))
        if version not in (LEGACY_STATE_VERSION, STATE_VERSION):
            raise ValueError(f"unsupported SVUMEManager state version: {version}")

        raw_stats = state.get("epoch_stats", {})
        self.epoch_stats = self._restore_epoch_stats(raw_stats)
        raw_config = state.get("config_snapshot", {})
        self.loaded_config_snapshot = (
            copy.deepcopy(dict(raw_config)) if isinstance(raw_config, Mapping) else {}
        )
        self.last_used_u_prev_epoch = self._optional_epoch(
            state.get("last_used_u_prev_epoch")
        )

        u_prev_state = state.get("u_prev_state", state.get("U_prev"))
        u_prev_epoch = self._resolve_memory_epoch(
            state.get("u_prev_epoch"),
            u_prev_state,
            state.get("_checkpoint_epoch"),
        )
        if u_prev_state is not None and u_prev_epoch is not None:
            restored = self._load_memory_state(u_prev_state, device=device, dtype=dtype)
            frozen = self.memory_builder.freeze_memory(restored)
            self._validate_memory_object(frozen)
            self._assert_memory_frozen(frozen)
            if self._memory_ready(frozen):
                self.U_prev = frozen
                self._u_prev_epoch = int(u_prev_epoch)

        if version >= STATE_VERSION:
            u_next_state = state.get("u_next_state", state.get("U_next"))
            u_next_epoch = self._resolve_memory_epoch(
                state.get("u_next_epoch"),
                u_next_state,
                None,
            )
            if u_next_state is not None and u_next_epoch is not None:
                restored_next = self._load_memory_state(
                    u_next_state,
                    device=device,
                    dtype=dtype,
                )
                self._validate_memory_object(restored_next)
                if self._memory_ready(restored_next):
                    self.U_next = restored_next
                    self._u_next_epoch = int(u_next_epoch)

            self._candidate_epoch = self._optional_epoch(state.get("candidate_epoch"))
            candidate_pool_state = state.get("candidate_pool_state")
            if candidate_pool_state is not None:
                self._load_candidate_pool(candidate_pool_state)

        temporal_cache = state.get("temporal_pseudo_label_cache")
        global_types = state.get("global_type_metadata")
        if temporal_cache is not None:
            self.temporal_pseudo_label_cache = self._normalize_temporal_cache(
                temporal_cache
            )
        elif self.U_prev is not None:
            self.temporal_pseudo_label_cache = copy.deepcopy(
                getattr(self.U_prev, "temporal_pseudo_label_cache", {})
            )
        if global_types is not None:
            self.global_type_metadata = self._normalize_global_type_metadata(
                global_types
            )
        elif self.U_prev is not None:
            self.global_type_metadata = copy.deepcopy(
                getattr(self.U_prev, "global_type_metadata", [])
            )

        if self.U_prev is not None:
            self.epoch_stats["u_prev_epoch"] = self._u_prev_epoch
            self.epoch_stats["status"] = "state_restored"
        elif u_prev_state is not None:
            self.epoch_stats["status"] = "state_restore_labeled_only"
        return self

    def _validate_enabled_config(self) -> None:
        for name, default in (
            ("use_aux_evidence_fusion", True),
            ("use_aux_feature_fusion", True),
            ("use_aux_source_penalty", False),
            ("allow_aux_dominate", True),
        ):
            if not isinstance(getattr(self.cfg, name, default), bool):
                raise TypeError(f"{name} must be a bool")
        checks = (
            (
                not bool(getattr(self.cfg, "sv_ume_require_svb_plr", True))
                or bool(getattr(self.cfg, "use_svb_plr", False)),
                "SV-UME requires use_svb_plr=True",
            ),
            (
                bool(getattr(self.cfg, "use_lagged_unlabeled_memory", True)),
                "use_lagged_unlabeled_memory must be True",
            ),
            (
                bool(getattr(self.cfg, "build_unlabeled_memory_after_epoch", True)),
                "build_unlabeled_memory_after_epoch must be True",
            ),
            (
                not bool(getattr(self.cfg, "use_unlabeled_memory_during_current_epoch", False)),
                "use_unlabeled_memory_during_current_epoch must be False",
            ),
            (
                bool(getattr(self.cfg, "rebuild_labeled_memory_each_epoch", True)),
                "rebuild_labeled_memory_each_epoch must be True",
            ),
            (
                bool(getattr(self.cfg, "do_not_update_labeled_memory_with_unlabeled", True)),
                "do_not_update_labeled_memory_with_unlabeled must be True",
            ),
            (
                str(getattr(self.cfg, "unlabeled_memory_source", ""))
                == "sam_refined_pseudo_label",
                "unlabeled_memory_source must be 'sam_refined_pseudo_label'",
            ),
            (
                str(getattr(self.cfg, "unlabeled_memory_feature_source", "")) == "teacher_p3",
                "unlabeled_memory_feature_source must be 'teacher_p3'",
            ),
            (
                not bool(getattr(self.cfg, "use_sam_embedding_as_memory_key", False)),
                "use_sam_embedding_as_memory_key must be False",
            ),
            (
                bool(getattr(self.cfg, "retrieve_labeled_and_unlabeled_separately", True)),
                "retrieve_labeled_and_unlabeled_separately must be True",
            ),
            (
                bool(getattr(self.cfg, "use_aux_evidence_fusion", True)),
                "use_aux_evidence_fusion must be True",
            ),
            (
                str(getattr(self.cfg, "aux_fusion_mode", ""))
                == "quality_adaptive_symmetric",
                "aux_fusion_mode must be 'quality_adaptive_symmetric'",
            ),
            (
                not bool(getattr(self.cfg, "use_fixed_matched_novel_ratio", False)),
                "use_fixed_matched_novel_ratio must be False",
            ),
        )
        for passed, message in checks:
            if not passed:
                raise ValueError(message)

        start_epoch = int(getattr(self.cfg, "sv_ume_start_epoch", 21))
        if start_epoch < 0:
            raise ValueError("sv_ume_start_epoch must be non-negative")
        raw_gamma_max = getattr(self.cfg, "gamma_max_final", 1.0)
        if isinstance(raw_gamma_max, bool):
            raise TypeError("gamma_max_final must be numeric, not bool")
        gamma_max = float(raw_gamma_max)
        if not math.isfinite(gamma_max) or not 0.0 <= gamma_max <= 1.0:
            raise ValueError("gamma_max_final must be finite and in [0, 1]")
        raw_source_penalty = getattr(self.cfg, "aux_source_penalty_value", 0.0)
        if isinstance(raw_source_penalty, bool):
            raise TypeError("aux_source_penalty_value must be numeric, not bool")
        source_penalty = float(raw_source_penalty)
        if not math.isfinite(source_penalty) or source_penalty < 0.0:
            raise ValueError("aux_source_penalty_value must be finite and non-negative")
        total_ratio = float(getattr(self.cfg, "unlabeled_to_labeled_ratio", 1.0))
        if not 0.0 <= total_ratio <= 1.0:
            raise ValueError("unlabeled_to_labeled_ratio must be in [0, 1]")
        region_ratios = getattr(self.cfg, "region_capacity_ratio", None)
        if not isinstance(region_ratios, Mapping):
            raise TypeError("region_capacity_ratio must be a mapping")
        for region in REGION_NAMES:
            if region not in region_ratios:
                raise KeyError(f"region_capacity_ratio is missing {region}")
            ratio = float(region_ratios[region])
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"region_capacity_ratio[{region!r}] must be in [0, 1]")

    @staticmethod
    def _validate_dependency(dependency, method_names: Sequence[str], name: str) -> None:
        if dependency is None:
            raise ValueError(f"{name} is required when SV-UME is enabled")
        missing = [method for method in method_names if not callable(getattr(dependency, method, None))]
        if missing:
            raise TypeError(f"{name} is missing required methods: {', '.join(missing)}")

    def _accumulate_candidate_diagnostics(self, result) -> None:
        if not isinstance(result, Mapping):
            return
        rejected = result.get("rejected", {})
        if isinstance(rejected, Mapping):
            for name, value in rejected.items():
                self.epoch_stats["rejected"][str(name)] = (
                    int(self.epoch_stats["rejected"].get(str(name), 0)) + int(value)
                )
        stats = result.get("stats", {})
        if not isinstance(stats, Mapping):
            return
        self.epoch_stats["diagnostic_batches"] += 1
        batch_size = max(1, int(stats.get("batch_size", 1)))
        image_score = self.epoch_stats["image_score"]
        previous_count = int(image_score["count"])
        image_score["count"] += batch_size
        image_score["sum"] += float(stats.get("image_score_mean", 0.0)) * batch_size
        batch_min = float(stats.get("image_score_min", 0.0))
        batch_max = float(stats.get("image_score_max", 0.0))
        image_score["min"] = batch_min if previous_count == 0 else min(image_score["min"], batch_min)
        image_score["max"] = batch_max if previous_count == 0 else max(image_score["max"], batch_max)
        image_score["mean"] = image_score["sum"] / max(image_score["count"], 1)
        batch_threshold = stats.get("image_threshold")
        if batch_threshold is not None:
            batch_threshold = float(batch_threshold)
            epoch_threshold = self.epoch_stats["image_threshold"]
            if epoch_threshold is None:
                self.epoch_stats["image_threshold"] = batch_threshold
            elif not math.isclose(
                float(epoch_threshold), batch_threshold, rel_tol=0.0, abs_tol=1.0e-8
            ):
                raise RuntimeError(
                    f"image threshold changed within collection: "
                    f"{epoch_threshold} -> {batch_threshold}"
                )
        for name in (
            "image_evidence_valid_count",
            "image_above_threshold_count",
            "image_allowed_count",
        ):
            self.epoch_stats[name] += int(stats.get(name, 0))
        self._merge_quantile_summary(
            self.epoch_stats["image_score_quantiles"],
            "score",
            stats.get("image_score_quantiles", {}),
        )
        source_image_components = stats.get("image_component_quantiles", {})
        if isinstance(source_image_components, Mapping):
            for name, summary in source_image_components.items():
                self._merge_quantile_summary(
                    self.epoch_stats["image_component_quantiles"],
                    str(name),
                    summary,
                )
        for field in (
            "region_pixel_counts",
            "region_image_counts",
            "raw_candidate_counts",
            "global_type_counts",
        ):
            source_name = "candidate_counts" if field == "raw_candidate_counts" else field
            source = stats.get(source_name, {})
            if isinstance(source, Mapping):
                for name, value in source.items():
                    self.epoch_stats[field][str(name)] = int(
                        self.epoch_stats[field].get(str(name), 0)
                    ) + int(value)
        source_region_scores = stats.get("region_score_mean", {})
        if isinstance(source_region_scores, Mapping):
            for region, value in source_region_scores.items():
                self.epoch_stats["region_score_sum"][region] += float(value) * batch_size
        self.epoch_stats["region_score_weight"] += batch_size
        weight = max(self.epoch_stats["region_score_weight"], 1)
        self.epoch_stats["region_score_mean"] = {
            region: value / weight for region, value in self.epoch_stats["region_score_sum"].items()
        }
        for field in ("token_score_quantiles", "token_component_quantiles"):
            source = stats.get(field, {})
            if isinstance(source, Mapping):
                for name, summary in source.items():
                    self._merge_quantile_summary(self.epoch_stats[field], str(name), summary)
        source_ratios = stats.get("cbm_valid_ratio", {})
        source_quantiles = stats.get("token_score_quantiles", {})
        if isinstance(source_ratios, Mapping):
            for region, value in source_ratios.items():
                summary = source_quantiles.get(region, {}) if isinstance(source_quantiles, Mapping) else {}
                count = max(0, int(summary.get("count", 0))) if isinstance(summary, Mapping) else 0
                target = self.epoch_stats["cbm_valid_ratio"].setdefault(
                    str(region), {"count": 0, "sum": 0.0, "mean": 0.0}
                )
                target["count"] += count
                target["sum"] += float(value) * count
                target["mean"] = target["sum"] / max(target["count"], 1)

    def _zero_candidate_message(self, epoch: int) -> str:
        return (
            f"[SV-UME][zero-candidates] epoch={epoch} profile="
            f"{getattr(self.cfg, 'sv_ume_profile_name', 'unversioned')} "
            f"config={getattr(self.cfg, 'run_cfg_path', '<unknown>')} "
            f"sha256={getattr(self.cfg, 'run_cfg_sha256', '<unknown>')} "
            f"image_threshold={self.epoch_stats.get('image_threshold')} "
            f"image_valid={self.epoch_stats.get('image_evidence_valid_count', 0)} "
            f"image_above={self.epoch_stats.get('image_above_threshold_count', 0)} "
            f"image_allowed={self.epoch_stats.get('image_allowed_count', 0)} "
            f"image_score={self.epoch_stats.get('image_score_quantiles', {})} "
            f"image_components={self.epoch_stats.get('image_component_quantiles', {})} "
            f"rejected={self.epoch_stats.get('rejected', {})} "
            f"token_score={self.epoch_stats.get('token_score_quantiles', {})}"
        )

    @staticmethod
    def _merge_quantile_summary(target: Dict[str, Any], name: str, summary) -> None:
        if not isinstance(summary, Mapping):
            return
        count = max(0, int(summary.get("count", 0)))
        current = target.setdefault(
            name,
            {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0},
        )
        old_count = int(current["count"])
        total = old_count + count
        if total <= 0:
            return
        for key in ("mean", "p50", "p90", "p99"):
            current[key] = (
                float(current[key]) * old_count + float(summary.get(key, 0.0)) * count
            ) / total
        current["max"] = max(float(current["max"]), float(summary.get("max", 0.0)))
        current["count"] = total

    def _append_batch_candidates(self, batch_candidates) -> None:
        if not isinstance(batch_candidates, Mapping):
            raise TypeError("candidate_builder.build_batch must return a mapping")
        unknown_regions = set(batch_candidates) - set(REGION_NAMES)
        if unknown_regions:
            raise KeyError(f"unknown candidate regions: {sorted(unknown_regions)}")
        for region in REGION_NAMES:
            values = batch_candidates.get(region, ())
            if values is None:
                continue
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise TypeError(f"candidates for {region} must be a sequence")
            for candidate in values:
                identity = self._candidate_identity(candidate, region)
                rank = self._candidate_rank(candidate)
                previous = self._candidate_index[region].get(identity)
                if previous is None:
                    self._candidate_index[region][identity] = (
                        len(self.candidate_pool[region]),
                        rank,
                    )
                    self.candidate_pool[region].append(candidate)
                elif rank > previous[1]:
                    index = previous[0]
                    self.candidate_pool[region][index] = candidate
                    self._candidate_index[region][identity] = (index, rank)

    def _memory_state_dict(self, memory) -> Dict[str, Any]:
        memory_state = self.memory_builder.memory_state_dict(memory)
        if not isinstance(memory_state, Mapping):
            raise TypeError("memory_builder.memory_state_dict must return a mapping")
        return dict(memory_state)

    def _load_memory_state(self, state, *, device=None, dtype=None):
        load_fn = self.memory_builder.load_memory_state_dict
        signature = inspect.signature(load_fn)
        supports_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        kwargs = {}
        if supports_kwargs or "device" in signature.parameters:
            kwargs["device"] = device
        if supports_kwargs or "dtype" in signature.parameters:
            kwargs["dtype"] = dtype
        restored = load_fn(state, **kwargs)
        self._validate_memory_object(restored)
        return restored

    def _reset_checkpoint_state(self) -> None:
        self.U_prev = None
        self.U_next = None
        self._u_prev_epoch = None
        self._u_next_epoch = None
        self._candidate_epoch = None
        self.last_used_u_prev_epoch = None
        self.temporal_pseudo_label_cache = {}
        self.global_type_metadata = []
        self.loaded_config_snapshot = {}
        self.clear_candidate_pool()

    def _sync_memory_metadata(self, memory) -> None:
        temporal = getattr(memory, "temporal_pseudo_label_cache", None)
        global_types = getattr(memory, "global_type_metadata", None)
        if isinstance(temporal, Mapping):
            self.temporal_pseudo_label_cache = copy.deepcopy(dict(temporal))
        else:
            self.temporal_pseudo_label_cache = self._temporal_cache_from_memory(memory)
        if isinstance(global_types, list):
            self.global_type_metadata = copy.deepcopy(global_types)
        else:
            raw_global_meta = getattr(memory, "global_meta", [])
            self.global_type_metadata = self._normalize_global_type_metadata(
                list(raw_global_meta) if isinstance(raw_global_meta, Sequence) else []
            )

    def _serialize_candidate_pool(self) -> Dict[str, List[dict]]:
        return {
            region: [self._candidate_to_state(candidate, region) for candidate in self.candidate_pool[region]]
            for region in REGION_NAMES
        }

    def _load_candidate_pool(self, state) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("candidate_pool_state must be a region-indexed mapping")
        unknown = set(state) - set(REGION_NAMES)
        if unknown:
            raise KeyError(f"unknown candidate-pool regions: {sorted(unknown)}")
        from CBM.sv_ume.unlabeled_dense_memory import UnlabeledMemoryToken

        self.clear_candidate_pool()
        restored = self._empty_candidate_pool()
        for region in REGION_NAMES:
            entries = state.get(region, [])
            if not isinstance(entries, list):
                raise TypeError(f"candidate_pool_state[{region!r}] must be list[dict]")
            for raw in entries:
                if not isinstance(raw, Mapping):
                    raise TypeError("candidate pool entries must be mappings")
                meta = copy.deepcopy(dict(raw.get("meta", {})))
                global_meta = raw.get("global_meta")
                restored[region].append(
                    UnlabeledMemoryToken(
                        key=self._candidate_tensor(raw.get("key"), "candidate.key"),
                        value=self._candidate_tensor(raw.get("value"), "candidate.value"),
                        global_key=self._candidate_tensor(
                            raw.get("global_key"),
                            "candidate.global_key",
                        ),
                        meta=meta,
                        reliability=float(raw.get("reliability")),
                        diversity=float(raw.get("diversity", 0.0)),
                        global_meta=(
                            copy.deepcopy(dict(global_meta))
                            if isinstance(global_meta, Mapping)
                            else None
                        ),
                    )
                )
        self._append_batch_candidates(restored)

    def _candidate_to_state(self, candidate, region: str) -> dict:
        self._candidate_identity(candidate, region)
        global_meta = getattr(candidate, "global_meta", None)
        if global_meta is not None and not isinstance(global_meta, Mapping):
            raise TypeError("candidate.global_meta must be a mapping or None")
        return {
            "key": self._candidate_tensor(getattr(candidate, "key", None), "candidate.key"),
            "value": self._candidate_tensor(getattr(candidate, "value", None), "candidate.value"),
            "global_key": self._candidate_tensor(
                getattr(candidate, "global_key", None),
                "candidate.global_key",
            ),
            "meta": copy.deepcopy(dict(candidate.meta)),
            "reliability": float(getattr(candidate, "reliability")),
            "diversity": float(getattr(candidate, "diversity", 0.0)),
            "global_meta": (
                copy.deepcopy(dict(global_meta)) if global_meta is not None else None
            ),
        }

    @staticmethod
    def _candidate_tensor(value, name: str) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor")
        return value.detach().cpu().clone()

    @staticmethod
    def _candidate_identity(candidate, region: str):
        meta = getattr(candidate, "meta", None)
        if not isinstance(meta, Mapping):
            raise TypeError("candidate.meta must be a mapping")
        required = (
            "image_id",
            "coord",
            "region",
            "epoch_added",
            "step_added",
            "global_type",
        )
        missing = [name for name in required if name not in meta]
        if missing:
            raise KeyError(f"candidate meta is missing checkpoint fields: {missing}")
        if str(meta["region"]) != region:
            raise ValueError("candidate meta region does not match candidate pool")
        global_type = str(meta["global_type"])
        if global_type not in {"matched", "expanded", "novel_pending"}:
            raise ValueError(f"unsupported candidate global_type: {global_type!r}")
        coord = meta["coord"]
        if not isinstance(coord, Sequence) or isinstance(coord, (str, bytes)) or len(coord) != 2:
            raise ValueError("candidate coord must have two elements")
        epoch_added = int(meta["epoch_added"])
        step_added = int(meta["step_added"])
        if epoch_added < 0 or step_added < 0:
            raise ValueError("candidate epoch_added/step_added must be non-negative")
        return (
            str(meta["image_id"]),
            (int(coord[0]), int(coord[1])),
            region,
            epoch_added,
        )

    @staticmethod
    def _candidate_rank(candidate):
        reliability = float(getattr(candidate, "reliability"))
        if not math.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
            raise ValueError("candidate reliability must be finite and in [0, 1]")
        return reliability, int(candidate.meta["step_added"])

    def _temporal_cache_from_memory(self, memory) -> Dict[str, List[dict]]:
        meta = getattr(memory, "meta", None)
        if not isinstance(meta, Mapping):
            return {}
        cache: Dict[str, List[dict]] = {}
        for region in REGION_NAMES:
            entries = meta.get(region, [])
            if not isinstance(entries, Sequence):
                continue
            for raw in entries:
                if not isinstance(raw, Mapping) or "p_ref_value" not in raw:
                    continue
                item = {
                    "image_id": str(raw["image_id"]),
                    "coord": tuple(raw["coord"]),
                    "region": region,
                    "epoch_added": raw.get("epoch_added"),
                    "p_ref_value": float(raw["p_ref_value"]),
                }
                if raw.get("conf_ref_value") is not None:
                    item["conf_ref_value"] = float(raw["conf_ref_value"])
                cache.setdefault(item["image_id"], []).append(item)
        return cache

    @staticmethod
    def _normalize_temporal_cache(value) -> Dict[str, List[dict]]:
        if not isinstance(value, Mapping):
            raise TypeError("temporal_pseudo_label_cache must be a mapping")
        result: Dict[str, List[dict]] = {}
        for image_id, entries in value.items():
            if not isinstance(entries, list):
                raise TypeError("temporal cache entries must be list[dict]")
            normalized_entries = []
            for raw in entries:
                if not isinstance(raw, Mapping):
                    raise TypeError("temporal cache entries must be list[dict]")
                item = copy.deepcopy(dict(raw))
                item["image_id"] = str(item.get("image_id", image_id))
                coord = item.get("coord")
                if not isinstance(coord, Sequence) or isinstance(coord, (str, bytes)) or len(coord) != 2:
                    raise ValueError("temporal cache coord must have two elements")
                item["coord"] = (int(coord[0]), int(coord[1]))
                region = str(item["region"])
                if region not in REGION_NAMES:
                    raise ValueError(f"unsupported temporal cache region: {region!r}")
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
            result[str(image_id)] = normalized_entries
        return result

    @staticmethod
    def _normalize_global_type_metadata(value) -> List[dict]:
        if not isinstance(value, list):
            raise TypeError("global_type_metadata must be list[dict]")
        result = []
        for raw in value:
            if not isinstance(raw, Mapping):
                raise TypeError("global_type_metadata must be list[dict]")
            item = copy.deepcopy(dict(raw))
            if "image_id" not in item:
                raise KeyError("global type metadata is missing image_id")
            item["image_id"] = str(item["image_id"])
            global_type = str(item.get("global_type"))
            if global_type not in {"matched", "expanded", "novel_pending"}:
                raise ValueError(f"unsupported global_type: {global_type!r}")
            item["global_type"] = global_type
            result.append(item)
        return result

    @classmethod
    def _resolve_memory_epoch(cls, explicit, memory_state, checkpoint_epoch):
        value = cls._optional_epoch(explicit)
        if value is not None:
            return value
        if isinstance(memory_state, Mapping):
            epochs = []
            raw_meta = memory_state.get("meta", {})
            if isinstance(raw_meta, Mapping):
                for entries in raw_meta.values():
                    if isinstance(entries, Sequence):
                        for item in entries:
                            if isinstance(item, Mapping) and item.get("epoch_added") is not None:
                                epochs.append(int(item["epoch_added"]))
            if epochs:
                return max(epochs)
        return cls._optional_epoch(checkpoint_epoch)

    @staticmethod
    def _optional_epoch(value) -> Optional[int]:
        if value is None:
            return None
        epoch = int(value)
        return epoch if epoch >= 0 else None

    @classmethod
    def _config_snapshot(cls, cfg) -> Dict[str, Any]:
        raw = dict(cfg) if isinstance(cfg, Mapping) else dict(vars(cfg))
        snapshot = {}
        for name, value in raw.items():
            if str(name).startswith("_") or callable(value) or inspect.ismodule(value):
                continue
            supported, normalized = cls._snapshot_value(value)
            if supported:
                snapshot[str(name)] = normalized
        return snapshot

    @classmethod
    def _snapshot_value(cls, value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return True, value
        if isinstance(value, (torch.device, torch.dtype)):
            return True, str(value)
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                supported, normalized = cls._snapshot_value(item)
                if supported:
                    result[str(key)] = normalized
            return True, result
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result = []
            for item in value:
                supported, normalized = cls._snapshot_value(item)
                if supported:
                    result.append(normalized)
            return True, result
        if hasattr(value, "__fspath__"):
            return True, str(value)
        return False, None

    def _region_capacities(self, labeled_memory) -> Dict[str, int]:
        labeled_counts = self._memory_region_counts(labeled_memory)
        global_ratio = float(getattr(self.cfg, "unlabeled_to_labeled_ratio", 1.0))
        region_ratios = getattr(self.cfg, "region_capacity_ratio")
        return {
            region: int(math.floor(labeled_counts[region] * global_ratio * float(region_ratios[region])))
            for region in REGION_NAMES
        }

    def _validate_capacity(self, counts: Mapping[str, int], capacities: Mapping[str, int]) -> None:
        exceeded = {
            region: (int(counts[region]), int(capacities[region]))
            for region in REGION_NAMES
            if int(counts[region]) > int(capacities[region])
        }
        if exceeded:
            raise ValueError(f"unlabeled memory exceeds per-region capacity: {exceeded}")

    def _validate_active_novel_entries(self, memory) -> None:
        meta = getattr(memory, "meta", None)
        if not isinstance(meta, Mapping):
            raise TypeError("built memory must expose a region-indexed meta mapping")
        for region in REGION_NAMES:
            region_meta = meta.get(region, ())
            if not isinstance(region_meta, Sequence):
                raise TypeError(f"memory.meta[{region!r}] must be a sequence")
            for item in region_meta:
                if not isinstance(item, Mapping):
                    raise TypeError(f"memory.meta[{region!r}] entries must be mappings")
                if item.get("global_type") == "novel_pending" and not bool(item.get("novel_activated", False)):
                    raise ValueError(
                        f"inactive novel_pending token found in active memory region {region}"
                    )

    @staticmethod
    def _validate_memory_object(memory) -> None:
        if memory is None:
            raise ValueError("memory_builder returned None")
        if not callable(getattr(memory, "is_ready", None)):
            raise TypeError("memory object must expose is_ready()")
        keys = getattr(memory, "keys", None)
        if not isinstance(keys, Mapping):
            raise TypeError("memory object must expose a region-indexed keys mapping")
        missing = [region for region in REGION_NAMES if region not in keys]
        if missing:
            raise KeyError(f"memory.keys is missing regions: {missing}")

    def _assert_memory_frozen(self, memory) -> None:
        tensor_count = 0
        roots = [
            getattr(memory, "image_keys", None),
            getattr(memory, "global_keys", None),
            getattr(memory, "keys", None),
            getattr(memory, "values", None),
        ]
        state = self.memory_builder.memory_state_dict(memory)
        if not isinstance(state, Mapping):
            raise TypeError("memory_builder.memory_state_dict must return a mapping")
        roots.append(state)
        for root in roots:
            for tensor in self._iter_tensors(root):
                tensor_count += 1
                if tensor.requires_grad or tensor.grad_fn is not None:
                    raise ValueError("frozen U_prev contains tensors attached to autograd")
        if tensor_count == 0:
            raise ValueError("frozen U_prev contains no tensors")

    @classmethod
    def _iter_tensors(cls, value, seen: Optional[set[int]] = None) -> Iterator[torch.Tensor]:
        if seen is None:
            seen = set()
        if value is None:
            return
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)
        if torch.is_tensor(value):
            yield value
            return
        if isinstance(value, Mapping):
            for item in value.values():
                yield from cls._iter_tensors(item, seen)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                yield from cls._iter_tensors(item, seen)

    @staticmethod
    def _memory_ready(memory) -> bool:
        if memory is None:
            return False
        ready = getattr(memory, "is_ready", None)
        if not callable(ready):
            return False
        try:
            return bool(ready())
        except Exception:
            return False

    @staticmethod
    def _entry_count(value) -> int:
        if torch.is_tensor(value):
            return int(value.size(0)) if value.dim() > 0 else int(value.numel())
        try:
            return int(len(value))
        except TypeError as exc:
            raise TypeError(f"memory region entries must be sized, got {type(value).__name__}") from exc

    def _memory_region_counts(self, memory) -> Dict[str, int]:
        keys = getattr(memory, "keys", None)
        if not isinstance(keys, Mapping):
            raise TypeError("memory must expose a region-indexed keys mapping")
        missing = [region for region in REGION_NAMES if region not in keys]
        if missing:
            raise KeyError(f"memory.keys is missing regions: {missing}")
        return {region: self._entry_count(keys[region]) for region in REGION_NAMES}

    def _candidate_counts(self) -> Dict[str, int]:
        return {region: len(self.candidate_pool[region]) for region in REGION_NAMES}

    @staticmethod
    def _empty_candidate_pool() -> Dict[str, List[Any]]:
        return {region: [] for region in REGION_NAMES}

    @staticmethod
    def _normalize_epoch(epoch: int) -> int:
        try:
            return int(epoch)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"epoch must be an integer-compatible value, got {epoch!r}") from exc

    @staticmethod
    def _new_epoch_stats(epoch: Optional[int], status: str) -> Dict[str, Any]:
        return {
            "epoch": epoch,
            "status": status,
            "batches_seen": 0,
            "diagnostic_batches": 0,
            "candidate_counts": {region: 0 for region in REGION_NAMES},
            "raw_candidate_counts": {region: 0 for region in REGION_NAMES},
            "rejected": {},
            "region_pixel_counts": {region: 0 for region in REGION_NAMES},
            "region_image_counts": {region: 0 for region in REGION_NAMES},
            "image_score": {
                "count": 0,
                "sum": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            },
            "image_threshold": None,
            "image_evidence_valid_count": 0,
            "image_above_threshold_count": 0,
            "image_allowed_count": 0,
            "image_score_quantiles": {},
            "image_component_quantiles": {},
            "region_score_sum": {region: 0.0 for region in REGION_NAMES},
            "region_score_mean": {region: 0.0 for region in REGION_NAMES},
            "region_score_weight": 0,
            "token_score_quantiles": {},
            "token_component_quantiles": {},
            "cbm_valid_ratio": {},
            "global_type_counts": {},
            "region_capacities": {region: 0 for region in REGION_NAMES},
            "memory_counts": {region: 0 for region in REGION_NAMES},
            "u_prev_epoch": None,
            "error": None,
        }

    @classmethod
    def _restore_epoch_stats(cls, raw_stats: Any) -> Dict[str, Any]:
        """Merge legacy checkpoint stats into the current diagnostic schema."""
        if not isinstance(raw_stats, Mapping):
            return cls._new_epoch_stats(epoch=None, status="state_restored")
        restored = cls._new_epoch_stats(
            epoch=raw_stats.get("epoch"),
            status=str(raw_stats.get("status", "state_restored")),
        )
        for name, value in raw_stats.items():
            if isinstance(restored.get(name), dict) and isinstance(value, Mapping):
                restored[name].update(copy.deepcopy(dict(value)))
            else:
                restored[name] = copy.deepcopy(value)
        return restored

    def _info(self, message: str) -> None:
        if not log_enabled(self.cfg):
            return
        if self.logger is None:
            LOGGER.info(message)
            return
        log_fn = getattr(self.logger, "info", None) or getattr(self.logger, "key_info", None)
        if callable(log_fn):
            log_fn(message)

    def _error(self, message: str) -> None:
        if not log_enabled(self.cfg):
            return
        if self.logger is None:
            LOGGER.error(message)
            return
        log_fn = (
            getattr(self.logger, "error", None)
            or getattr(self.logger, "warning", None)
            or getattr(self.logger, "warn_info", None)
            or getattr(self.logger, "info", None)
        )
        if callable(log_fn):
            log_fn(message)


__all__ = ["SVUMEManager", "SVUMEZeroCandidatesError"]
