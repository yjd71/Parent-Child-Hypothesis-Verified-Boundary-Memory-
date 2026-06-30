from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional


REGIONS = ("fg_core", "fg_boundary", "bg_near", "bg_far")

_FAIR_MAX_SIZES = {
    "fg_core": 8192,
    "fg_boundary": 16384,
    "bg_near": 16384,
    "bg_far": 8192,
}

_MIN_TOKENS_PER_COMPONENT = {
    "fg_core": 2,
    "fg_boundary": 4,
    "bg_near": 4,
    "bg_far": 2,
}


@dataclass(frozen=True)
class LabeledMemorySelectionConfig:
    profile_name: str
    split: Optional[float]
    sample_per_image: Dict[str, int]
    max_sizes: Dict[str, int]
    top_img_k: int
    grid_size: int
    min_tokens_per_component: Dict[str, int]
    min_spatial_dist: Dict[str, float]
    grid_quota_ratio: Dict[str, float]
    max_feature_sim: float
    relaxed_min_spatial_dist: float
    relaxed_max_feature_sim: float
    allow_underfill: bool
    use_component_quota: bool
    use_grid_quota: bool
    use_spatial_diversity: bool
    use_feature_diversity: bool
    relax_diversity_if_underfilled: bool
    global_fill_max_per_image: Optional[Dict[str, int]]


def _profile(
    name: str,
    split: float,
    sample_per_image: Dict[str, int],
    top_img_k: int,
    max_feature_sim: float,
    min_spatial_dist: Dict[str, float],
    grid_quota_ratio: Dict[str, float],
    max_sizes: Optional[Dict[str, int]] = None,
) -> LabeledMemorySelectionConfig:
    return LabeledMemorySelectionConfig(
        profile_name=name,
        split=float(split),
        sample_per_image=dict(sample_per_image),
        max_sizes=dict(max_sizes or _FAIR_MAX_SIZES),
        top_img_k=int(top_img_k),
        grid_size=4,
        min_tokens_per_component=dict(_MIN_TOKENS_PER_COMPONENT),
        min_spatial_dist=dict(min_spatial_dist),
        grid_quota_ratio=dict(grid_quota_ratio),
        max_feature_sim=float(max_feature_sim),
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


_AUTO_PROFILES = {
    0.01: _profile(
        "1p",
        0.01,
        {"fg_core": 256, "fg_boundary": 640, "bg_near": 640, "bg_far": 256},
        8,
        0.98,
        {"fg_core": 2, "fg_boundary": 2, "bg_near": 2, "bg_far": 3},
        {"fg_core": 0.30, "fg_boundary": 0.18, "bg_near": 0.18, "bg_far": 0.15},
    ),
    0.05: _profile(
        "5p",
        0.05,
        {"fg_core": 128, "fg_boundary": 384, "bg_near": 384, "bg_far": 128},
        32,
        0.98,
        {"fg_core": 2, "fg_boundary": 2, "bg_near": 2, "bg_far": 3},
        {"fg_core": 0.35, "fg_boundary": 0.20, "bg_near": 0.20, "bg_far": 0.15},
    ),
    0.10: _profile(
        "10p",
        0.10,
        {"fg_core": 128, "fg_boundary": 384, "bg_near": 384, "bg_far": 128},
        32,
        0.985,
        {"fg_core": 2, "fg_boundary": 2, "bg_near": 2, "bg_far": 3},
        {"fg_core": 0.35, "fg_boundary": 0.25, "bg_near": 0.25, "bg_far": 0.20},
    ),
    0.20: _profile(
        "20p_fair",
        0.20,
        {"fg_core": 96, "fg_boundary": 256, "bg_near": 256, "bg_far": 96},
        32,
        0.99,
        {"fg_core": 1, "fg_boundary": 2, "bg_near": 2, "bg_far": 2},
        {"fg_core": 0.40, "fg_boundary": 0.30, "bg_near": 0.30, "bg_far": 0.25},
    ),
}

_PERFORMANCE_20P = _profile(
    "20p_performance",
    0.20,
    {"fg_core": 128, "fg_boundary": 384, "bg_near": 384, "bg_far": 128},
    48,
    0.99,
    {"fg_core": 1, "fg_boundary": 2, "bg_near": 2, "bg_far": 2},
    {"fg_core": 0.40, "fg_boundary": 0.30, "bg_near": 0.30, "bg_far": 0.25},
    {"fg_core": 12288, "fg_boundary": 24576, "bg_near": 24576, "bg_far": 12288},
)


def _current_split(config: Any) -> Optional[float]:
    value = getattr(config, "cbm_labeled_split", None)
    if value is not None:
        return float(value)
    configured = getattr(config, "data_split", None)
    if isinstance(configured, (list, tuple)) and len(configured) == 1:
        return float(configured[0])
    return None


def _match_split(split: Optional[float]) -> float:
    if split is None:
        raise ValueError(
            "cbm_labeled_memory_profile='auto' requires config.cbm_labeled_split or a single data_split value"
        )
    for supported in _AUTO_PROFILES:
        if abs(float(split) - supported) < 1e-8:
            return supported
    raise ValueError(f"Unsupported labeled split {split}; expected one of {sorted(_AUTO_PROFILES)}")


def _region_dict(value: Any, name: str, cast) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping keyed by {REGIONS}")
    missing = [region for region in REGIONS if region not in value]
    if missing:
        raise ValueError(f"{name} is missing regions: {missing}")
    return {region: cast(value[region]) for region in REGIONS}


def _manual_profile(config: Any, split: Optional[float]) -> LabeledMemorySelectionConfig:
    return LabeledMemorySelectionConfig(
        profile_name="manual",
        split=split,
        sample_per_image=_region_dict(
            getattr(config, "cbm_memory_sample_per_image_labeled"),
            "cbm_memory_sample_per_image_labeled",
            int,
        ),
        max_sizes=_region_dict(getattr(config, "cbm_memory_max_sizes"), "cbm_memory_max_sizes", int),
        top_img_k=int(getattr(config, "cbm_top_img_k", 8)),
        grid_size=int(getattr(config, "cbm_memory_grid_size", 4)),
        min_tokens_per_component=_region_dict(
            getattr(config, "cbm_memory_min_tokens_per_component"),
            "cbm_memory_min_tokens_per_component",
            int,
        ),
        min_spatial_dist=_region_dict(
            getattr(config, "cbm_memory_min_spatial_dist"), "cbm_memory_min_spatial_dist", float
        ),
        grid_quota_ratio=_region_dict(
            getattr(config, "cbm_memory_grid_quota_ratio"), "cbm_memory_grid_quota_ratio", float
        ),
        max_feature_sim=float(getattr(config, "cbm_memory_max_feature_sim", 0.98)),
        relaxed_min_spatial_dist=float(getattr(config, "cbm_memory_relaxed_min_spatial_dist", 1.0)),
        relaxed_max_feature_sim=float(getattr(config, "cbm_memory_relaxed_max_feature_sim", 0.995)),
        allow_underfill=bool(getattr(config, "cbm_memory_allow_underfill", True)),
        use_component_quota=bool(getattr(config, "cbm_memory_use_component_quota", True)),
        use_grid_quota=bool(getattr(config, "cbm_memory_use_grid_quota", True)),
        use_spatial_diversity=bool(getattr(config, "cbm_memory_use_spatial_diversity", True)),
        use_feature_diversity=bool(getattr(config, "cbm_memory_use_feature_diversity", True)),
        relax_diversity_if_underfilled=bool(
            getattr(config, "cbm_memory_relax_diversity_if_underfilled", True)
        ),
        global_fill_max_per_image=getattr(config, "cbm_memory_global_fill_max_per_image", None),
    )


def _apply_overrides(
    profile: LabeledMemorySelectionConfig, overrides: Any
) -> LabeledMemorySelectionConfig:
    if not overrides:
        return profile
    if not isinstance(overrides, Mapping):
        raise TypeError("cbm_labeled_memory_profile_overrides must be a mapping")
    allowed = set(profile.__dataclass_fields__) - {"profile_name", "split"}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown labeled memory profile overrides: {unknown}")
    values = dict(overrides)
    for name in (
        "sample_per_image",
        "max_sizes",
        "min_tokens_per_component",
        "min_spatial_dist",
        "grid_quota_ratio",
    ):
        if name in values:
            cast = float if name in ("min_spatial_dist", "grid_quota_ratio") else int
            values[name] = _region_dict(values[name], name, cast)
    if values.get("global_fill_max_per_image") is not None:
        values["global_fill_max_per_image"] = _region_dict(
            values["global_fill_max_per_image"], "global_fill_max_per_image", int
        )
    return replace(profile, **values)


def _apply_common_config(
    profile: LabeledMemorySelectionConfig, config: Any
) -> LabeledMemorySelectionConfig:
    global_cap = getattr(config, "cbm_memory_global_fill_max_per_image", None)
    if global_cap is not None:
        global_cap = _region_dict(global_cap, "cbm_memory_global_fill_max_per_image", int)
    return replace(
        profile,
        grid_size=int(getattr(config, "cbm_memory_grid_size", profile.grid_size)),
        min_tokens_per_component=_region_dict(
            getattr(config, "cbm_memory_min_tokens_per_component", profile.min_tokens_per_component),
            "cbm_memory_min_tokens_per_component",
            int,
        ),
        relaxed_min_spatial_dist=float(
            getattr(config, "cbm_memory_relaxed_min_spatial_dist", profile.relaxed_min_spatial_dist)
        ),
        relaxed_max_feature_sim=float(
            getattr(config, "cbm_memory_relaxed_max_feature_sim", profile.relaxed_max_feature_sim)
        ),
        allow_underfill=bool(getattr(config, "cbm_memory_allow_underfill", profile.allow_underfill)),
        use_component_quota=bool(
            getattr(config, "cbm_memory_use_component_quota", profile.use_component_quota)
        ),
        use_grid_quota=bool(getattr(config, "cbm_memory_use_grid_quota", profile.use_grid_quota)),
        use_spatial_diversity=bool(
            getattr(config, "cbm_memory_use_spatial_diversity", profile.use_spatial_diversity)
        ),
        use_feature_diversity=bool(
            getattr(config, "cbm_memory_use_feature_diversity", profile.use_feature_diversity)
        ),
        relax_diversity_if_underfilled=bool(
            getattr(
                config,
                "cbm_memory_relax_diversity_if_underfilled",
                profile.relax_diversity_if_underfilled,
            )
        ),
        global_fill_max_per_image=global_cap,
    )


def resolve_labeled_memory_profile(config: Any) -> LabeledMemorySelectionConfig:
    strategy = str(
        getattr(config, "cbm_memory_finalize_strategy", "image_balanced_reliability_diversity")
    ).strip().lower()
    if strategy != "image_balanced_reliability_diversity":
        raise ValueError(
            "cbm_memory_finalize_strategy must be 'image_balanced_reliability_diversity'"
        )
    mode = str(getattr(config, "cbm_labeled_memory_profile", "auto")).strip().lower()
    split = _current_split(config)
    if mode == "auto":
        profile = _AUTO_PROFILES[_match_split(split)]
    elif mode == "20p_performance":
        if abs(_match_split(split) - 0.20) >= 1e-8:
            raise ValueError("20p_performance profile requires labeled split 0.20")
        profile = _PERFORMANCE_20P
    elif mode == "manual":
        profile = _manual_profile(config, split)
    else:
        raise ValueError("cbm_labeled_memory_profile must be 'auto', 'manual', or '20p_performance'")
    profile = _apply_common_config(profile, config)
    return _apply_overrides(profile, getattr(config, "cbm_labeled_memory_profile_overrides", None))


__all__ = ["LabeledMemorySelectionConfig", "resolve_labeled_memory_profile"]
