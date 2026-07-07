"""Core PC-HBM orchestration and configuration."""

from .pc_config import (
    apply_pc_hbm_defaults,
    pc_hbm_enabled,
    pc_hbm_should_rebuild_memory,
    pc_hbm_stage,
    pc_hbm_unlabeled_enabled,
)


def build_pc_hbm(*args, **kwargs):
    """Lazy factory to avoid import cycles through models.build_model."""

    from .engine import build_pc_hbm as _build_pc_hbm

    return _build_pc_hbm(*args, **kwargs)


def __getattr__(name):
    if name == "PCHBMEngine":
        from .engine import PCHBMEngine

        return PCHBMEngine
    raise AttributeError(name)


__all__ = [
    "PCHBMEngine",
    "apply_pc_hbm_defaults",
    "build_pc_hbm",
    "pc_hbm_enabled",
    "pc_hbm_should_rebuild_memory",
    "pc_hbm_stage",
    "pc_hbm_unlabeled_enabled",
]
