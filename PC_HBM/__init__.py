"""PC-HBM package entry points.

The package implements Parent-Child Hypothesis-Verified Boundary Memory as an
optional, gated extension for TALNet.  Production forward paths never fabricate
memory contents; if memory is empty or disabled the caller receives baseline
TALNet outputs with a fallback reason in the auxiliary dictionary.
"""

from .pc_config import apply_pc_hbm_defaults, pc_hbm_enabled
from .pc_memory import PCHBMMemory

def build_pc_hbm(*args, **kwargs):
    """Lazy factory to avoid import cycles with models.build_model."""

    from .engine import build_pc_hbm as _build_pc_hbm

    return _build_pc_hbm(*args, **kwargs)


def __getattr__(name):
    if name == "PCHBMEngine":
        from .engine import PCHBMEngine

        return PCHBMEngine
    raise AttributeError(name)


__all__ = [
    "PCHBMEngine",
    "PCHBMMemory",
    "apply_pc_hbm_defaults",
    "build_pc_hbm",
    "pc_hbm_enabled",
]
