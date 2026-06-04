from __future__ import annotations

from .config.defaults import apply_cbm_defaults
from .engine import CBMPFIEngine


def build_cbm_pfi(config, device=None, logger=None) -> CBMPFIEngine:
    apply_cbm_defaults(config)
    return CBMPFIEngine(config=config, device=device, logger=logger)


__all__ = ["apply_cbm_defaults", "build_cbm_pfi"]
