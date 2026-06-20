from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from runpy import run_path
from typing import Any

_CBM_BASE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "base" / "cbm.py"
CBM_DEFAULTS = dict(run_path(str(_CBM_BASE_CONFIG))["CBM_DEFAULTS"])


def apply_cbm_defaults(config: Any) -> Any:
    """Fill missing CBM config fields without overwriting user-provided values."""
    if isinstance(config, MutableMapping):
        for key, value in CBM_DEFAULTS.items():
            config.setdefault(key, value)
        return config

    for key, value in CBM_DEFAULTS.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    return config
