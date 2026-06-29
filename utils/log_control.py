from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEFAULT_LOG_INTERVAL = 200


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def log_enabled(config: Any) -> bool:
    return bool(_config_value(config, "log_enable", True))


def log_interval(config: Any) -> int:
    try:
        return max(1, int(_config_value(config, "log_interval", DEFAULT_LOG_INTERVAL)))
    except (TypeError, ValueError):
        return DEFAULT_LOG_INTERVAL


def should_log(config: Any, step: Any = None) -> bool:
    if not log_enabled(config):
        return False
    if step is None:
        return True
    try:
        return int(step) % log_interval(config) == 0
    except (TypeError, ValueError):
        return True
