from __future__ import annotations

from typing import Any, Optional


def _as_epoch(epoch: int) -> int:
    try:
        return int(epoch)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"epoch must be an integer-compatible value, got {epoch!r}") from exc


def sv_ume_enabled(cfg: Any) -> bool:
    """Return whether SV-UME is explicitly enabled."""
    return bool(getattr(cfg, "use_sv_ume", False))


def should_build_after_epoch(cfg: Any, epoch: int) -> bool:
    """Return whether epoch ``t`` should build the frozen candidate memory ``U_t``."""
    if not sv_ume_enabled(cfg):
        return False
    if not bool(getattr(cfg, "use_lagged_unlabeled_memory", True)):
        return False
    if bool(getattr(cfg, "use_unlabeled_memory_during_current_epoch", False)):
        return False
    if not bool(getattr(cfg, "build_unlabeled_memory_after_epoch", True)):
        return False
    return _as_epoch(epoch) >= int(getattr(cfg, "sv_ume_start_epoch", 21))


def expected_unlabeled_source_epoch(cfg: Any, epoch: int) -> Optional[int]:
    """Return the only U-memory epoch that may be used during ``epoch``."""
    if not sv_ume_enabled(cfg):
        return None
    if not bool(getattr(cfg, "use_lagged_unlabeled_memory", True)):
        return None
    if bool(getattr(cfg, "use_unlabeled_memory_during_current_epoch", False)):
        return None
    current_epoch = _as_epoch(epoch)
    start_epoch = int(getattr(cfg, "sv_ume_start_epoch", 21))
    if current_epoch <= start_epoch:
        return None
    return current_epoch - 1


def can_use_lagged_memory(cfg: Any, epoch: int, source_epoch: Optional[int]) -> bool:
    """Check that a frozen U-memory comes from exactly the previous epoch."""
    expected_epoch = expected_unlabeled_source_epoch(cfg, epoch)
    if expected_epoch is None or source_epoch is None:
        return False
    try:
        return int(source_epoch) == expected_epoch
    except (TypeError, ValueError):
        return False


__all__ = [
    "sv_ume_enabled",
    "should_build_after_epoch",
    "expected_unlabeled_source_epoch",
    "can_use_lagged_memory",
]
