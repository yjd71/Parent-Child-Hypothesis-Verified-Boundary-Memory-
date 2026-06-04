from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CBMState:
    epoch: Optional[int] = None
    stage_epoch: Optional[int] = None
    stage_name: str = "unknown"
    memory_ready: bool = False
    memory_build_failed: bool = False
    memory_build_error: Optional[str] = None
    last_aux: Optional[Dict[str, Any]] = None
    loss_dict: Dict[str, float] = field(default_factory=dict)
