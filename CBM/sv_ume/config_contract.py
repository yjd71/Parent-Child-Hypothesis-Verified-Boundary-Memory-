from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _matches(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return (
            isinstance(actual, Sequence)
            and not isinstance(actual, (str, bytes))
            and len(actual) == len(expected)
            and all(_matches(left, right) for left, right in zip(actual, expected))
        )
    if isinstance(expected, float):
        if isinstance(actual, bool):
            return False
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-8)
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_sv_ume_profile_contract(cfg) -> Dict[str, Any]:
    """Validate an optional run-level SV-UME effective-config contract."""
    contract = getattr(cfg, "sv_ume_profile_contract", None)
    if contract is None:
        return {}
    if not isinstance(contract, Mapping) or not contract:
        raise TypeError("sv_ume_profile_contract must be a non-empty mapping")
    profile_name = getattr(cfg, "sv_ume_profile_name", None)
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise ValueError("sv_ume_profile_name must be a non-empty string")

    mismatches = []
    effective = {}
    for name, expected in contract.items():
        if not isinstance(name, str) or not name:
            raise TypeError("sv_ume_profile_contract keys must be non-empty strings")
        if not hasattr(cfg, name):
            mismatches.append(f"{name}: missing (expected {expected!r})")
            continue
        actual = getattr(cfg, name)
        effective[name] = actual
        if not _matches(actual, expected):
            mismatches.append(f"{name}: actual={actual!r}, expected={expected!r}")
    if mismatches:
        source = getattr(cfg, "run_cfg_path", "<unknown>")
        digest = getattr(cfg, "run_cfg_sha256", "<unknown>")
        raise ValueError(
            f"SV-UME profile {profile_name!r} mismatch; config={source}; "
            f"sha256={digest}; " + "; ".join(mismatches)
        )
    return effective


__all__ = ["validate_sv_ume_profile_contract"]
