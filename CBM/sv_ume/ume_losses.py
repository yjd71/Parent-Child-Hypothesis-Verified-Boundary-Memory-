from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import torch

from CBM.memory.labels import VALUE_LAYOUT


_EPS = 1.0e-6
_REGION_CHANNELS = 4
_RELIABILITY_INDEX = VALUE_LAYOUT.index("reliability")


def compute_ume_evidence_loss(
    retrieval_aux_s: Any,
    memory_t: Any,
    cfg: Any,
) -> torch.Tensor:
    """Return the weighted unlabeled-memory evidence loss."""
    zero = _graph_zero(retrieval_aux_s)
    if not _loss_enabled(cfg, "use_ume_evidence_loss"):
        return zero
    if not _has_ready_unlabeled_memory(memory_t):
        return zero

    retrieval = _as_mapping(retrieval_aux_s)
    ret_u = _as_mapping(retrieval.get("ret_u"))
    y_u = _retrieval_map(ret_u, "Y_map", "Y")
    valid_u = ret_u.get("valid_map")
    if not _valid_evidence_inputs(y_u, valid_u, require_reliability=True):
        return zero

    probs_u, mass_valid_u = _region_distribution(y_u)
    valid_mask = _valid_mask(valid_u, probs_u) & mass_valid_u

    pseudo_region_label = probs_u.detach().argmax(dim=1, keepdim=True)
    target_prob = probs_u.gather(1, pseudo_region_label).clamp_min(_EPS)
    ce_map = -target_prob.log()

    r_token = torch.nan_to_num(
        y_u[:, _RELIABILITY_INDEX : _RELIABILITY_INDEX + 1].float(),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    raw_loss = _masked_mean(r_token * ce_map, valid_mask)
    weighted_loss = _loss_weight(cfg, "lambda_ume_evi", 0.05) * raw_loss
    return _finite_loss(weighted_loss)


def compute_source_consistency_loss(
    retrieval_aux_s: Any,
    memory_t: Any,
    cfg: Any,
) -> torch.Tensor:
    """Return weighted JS consistency on confident, agreeing source evidence."""
    zero = _graph_zero(retrieval_aux_s)
    if not _loss_enabled(cfg, "use_source_consistency_loss"):
        return zero
    if not _has_ready_unlabeled_memory(memory_t):
        return zero

    retrieval = _as_mapping(retrieval_aux_s)
    ret_l = _as_mapping(retrieval.get("ret_l"))
    ret_u = _as_mapping(retrieval.get("ret_u"))
    y_l = _retrieval_map(ret_l, "Y_map", "Y")
    y_u = _retrieval_map(ret_u, "Y_map", "Y")
    valid_l = ret_l.get("valid_map")
    valid_u = ret_u.get("valid_map")
    if not _valid_source_inputs(y_l, valid_l, y_u, valid_u):
        return zero

    probs_l, mass_valid_l = _region_distribution(y_l)
    probs_u, mass_valid_u = _region_distribution(y_u)
    valid_mask = (
        _valid_mask(valid_l, probs_l)
        & _valid_mask(valid_u, probs_u)
        & mass_valid_l
        & mass_valid_u
    )

    confidence_l, region_l = probs_l.detach().max(dim=1, keepdim=True)
    confidence_u, region_u = probs_u.detach().max(dim=1, keepdim=True)
    tau = _confidence_threshold(cfg)
    agreement_mask = (
        valid_mask
        & (confidence_l > tau)
        & (confidence_u > tau)
        & (region_l == region_u)
    )

    midpoint = 0.5 * (probs_l + probs_u)
    log_l = probs_l.clamp_min(_EPS).log()
    log_u = probs_u.clamp_min(_EPS).log()
    log_midpoint = midpoint.clamp_min(_EPS).log()
    js_map = 0.5 * (
        (probs_l * (log_l - log_midpoint)).sum(dim=1, keepdim=True)
        + (probs_u * (log_u - log_midpoint)).sum(dim=1, keepdim=True)
    )
    js_map = torch.nan_to_num(js_map, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    raw_loss = _masked_mean(js_map, agreement_mask)
    weighted_loss = _loss_weight(cfg, "lambda_source_cons", 0.02) * raw_loss
    return _finite_loss(weighted_loss)


def compute_total_sv_ume_loss(
    *,
    aux_s: Any = None,
    retrieval_aux_s: Any = None,
    memory_t: Any = None,
    cfg: Any = None,
) -> Dict[str, torch.Tensor]:
    """Compute optional SV-UME losses from student retrieval evidence."""
    retrieval = _resolve_retrieval_aux(aux_s, retrieval_aux_s)
    zero = _graph_zero(retrieval, aux_s)
    if not bool(_cfg_get(cfg, "use_sv_ume", False)):
        return {
            "loss_ume_evi": zero,
            "loss_source_cons": zero,
            "loss_sv_ume": zero,
        }

    loss_ume_evi = compute_ume_evidence_loss(retrieval, memory_t, cfg)
    loss_source_cons = compute_source_consistency_loss(retrieval, memory_t, cfg)
    loss_sv_ume = _finite_loss(loss_ume_evi + loss_source_cons)
    return {
        "loss_ume_evi": loss_ume_evi,
        "loss_source_cons": loss_source_cons,
        "loss_sv_ume": loss_sv_ume,
    }


def _resolve_retrieval_aux(aux_s: Any, retrieval_aux_s: Any) -> Mapping[str, Any]:
    if isinstance(retrieval_aux_s, Mapping):
        return retrieval_aux_s
    aux = _as_mapping(aux_s)
    retrieval = aux.get("retrieval")
    if isinstance(retrieval, Mapping):
        return retrieval
    return aux


def _region_distribution(evidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    region_evidence = torch.nan_to_num(
        evidence[:, :_REGION_CHANNELS].float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    evidence_mass = region_evidence.sum(dim=1, keepdim=True)
    probabilities = region_evidence / evidence_mass.clamp_min(_EPS)
    return probabilities, evidence_mass > _EPS


def _valid_mask(valid_map: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    valid = torch.nan_to_num(
        valid_map.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return valid.to(device=reference.device) > 0.5


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(device=value.device, dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _valid_evidence_inputs(
    evidence: Any,
    valid_map: Any,
    *,
    require_reliability: bool,
) -> bool:
    if not torch.is_tensor(evidence) or not torch.is_tensor(valid_map):
        return False
    required_channels = _RELIABILITY_INDEX + 1 if require_reliability else _REGION_CHANNELS
    return bool(
        evidence.dim() == 4
        and evidence.size(1) >= required_channels
        and valid_map.dim() == 4
        and valid_map.size(1) == 1
        and evidence.size(0) == valid_map.size(0)
        and tuple(evidence.shape[-2:]) == tuple(valid_map.shape[-2:])
        and evidence.device == valid_map.device
    )


def _valid_source_inputs(
    y_l: Any,
    valid_l: Any,
    y_u: Any,
    valid_u: Any,
) -> bool:
    if not _valid_evidence_inputs(y_l, valid_l, require_reliability=False):
        return False
    if not _valid_evidence_inputs(y_u, valid_u, require_reliability=False):
        return False
    return bool(y_l.shape == y_u.shape and y_l.device == y_u.device)


def _retrieval_map(retrieval: Mapping[str, Any], *aliases: str) -> Optional[torch.Tensor]:
    for name in aliases:
        value = retrieval.get(name)
        if torch.is_tensor(value):
            return value
    return None


def _has_ready_unlabeled_memory(memory_t: Any) -> bool:
    if not isinstance(memory_t, Mapping):
        return False
    memory = memory_t.get("unlabeled_memory")
    if memory is None:
        memory = memory_t.get("U_prev")
    if memory is None:
        return False
    is_ready = getattr(memory, "is_ready", None)
    return bool(is_ready()) if callable(is_ready) else True


def _loss_enabled(cfg: Any, flag: str) -> bool:
    return bool(
        _cfg_get(cfg, "use_sv_ume", False)
        and _cfg_get(cfg, flag, False)
    )


def _loss_weight(cfg: Any, name: str, default: float) -> float:
    value = float(_cfg_get(cfg, name, default))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _confidence_threshold(cfg: Any) -> float:
    value = float(_cfg_get(cfg, "source_consistency_tau", 0.70))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("source_consistency_tau must be finite and in [0, 1]")
    return value


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default) if cfg is not None else default


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _graph_zero(*values: Any) -> torch.Tensor:
    for value in values:
        reference = _find_floating_tensor(value)
        if reference is not None:
            return reference.float().reshape(-1)[:1].sum() * 0.0
    return torch.zeros((), dtype=torch.float32)


def _find_floating_tensor(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value if value.is_floating_point() else None
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _find_floating_tensor(item)
            if tensor is not None:
                return tensor
    elif isinstance(value, (tuple, list)):
        for item in value:
            tensor = _find_floating_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _finite_loss(loss: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(loss.float(), nan=0.0, posinf=0.0, neginf=0.0)


__all__ = [
    "compute_ume_evidence_loss",
    "compute_source_consistency_loss",
    "compute_total_sv_ume_loss",
]
