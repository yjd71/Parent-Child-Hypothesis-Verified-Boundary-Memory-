from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List

RETRIEVAL_AUX_KEYS = (
    "Y_map",
    "Y_ctx",
    "R_map",
    "R_ctx",
    "U_map",
    "valid_map",
    "cons_map",
    "gate3",
    "B3",
    "prob3",
    "p_main",
    "p_final",
    "B_query",
    "boundary_mask",
    "z_mem3",
    "top_img_ids",
    "img_scores",
    "ret_l",
    "ret_u",
    "w_l",
    "w_u",
    "w_l_map",
    "w_u_map",
    "score_l",
    "score_u",
    "used_unlabeled_memory",
    "source_entropy",
)

REQUIRED_CBM_EVIDENCE_KEYS = (
    "Y_map",
    "Y_ctx",
    "R_map",
    "R_ctx",
    "U_map",
    "valid_map",
    "cons_map",
    "gate3",
    "B3",
    "prob3",
)


def build_retrieval_aux_from_cbm_aux(aux_t: Any) -> Dict[str, Any]:
    """Normalize CBM-PFI aux evidence into the SVB-PLR retrieval_aux schema.

    Shape:
        aux_t: mapping-like CBM aux dictionary from teacher forward.
        return: dict with fixed SVB-PLR retrieval evidence keys.
    """
    aux = _as_mapping(aux_t)
    retrieval = _as_mapping(aux.get("retrieval"))

    retrieval_aux = {
        "Y_map": _pick_value(retrieval, aux, "Y_map"),
        "Y_ctx": _pick_value(retrieval, aux, "Y_ctx"),
        "R_map": _pick_value(retrieval, aux, "R_map"),
        "R_ctx": _pick_value(retrieval, aux, "R_ctx"),
        "U_map": _pick_value(retrieval, aux, "U_map"),
        "valid_map": _pick_value(retrieval, aux, "valid_map"),
        "cons_map": _pick_value(retrieval, aux, "cons_map"),
        "gate3": _pick_value(retrieval, aux, "gate3"),
        "B3": _pick_b3(retrieval, aux),
        "prob3": _pick_value(retrieval, aux, "prob3"),
        "p_main": _pick_value(retrieval, aux, "p_main"),
        "p_final": _pick_p_final(retrieval, aux),
    }
    for key in (
        "B_query",
        "boundary_mask",
        "z_mem3",
        "top_img_ids",
        "img_scores",
        "ret_l",
        "ret_u",
        "w_l",
        "w_u",
        "w_l_map",
        "w_u_map",
        "score_l",
        "score_u",
        "used_unlabeled_memory",
        "source_entropy",
    ):
        retrieval_aux[key] = _pick_value(retrieval, aux, key)
    validate_retrieval_aux(retrieval_aux)
    return retrieval_aux


def validate_retrieval_aux(retrieval_aux: Any) -> Dict[str, Any]:
    """Validate required CBM evidence keys for SVB-PLR.

    Shape:
        retrieval_aux: mapping-like object returned by build_retrieval_aux_from_cbm_aux.
        return: {"valid": bool, "missing_keys": List[str], "present_keys": List[str]}.
    """
    aux = _as_mapping(retrieval_aux)
    present_keys = [key for key in REQUIRED_CBM_EVIDENCE_KEYS if aux.get(key) is not None]
    missing_keys = [key for key in REQUIRED_CBM_EVIDENCE_KEYS if aux.get(key) is None]

    return {
        "valid": not missing_keys,
        "missing_keys": missing_keys,
        "present_keys": present_keys,
    }


def has_valid_cbm_evidence(retrieval_aux: Any) -> bool:
    """Return whether retrieval_aux contains all required CBM evidence.

    Shape:
        retrieval_aux: mapping-like object.
        return: bool.
    """
    return bool(validate_retrieval_aux(retrieval_aux)["valid"])


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _pick_value(primary: Mapping[str, Any], secondary: Mapping[str, Any], key: str) -> Any:
    value = primary.get(key)
    if value is not None:
        return value
    value = secondary.get(key)
    if value is not None:
        return value
    return None


def _pick_b3(retrieval: Mapping[str, Any], aux: Mapping[str, Any]) -> Any:
    for key in ("B3",):
        value = retrieval.get(key)
        if value is not None:
            return value
    for key in ("B3", "B_query"):
        value = aux.get(key)
        if value is not None:
            return value
    return None


def _pick_p_final(retrieval: Mapping[str, Any], aux: Mapping[str, Any]) -> Any:
    value = aux.get("p_final")
    if value is not None:
        return value
    value = retrieval.get("p_final")
    if value is not None:
        return value
    return None


__all__ = [
    "RETRIEVAL_AUX_KEYS",
    "REQUIRED_CBM_EVIDENCE_KEYS",
    "build_retrieval_aux_from_cbm_aux",
    "validate_retrieval_aux",
    "has_valid_cbm_evidence",
]
