from __future__ import annotations


def format_cbm_diagnostics(aux, memory=None) -> str:
    aux = aux or {}
    if not aux.get("cbm_used", False):
        return f"[CBM] disabled fallback={aux.get('fallback_reason', 'unknown')}"
    memory_text = memory.diagnostic_string() if memory is not None and hasattr(memory, "diagnostic_string") else ""
    return (
        "[CBM] used "
        f"tokens={aux.get('num_memory_tokens', 0)}, "
        f"valid={aux.get('num_valid_boundary_tokens', 0)}, "
        f"gate={aux.get('gate_mean', 0.0):.4f}, "
        f"valid_ratio={aux.get('valid_ratio', 0.0):.4f}, "
        f"retrieval_uncertainty={aux.get('u_mean', 0.0):.4f}, "
        f"cons={aux.get('cons_mean', 0.0):.4f} "
        f"{memory_text}"
    ).strip()
