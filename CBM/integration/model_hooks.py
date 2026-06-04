from __future__ import annotations


def apply_p3_hook(cbm, *, x, x3, p3, m3, training=False):
    if cbm is None:
        return p3, None
    return cbm.apply_p3_hook(x=x, x3=x3, p3=p3, m3=m3, training=training)


def apply_final_fusion(cbm, p1_out, aux):
    if cbm is None:
        return p1_out
    return cbm.apply_final_fusion(p1_out, aux)
