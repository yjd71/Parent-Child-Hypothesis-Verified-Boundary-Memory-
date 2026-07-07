"""Diagnostics aggregation for PC-HBM forward/loss outputs."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


DIAGNOSTIC_KEYS = (
    "parent_top1_region_acc",
    "parent_topk_region_acc",
    "parent_entropy",
    "route_entropy",
    "child_verify_auc",
    "child_score_fg_boundary_mean",
    "child_score_bg_near_mean",
    "geo_sdf_l1",
    "geo_offset_l1",
    "C23_mean",
    "C23_boundary_mean",
    "gate_pc_mean",
    "gate_pc_on_error",
    "gate_pc_on_correct",
    "pi_keep_mean",
    "pi_res_mean",
    "pi_def_mean",
    "pi_sup_mean",
    "pi_res_on_FN",
    "pi_sup_on_FP",
    "pi_def_on_misalign",
    "branch_oracle_keep_ratio",
    "branch_oracle_res_ratio",
    "branch_oracle_def_ratio",
    "branch_oracle_sup_ratio",
    "z_main_loss",
    "z_final_loss",
    "pseudo_conf_mean",
    "pseudo_conf_boundary_mean",
)


def collect_pc_hbm_diagnostics(aux: Dict[str, Any], gt: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    """Collect scalar diagnostics from PC-HBM aux outputs."""

    ref = _ref(aux)
    diag = {key: ref.new_zeros(()) for key in DIAGNOSTIC_KEYS}
    pc = aux.get("pc_hbm", {}) or {}
    mix = aux.get("mixture", {}) or {}
    diag["parent_entropy"] = _mean(pc.get("parent_entropy"), ref)
    diag["route_entropy"] = _mean(pc.get("route_entropy"), ref)
    diag["C23_mean"] = _mean(pc.get("C23_map"), ref)
    c23 = pc.get("C23_map")
    b3 = pc.get("B3")
    if torch.is_tensor(c23) and torch.is_tensor(b3):
        diag["C23_boundary_mean"] = _masked_mean(c23, b3, ref)
    gate = pc.get("gate_pc_map")
    diag["gate_pc_mean"] = _mean(gate, ref)
    pi = mix.get("pi")
    if torch.is_tensor(pi) and pi.size(1) >= 4:
        diag["pi_keep_mean"] = pi[:, 0].mean()
        diag["pi_res_mean"] = pi[:, 1].mean()
        diag["pi_def_mean"] = pi[:, 2].mean()
        diag["pi_sup_mean"] = pi[:, 3].mean()
        diag["branch_oracle_keep_ratio"] = (pi.argmax(dim=1) == 0).float().mean()
        diag["branch_oracle_res_ratio"] = (pi.argmax(dim=1) == 1).float().mean()
        diag["branch_oracle_def_ratio"] = (pi.argmax(dim=1) == 2).float().mean()
        diag["branch_oracle_sup_ratio"] = (pi.argmax(dim=1) == 3).float().mean()
    s_child = pc.get("S_child")
    values = pc.get("top_parent_values")
    if torch.is_tensor(s_child) and torch.is_tensor(values) and s_child.numel() > 0:
        support = values[..., 1].clamp(0.0, 1.0)
        diag["child_verify_auc"] = _approx_auc(torch.sigmoid(s_child).reshape(-1), support.reshape(-1), ref)
        diag["child_score_fg_boundary_mean"] = _masked_mean(torch.sigmoid(s_child), support, ref)
        bg_near = values[..., 2].clamp(0.0, 1.0)
        diag["child_score_bg_near_mean"] = _masked_mean(torch.sigmoid(s_child), bg_near, ref)
    g_attn = pc.get("G_attn")
    g_child = pc.get("G_child_attn")
    if torch.is_tensor(g_attn) and torch.is_tensor(g_child) and g_attn.numel() > 0:
        diag["geo_sdf_l1"] = (g_attn[:, :1] - g_child[:, :1]).abs().mean()
        diag["geo_offset_l1"] = (g_attn[:, 3:5] - g_child[:, 3:5]).abs().mean()
    if gt is not None:
        z_main = aux.get("z_main")
        z_final = aux.get("z_final")
        if torch.is_tensor(z_main):
            diag["z_main_loss"] = F.binary_cross_entropy_with_logits(z_main, _resize_gt(gt, z_main))
        if torch.is_tensor(z_final):
            diag["z_final_loss"] = F.binary_cross_entropy_with_logits(z_final, _resize_gt(gt, z_final))
        if torch.is_tensor(gate) and torch.is_tensor(z_main):
            pred = torch.sigmoid(z_main.detach())
            target = _resize_gt(gt, z_main)
            err = (pred - target).abs()
            err_gate = F.interpolate(gate, size=err.shape[-2:], mode="bilinear", align_corners=False)
            diag["gate_pc_on_error"] = _masked_mean(err_gate, (err > 0.3).float(), ref)
            diag["gate_pc_on_correct"] = _masked_mean(err_gate, (err <= 0.3).float(), ref)
        if torch.is_tensor(pi) and torch.is_tensor(z_main):
            target = _resize_gt(gt, z_main)
            pred = torch.sigmoid(z_main.detach())
            fn = ((target > 0.5) & (pred < 0.4)).float()
            fp = ((target < 0.5) & (pred > 0.6)).float()
            edge = _boundary(target)
            diag["pi_res_on_FN"] = _masked_mean(pi[:, 1:2], fn, ref)
            diag["pi_sup_on_FP"] = _masked_mean(pi[:, 3:4], fp, ref)
            diag["pi_def_on_misalign"] = _masked_mean(pi[:, 2:3], edge, ref)
    p_final = aux.get("p_final")
    if torch.is_tensor(p_final):
        conf = (2.0 * (p_final - 0.5).abs()).clamp(0.0, 1.0)
        diag["pseudo_conf_mean"] = conf.mean()
        diag["pseudo_conf_boundary_mean"] = _masked_mean(conf, _boundary(p_final), ref)
    return diag


def _ref(aux):
    for value in aux.values():
        if torch.is_tensor(value):
            return value
    pc = aux.get("pc_hbm", {}) or {}
    for value in pc.values():
        if torch.is_tensor(value):
            return value
    return torch.tensor(0.0)


def _mean(value, ref):
    if torch.is_tensor(value) and value.numel() > 0:
        return value.mean()
    return ref.new_zeros(())


def _masked_mean(value, mask, ref):
    if not torch.is_tensor(value) or not torch.is_tensor(mask) or value.numel() == 0:
        return ref.new_zeros(())
    if mask.shape != value.shape:
        mask = F.interpolate(mask.float(), size=value.shape[-2:], mode="nearest") if mask.dim() == 4 else mask.float()
        if mask.shape != value.shape:
            mask = mask.expand_as(value)
    mask = mask.to(device=value.device, dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _approx_auc(score, label, ref):
    pos = label > 0.5
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return ref.new_zeros(())
    return (score[pos].mean() - score[neg].mean()).sigmoid()


def _resize_gt(gt, logits):
    return F.interpolate(gt.float(), size=logits.shape[-2:], mode="nearest")


def _boundary(prob):
    dil = F.max_pool2d(prob.float(), 3, stride=1, padding=1)
    ero = -F.max_pool2d(-prob.float(), 3, stride=1, padding=1)
    return (dil - ero).clamp(0.0, 1.0)
