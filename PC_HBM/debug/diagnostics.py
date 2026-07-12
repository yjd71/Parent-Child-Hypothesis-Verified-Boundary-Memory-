"""Diagnostics aggregation for PC-HBM forward/loss outputs."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from ..training.pc_supervision import (
    REGION_BG_NEAR,
    REGION_FG_BOUNDARY,
    build_geometry_target,
    build_need_correction_map,
    build_region_label_map,
    gather_by_boundary_indices,
)


DIAGNOSTIC_KEYS = (
    "parent_top1_region_acc",
    "parent_topk_region_acc",
    "parent_entropy",
    "route_entropy",
    "route_entropy_norm",
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
    "pseudo_certainty_mean",
    "pseudo_certainty_boundary_mean",
)


def collect_pc_hbm_diagnostics(aux: Dict[str, Any], gt: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    """Collect scalar diagnostics from PC-HBM aux outputs."""

    ref = _ref(aux)
    diag = {key: ref.new_zeros(()) for key in DIAGNOSTIC_KEYS}
    pc = aux.get("pc_hbm", {}) or {}
    mix = aux.get("mixture", {}) or {}
    diag["parent_entropy"] = _mean(pc.get("parent_entropy"), ref)
    diag["route_entropy"] = _mean(pc.get("route_entropy"), ref)
    diag["route_entropy_norm"] = _mean(pc.get("route_entropy_norm"), ref)
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
    parent_region = pc.get("top_parent_region_ids")
    region_label3 = None
    if gt is not None and torch.is_tensor(parent_region) and parent_region.numel() > 0:
        boundary = pc.get("boundary_indices3")
        if isinstance(boundary, dict):
            region_label_map = build_region_label_map(gt.to(device=parent_region.device), _pc_size(pc))
            region_label3 = gather_by_boundary_indices(region_label_map, boundary).long().clamp(0, 3)
            valid_parent = parent_region.ge(0)
            diag["parent_top1_region_acc"] = ((parent_region[:, 0] == region_label3) & valid_parent[:, 0]).float().mean()
            diag["parent_topk_region_acc"] = ((parent_region == region_label3[:, None]) & valid_parent).any(dim=1).float().mean()
    if torch.is_tensor(s_child) and torch.is_tensor(parent_region) and s_child.numel() > 0:
        score = torch.sigmoid(s_child)
        if region_label3 is not None:
            support = (parent_region == region_label3[:, None]).to(dtype=score.dtype)
            valid = parent_region.ge(0)
            diag["child_verify_auc"] = _approx_auc(score[valid].reshape(-1), support[valid].reshape(-1), ref) if valid.any() else ref.new_zeros(())
        diag["child_score_fg_boundary_mean"] = _masked_mean(score, (parent_region == REGION_FG_BOUNDARY).float(), ref)
        diag["child_score_bg_near_mean"] = _masked_mean(score, (parent_region == REGION_BG_NEAR).float(), ref)
    g_attn = pc.get("G_attn")
    g_child = pc.get("G_child_attn")
    if torch.is_tensor(g_attn) and g_attn.numel() > 0:
        boundary = pc.get("boundary_indices3")
        if gt is not None and isinstance(boundary, dict):
            geo = build_geometry_target(gt.to(device=g_attn.device), _pc_size(pc))
            gt_sdf = gather_by_boundary_indices(geo["sdf"], boundary)
            gt_offset = gather_by_boundary_indices(geo["offset"], boundary)
            diag["geo_sdf_l1"] = (g_attn[:, :1] - gt_sdf).abs().mean()
            o_pc = pc.get("O_pc_token")
            if torch.is_tensor(o_pc) and o_pc.numel() > 0:
                diag["geo_offset_l1"] = (o_pc - gt_offset).abs().mean()
        elif torch.is_tensor(g_child) and g_child.numel() > 0:
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
            need = build_need_correction_map(z_main.detach(), gt.to(device=z_main.device), gate.shape[-2:], threshold=0.25)
            diag["gate_pc_on_error"] = _masked_mean(gate, need, ref)
            diag["gate_pc_on_correct"] = _masked_mean(gate, 1.0 - need, ref)
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
        certainty = (2.0 * (p_final - 0.5).abs()).clamp(0.0, 1.0)
        diag["pseudo_certainty_mean"] = certainty.mean()
        diag["pseudo_certainty_boundary_mean"] = _masked_mean(
            certainty,
            _boundary(p_final),
            ref,
        )
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


def _pc_size(pc):
    for key in ("B3", "valid3_map", "G_attn_map", "M_pc_map"):
        value = pc.get(key)
        if torch.is_tensor(value) and value.dim() >= 2:
            return tuple(int(v) for v in value.shape[-2:])
    return (40, 40)
