"""PC-HBM labelled and semi-supervised losses.

All disabled or unavailable terms return tensor zeros on the active device, so
training code can log a stable dictionary across stages.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F

from .branch_oracle import oracle_distribution
from .pc_supervision import build_geometry_target, build_need_correction_map, build_region_label_map, gather_by_boundary_indices


def zero_like_loss(ref: torch.Tensor) -> torch.Tensor:
    return ref.sum() * 0.0


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    inter = (prob * target).sum(dim=(-2, -1))
    denom = prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def iou_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    inter = (prob * target).sum(dim=(-2, -1))
    union = (prob + target - prob * target).sum(dim=(-2, -1))
    return (1.0 - (inter + eps) / (union + eps)).mean()


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    return bce + dice_loss_with_logits(logits, target) + iou_loss_with_logits(logits, target)


def compute_pc_hbm_labeled_loss(outputs, aux: Dict[str, Any] | None, gt: torch.Tensor, config: Any) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute full labelled PC-HBM loss and logging tensors."""

    if aux is None:
        ref = outputs[-1] if isinstance(outputs, (list, tuple)) else gt
        z = zero_like_loss(ref)
        return z, _zero_log(z)
    m4, m3, m2, z_main = outputs
    ref = z_main
    L_ms = (
        0.4 * seg_loss(m4, gt)
        + 0.6 * seg_loss(m3, gt)
        + 0.8 * seg_loss(m2, gt)
        + float(getattr(config, "lambda_main", 1.0)) * seg_loss(z_main, gt)
    )
    z_final = aux.get("z_final", z_main)
    z_nomix = aux.get("z_nomix", z_main)
    L_final = float(getattr(config, "lambda_final", 1.0)) * seg_loss(z_final, gt)
    L_nomix = float(getattr(config, "lambda_nomix", 0.5)) * seg_loss(z_nomix, gt)
    L_seg_total = L_ms + L_final + L_nomix
    pc = aux.get("pc_hbm", {}) or {}
    mix = aux.get("mixture", {}) or {}
    p2 = aux.get("p2_bra", {}) or {}
    p1 = aux.get("p1_pra", {}) or {}
    L_parent_ce = _parent_ce(pc, gt)
    L_child_verify = _child_verify(pc, gt)
    L_geometry = _geometry_loss(pc, gt)
    L_gate = _gate_loss(pc, aux, gt)
    L_mem = L_parent_ce + L_child_verify + L_geometry + L_gate
    L_boundary_aux = _boundary_aux(pc, p2, p1, gt, ref)
    L_mix_oracle, oracle = _mix_oracle(mix, gt, config, ref)
    L_branch = _branch_loss(mix, gt, ref)
    L_quality = _quality_loss(mix, oracle, ref)
    L_usage = _usage_loss(mix, gt, ref)
    L_reg = _regularization(mix, ref)
    total = (
        L_seg_total
        + float(getattr(config, "lambda_mem", 0.2)) * L_mem
        + float(getattr(config, "lambda_boundary_aux", 0.2)) * L_boundary_aux
        + float(getattr(config, "lambda_mix_oracle", 0.2)) * L_mix_oracle
        + float(getattr(config, "lambda_branch", 0.2)) * L_branch
        + float(getattr(config, "lambda_quality", 0.05)) * L_quality
        + float(getattr(config, "lambda_usage", 0.02)) * L_usage
        + float(getattr(config, "lambda_reg", 0.05)) * L_reg
    )
    log = {
        "L_seg_total": L_seg_total.detach(),
        "L_parent_ce": L_parent_ce.detach(),
        "L_child_verify": L_child_verify.detach(),
        "L_geometry": L_geometry.detach(),
        "L_gate": L_gate.detach(),
        "L_boundary_aux": L_boundary_aux.detach(),
        "L_mix_oracle": L_mix_oracle.detach(),
        "L_branch": L_branch.detach(),
        "L_quality": L_quality.detach(),
        "L_usage": L_usage.detach(),
        "L_reg": L_reg.detach(),
        "pi_keep_mean": _mean_or_zero(mix.get("pi", ref.new_zeros(ref.size(0), 4, *ref.shape[-2:]))[:, 0:1], ref).detach(),
        "pi_res_mean": _mean_or_zero(mix.get("pi", ref.new_zeros(ref.size(0), 4, *ref.shape[-2:]))[:, 1:2], ref).detach(),
        "pi_def_mean": _mean_or_zero(mix.get("pi", ref.new_zeros(ref.size(0), 4, *ref.shape[-2:]))[:, 2:3], ref).detach(),
        "pi_sup_mean": _mean_or_zero(mix.get("pi", ref.new_zeros(ref.size(0), 4, *ref.shape[-2:]))[:, 3:4], ref).detach(),
        "gate_pc_mean": _mean_or_zero(pc.get("gate_pc_map"), ref).detach(),
        "C23_mean": _mean_or_zero(pc.get("C23_map"), ref).detach(),
        "route_entropy": _mean_or_zero(pc.get("route_entropy"), ref).detach(),
        "parent_entropy": _mean_or_zero(pc.get("parent_entropy"), ref).detach(),
    }
    return total, log


def compute_pc_hbm_unlabeled_loss(student_aux: Dict[str, Any], pseudo_prob: torch.Tensor, confidence: torch.Tensor, config: Any) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Stage-4 unlabeled loss: supervise student z_nomix/z_main, not z_final."""

    z_student = student_aux.get("z_nomix", student_aux.get("z_main"))
    if z_student is None:
        z_student = pseudo_prob
    if pseudo_prob.shape[-2:] != z_student.shape[-2:]:
        pseudo_prob = F.interpolate(pseudo_prob, size=z_student.shape[-2:], mode="nearest")
    if confidence.shape[-2:] != z_student.shape[-2:]:
        confidence = F.interpolate(confidence, size=z_student.shape[-2:], mode="bilinear", align_corners=False)
    bce = F.binary_cross_entropy_with_logits(z_student, pseudo_prob.detach(), reduction="none")
    loss = (bce * confidence.detach()).sum() / confidence.sum().clamp_min(1.0)
    final_weight = float(getattr(config, "pc_hbm_unsup_final_consistency_weight", 0.1))
    if bool(student_aux.get("mixture_skipped", False)) or str(student_aux.get("forward_mode", "")) == "student_core":
        final_weight = 0.0
    z_final = student_aux.get("z_final")
    if z_final is not None and final_weight > 0:
        final_bce = F.binary_cross_entropy_with_logits(z_final, pseudo_prob.detach(), reduction="none")
        high = (confidence > 0.8).float()
        loss = loss + final_weight * (final_bce * high).sum() / high.sum().clamp_min(1.0)
    return float(getattr(config, "lambda_u", 1.0)) * loss, {"L_u": loss.detach(), "pseudo_conf_mean": confidence.mean().detach()}


def structure_aware_confidence(teacher_aux: Dict[str, Any]) -> torch.Tensor:
    """Confidence from probability certainty, PC agreement, mixture and route entropy."""

    p_mix = teacher_aux.get("p_final")
    if p_mix is None:
        p_mix = torch.sigmoid(teacher_aux.get("z_final", teacher_aux["z_main"]))
    z_main = teacher_aux.get("z_main")
    if z_main is not None:
        p_main = torch.sigmoid(z_main)
    else:
        p_main = teacher_aux.get("p_main", p_mix)
    if p_main.shape[-2:] != p_mix.shape[-2:]:
        p_main = F.interpolate(p_main, size=p_mix.shape[-2:], mode="bilinear", align_corners=False)
    certainty = (2.0 * (p_mix - 0.5).abs()).clamp(0.0, 1.0)
    agreement = (1.0 - (p_mix - p_main).abs()).clamp(0.0, 1.0)
    mix = teacher_aux.get("mixture", {}) or {}
    pc = teacher_aux.get("pc_hbm", {}) or {}
    pi = mix.get("pi")
    if pi is not None:
        mix_ent = -(pi * pi.clamp_min(1e-6).log()).sum(dim=1, keepdim=True) / torch.log(torch.tensor(4.0, device=pi.device, dtype=pi.dtype))
        if mix_ent.shape[-2:] != certainty.shape[-2:]:
            mix_ent = F.interpolate(mix_ent, size=certainty.shape[-2:], mode="bilinear", align_corners=False)
    else:
        mix_ent = torch.zeros_like(certainty)
    c23 = pc.get("C23_map")
    if c23 is None:
        c23_up = torch.zeros_like(certainty)
    else:
        c23_up = F.interpolate(c23, size=certainty.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    route_ent = pc.get("route_entropy")
    if isinstance(route_ent, torch.Tensor) and route_ent.numel() > 0:
        route_penalty = route_ent.view(route_ent.size(0), 1, 1, 1).to(device=certainty.device, dtype=certainty.dtype)
    else:
        route_penalty = torch.zeros_like(certainty[:, :, :1, :1])
    return (certainty * agreement * (1.0 - 0.5 * mix_ent) * (1.0 - 0.5 * c23_up) * (1.0 - 0.25 * route_penalty)).clamp(0.0, 1.0)


def _parent_ce(pc: Dict[str, Any], gt: torch.Tensor) -> torch.Tensor:
    p3_group = pc.get("P3_group")
    boundary = pc.get("boundary_indices3")
    if p3_group is None or boundary is None or p3_group.numel() == 0:
        return zero_like_loss(gt)
    region_label_map = build_region_label_map(gt, _pc_size3(pc))
    region_label3 = gather_by_boundary_indices(region_label_map, boundary).long().clamp(0, 3)
    probs = p3_group.clamp_min(1e-6)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return F.nll_loss(probs.log(), region_label3)


def _child_verify(pc: Dict[str, Any], gt: torch.Tensor) -> torch.Tensor:
    s_child = pc.get("S_child")
    parent_region = pc.get("top_parent_region_ids")
    boundary = pc.get("boundary_indices3")
    if s_child is None or parent_region is None or boundary is None or s_child.numel() == 0:
        ref = next((v for v in pc.values() if isinstance(v, torch.Tensor)), gt)
        return zero_like_loss(ref)
    region_label_map = build_region_label_map(gt, _pc_size3(pc))
    region_label3 = gather_by_boundary_indices(region_label_map, boundary).long().clamp(0, 3)
    support = (parent_region == region_label3[:, None]).to(dtype=s_child.dtype)
    valid = parent_region.ge(0).to(dtype=s_child.dtype)
    hard_neg = (
        ((region_label3[:, None] == 1) & (parent_region == 2))
        | ((region_label3[:, None] == 2) & (parent_region == 1))
    ).to(dtype=s_child.dtype)
    weight = valid * (1.0 + hard_neg)
    loss = F.binary_cross_entropy_with_logits(s_child, support, weight=weight, reduction="sum")
    return loss / weight.sum().clamp_min(1.0)


def _geometry_loss(pc: Dict[str, Any], gt: torch.Tensor) -> torch.Tensor:
    g_parent = pc.get("G_attn")
    g_child = pc.get("G_child_attn")
    o_pc = pc.get("O_pc_token")
    boundary = pc.get("boundary_indices3")
    if g_parent is None or boundary is None or g_parent.numel() == 0:
        ref = next((v for v in pc.values() if isinstance(v, torch.Tensor)), gt)
        return zero_like_loss(ref)
    geo = build_geometry_target(gt, _pc_size3(pc))
    gt_sdf = gather_by_boundary_indices(geo["sdf"], boundary).view(-1)
    gt_normal = gather_by_boundary_indices(geo["normal"], boundary)
    gt_offset = gather_by_boundary_indices(geo["offset"], boundary)
    l_sdf = F.l1_loss(g_parent[:, 0], gt_sdf)
    l_normal = 1.0 - F.cosine_similarity(g_parent[:, 1:3], gt_normal, dim=-1).mean()
    l_offset = F.l1_loss(o_pc, gt_offset) if torch.is_tensor(o_pc) and o_pc.numel() > 0 else zero_like_loss(g_parent)
    l_cons = 0.0
    if torch.is_tensor(g_child) and g_child.numel() > 0 and g_child.shape == g_parent.shape:
        l_cons = 0.1 * (g_parent - g_child).abs().mean()
    return l_sdf + 0.5 * l_normal + 0.5 * l_offset + l_cons


def _gate_loss(pc: Dict[str, Any], aux: Dict[str, Any], gt: torch.Tensor) -> torch.Tensor:
    gate = pc.get("gate_pc_token")
    c23 = pc.get("C23_token")
    boundary = pc.get("boundary_indices3")
    z_main = aux.get("z_main")
    if boundary is None or z_main is None:
        ref = next((v for v in pc.values() if isinstance(v, torch.Tensor)), gt)
        return zero_like_loss(ref)
    if gate is None:
        gate_map = pc.get("gate_pc_map")
        gate = gather_by_boundary_indices(gate_map, boundary) if torch.is_tensor(gate_map) else None
    if c23 is None:
        c23_map = pc.get("C23_map")
        c23 = gather_by_boundary_indices(c23_map, boundary) if torch.is_tensor(c23_map) else None
    if gate is None or c23 is None or gate.numel() == 0:
        ref = next((v for v in pc.values() if isinstance(v, torch.Tensor)), gt)
        return zero_like_loss(ref)
    need = build_need_correction_map(z_main, gt, _pc_size3(pc), threshold=0.25)
    gate_target = gather_by_boundary_indices(need, boundary).view(-1, 1)
    gate_target = gate_target * (1.0 - c23.detach()).clamp(0.0, 1.0)
    return F.binary_cross_entropy(gate.view_as(gate_target).clamp(1e-6, 1.0 - 1e-6), gate_target.detach())


def _pc_size3(pc: Dict[str, Any]) -> tuple[int, int]:
    for key in ("B3", "valid3_map", "G_attn_map", "M_pc_map"):
        value = pc.get(key)
        if torch.is_tensor(value) and value.dim() >= 2:
            return tuple(int(v) for v in value.shape[-2:])
    return (40, 40)


def _boundary_aux(pc, p2, p1, gt, ref):
    losses = []
    for key, aux in (("B3", pc), ("B2", p2), ("B1", p1), ("B2_refined_map", p2)):
        pred = aux.get(key)
        if pred is None:
            continue
        target = _gt_boundary(gt, pred.shape[-2:])
        losses.append(F.binary_cross_entropy(pred.clamp(1e-6, 1.0 - 1e-6), target))
    g2_ref = p2.get("G2_refined_map")
    if torch.is_tensor(g2_ref):
        need2 = build_need_correction_map(ref, gt, g2_ref.shape[-2:], threshold=0.25)
        losses.append(F.binary_cross_entropy(g2_ref.clamp(1e-6, 1.0 - 1e-6), need2))
    o2_ref = p2.get("O2_refined_map")
    if torch.is_tensor(o2_ref):
        geo2 = build_geometry_target(gt, o2_ref.shape[-2:])
        valid2 = p2.get("valid2_map", torch.ones_like(o2_ref[:, :1])).to(device=o2_ref.device, dtype=o2_ref.dtype)
        offset_loss = F.smooth_l1_loss(o2_ref, geo2["offset"].to(device=o2_ref.device, dtype=o2_ref.dtype), reduction="none")
        losses.append(0.25 * (offset_loss * valid2).sum() / valid2.sum().clamp_min(1.0))
    return sum(losses) if losses else zero_like_loss(ref)


def _gt_boundary(gt, size):
    target = F.interpolate(gt.float(), size=size, mode="nearest")
    dil = F.max_pool2d(target, 3, stride=1, padding=1)
    ero = -F.max_pool2d(-target, 3, stride=1, padding=1)
    return (dil - ero).clamp(0.0, 1.0)


def _mix_oracle(mix, gt, config, ref):
    if not mix or "pi" not in mix:
        return zero_like_loss(ref), {}
    oracle = oracle_distribution(mix, gt, tau=float(getattr(config, "pc_hbm_tau_oracle", 0.5)))
    target_mix = oracle["target_mix"].detach()
    mask = oracle["oracle_mask"].detach()
    pi = mix["pi"].clamp_min(1e-6)
    kl = (target_mix * (target_mix.clamp_min(1e-6).log() - pi.log())).sum(dim=1, keepdim=True)
    loss = (kl * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, oracle


def _branch_loss(mix, gt, ref):
    if not mix or "z_keep" not in mix:
        return zero_like_loss(ref)
    return 0.25 * sum(seg_loss(mix[name], gt) for name in ("z_keep", "z_res", "z_def", "z_sup"))


def _quality_loss(mix, oracle, ref):
    quality = mix.get("branch_quality")
    if quality is None or not oracle:
        return zero_like_loss(ref)
    err = oracle["pixel_error"].detach()
    target_gain = err[:, 0:1] - err
    weight = mix.get("B_pix", torch.ones_like(target_gain[:, :1])).detach()
    return (F.smooth_l1_loss(quality, target_gain, reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)


def _usage_loss(mix, gt, ref):
    pi = mix.get("pi")
    if pi is None:
        return zero_like_loss(ref)
    target = F.interpolate(gt.float(), size=pi.shape[-2:], mode="nearest")
    p_keep = torch.sigmoid(mix.get("z_keep", ref))
    if p_keep.shape[-2:] != pi.shape[-2:]:
        p_keep = F.interpolate(p_keep, size=pi.shape[-2:], mode="bilinear", align_corners=False)
    fn = ((target > 0.5) & (p_keep < 0.4)).float()
    fp = ((target < 0.5) & (p_keep > 0.6)).float()
    grad = _gt_boundary(target, pi.shape[-2:])
    mis = grad * (1.0 - fn) * (1.0 - fp)
    stable = (1.0 - torch.maximum(torch.maximum(fn, fp), mis)).clamp(0.0, 1.0)
    targets = [
        (stable, pi.new_tensor([0.90, 0.03, 0.04, 0.03])),
        (fn, pi.new_tensor([0.20, 0.60, 0.15, 0.05])),
        (fp, pi.new_tensor([0.20, 0.05, 0.15, 0.60])),
        (mis, pi.new_tensor([0.25, 0.15, 0.50, 0.10])),
    ]
    loss = zero_like_loss(pi)
    for mask, vec in targets:
        tgt = vec.view(1, 4, 1, 1).expand_as(pi)
        ce = -(tgt * pi.clamp_min(1e-6).log()).sum(dim=1, keepdim=True)
        loss = loss + (ce * mask).sum() / mask.sum().clamp_min(1.0)
    return loss


def _regularization(mix, ref):
    if not mix:
        return zero_like_loss(ref)
    reg = zero_like_loss(ref)
    if "O_pix" in mix:
        off = mix["O_pix"]
        reg = reg + off.abs().mean()
        reg = reg + (off[..., 1:, :] - off[..., :-1, :]).abs().mean() + (off[..., :, 1:] - off[..., :, :-1]).abs().mean()
    if "Mask_corr" in mix:
        reg = reg + mix["Mask_corr"].mean() * 0.1
    if "z_final" in mix and "z_keep" in mix:
        reg = reg + (mix["z_final"] - mix["z_keep"]).abs().mean() * 0.01
    return reg


def _mean_or_zero(value, ref):
    if isinstance(value, torch.Tensor) and value.numel() > 0:
        return value.mean()
    return zero_like_loss(ref)


def _zero_log(z):
    names = [
        "L_seg_total",
        "L_parent_ce",
        "L_child_verify",
        "L_geometry",
        "L_gate",
        "L_boundary_aux",
        "L_mix_oracle",
        "L_branch",
        "L_quality",
        "L_usage",
        "L_reg",
        "pi_keep_mean",
        "pi_res_mean",
        "pi_def_mean",
        "pi_sup_mean",
        "gate_pc_mean",
        "C23_mean",
        "route_entropy",
        "parent_entropy",
    ]
    return {name: z.detach() for name in names}
