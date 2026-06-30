from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .svb_utils import (
        compute_connected_component_boxes,
        image_gradient_magnitude,
        merge_pos_neg_points,
        normalize_01,
        resize_like,
        sample_topk_points,
    )
except ImportError:
    from SAM.SAM_refinement.svb_utils import (
        compute_connected_component_boxes,
        image_gradient_magnitude,
        merge_pos_neg_points,
        normalize_01,
        resize_like,
        sample_topk_points,
    )


class CBMPromptGenerator(nn.Module):
    """Generate CBM-guided hybrid prompts for SVB-PLR.

    Shape:
        teacher_prob: [B, 1, H, W]
        retrieval_aux: mapping with CBM evidence maps
        return: prompt_pack dict consumed by ExistingSAMBackendAdapter
    """

    DEFAULTS = {
        "sam_cbm_boundary_weight": 1.0,
        "sam_unc_weight": 0.5,
        "sam_grad_weight": 0.5,
        "sam_cons_weight": 0.5,
        "sam_gate_weight": 0.5,
        "sam_refine_theta": 0.25,
        "sam_num_pos_points": 8,
        "sam_num_neg_points": 8,
        "sam_num_boundary_points": 12,
        "sam_box_expand_ratio": 0.05,
        "sam_prompt_min_area": 32,
        "sam_refine_boundary_only": True,
        "sam_use_box_prompt": True,
        "sam_use_point_prompt": True,
        "sam_use_mask_prompt": True,
        "sam_mask_prompt_eps": 0.05,
        "sam_mask_prompt_strength": 1.0,
        "sam_use_boundary_points": True,
    }

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(self, teacher_prob: torch.Tensor, retrieval_aux: Any) -> Dict[str, Any]:
        """Build SAM prompts from teacher probability and CBM retrieval evidence.

        Shape:
            teacher_prob: [B, 1, H, W]
            retrieval_aux: dict-like evidence from cbm_aux_adapter
            return: prompt_pack with boxes/points/mask/refine band/evidence
        """
        p_t = self._as_teacher_prob(teacher_prob)
        evidence = self.parse_cbm_evidence(p_t, retrieval_aux)
        refine_band, band_extra = self.build_refinement_band(p_t, evidence)
        evidence.update(band_extra)

        pos_points = self._build_positive_points(p_t, evidence)
        neg_points = self._build_negative_points(p_t, evidence)
        boundary_points = self._build_boundary_points(p_t, evidence, refine_band)

        if self._cfg_bool("sam_use_boundary_points"):
            pos_with_boundary = [
                torch.cat((pos, bd.to(device=pos.device, dtype=pos.dtype)), dim=0)
                if pos.numel() or bd.numel()
                else pos
                for pos, bd in zip(pos_points, boundary_points)
            ]
        else:
            pos_with_boundary = pos_points
        point_coords, point_labels = merge_pos_neg_points(pos_with_boundary, neg_points)
        point_coords = self._clamp_point_list(point_coords, p_t)

        boxes = self._build_boxes(p_t, evidence)
        mask_prompt = self._build_mask_prompt(p_t, refine_band)

        prompt_pack = {
            "boxes": boxes,
            "pos_points": pos_points,
            "neg_points": neg_points,
            "boundary_points": boundary_points,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_prompt": mask_prompt,
            "mask_inputs": mask_prompt,
            "refine_band": refine_band,
            "evidence": evidence,
        }
        return self._apply_prompt_switches(prompt_pack, p_t)

    def parse_cbm_evidence(self, teacher_prob: torch.Tensor, retrieval_aux: Any) -> Dict[str, torch.Tensor]:
        """Parse and resize CBM evidence maps to teacher_prob spatial size.

        Shape:
            teacher_prob: [B, 1, H, W]
            retrieval_aux: mapping with optional Y_ctx/U_map/cons_map/gate3/B3/B_query/valid_map
            return: evidence maps, each [B, 1, H, W]
        """
        aux = retrieval_aux if isinstance(retrieval_aux, Mapping) else {}
        p_t = teacher_prob
        uncertainty = self._uncertainty(p_t)
        gradient = image_gradient_magnitude(p_t)

        y_ctx = self._resize_map(aux.get("Y_ctx"), p_t, mode="bilinear")
        if y_ctx is not None and y_ctx.dim() == 4 and y_ctx.size(1) >= 4:
            s_fg = y_ctx[:, 0:1] + y_ctx[:, 1:2]
            s_bg = y_ctx[:, 2:3] + y_ctx[:, 3:4]
            m_bd = y_ctx[:, 1:2] - y_ctx[:, 2:3]
        else:
            s_fg = p_t
            s_bg = 1.0 - p_t
            m_bd = torch.zeros_like(p_t)

        u_map = self._resize_map(aux.get("U_map"), p_t, mode="bilinear")
        cons_map = self._resize_map(aux.get("cons_map"), p_t, mode="bilinear")
        gate3 = self._resize_map(aux.get("gate3"), p_t, mode="bilinear")
        b3_value = aux.get("B3")
        if b3_value is None:
            b3_value = aux.get("B_query")
        b3 = self._resize_map(b3_value, p_t, mode="bilinear")
        valid_map = self._resize_map(aux.get("valid_map"), p_t, mode="nearest")

        u_up = uncertainty if u_map is None else u_map.clamp(0.0, 1.0)
        cons_up = torch.ones_like(p_t) if cons_map is None else cons_map.clamp(0.0, 1.0)
        gate_up = torch.ones_like(p_t) if gate3 is None else gate3.clamp(0.0, 1.0)
        b3_up = normalize_01(uncertainty + gradient) if b3 is None else b3.clamp(0.0, 1.0)
        valid_up = torch.ones_like(p_t) if valid_map is None else (valid_map > 0.5).to(dtype=p_t.dtype)

        return {
            "S_fg_up": s_fg.to(device=p_t.device, dtype=p_t.dtype),
            "S_bg_up": s_bg.to(device=p_t.device, dtype=p_t.dtype),
            "M_bd_up": m_bd.to(device=p_t.device, dtype=p_t.dtype),
            "U_up": u_up.to(device=p_t.device, dtype=p_t.dtype),
            "cons_up": cons_up.to(device=p_t.device, dtype=p_t.dtype),
            "gate_up": gate_up.to(device=p_t.device, dtype=p_t.dtype),
            "B3_up": b3_up.to(device=p_t.device, dtype=p_t.dtype),
            "valid_up": valid_up.to(device=p_t.device, dtype=p_t.dtype),
        }

    def build_refinement_band(
        self,
        teacher_prob: torch.Tensor,
        evidence: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Build refinement band from teacher uncertainty, gradient, and CBM evidence.

        Shape:
            teacher_prob: [B, 1, H, W]
            evidence maps: [B, 1, H, W]
            return: refine_band [B, 1, H, W], extra evidence dict
        """
        p_t = teacher_prob
        b_unc = self._uncertainty(p_t)
        b_grad = image_gradient_magnitude(p_t)
        r_band_score = (
            self._cfg_float("sam_cbm_boundary_weight") * evidence["B3_up"]
            + self._cfg_float("sam_unc_weight") * b_unc
            + self._cfg_float("sam_grad_weight") * b_grad
            + self._cfg_float("sam_cons_weight") * (1.0 - evidence["cons_up"])
            + self._cfg_float("sam_gate_weight") * evidence["gate_up"]
        )
        r_band_score = normalize_01(r_band_score)
        refine_band = (r_band_score > self._cfg_float("sam_refine_theta")).to(dtype=p_t.dtype)
        return refine_band, {
            "B_unc": b_unc,
            "B_grad": b_grad,
            "R_band_score": r_band_score,
        }

    def _build_positive_points(self, p_t: torch.Tensor, evidence: Mapping[str, torch.Tensor]) -> List[torch.Tensor]:
        pos_score = (
            0.5 * p_t
            + 0.3 * torch.sigmoid(evidence["S_fg_up"] - evidence["S_bg_up"])
            + 0.2 * torch.sigmoid(evidence["M_bd_up"])
        ) * evidence["cons_up"]
        pos_mask = (
            (p_t > 0.65)
            | ((evidence["S_fg_up"] > evidence["S_bg_up"] + 0.15) & (evidence["M_bd_up"] > 0))
        ) & (evidence["cons_up"] > 0.4)
        return self._clamp_point_list(
            sample_topk_points(pos_score, pos_mask, self._cfg_int("sam_num_pos_points")),
            p_t,
        )

    def _build_negative_points(self, p_t: torch.Tensor, evidence: Mapping[str, torch.Tensor]) -> List[torch.Tensor]:
        neg_score = (
            0.5 * (1.0 - p_t)
            + 0.3 * torch.sigmoid(evidence["S_bg_up"] - evidence["S_fg_up"])
            + 0.2 * torch.sigmoid(-evidence["M_bd_up"])
        ) * evidence["cons_up"]
        neg_mask = (
            (p_t < 0.25)
            | ((evidence["S_bg_up"] > evidence["S_fg_up"] + 0.15) & (evidence["M_bd_up"] < 0))
        )
        return self._clamp_point_list(
            sample_topk_points(neg_score, neg_mask, self._cfg_int("sam_num_neg_points")),
            p_t,
        )

    def _build_boundary_points(
        self,
        p_t: torch.Tensor,
        evidence: Mapping[str, torch.Tensor],
        refine_band: torch.Tensor,
    ) -> List[torch.Tensor]:
        boundary_score = refine_band * (
            0.4 * evidence["B3_up"]
            + 0.3 * evidence["U_up"]
            + 0.3 * image_gradient_magnitude(p_t)
        )
        return self._clamp_point_list(
            sample_topk_points(boundary_score, refine_band > 0, self._cfg_int("sam_num_boundary_points")),
            p_t,
        )

    def _build_boxes(self, p_t: torch.Tensor, evidence: Mapping[str, torch.Tensor]) -> List[torch.Tensor]:
        min_area = self._cfg_int("sam_prompt_min_area")
        expand_ratio = self._cfg_float("sam_box_expand_ratio")

        boxes = compute_connected_component_boxes(p_t > 0.5, min_area, expand_ratio)
        boxes = self._fill_empty_boxes(boxes, p_t > 0.4, min_area, expand_ratio)

        evidence_score = normalize_01(evidence["S_fg_up"] - evidence["S_bg_up"])
        evidence_mask = self._top_fraction_mask(evidence_score, fraction=0.10, min_score=0.5)
        boxes = self._fill_empty_boxes(boxes, evidence_mask, min_area, expand_ratio)
        return [box.to(device=p_t.device, dtype=torch.float32) for box in boxes]

    def _build_mask_prompt(self, p_t: torch.Tensor, refine_band: torch.Tensor) -> torch.Tensor:
        probability = p_t.detach().clamp(0.0, 1.0)
        if self._cfg_bool("sam_refine_boundary_only"):
            blurred_teacher = F.avg_pool2d(probability, kernel_size=3, stride=1, padding=1)
            probability = (
                probability * (1.0 - refine_band) + blurred_teacher * refine_band
            ).clamp(0.0, 1.0)
        eps = min(max(self._cfg_float("sam_mask_prompt_eps"), 1e-6), 0.499999)
        strength = self._cfg_float("sam_mask_prompt_strength")
        return torch.logit(probability.clamp(eps, 1.0 - eps)) * strength

    def _apply_prompt_switches(self, prompt_pack: Dict[str, Any], ref: torch.Tensor) -> Dict[str, Any]:
        """Apply user-facing prompt switches without dropping evidence maps."""
        prompt = dict(prompt_pack)
        switches = {
            "sam_use_box_prompt": self._cfg_bool("sam_use_box_prompt"),
            "sam_use_point_prompt": self._cfg_bool("sam_use_point_prompt"),
            "sam_use_mask_prompt": self._cfg_bool("sam_use_mask_prompt"),
            "sam_use_boundary_points": self._cfg_bool("sam_use_boundary_points"),
        }

        if not switches["sam_use_box_prompt"]:
            prompt["boxes"] = self._empty_boxes(ref)
        if not switches["sam_use_boundary_points"]:
            prompt["boundary_points"] = self._empty_points(ref)
        if not switches["sam_use_point_prompt"]:
            prompt["pos_points"] = self._empty_points(ref)
            prompt["neg_points"] = self._empty_points(ref)
            prompt["boundary_points"] = self._empty_points(ref)
            prompt["point_coords"] = self._empty_points(ref)
            prompt["point_labels"] = self._empty_labels(ref)
        if not switches["sam_use_mask_prompt"]:
            prompt["mask_prompt"] = None
            prompt["mask_inputs"] = None

        prompt["prompt_switches"] = switches
        prompt["prompt_stats"] = self._prompt_stats(prompt, ref)
        return prompt

    def _prompt_stats(self, prompt: Mapping[str, Any], ref: torch.Tensor) -> Dict[str, float]:
        boxes = prompt.get("boxes")
        point_coords = prompt.get("point_coords")
        boundary_points = prompt.get("boundary_points")
        mask_inputs = prompt.get("mask_inputs", prompt.get("mask_prompt"))

        box_counts = self._count_tensor_list(boxes, ref.size(0))
        point_counts = self._count_tensor_list(point_coords, ref.size(0))
        boundary_counts = self._count_tensor_list(boundary_points, ref.size(0))
        has_mask = 1.0 if torch.is_tensor(mask_inputs) and mask_inputs.numel() > 0 else 0.0

        empty_box_ratio = sum(1 for count in box_counts if count == 0) / max(1, len(box_counts))
        empty_point_ratio = sum(1 for count in point_counts if count == 0) / max(1, len(point_counts))
        all_prompt_empty = (
            all(count == 0 for count in box_counts)
            and all(count == 0 for count in point_counts)
            and has_mask == 0.0
        )
        return {
            "box_count": float(sum(box_counts)),
            "point_count": float(sum(point_counts)),
            "boundary_point_count": float(sum(boundary_counts)),
            "has_mask": float(has_mask),
            "empty_box_ratio": float(empty_box_ratio),
            "empty_point_ratio": float(empty_point_ratio),
            "all_prompt_empty": 1.0 if all_prompt_empty else 0.0,
        }

    def _fill_empty_boxes(
        self,
        boxes: List[torch.Tensor],
        mask: torch.Tensor,
        min_area: int,
        expand_ratio: float,
    ) -> List[torch.Tensor]:
        fallback_boxes = compute_connected_component_boxes(mask, min_area, expand_ratio)
        out: List[torch.Tensor] = []
        for current, fallback in zip(boxes, fallback_boxes):
            out.append(fallback if current.numel() == 0 else current)
        return out

    @staticmethod
    def _top_fraction_mask(score: torch.Tensor, fraction: float, min_score: float) -> torch.Tensor:
        score4 = score if score.dim() == 4 else score.reshape(1, 1, *score.shape[-2:])
        batch_size, _, height, width = score4.shape
        masks = torch.zeros_like(score4, dtype=torch.bool)
        flat_count = height * width
        k = max(1, int(round(flat_count * float(fraction))))
        for batch_idx in range(batch_size):
            flat = score4[batch_idx, 0].flatten()
            if flat.numel() == 0:
                continue
            top_idx = torch.topk(flat, k=min(k, flat.numel()), dim=0).indices
            selected = torch.zeros_like(flat, dtype=torch.bool)
            selected[top_idx] = True
            selected &= flat > float(min_score)
            masks[batch_idx, 0] = selected.reshape(height, width)
        return masks

    @staticmethod
    def _uncertainty(p_t: torch.Tensor) -> torch.Tensor:
        return (4.0 * p_t * (1.0 - p_t)).clamp(0.0, 1.0)

    @staticmethod
    def _as_teacher_prob(teacher_prob: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(teacher_prob):
            raise TypeError("teacher_prob must be a torch.Tensor")
        if teacher_prob.dim() != 4 or teacher_prob.size(1) != 1:
            raise ValueError("teacher_prob must have shape [B,1,H,W]")
        return teacher_prob.detach().clamp(0.0, 1.0)

    @staticmethod
    def _clamp_point_list(points: List[torch.Tensor], ref: torch.Tensor) -> List[torch.Tensor]:
        height, width = ref.shape[-2:]
        clamped: List[torch.Tensor] = []
        for point in points:
            if point.numel() == 0:
                clamped.append(point.to(device=ref.device, dtype=torch.float32).reshape(0, 2))
                continue
            current = point.to(device=ref.device, dtype=torch.float32).reshape(-1, 2).clone()
            current[:, 0].clamp_(0, max(width - 1, 0))
            current[:, 1].clamp_(0, max(height - 1, 0))
            clamped.append(current)
        return clamped

    @staticmethod
    def _empty_points(ref: torch.Tensor) -> List[torch.Tensor]:
        return [ref.new_empty((0, 2), dtype=torch.float32) for _ in range(ref.size(0))]

    @staticmethod
    def _empty_labels(ref: torch.Tensor) -> List[torch.Tensor]:
        return [torch.empty((0,), device=ref.device, dtype=torch.long) for _ in range(ref.size(0))]

    @staticmethod
    def _empty_boxes(ref: torch.Tensor) -> List[torch.Tensor]:
        return [ref.new_empty((0, 4), dtype=torch.float32) for _ in range(ref.size(0))]

    @staticmethod
    def _count_tensor_list(value: Any, batch_size: int) -> List[int]:
        if isinstance(value, (list, tuple)):
            counts: List[int] = []
            for item in value:
                if torch.is_tensor(item):
                    if item.numel() == 0:
                        counts.append(0)
                    elif item.dim() >= 2:
                        counts.append(int(item.reshape(-1, item.shape[-1]).size(0)))
                    else:
                        counts.append(int(item.numel()))
                else:
                    counts.append(0)
            if len(counts) < batch_size:
                counts.extend([0] * (batch_size - len(counts)))
            return counts[:batch_size]
        if torch.is_tensor(value):
            if value.numel() == 0:
                return [0] * batch_size
            if value.dim() >= 3 and value.size(0) == batch_size:
                return [int(value[idx].reshape(-1, value.shape[-1]).size(0)) for idx in range(batch_size)]
            if value.dim() >= 2:
                return [int(value.reshape(-1, value.shape[-1]).size(0))] * batch_size
        return [0] * batch_size

    def _resize_map(self, value: Any, ref: torch.Tensor, mode: str) -> Optional[torch.Tensor]:
        if value is None or not torch.is_tensor(value):
            return None
        x = value.detach().to(device=ref.device)
        if x.numel() == 0:
            return None
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            return None
        if x.size(0) != ref.size(0):
            if x.size(0) == 1:
                x = x.expand(ref.size(0), -1, -1, -1)
            else:
                return None
        x = resize_like(x, ref, mode=mode)
        return x.to(device=ref.device, dtype=ref.dtype)

    def _cfg(self, name: str) -> Any:
        if self.cfg is not None and hasattr(self.cfg, name):
            return getattr(self.cfg, name)
        return self.DEFAULTS[name]

    def _cfg_float(self, name: str) -> float:
        return float(self._cfg(name))

    def _cfg_int(self, name: str) -> int:
        return int(self._cfg(name))

    def _cfg_bool(self, name: str) -> bool:
        return bool(self._cfg(name))


__all__ = ["CBMPromptGenerator"]
