from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .svb_utils import resize_like, soft_boundary_alignment, soft_iou
except ImportError:
    from SAM.SAM_refinement.svb_utils import resize_like, soft_boundary_alignment, soft_iou


class PromptExpertSelector(nn.Module):
    """Build prompt experts and select the best SAM candidate per image.

    Shape:
        prompt_pack: dict from CBMPromptGenerator
        sam_candidates: list of dicts with masks [B, K, H, W], scores [B, K]
        teacher_prob: [B, 1, H, W]
    """

    DEFAULTS = {
        "use_prompt_expert": True,
        "sam_prompt_experts": ("box", "box_point", "mask", "boundary"),
        "sam_prompt_select_tau": 0.1,
        "sam_selector_lambda_iou": 0.25,
        "sam_selector_lambda_boundary": 0.25,
        "sam_selector_lambda_cbm": 0.40,
        "sam_selector_lambda_over": 0.10,
    }

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def build_expert_prompts(self, prompt_pack: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Build prompt variants for SAM prompt expert inference.

        Shape:
            prompt_pack: dict with boxes/points/mask/refine_band/evidence
            return: list of expert-specific prompt packs
        """
        base = dict(prompt_pack or {})
        if not self._cfg_bool("use_prompt_expert"):
            default_prompt = dict(base)
            default_prompt["expert"] = "default"
            return [default_prompt]

        expert_prompts: List[Dict[str, Any]] = []
        for expert in self._cfg_experts():
            if expert == "box":
                expert_prompts.append(self._box_prompt(base))
            elif expert == "box_point":
                expert_prompts.append(self._box_point_prompt(base))
            elif expert == "mask":
                expert_prompts.append(self._mask_prompt(base))
            elif expert == "boundary":
                expert_prompts.append(self._boundary_prompt(base))
        if not expert_prompts:
            default_prompt = dict(base)
            default_prompt["expert"] = "default"
            return [default_prompt]
        return expert_prompts

    @torch.no_grad()
    def select(
        self,
        sam_candidates: Sequence[Mapping[str, Any]],
        teacher_prob: torch.Tensor,
        prompt_pack: Mapping[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Select the best candidate mask for each image.

        Shape:
            sam_candidates: list of {"expert", "masks": [B,K,H,W], "scores": [B,K]}
            teacher_prob: [B, 1, H, W]
            return: best_mask [B,1,H,W], best_score [B], selector_aux dict
        """
        p_t = self._as_teacher_prob(teacher_prob)
        if not sam_candidates:
            return self._fallback_selection(p_t, "empty_sam_candidates")

        expert_scores: Dict[str, torch.Tensor] = {}
        expert_components: Dict[str, Dict[str, torch.Tensor]] = {}
        masks_by_expert: List[torch.Tensor] = []
        logits_by_expert: List[Optional[torch.Tensor]] = []
        names_by_expert: List[str] = []
        local_indices_by_expert: List[torch.Tensor] = []

        for candidate_idx, candidate in enumerate(sam_candidates):
            candidate_map = candidate if isinstance(candidate, Mapping) else {}
            expert_name = str(candidate_map.get("expert", "candidate_{}".format(candidate_idx)))
            if expert_name in expert_scores:
                expert_name = "{}_{}".format(expert_name, candidate_idx)
            masks = self._as_bkhw_masks(candidate_map.get("masks"), p_t)
            if masks is None or masks.numel() == 0 or masks.size(1) == 0:
                continue

            scores = self._as_bk_scores(candidate_map.get("scores"), masks, p_t)
            components = self._score_components(masks, scores, p_t, prompt_pack)
            total = (
                self._cfg_float("sam_selector_lambda_iou") * components["iou"]
                + self._cfg_float("sam_selector_lambda_boundary") * components["boundary"]
                + self._cfg_float("sam_selector_lambda_cbm") * components["cbm"]
                - self._cfg_float("sam_selector_lambda_over") * components["over"]
                + self._cfg_float("sam_prompt_select_tau") * components["sam_score"]
            )

            expert_scores[expert_name] = total
            expert_components[expert_name] = components
            masks_by_expert.append(masks)
            logits_by_expert.append(self._as_candidate_logits(candidate_map.get("logits"), masks, p_t))
            names_by_expert.append(expert_name)
            local_indices_by_expert.append(torch.arange(masks.size(1), device=p_t.device, dtype=torch.long))

        if not masks_by_expert:
            return self._fallback_selection(p_t, "empty_candidate_masks")

        all_masks = torch.cat(masks_by_expert, dim=1)
        all_scores = torch.cat([expert_scores[name] for name in names_by_expert], dim=1)
        all_local_indices = torch.cat(local_indices_by_expert, dim=0)
        best_flat = all_scores.argmax(dim=1)
        batch_indices = torch.arange(p_t.size(0), device=p_t.device)
        best_mask = all_masks[batch_indices, best_flat].unsqueeze(1).clamp(0.0, 1.0)
        best_score = all_scores[batch_indices, best_flat]

        expert_offsets = self._expert_offsets(masks_by_expert)
        best_expert = self._best_expert_names(best_flat, names_by_expert, expert_offsets)
        best_candidate_index = all_local_indices[best_flat]
        selected_logits = self._select_logits(logits_by_expert, masks_by_expert, best_flat, batch_indices, p_t)

        selector_aux = {
            "expert_scores": expert_scores,
            "expert_components": expert_components,
            "best_expert": best_expert,
            "best_candidate_index": best_candidate_index,
            "use_prompt_expert": self._cfg_bool("use_prompt_expert"),
            "selected_logits": selected_logits,
            "used_fallback": False,
        }
        return best_mask, best_score, selector_aux

    def _score_components(
        self,
        masks: torch.Tensor,
        sam_scores: torch.Tensor,
        teacher_prob: torch.Tensor,
        prompt_pack: Mapping[str, Any],
    ) -> Dict[str, torch.Tensor]:
        teacher = resize_like(teacher_prob, masks, mode="bilinear").clamp(0.0, 1.0)
        iou_score = soft_iou(masks, teacher)
        refine_band = self._refine_band(prompt_pack, teacher)
        boundary_score = soft_boundary_alignment(masks, refine_band)
        cbm_score = self._cbm_agreement(masks, teacher, prompt_pack)
        over_penalty = self._over_seg_penalty(masks, teacher)
        return {
            "iou": iou_score,
            "boundary": boundary_score,
            "cbm": cbm_score,
            "over": over_penalty,
            "sam_score": sam_scores.clamp(0.0, 1.0),
        }

    def _box_prompt(self, base: Mapping[str, Any]) -> Dict[str, Any]:
        prompt = self._common_prompt(base, "box")
        prompt["boxes"] = base.get("boxes")
        return prompt

    def _box_point_prompt(self, base: Mapping[str, Any]) -> Dict[str, Any]:
        prompt = self._common_prompt(base, "box_point")
        prompt["boxes"] = base.get("boxes")
        prompt["point_coords"] = base.get("point_coords")
        prompt["point_labels"] = base.get("point_labels")
        return prompt

    def _mask_prompt(self, base: Mapping[str, Any]) -> Dict[str, Any]:
        prompt = self._common_prompt(base, "mask")
        mask_prompt = base.get("mask_inputs", base.get("mask_prompt"))
        prompt["mask_prompt"] = mask_prompt
        prompt["mask_inputs"] = mask_prompt
        return prompt

    def _boundary_prompt(self, base: Mapping[str, Any]) -> Dict[str, Any]:
        prompt = self._common_prompt(base, "boundary")
        boundary_points = base.get("boundary_points")
        prompt["point_coords"] = boundary_points
        prompt["point_labels"] = self._positive_labels_like(boundary_points)
        mask_prompt = base.get("mask_inputs", base.get("mask_prompt"))
        prompt["mask_prompt"] = mask_prompt
        prompt["mask_inputs"] = mask_prompt
        return prompt

    @staticmethod
    def _common_prompt(base: Mapping[str, Any], expert: str) -> Dict[str, Any]:
        return {
            "expert": expert,
            "refine_band": base.get("refine_band"),
            "evidence": base.get("evidence", {}),
        }

    @staticmethod
    def _positive_labels_like(points: Any):
        if points is None:
            return None
        if isinstance(points, (list, tuple)):
            labels = []
            for point in points:
                if torch.is_tensor(point):
                    labels.append(torch.ones(point.reshape(-1, 2).size(0), device=point.device, dtype=torch.long))
                else:
                    labels.append(None)
            return labels
        if torch.is_tensor(points):
            if points.dim() == 2:
                return torch.ones(points.reshape(-1, 2).size(0), device=points.device, dtype=torch.long)
            if points.dim() >= 3:
                return torch.ones(points.shape[:-1], device=points.device, dtype=torch.long)
        return None

    def _cbm_agreement(
        self,
        masks: torch.Tensor,
        teacher_prob: torch.Tensor,
        prompt_pack: Mapping[str, Any],
    ) -> torch.Tensor:
        evidence = prompt_pack.get("evidence", {}) if isinstance(prompt_pack, Mapping) else {}
        s_fg = evidence.get("S_fg_up") if isinstance(evidence, Mapping) else None
        s_bg = evidence.get("S_bg_up") if isinstance(evidence, Mapping) else None
        if torch.is_tensor(s_fg) and torch.is_tensor(s_bg):
            cbm_fg = torch.sigmoid(
                self._resize_evidence(s_fg, teacher_prob) - self._resize_evidence(s_bg, teacher_prob)
            ).clamp(0.0, 1.0)
        else:
            cbm_fg = teacher_prob
        return soft_iou(masks, cbm_fg)

    @staticmethod
    def _over_seg_penalty(masks: torch.Tensor, teacher_prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mask_area = masks.float().sum(dim=(-2, -1))
        teacher_area = (teacher_prob > 0.5).float().sum(dim=(-2, -1)).clamp_min(eps)
        area_ratio = mask_area / teacher_area
        return F.relu(area_ratio - 1.5)

    def _refine_band(self, prompt_pack: Mapping[str, Any], ref: torch.Tensor) -> torch.Tensor:
        band = prompt_pack.get("refine_band") if isinstance(prompt_pack, Mapping) else None
        if torch.is_tensor(band):
            band = self._resize_evidence(band, ref)
            return band.clamp(0.0, 1.0)
        return ref.new_zeros(ref.shape)

    @staticmethod
    def _resize_evidence(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        evidence = value.detach().to(device=ref.device, dtype=ref.dtype)
        if evidence.dim() == 2:
            evidence = evidence.unsqueeze(0).unsqueeze(0)
        elif evidence.dim() == 3:
            evidence = evidence.unsqueeze(1)
        elif evidence.dim() != 4:
            return ref
        if evidence.size(0) != ref.size(0):
            if evidence.size(0) == 1:
                evidence = evidence.expand(ref.size(0), -1, -1, -1)
            else:
                return ref
        if evidence.size(1) != 1:
            evidence = evidence[:, :1]
        return resize_like(evidence, ref, mode="bilinear").to(device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _as_teacher_prob(teacher_prob: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(teacher_prob):
            raise TypeError("teacher_prob must be a torch.Tensor")
        if teacher_prob.dim() != 4 or teacher_prob.size(1) != 1:
            raise ValueError("teacher_prob must have shape [B,1,H,W]")
        return teacher_prob.detach().clamp(0.0, 1.0)

    @staticmethod
    def _as_bkhw_masks(value: Any, ref: torch.Tensor) -> Optional[torch.Tensor]:
        if not torch.is_tensor(value):
            return None
        masks = value.detach().to(device=ref.device, dtype=ref.dtype)
        if masks.numel() == 0:
            return None
        if masks.dim() == 2:
            masks = masks.reshape(1, 1, *masks.shape[-2:])
        elif masks.dim() == 3:
            if masks.size(0) == ref.size(0):
                masks = masks.unsqueeze(1)
            else:
                masks = masks.unsqueeze(0)
        elif masks.dim() != 4:
            return None
        if masks.size(0) != ref.size(0):
            if masks.size(0) == 1:
                masks = masks.expand(ref.size(0), -1, -1, -1)
            else:
                return None
        if tuple(masks.shape[-2:]) != tuple(ref.shape[-2:]):
            masks = resize_like(masks, ref, mode="bilinear")
        return masks.clamp(0.0, 1.0)

    @staticmethod
    def _as_bk_scores(value: Any, masks: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        batch_size, num_candidates = masks.shape[:2]
        if not torch.is_tensor(value):
            return ref.new_zeros((batch_size, num_candidates))
        scores = value.detach().to(device=ref.device, dtype=ref.dtype)
        if scores.dim() == 0:
            scores = scores.reshape(1, 1).expand(batch_size, num_candidates)
        elif scores.dim() == 1:
            if scores.numel() == num_candidates:
                scores = scores.reshape(1, num_candidates).expand(batch_size, -1)
            elif scores.numel() == batch_size:
                scores = scores.reshape(batch_size, 1).expand(-1, num_candidates)
            else:
                scores = scores.reshape(1, -1)
        else:
            scores = scores.reshape(scores.size(0), -1)
        if scores.size(0) != batch_size:
            if scores.size(0) == 1:
                scores = scores.expand(batch_size, -1)
            else:
                return ref.new_zeros((batch_size, num_candidates))
        if scores.size(1) < num_candidates:
            pad = ref.new_zeros((batch_size, num_candidates - scores.size(1)))
            scores = torch.cat((scores, pad), dim=1)
        return scores[:, :num_candidates].clamp(0.0, 1.0)

    @staticmethod
    def _as_candidate_logits(value: Any, masks: torch.Tensor, ref: torch.Tensor) -> Optional[torch.Tensor]:
        if not torch.is_tensor(value):
            return None
        logits = value.detach().to(device=ref.device, dtype=ref.dtype)
        if logits.numel() == 0:
            return None
        if logits.dim() == 2:
            logits = logits.reshape(1, 1, *logits.shape[-2:])
        elif logits.dim() == 3:
            if logits.size(0) == ref.size(0):
                logits = logits.unsqueeze(1)
            else:
                logits = logits.unsqueeze(0)
        elif logits.dim() != 4:
            return None
        if logits.size(0) != ref.size(0):
            if logits.size(0) == 1:
                logits = logits.expand(ref.size(0), -1, -1, -1)
            else:
                return None
        if logits.size(1) < masks.size(1):
            return None
        return logits[:, : masks.size(1)]

    @staticmethod
    def _expert_offsets(masks_by_expert: Sequence[torch.Tensor]) -> List[Tuple[int, int]]:
        offsets: List[Tuple[int, int]] = []
        start = 0
        for masks in masks_by_expert:
            end = start + masks.size(1)
            offsets.append((start, end))
            start = end
        return offsets

    @staticmethod
    def _best_expert_names(best_flat: torch.Tensor, names: Sequence[str], offsets: Sequence[Tuple[int, int]]) -> List[str]:
        selected: List[str] = []
        best_cpu = best_flat.detach().cpu().tolist()
        for index in best_cpu:
            expert_name = names[-1] if names else "unknown"
            for name, (start, end) in zip(names, offsets):
                if start <= int(index) < end:
                    expert_name = name
                    break
            selected.append(expert_name)
        return selected

    @staticmethod
    def _select_logits(
        logits_by_expert: Sequence[Optional[torch.Tensor]],
        masks_by_expert: Sequence[torch.Tensor],
        best_flat: torch.Tensor,
        batch_indices: torch.Tensor,
        ref: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        selected: List[torch.Tensor] = []
        start = 0
        for batch_idx, flat_idx in enumerate(best_flat.detach().cpu().tolist()):
            chosen: Optional[torch.Tensor] = None
            for logits, masks in zip(logits_by_expert, masks_by_expert):
                end = start + masks.size(1)
                if start <= int(flat_idx) < end and logits is not None:
                    local_idx = int(flat_idx) - start
                    chosen = logits[batch_indices[batch_idx], local_idx]
                    break
                start = end
            start = 0
            if chosen is None:
                return None
            selected.append(chosen)
        return torch.stack(selected, dim=0).unsqueeze(1).to(device=ref.device, dtype=ref.dtype)

    def _fallback_selection(self, teacher_prob: torch.Tensor, reason: str):
        best_mask = teacher_prob.detach().clamp(0.0, 1.0)
        best_score = teacher_prob.new_zeros((teacher_prob.size(0),))
        selector_aux = {
            "expert_scores": {},
            "expert_components": {},
            "best_expert": ["fallback"] * teacher_prob.size(0),
            "best_candidate_index": teacher_prob.new_zeros((teacher_prob.size(0),), dtype=torch.long),
            "use_prompt_expert": self._cfg_bool("use_prompt_expert"),
            "selected_logits": None,
            "used_fallback": True,
            "fallback_reason": reason,
        }
        return best_mask, best_score, selector_aux

    def _cfg(self, name: str) -> Any:
        if self.cfg is not None and hasattr(self.cfg, name):
            return getattr(self.cfg, name)
        return self.DEFAULTS[name]

    def _cfg_bool(self, name: str) -> bool:
        return bool(self._cfg(name))

    def _cfg_float(self, name: str) -> float:
        return float(self._cfg(name))

    def _cfg_experts(self) -> List[str]:
        experts = self._cfg("sam_prompt_experts")
        if isinstance(experts, str):
            experts = [item.strip() for item in experts.split(",")]
        return [str(item) for item in experts if str(item) in {"box", "box_point", "mask", "boundary"}]


__all__ = ["PromptExpertSelector"]
