from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from utils.log_control import should_log

try:
    from .cbm_prompt_generator import CBMPromptGenerator
    from .conformal_sam_calibrator import ConformalSAMCalibrator
    from .prompt_expert_selector import PromptExpertSelector
    from .sam_backend_adapter import ExistingSAMBackendAdapter
    from .sam_refine_visualizer import SamRefineVisualizer
    from .sam_reliability_filter import SAMCBMReliabilityFilter
    from .svb_cache import SVBPLRCache
    from .svb_utils import SAMInferenceError, binary_reliability, resize_like
except ImportError:
    from SAM.SAM_refinement.cbm_prompt_generator import CBMPromptGenerator
    from SAM.SAM_refinement.conformal_sam_calibrator import ConformalSAMCalibrator
    from SAM.SAM_refinement.prompt_expert_selector import PromptExpertSelector
    from SAM.SAM_refinement.sam_backend_adapter import ExistingSAMBackendAdapter
    from SAM.SAM_refinement.sam_refine_visualizer import SamRefineVisualizer
    from SAM.SAM_refinement.sam_reliability_filter import SAMCBMReliabilityFilter
    from SAM.SAM_refinement.svb_cache import SVBPLRCache
    from SAM.SAM_refinement.svb_utils import SAMInferenceError, binary_reliability, resize_like


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SVBAblationPolicy:
    mode: str
    enabled: bool
    use_boundary_band: bool
    use_cbm_points: bool
    use_reliability: bool
    use_prompt_expert: bool
    use_conformal: bool
    full_image_fusion: bool


class SAMVerifiedBoundaryPseudoLabelRefinement(nn.Module):
    """SVB-PLR orchestrator for SAM-verified boundary pseudo labels.

    Shape:
        images: [B, 3, H_img, W_img]
        teacher_prob: [B, 1, H, W]
        return: p_ref [B,1,H,W], conf_ref [B,1,H,W], sam_aux dict
    """

    DEFAULTS = {
        "use_svb_plr": False,
        "use_sam_refine_unlabeled": False,
        "sam_start_epoch": 16,
        "sam_refine_interval": 1,
        "use_prompt_expert": True,
        "sam_use_conformal": True,
        "use_sam_cache": False,
        "use_svb_output_cache": False,
        "sam_cache_dir": "./cache/sam_refined_pseudo",
        "cache_refined_masks": True,
        "cache_prompt_debug": True,
        "vis_sam_refinement": True,
        "vis_sam_refine_interval": 200,
        "vis_sam_refine_max_samples": 2,
        "sam_refine_vis_dir": "outputs/svb_plr_visualization",
        "svb_ablation_mode": "full",
        "sam_beta_max": 0.75,
    }

    def __init__(self, cfg, device=None, logger=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self._sam_backend_init_error: Optional[str] = None
        self._current_step: Optional[int] = None
        self.ablation_policy = self._build_ablation_policy()
        self.ablation_mode = self.ablation_policy.mode
        self._conformal_enabled = bool(self.ablation_policy.use_conformal and self._cfg_bool("sam_use_conformal"))
        self._log_info("[SVB-PLR] ablation_mode={}".format(self.ablation_mode))

        self.sam_backend = self._build_sam_backend(cfg, device, logger)
        self.prompt_generator = CBMPromptGenerator(cfg)
        selector_cfg = self._config_overlay(use_prompt_expert=True)
        reliability_cfg = self._config_overlay(sam_use_conformal=self._conformal_enabled)
        calibrator_cfg = self._config_overlay(sam_use_conformal=True)
        self.prompt_selector = PromptExpertSelector(selector_cfg) if self.ablation_policy.use_prompt_expert else None
        self.reliability_filter = SAMCBMReliabilityFilter(reliability_cfg)
        self.calibrator = ConformalSAMCalibrator(calibrator_cfg) if self._conformal_enabled else None
        self.cache = SVBPLRCache(cfg, logger=logger) if self._cfg_bool("use_svb_output_cache") else None
        self.visualizer = SamRefineVisualizer(cfg, logger=logger) if self._cfg_bool("vis_sam_refinement") else None

    @torch.no_grad()
    def refine(
        self,
        images: torch.Tensor,
        teacher_prob: Optional[torch.Tensor],
        retrieval_aux,
        image_ids=None,
        epoch=None,
        step=None,
        student_pred=None,
    ):
        """Run full SVB-PLR refinement on unlabeled teacher probabilities."""
        self._current_step = self._normalize_step(step)
        if teacher_prob is None:
            return None, None, {"used_sam": False, "fallback_reason": "teacher_prob_none"}

        try:
            p_t = self._as_teacher_prob(teacher_prob)
            enabled, reason = self._enabled_for_epoch(epoch)
            if not enabled:
                return self._fallback_output(p_t, reason)

            backend_name = self._backend_name()
            cache_hit = (
                self.cache.read(
                    image_ids,
                    p_t,
                    epoch=epoch,
                    backend=backend_name,
                    prompt_mode=self.ablation_mode,
                )
                if self.cache is not None
                else None
            )
            if cache_hit is not None:
                p_ref, conf_ref, cached_aux = cache_hit
                cached_aux["cache_hit"] = True
                cached_aux["svb_ablation_mode"] = self.ablation_mode
                return p_ref.detach(), conf_ref.detach(), cached_aux

            if not self._interval_allows_refine(epoch):
                return self._fallback_output(p_t, "sam_refine_interval_skipped")
            if self.sam_backend is None:
                reason = "sam_backend_unavailable"
                if self._sam_backend_init_error:
                    reason = "{}: {}".format(reason, self._sam_backend_init_error)
                raise SAMInferenceError(
                    reason,
                    epoch=epoch,
                    step=step,
                    sample_indices=list(range(p_t.size(0))),
                )

            prompt_pack = self.prompt_generator(p_t, retrieval_aux or {})
            prompt_pack = self._apply_ablation_prompt(prompt_pack, p_t)
            if self.ablation_policy.use_prompt_expert and self.prompt_selector is not None:
                sam_mask, sam_score, selector_aux, backend_aux = self._run_prompt_experts(
                    images, p_t, prompt_pack, epoch=epoch, step=step
                )
            else:
                sam_mask, sam_score, selector_aux, backend_aux = self._run_default_prompt(
                    images, p_t, prompt_pack, epoch=epoch, step=step
                )

            if self.ablation_policy.use_reliability:
                p_ref, conf_ref, filter_aux = self.reliability_filter(
                    p_t,
                    sam_mask,
                    sam_score,
                    prompt_pack,
                    retrieval_aux=retrieval_aux,
                    calibrator=self.calibrator if self._conformal_enabled else None,
                    epoch=epoch,
                )
            else:
                p_ref, conf_ref, filter_aux = self._simple_fusion(p_t, sam_mask, prompt_pack)

            sam_aux = {
                "used_sam": True,
                "svb_ablation_mode": self.ablation_mode,
                "sam_mask": sam_mask.detach(),
                "sam_score": None if sam_score is None else sam_score.detach(),
                "prompt_pack": prompt_pack,
                "selector_aux": selector_aux,
                "backend_aux": backend_aux,
                "cache_hit": False,
            }
            sam_aux.update(filter_aux)

            if self.cache is not None:
                self.cache.write(
                    image_ids,
                    p_ref,
                    conf_ref,
                    sam_aux,
                    teacher_prob=p_t,
                    epoch=epoch,
                    backend=backend_name,
                    prompt_mode=self.ablation_mode,
                )
            if self.visualizer is not None:
                self.visualizer.save(
                    images=images,
                    teacher_prob=p_t,
                    sam_mask=sam_mask,
                    p_ref=p_ref,
                    conf_ref=conf_ref,
                    sam_aux=sam_aux,
                    image_ids=image_ids,
                    epoch=epoch,
                    step=step,
                    student_pred=student_pred,
                )
            return p_ref.detach(), conf_ref.detach(), sam_aux
        except SAMInferenceError as exc:
            if exc.epoch is None and exc.step is None:
                raise SAMInferenceError(
                    exc.message,
                    epoch=epoch,
                    step=step,
                    sample_indices=exc.sample_indices,
                    failures=exc.failures,
                ) from exc
            raise
        except Exception as exc:
            reason = "svb_plr_exception: {}".format(exc)
            self._warn("[SVB-PLR] refinement fallback: {}".format(reason))
            return self._fallback_output(self._as_teacher_prob(teacher_prob), reason)

    def _run_prompt_experts(
        self,
        images: torch.Tensor,
        teacher_prob: torch.Tensor,
        prompt_pack: Dict[str, Any],
        epoch=None,
        step=None,
    ):
        expert_prompts = self.prompt_selector.build_expert_prompts(prompt_pack)
        sam_candidates: List[Dict[str, Any]] = []
        backend_aux: Dict[str, Any] = {}
        candidate_masks: List[torch.Tensor] = []

        for prompt in expert_prompts:
            expert = str(prompt.get("expert", "unknown"))
            sam_out = self.sam_backend.predict(
                images,
                teacher_prob,
                prompt_pack=prompt,
                epoch=epoch,
                step=step,
            )
            masks = self._as_bkhw_masks(sam_out.get("masks"), teacher_prob)
            scores = self._as_bk_scores(sam_out.get("scores"), masks, teacher_prob) if masks is not None else None
            valid_candidates = (
                PromptExpertSelector._as_bk_valid(sam_out.get("valid_candidates"), masks, teacher_prob)
                if masks is not None
                else None
            )
            expert_backend_aux = sam_out.get("backend_aux", {})
            candidate = {
                "expert": expert,
                "masks": masks,
                "scores": scores,
                "valid_candidates": valid_candidates,
                "logits": sam_out.get("logits"),
                "backend_aux": expert_backend_aux,
            }
            sam_candidates.append(candidate)
            if masks is not None:
                candidate_masks.append(masks)
            backend_aux[expert] = expert_backend_aux

        if candidate_masks:
            prompt_pack["candidate_masks"] = torch.cat(candidate_masks, dim=1).detach()
        sam_mask, sam_score, selector_aux = self.prompt_selector.select(sam_candidates, teacher_prob, prompt_pack)
        return sam_mask, sam_score, selector_aux, backend_aux

    def _run_default_prompt(
        self,
        images: torch.Tensor,
        teacher_prob: torch.Tensor,
        prompt_pack: Dict[str, Any],
        epoch=None,
        step=None,
    ):
        sam_out = self.sam_backend.predict(
            images,
            teacher_prob,
            prompt_pack=prompt_pack,
            epoch=epoch,
            step=step,
        )
        sam_mask, sam_score, selector_aux = self._default_select(sam_out, teacher_prob)
        masks = self._as_bkhw_masks(sam_out.get("masks"), teacher_prob)
        if masks is not None:
            prompt_pack["candidate_masks"] = masks.detach()
        return sam_mask, sam_score, selector_aux, sam_out.get("backend_aux", {})

    def _default_select(self, sam_out: Dict[str, Any], teacher_prob: torch.Tensor):
        masks = self._as_bkhw_masks(sam_out.get("masks"), teacher_prob)
        if masks is None or masks.numel() == 0 or masks.size(1) == 0:
            raise SAMInferenceError(
                "Default SAM prompt returned no candidate masks",
                sample_indices=list(range(teacher_prob.size(0))),
                failures=[sam_out.get("backend_aux", {})],
            )

        scores = self._as_bk_scores(sam_out.get("scores"), masks, teacher_prob)
        valid_candidates = PromptExpertSelector._as_bk_valid(
            sam_out.get("valid_candidates"), masks, teacher_prob
        )
        invalid_samples = (~valid_candidates.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
        if invalid_samples:
            raise SAMInferenceError(
                "Default SAM prompt has no valid candidates",
                sample_indices=invalid_samples,
                failures=[sam_out.get("backend_aux", {})],
            )
        if scores is None:
            ranked_scores = teacher_prob.new_zeros(valid_candidates.shape)
        else:
            ranked_scores = scores
        ranked_scores = ranked_scores.masked_fill(~valid_candidates, float("-inf"))
        best_idx = ranked_scores.argmax(dim=1)
        sam_score = ranked_scores[torch.arange(ranked_scores.size(0), device=ranked_scores.device), best_idx]
        batch_idx = torch.arange(teacher_prob.size(0), device=teacher_prob.device)
        sam_mask = masks[batch_idx, best_idx].unsqueeze(1).clamp(0.0, 1.0)
        selector_aux = {
            "use_prompt_expert": False,
            "used_fallback": False,
            "best_expert": ["default"] * teacher_prob.size(0),
            "best_candidate_index": best_idx.detach(),
            "expert_scores": {"default": scores.detach() if scores is not None else None},
            "expert_components": {},
            "selected_logits": None,
            "valid_candidates": valid_candidates.detach(),
            "valid_candidate_ratio": valid_candidates.float().mean().detach(),
        }
        return sam_mask, sam_score.detach(), selector_aux

    def _apply_ablation_prompt(self, prompt_pack: Dict[str, Any], teacher_prob: torch.Tensor) -> Dict[str, Any]:
        prompt = dict(prompt_pack or {})
        policy = self.ablation_policy

        if policy.full_image_fusion:
            prompt["refine_band"] = torch.ones_like(teacher_prob)
            prompt["mask_prompt"] = teacher_prob.detach().clamp(0.0, 1.0)
            prompt["mask_inputs"] = prompt["mask_prompt"]

        if not policy.use_cbm_points:
            prompt["pos_points"] = self._empty_points(teacher_prob)
            prompt["neg_points"] = self._empty_points(teacher_prob)
            prompt["boundary_points"] = self._empty_points(teacher_prob)
            prompt["point_coords"] = self._empty_points(teacher_prob)
            prompt["point_labels"] = self._empty_labels(teacher_prob)

        if self.ablation_mode == "teacher_sam_full":
            prompt["boxes"] = self._empty_boxes(teacher_prob)
            prompt["refine_band"] = torch.ones_like(teacher_prob)
            prompt["mask_prompt"] = teacher_prob.detach().clamp(0.0, 1.0)
            prompt["mask_inputs"] = prompt["mask_prompt"]

        prompt["svb_ablation_mode"] = self.ablation_mode
        return prompt

    def _simple_fusion(self, teacher_prob: torch.Tensor, sam_mask: torch.Tensor, prompt_pack: Dict[str, Any]):
        p_t = self._as_teacher_prob(teacher_prob)
        masks = self._as_bkhw_masks(sam_mask, p_t)
        sam = p_t if masks is None or masks.numel() == 0 else masks[:, 0:1].clamp(0.0, 1.0)
        if self.ablation_policy.full_image_fusion:
            refine_band = torch.ones_like(p_t)
        elif self.ablation_policy.use_boundary_band:
            refine_band = self._map_like(prompt_pack.get("refine_band"), p_t, fallback=p_t.new_ones(p_t.shape), mode="nearest")
        else:
            refine_band = torch.ones_like(p_t)

        r_sam = torch.ones_like(p_t)
        beta = (self._cfg_float("sam_beta_max") * refine_band).clamp(0.0, 1.0)
        p_ref = ((1.0 - beta) * p_t + beta * sam).clamp(0.0, 1.0)
        conf_ref = binary_reliability(p_t).detach()
        filter_aux = {
            "R_teacher": (1.0 - (sam - p_t).abs()).clamp(0.0, 1.0).detach(),
            "R_cbm": r_sam.detach(),
            "R_stability": r_sam.detach(),
            "R_conformal": p_t.new_zeros(p_t.shape),
            "R_sam": r_sam.detach(),
            "beta": beta.detach(),
            "refine_band": refine_band.detach(),
            "fg_support": p_t.detach(),
            "bg_support": (1.0 - p_t).detach(),
            "lambda_epoch": 1.0,
            "used_conformal": False,
            "stability_source": "ablation_simple_fusion",
        }
        return p_ref.detach(), conf_ref.detach(), filter_aux

    def _enabled_for_epoch(self, epoch) -> Tuple[bool, str]:
        if not self._cfg_bool("use_svb_plr"):
            return False, "use_svb_plr_false"
        if not self.ablation_policy.enabled:
            return False, "svb_ablation_off"
        if not self._cfg_bool("use_sam_refine_unlabeled"):
            return False, "use_sam_refine_unlabeled_false"
        if epoch is None:
            return False, "epoch_none"
        try:
            if int(epoch) < self._cfg_int("sam_start_epoch"):
                return False, "before_sam_start_epoch"
        except (TypeError, ValueError):
            return False, "invalid_epoch"
        return True, "enabled"

    def _interval_allows_refine(self, epoch) -> bool:
        interval = max(1, self._cfg_int("sam_refine_interval"))
        try:
            return int(epoch) % interval == 0
        except (TypeError, ValueError):
            return False

    def _fallback_output(self, teacher_prob: torch.Tensor, reason: str):
        p_t = self._as_teacher_prob(teacher_prob)
        return (
            p_t.detach(),
            binary_reliability(p_t).detach(),
            {
                "used_sam": False,
                "fallback_reason": reason,
                "cache_hit": False,
                "svb_ablation_mode": self.ablation_mode,
            },
        )

    def _build_sam_backend(self, cfg, device, logger):
        try:
            backend = ExistingSAMBackendAdapter(cfg, device=device, logger=logger)
            backend.eval()
            for param in backend.parameters():
                param.requires_grad = False
            return backend
        except Exception as exc:
            self._sam_backend_init_error = str(exc)
            self._warn("[SVB-PLR] SAM backend initialization failed: {}".format(exc))
            return None

    @staticmethod
    def _as_teacher_prob(teacher_prob: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(teacher_prob):
            raise TypeError("teacher_prob must be a torch.Tensor")
        if teacher_prob.dim() != 4 or teacher_prob.size(1) != 1:
            raise ValueError("teacher_prob must have shape [B,1,H,W]")
        return teacher_prob.detach().clamp(0.0, 1.0)

    @staticmethod
    def _as_bkhw_masks(value, ref: torch.Tensor) -> Optional[torch.Tensor]:
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
    def _as_bk_scores(value, masks: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
        if masks is None:
            return None
        batch_size, num_candidates = masks.shape[:2]
        if not torch.is_tensor(value):
            return ref.new_zeros((batch_size, num_candidates))
        scores = value.detach().to(device=ref.device, dtype=ref.dtype)
        if scores.numel() == 0:
            return ref.new_zeros((batch_size, num_candidates))
        if scores.dim() == 0:
            scores = scores.reshape(1, 1).expand(batch_size, num_candidates)
        elif scores.dim() == 1:
            if scores.numel() == batch_size:
                scores = scores.reshape(batch_size, 1).expand(-1, num_candidates)
            elif scores.numel() == num_candidates:
                scores = scores.reshape(1, num_candidates).expand(batch_size, -1)
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
            scores = torch.cat((scores, ref.new_zeros((batch_size, num_candidates - scores.size(1)))), dim=1)
        return scores[:, :num_candidates].clamp(0.0, 1.0)

    @staticmethod
    def _empty_points(ref: torch.Tensor) -> List[torch.Tensor]:
        return [ref.new_zeros((0, 2), dtype=torch.float32) for _ in range(ref.size(0))]

    @staticmethod
    def _empty_labels(ref: torch.Tensor) -> List[torch.Tensor]:
        return [torch.zeros((0,), device=ref.device, dtype=torch.int64) for _ in range(ref.size(0))]

    @staticmethod
    def _empty_boxes(ref: torch.Tensor) -> List[torch.Tensor]:
        return [ref.new_zeros((0, 4), dtype=torch.float32) for _ in range(ref.size(0))]

    @staticmethod
    def _map_like(value, ref: torch.Tensor, fallback: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        if not torch.is_tensor(value):
            return fallback.detach().to(device=ref.device, dtype=ref.dtype)
        x = value.detach().to(device=ref.device, dtype=ref.dtype)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.size(1) != 1:
            x = x[:, :1]
        if x.dim() != 4:
            return fallback.detach().to(device=ref.device, dtype=ref.dtype)
        if x.size(0) != ref.size(0):
            if x.size(0) == 1:
                x = x.expand(ref.size(0), -1, -1, -1)
            else:
                return fallback.detach().to(device=ref.device, dtype=ref.dtype)
        if tuple(x.shape[-2:]) != tuple(ref.shape[-2:]):
            x = resize_like(x, ref, mode=mode)
        return x.clamp(0.0, 1.0)

    def _backend_name(self) -> str:
        if self.sam_backend is not None and hasattr(self.sam_backend, "backend_name"):
            return str(getattr(self.sam_backend, "backend_name"))
        return str(getattr(self.cfg, "sam_pseudo_backend", "sam1"))

    def _build_ablation_policy(self) -> SVBAblationPolicy:
        mode = self._normalize_ablation_mode(self._cfg("svb_ablation_mode"))
        if mode == "off":
            return SVBAblationPolicy(mode, False, False, False, False, False, False, False)
        if mode == "teacher_sam_full":
            return SVBAblationPolicy(mode, True, False, False, False, False, False, True)
        if mode == "boundary_only":
            return SVBAblationPolicy(mode, True, True, False, False, False, False, False)
        if mode == "cbm_points":
            return SVBAblationPolicy(mode, True, True, True, False, False, False, False)
        if mode == "reliability":
            return SVBAblationPolicy(mode, True, True, True, True, False, False, False)
        if mode == "prompt_expert":
            return SVBAblationPolicy(mode, True, True, True, True, True, False, False)
        if mode == "conformal":
            return SVBAblationPolicy(mode, True, True, True, True, False, True, False)
        return SVBAblationPolicy("full", True, True, True, True, True, True, False)

    def _normalize_ablation_mode(self, value) -> str:
        mode = str(value or "full").strip().lower()
        valid = {
            "off",
            "teacher_sam_full",
            "boundary_only",
            "cbm_points",
            "reliability",
            "prompt_expert",
            "conformal",
            "full",
        }
        if mode not in valid:
            self._warn("[SVB-PLR] unknown svb_ablation_mode='{}'; falling back to full.".format(mode))
            return "full"
        return mode

    def _config_overlay(self, **overrides):
        return _ConfigOverlay(self.cfg, overrides)

    def _cfg(self, name: str) -> Any:
        if self.cfg is not None and hasattr(self.cfg, name):
            return getattr(self.cfg, name)
        return self.DEFAULTS[name]

    def _cfg_bool(self, name: str) -> bool:
        return bool(self._cfg(name))

    def _cfg_int(self, name: str) -> int:
        return int(self._cfg(name))

    def _cfg_float(self, name: str) -> float:
        return float(self._cfg(name))

    def _should_log(self) -> bool:
        return should_log(self.cfg, self._current_step)

    @staticmethod
    def _normalize_step(step) -> Optional[int]:
        try:
            return int(step)
        except (TypeError, ValueError):
            return None

    def _log_info(self, message: str) -> None:
        if not self._should_log():
            return
        if self.logger is not None:
            method = getattr(self.logger, "info", None) or getattr(self.logger, "key_info", None)
            if callable(method):
                method(message)
                return
        LOGGER.info(message)

    def _warn(self, message: str) -> None:
        if not self._should_log():
            return
        if self.logger is not None:
            for method_name in ("warn_info", "warning", "warn", "info"):
                method = getattr(self.logger, method_name, None)
                if callable(method):
                    method(message)
                    return
        LOGGER.warning(message)

class _ConfigOverlay:
    def __init__(self, base, overrides: Dict[str, Any]) -> None:
        self._base = base
        self._overrides = dict(overrides)

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        if self._base is not None:
            return getattr(self._base, name)
        raise AttributeError(name)


__all__ = [
    "SAMVerifiedBoundaryPseudoLabelRefinement",
    "SVBPLRCache",
    "SamRefineVisualizer",
]
