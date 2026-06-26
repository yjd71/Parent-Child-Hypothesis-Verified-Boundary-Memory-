from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    from .svb_utils import binary_reliability, resize_like
except ImportError:
    from SAM.SAM_refinement.svb_utils import binary_reliability, resize_like


class SAMCBMReliabilityFilter(nn.Module):
    """Fuse teacher probability and SAM mask with SAM-CBM reliability.

    Shape:
        teacher_prob: [B, 1, H, W]
        sam_mask: [B, 1, H, W] or resize-compatible mask tensor
        sam_score: optional [B], [B, 1], or [B, K]
        return: p_ref [B, 1, H, W], conf_ref [B, 1, H, W], filter_aux
    """

    DEFAULTS = {
        "sam_teacher_agree_weight": 0.25,
        "sam_cbm_agree_weight": 0.45,
        "sam_stability_weight": 0.20,
        "sam_conformal_weight": 0.10,
        "sam_use_conformal": True,
        "sam_beta_max": 0.75,
        "sam_lambda_start": 1.0,
        "sam_lambda_end": 0.3,
        "sam_lambda_decay": True,
        "sam_min_reliability": 0.3,
        "sam_start_epoch": 16,
    }

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(
        self,
        teacher_prob: torch.Tensor,
        sam_mask: torch.Tensor,
        sam_score=None,
        prompt_pack: Optional[Mapping[str, Any]] = None,
        retrieval_aux=None,
        calibrator=None,
        epoch=None,
    ):
        """Apply reliability-gated soft pseudo-label refinement.

        Shape:
            teacher_prob: [B, 1, H, W]
            sam_mask: [B, 1, H, W]
            return: p_ref, conf_ref, filter_aux, all detached
        """
        p_t = self._as_teacher_prob(teacher_prob)
        sam = self._as_b1hw(sam_mask, p_t, "sam_mask").clamp(0.0, 1.0)
        pack = prompt_pack if isinstance(prompt_pack, Mapping) else {}
        evidence = pack.get("evidence", {}) if isinstance(pack.get("evidence", {}), Mapping) else {}

        refine_band = self._optional_map(pack.get("refine_band"), p_t, p_t.new_zeros(p_t.shape), mode="nearest").clamp(0.0, 1.0)
        s_fg = self._evidence_map(evidence, retrieval_aux, "S_fg_up", p_t, p_t)
        s_bg = self._evidence_map(evidence, retrieval_aux, "S_bg_up", p_t, 1.0 - p_t)
        m_bd = self._evidence_map(evidence, retrieval_aux, "M_bd_up", p_t, p_t.new_zeros(p_t.shape))

        r_teacher = (1.0 - (sam - p_t).abs()).clamp(0.0, 1.0)
        fg_support = torch.sigmoid(s_fg - s_bg).clamp(0.0, 1.0)
        bg_support = torch.sigmoid(s_bg - s_fg).clamp(0.0, 1.0)
        r_cbm = (sam * fg_support + (1.0 - sam) * bg_support).clamp(0.0, 1.0)
        r_stability, stability_source = self._stability_map(sam_score, pack, p_t)
        r_conformal, used_conformal = self._conformal_map(
            calibrator=calibrator,
            teacher_prob=p_t,
            sam_mask=sam,
            prompt_pack=pack,
            retrieval_aux=retrieval_aux,
            epoch=epoch,
        )

        r_sam = (
            self._cfg_float("sam_teacher_agree_weight") * r_teacher
            + self._cfg_float("sam_cbm_agree_weight") * r_cbm
            + self._cfg_float("sam_stability_weight") * r_stability
            + self._cfg_float("sam_conformal_weight") * r_conformal
        ).clamp(0.0, 1.0)

        lambda_epoch = self._lambda_epoch(epoch)
        beta = (lambda_epoch * self._cfg_float("sam_beta_max") * r_sam * refine_band).clamp(0.0, 1.0)
        p_ref = ((1.0 - beta) * p_t + beta * sam).clamp(0.0, 1.0)

        conf_teacher = binary_reliability(p_t)
        conf_ref = conf_teacher * (1.0 - refine_band) + r_sam * refine_band
        conf_ref = conf_ref.clamp(min=self._cfg_float("sam_min_reliability"), max=1.0)

        filter_aux = {
            "R_teacher": r_teacher.detach(),
            "R_cbm": r_cbm.detach(),
            "R_stability": r_stability.detach(),
            "R_conformal": r_conformal.detach(),
            "R_sam": r_sam.detach(),
            "beta": beta.detach(),
            "refine_band": refine_band.detach(),
            "fg_support": fg_support.detach(),
            "bg_support": bg_support.detach(),
            "lambda_epoch": float(lambda_epoch),
            "M_bd_up": m_bd.detach(),
            "used_conformal": bool(used_conformal),
            "stability_source": stability_source,
        }
        return p_ref.detach(), conf_ref.detach(), filter_aux

    def _stability_map(
        self,
        sam_score,
        prompt_pack: Mapping[str, Any],
        ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, str]:
        score_map = self._score_to_map(sam_score, ref)
        if score_map is not None:
            return score_map.clamp(0.0, 1.0), "sam_score"

        candidate_masks = prompt_pack.get("candidate_masks") if isinstance(prompt_pack, Mapping) else None
        if torch.is_tensor(candidate_masks):
            candidates = self._as_bkhw(candidate_masks, ref)
            if candidates is not None and candidates.size(1) > 1:
                stability = 1.0 - candidates.float().var(dim=1, keepdim=True, unbiased=False)
                return stability.clamp(0.0, 1.0).to(device=ref.device, dtype=ref.dtype), "candidate_variance"

        return ref.new_full(ref.shape, 0.5), "constant"

    def _conformal_map(
        self,
        calibrator,
        teacher_prob: torch.Tensor,
        sam_mask: torch.Tensor,
        prompt_pack: Mapping[str, Any],
        retrieval_aux,
        epoch,
    ) -> Tuple[torch.Tensor, bool]:
        if not self._cfg_bool("sam_use_conformal") or calibrator is None:
            return teacher_prob.new_zeros(teacher_prob.shape), False
        estimate = getattr(calibrator, "estimate_reliability", None)
        if not callable(estimate):
            return teacher_prob.new_zeros(teacher_prob.shape), False
        try:
            value = estimate(
                teacher_prob=teacher_prob,
                sam_mask=sam_mask,
                prompt_pack=prompt_pack,
                retrieval_aux=retrieval_aux,
                epoch=epoch,
            )
        except TypeError:
            try:
                value = estimate(teacher_prob, sam_mask, prompt_pack)
            except Exception:
                return teacher_prob.new_zeros(teacher_prob.shape), False
        except Exception:
            return teacher_prob.new_zeros(teacher_prob.shape), False
        conformal = self._optional_map(value, teacher_prob, teacher_prob.new_zeros(teacher_prob.shape), mode="bilinear")
        return conformal.clamp(0.0, 1.0), True

    def _lambda_epoch(self, epoch) -> float:
        start = self._cfg_float("sam_lambda_start")
        end = self._cfg_float("sam_lambda_end")
        if not self._cfg_bool("sam_lambda_decay") or epoch is None:
            return start

        try:
            epoch_value = float(epoch)
        except (TypeError, ValueError):
            return start

        sam_start_epoch = self._cfg_float("sam_start_epoch")
        if epoch_value < sam_start_epoch:
            return start

        # Hold the start value until SVB-PLR becomes active, then decay over the
        # remaining training epochs so the ramp is anchored at sam_start_epoch.
        total_epochs = getattr(self.cfg, "tot_epochs", None)
        try:
            total_epochs = float(total_epochs)
        except (TypeError, ValueError):
            total_epochs = sam_start_epoch
        decay_span = max(1.0, total_epochs - sam_start_epoch)
        progress = max(0.0, min(1.0, (epoch_value - sam_start_epoch) / decay_span))

        if start >= end:
            return max(end, start - progress * (start - end))
        return min(end, start + progress * (end - start))

    def _evidence_map(
        self,
        evidence: Mapping[str, Any],
        retrieval_aux,
        key: str,
        ref: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        value = evidence.get(key) if isinstance(evidence, Mapping) else None
        if value is None and isinstance(retrieval_aux, Mapping):
            value = retrieval_aux.get(key)
        return self._optional_map(value, ref, fallback, mode="bilinear")

    def _optional_map(self, value, ref: torch.Tensor, fallback: torch.Tensor, mode: str) -> torch.Tensor:
        if value is None or not torch.is_tensor(value):
            return fallback.detach().to(device=ref.device, dtype=ref.dtype)
        try:
            return self._as_b1hw(value, ref, "optional_map", mode=mode)
        except (TypeError, ValueError):
            return fallback.detach().to(device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _as_teacher_prob(teacher_prob: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(teacher_prob):
            raise TypeError("teacher_prob must be a torch.Tensor")
        if teacher_prob.dim() != 4 or teacher_prob.size(1) != 1:
            raise ValueError("teacher_prob must have shape [B,1,H,W]")
        return teacher_prob.detach().clamp(0.0, 1.0)

    @staticmethod
    def _as_b1hw(value: torch.Tensor, ref: torch.Tensor, name: str, mode: str = "bilinear") -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError("{} must be a torch.Tensor".format(name))
        x = value.detach().to(device=ref.device, dtype=ref.dtype)
        if x.numel() == 0:
            raise ValueError("{} must be non-empty".format(name))
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            if x.size(1) != 1:
                x = x[:, :1]
        else:
            raise ValueError("{} must have shape [H,W], [B,H,W], or [B,C,H,W]".format(name))
        if x.size(0) != ref.size(0):
            if x.size(0) == 1:
                x = x.expand(ref.size(0), -1, -1, -1)
            else:
                raise ValueError("{} batch size must match teacher_prob".format(name))
        if tuple(x.shape[-2:]) != tuple(ref.shape[-2:]):
            x = resize_like(x, ref, mode=mode)
        return x.to(device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _as_bkhw(value: torch.Tensor, ref: torch.Tensor) -> Optional[torch.Tensor]:
        if not torch.is_tensor(value):
            return None
        x = value.detach().to(device=ref.device, dtype=ref.dtype)
        if x.numel() == 0:
            return None
        if x.dim() == 2:
            x = x.reshape(1, 1, *x.shape[-2:])
        elif x.dim() == 3:
            if x.size(0) == ref.size(0):
                x = x.unsqueeze(1)
            else:
                x = x.unsqueeze(0)
        elif x.dim() != 4:
            return None
        if x.size(0) != ref.size(0):
            if x.size(0) == 1:
                x = x.expand(ref.size(0), -1, -1, -1)
            else:
                return None
        if tuple(x.shape[-2:]) != tuple(ref.shape[-2:]):
            x = resize_like(x, ref, mode="bilinear")
        return x.clamp(0.0, 1.0)

    @staticmethod
    def _score_to_map(score, ref: torch.Tensor) -> Optional[torch.Tensor]:
        if score is None or not torch.is_tensor(score):
            return None
        x = score.detach().to(device=ref.device, dtype=ref.dtype)
        if x.numel() == 0:
            return None
        if x.dim() == 0:
            x = x.reshape(1, 1).expand(ref.size(0), 1)
        elif x.dim() == 1:
            if x.numel() == ref.size(0):
                x = x.reshape(ref.size(0), 1)
            else:
                x = x.reshape(1, -1)[:, :1].expand(ref.size(0), 1)
        else:
            x = x.reshape(x.size(0), -1)[:, :1]
            if x.size(0) != ref.size(0):
                if x.size(0) == 1:
                    x = x.expand(ref.size(0), -1)
                else:
                    return None
        return x.reshape(ref.size(0), 1, 1, 1).expand_as(ref)

    def _cfg(self, name: str) -> Any:
        if self.cfg is not None and hasattr(self.cfg, name):
            return getattr(self.cfg, name)
        return self.DEFAULTS[name]

    def _cfg_bool(self, name: str) -> bool:
        return bool(self._cfg(name))

    def _cfg_float(self, name: str) -> float:
        return float(self._cfg(name))


__all__ = ["SAMCBMReliabilityFilter"]
