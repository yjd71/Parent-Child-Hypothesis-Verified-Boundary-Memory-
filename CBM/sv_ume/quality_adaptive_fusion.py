from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Number
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from CBM.core.tensor_ops import entropy_uncertainty


DEFAULT_SIM_WEIGHT = 1.0
DEFAULT_CONS_WEIGHT = 1.0
DEFAULT_REL_WEIGHT = 1.0
DEFAULT_UNC_WEIGHT = 0.5


class QualityAdaptiveSourceFusion(nn.Module):
    """Symmetrically fuse labeled and lagged-unlabeled retrieval results."""

    def __init__(self, cfg, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.cfg = cfg
        self.eps = float(eps)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be finite and positive")

        self.sim_weight = self._non_negative_weight(
            getattr(cfg, "fusion_score_sim_weight", DEFAULT_SIM_WEIGHT),
            "fusion_score_sim_weight",
        )
        self.cons_weight = self._non_negative_weight(
            getattr(cfg, "fusion_score_cons_weight", DEFAULT_CONS_WEIGHT),
            "fusion_score_cons_weight",
        )
        self.rel_weight = self._non_negative_weight(
            getattr(cfg, "fusion_score_rel_weight", DEFAULT_REL_WEIGHT),
            "fusion_score_rel_weight",
        )
        self.unc_weight = self._non_negative_weight(
            getattr(cfg, "fusion_score_unc_weight", DEFAULT_UNC_WEIGHT),
            "fusion_score_unc_weight",
        )
        self.use_evidence_fusion = bool(
            getattr(cfg, "use_aux_evidence_fusion", True)
        )
        self.use_feature_fusion = bool(
            getattr(cfg, "use_aux_feature_fusion", True)
        )

        if not math.isclose(
            float(getattr(cfg, "gamma_max_final", 1.0)),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise ValueError("gamma_max_final must be 1.0")
        if bool(getattr(cfg, "use_aux_source_penalty", False)):
            raise ValueError("use_aux_source_penalty must be False")
        if not bool(getattr(cfg, "allow_aux_dominate", True)):
            raise ValueError("allow_aux_dominate must be True")

    def compute_score(
        self,
        retrieval: Mapping[str, Any],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return a float32 per-pixel quality score with shape ``[B,1,H,W]``."""
        self._validate_retrieval(retrieval, "retrieval")
        self._validate_reference(reference)
        sim = self._as_quality_map(
            self._required(retrieval, ("sim_mean",), "sim_mean"),
            reference,
            "sim_mean",
        )
        consistency = self._as_quality_map(
            self._required(
                retrieval,
                ("topk_consistency",),
                "topk_consistency",
            ),
            reference,
            "topk_consistency",
        )
        reliability = self._as_quality_map(
            self._required(
                retrieval,
                ("memory_reliability",),
                "memory_reliability",
            ),
            reference,
            "memory_reliability",
        )
        uncertainty = self._as_quality_map(
            self._required(
                retrieval,
                ("U_map", "uncertainty", "U"),
                "U_map/uncertainty",
            ),
            reference,
            "uncertainty",
        )
        score = (
            self.sim_weight * sim
            + self.cons_weight * consistency
            + self.rel_weight * reliability
            - self.unc_weight * uncertainty
        )
        self._ensure_finite(score, "quality score")
        return score

    def forward(
        self,
        ret_l: Mapping[str, Any],
        ret_u: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._validate_retrieval(ret_l, "ret_l")
        y_l = self._feature_map(ret_l, ("Y_map", "Y"), "ret_l Y")
        r_l = self._feature_map(ret_l, ("R_map", "R"), "ret_l R")
        self._validate_y_r_pair(y_l, r_l, "ret_l")
        score_l = self.compute_score(ret_l, y_l)
        valid_l = self._valid_map(ret_l, y_l, "ret_l.valid_map")

        y_u = None
        r_u = None
        if ret_u is None:
            score_u = torch.zeros_like(score_l)
            valid_u = torch.zeros_like(valid_l)
            w_l = torch.ones_like(score_l)
            w_u = torch.zeros_like(score_l)
        else:
            self._validate_retrieval(ret_u, "ret_u")
            y_u = self._feature_map(ret_u, ("Y_map", "Y"), "ret_u Y")
            r_u = self._feature_map(ret_u, ("R_map", "R"), "ret_u R")
            self._validate_y_r_pair(y_u, r_u, "ret_u")
            self._validate_source_shapes(y_l, y_u, r_l, r_u)
            score_u = self.compute_score(ret_u, y_l)
            valid_u = self._valid_map(ret_u, y_l, "ret_u.valid_map")
            w_l, w_u = self._source_weights(
                score_l,
                score_u,
                valid_l,
                valid_u,
            )

        weight_l_y = w_l.to(dtype=y_l.dtype)
        weight_u_y = w_u.to(dtype=y_l.dtype)
        weight_l_r = w_l.to(dtype=r_l.dtype)
        weight_u_r = w_u.to(dtype=r_l.dtype)

        if y_u is not None and self.use_evidence_fusion:
            y_fused = weight_l_y * y_l + weight_u_y * y_u
        else:
            y_fused = y_l
        if r_u is not None and self.use_feature_fusion:
            r_fused = weight_l_r * r_l + weight_u_r * r_u
        else:
            r_fused = r_l

        uncertainty = entropy_uncertainty(
            y_fused[:, :4].float(),
            eps=self.eps,
        ).unsqueeze(1).to(dtype=y_fused.dtype)
        valid_map = (valid_l | valid_u).to(dtype=y_l.dtype)
        source_entropy = self._source_entropy(w_l, w_u)

        for name, tensor in (
            ("Y_fused", y_fused),
            ("R_fused", r_fused),
            ("U_fused", uncertainty),
            ("w_l_map", w_l),
            ("w_u_map", w_u),
            ("score_l", score_l),
            ("score_u", score_u),
            ("source_entropy", source_entropy),
        ):
            self._ensure_finite(tensor, name)

        output = dict(ret_l)
        output.update(
            {
                "Y_map": y_fused,
                "Y": y_fused,
                "R_map": r_fused,
                "R": r_fused,
                "U_map": uncertainty,
                "U": uncertainty,
                "uncertainty": uncertainty,
                "valid_map": valid_map,
                "w_l_map": w_l,
                "w_l": w_l,
                "w_u_map": w_u,
                "w_u": w_u,
                "score_l": score_l,
                "score_u": score_u,
                "source_entropy": source_entropy,
            }
        )
        return output

    def _source_weights(
        self,
        score_l: torch.Tensor,
        score_u: torch.Tensor,
        valid_l: torch.Tensor,
        valid_u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        soft_weights = F.softmax(torch.cat((score_l, score_u), dim=1), dim=1)
        both_valid = valid_l & valid_u
        only_u_valid = ~valid_l & valid_u
        ones = torch.ones_like(score_l)
        zeros = torch.zeros_like(score_l)
        w_l = torch.where(
            both_valid,
            soft_weights[:, 0:1],
            torch.where(only_u_valid, zeros, ones),
        )
        w_u = ones - w_l
        return w_l, w_u

    def _source_entropy(
        self,
        w_l: torch.Tensor,
        w_u: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.cat((w_l, w_u), dim=1)
        entropy = -(weights * weights.clamp_min(self.eps).log()).sum(
            dim=1,
            keepdim=True,
        )
        return (entropy / math.log(2.0)).clamp(0.0, 1.0)

    def _valid_map(
        self,
        retrieval: Mapping[str, Any],
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        value = self._required(retrieval, ("valid_map",), name)
        return self._as_quality_map(value, reference, name) > 0.5

    def _as_quality_map(
        self,
        value,
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        bsz, _, height, width = reference.shape
        if isinstance(value, Number):
            tensor = torch.tensor(
                float(value),
                device=reference.device,
                dtype=torch.float32,
            )
        elif isinstance(value, torch.Tensor):
            tensor = value.to(device=reference.device, dtype=torch.float32)
        else:
            raise TypeError(f"{name} must be a number or tensor")

        if tensor.dim() == 0:
            tensor = tensor.reshape(1, 1, 1, 1)
        elif tensor.dim() == 1:
            tensor = tensor.reshape(tensor.size(0), 1, 1, 1)
        elif tensor.dim() == 2:
            if tensor.size(1) != 1:
                raise ValueError(f"{name} 2D input must have shape [B,1]")
            tensor = tensor.reshape(tensor.size(0), 1, 1, 1)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
        elif tensor.dim() == 4:
            if tensor.size(1) != 1:
                raise ValueError(f"{name} must be single-channel")
        else:
            raise ValueError(
                f"{name} must be scalar, [B], [B,1], [B,H,W], or [B,1,H,W]"
            )

        target_shape = (bsz, 1, height, width)
        for actual, target, axis in zip(tensor.shape, target_shape, ("B", "C", "H", "W")):
            if actual not in (1, target):
                raise ValueError(
                    f"{name} {axis} dimension {actual} cannot broadcast to {target}"
                )
        tensor = tensor.expand(target_shape)
        self._ensure_finite(tensor, name)
        return tensor

    @staticmethod
    def _feature_map(
        retrieval: Mapping[str, Any],
        aliases: Sequence[str],
        name: str,
    ) -> torch.Tensor:
        value = QualityAdaptiveSourceFusion._required(retrieval, aliases, name)
        if not isinstance(value, torch.Tensor) or value.dim() != 4:
            raise ValueError(f"{name} must be a tensor with shape [B,C,H,W]")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        QualityAdaptiveSourceFusion._ensure_finite(value, name)
        return value

    @staticmethod
    def _validate_y_r_pair(y_map: torch.Tensor, r_map: torch.Tensor, name: str) -> None:
        if y_map.size(1) < 4:
            raise ValueError(f"{name} Y must have at least four evidence channels")
        if y_map.size(0) != r_map.size(0) or y_map.shape[-2:] != r_map.shape[-2:]:
            raise ValueError(f"{name} Y/R batch and spatial shapes must match")
        if y_map.device != r_map.device:
            raise ValueError(f"{name} Y/R must be on the same device")

    @staticmethod
    def _validate_source_shapes(
        y_l: torch.Tensor,
        y_u: torch.Tensor,
        r_l: torch.Tensor,
        r_u: torch.Tensor,
    ) -> None:
        if y_l.shape != y_u.shape:
            raise ValueError("ret_l and ret_u Y shapes must match")
        if r_l.shape != r_u.shape:
            raise ValueError("ret_l and ret_u R shapes must match")
        if y_l.device != y_u.device or r_l.device != r_u.device:
            raise ValueError("ret_l and ret_u tensors must be on the same device")
        if y_l.dtype != y_u.dtype or r_l.dtype != r_u.dtype:
            raise ValueError("ret_l and ret_u tensor dtypes must match")

    @staticmethod
    def _validate_reference(reference: torch.Tensor) -> None:
        if not isinstance(reference, torch.Tensor) or reference.dim() != 4:
            raise ValueError("reference must have shape [B,C,H,W]")
        if not reference.is_floating_point():
            raise TypeError("reference must be floating point")

    @staticmethod
    def _validate_retrieval(retrieval, name: str) -> None:
        if not isinstance(retrieval, Mapping):
            raise TypeError(f"{name} must be a mapping")

    @staticmethod
    def _required(
        retrieval: Mapping[str, Any],
        aliases: Sequence[str],
        name: str,
    ):
        for key in aliases:
            if key in retrieval and retrieval[key] is not None:
                return retrieval[key]
        raise KeyError(f"retrieval is missing required field {name!r}")

    @staticmethod
    def _ensure_finite(value: torch.Tensor, name: str) -> None:
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")

    @staticmethod
    def _non_negative_weight(value, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result


__all__ = [
    "QualityAdaptiveSourceFusion",
    "DEFAULT_SIM_WEIGHT",
    "DEFAULT_CONS_WEIGHT",
    "DEFAULT_REL_WEIGHT",
    "DEFAULT_UNC_WEIGHT",
]
