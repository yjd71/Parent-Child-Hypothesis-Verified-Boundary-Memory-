from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .cbm_aux_adapter import build_retrieval_aux_from_cbm_aux
    from .cbm_prompt_generator import CBMPromptGenerator
    from .svb_utils import resize_like
except ImportError:
    from SAM.SAM_refinement.cbm_aux_adapter import build_retrieval_aux_from_cbm_aux
    from SAM.SAM_refinement.cbm_prompt_generator import CBMPromptGenerator
    from SAM.SAM_refinement.svb_utils import resize_like


class ConformalSAMCalibrator(nn.Module):
    """Labeled-domain conformal calibration for SAM-CBM reliability.

    Shape:
        fit samples nonconformity from labeled COD boundary/refinement bands.
        estimate_reliability returns [B, 1, H, W] conformal reliability maps.
    """

    DEFAULTS = {
        "sam_use_conformal": True,
        "sam_conformal_alpha": 0.1,
        "sam_conformal_eps": 1e-6,
        "sam_conformal_max_samples": 200000,
    }

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.prompt_generator = CBMPromptGenerator(cfg)
        self.register_buffer("q_alpha", torch.tensor(float("nan"), dtype=torch.float32))
        self.register_buffer("num_calibration_pixels", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def fit(self, model, memory, labeled_loader, sam_backend, device, max_batches=None):
        """Estimate q_alpha on labeled batches without modifying model memory.

        Shape:
            labeled_loader batch: expects image at batch[0], GT at batch[1].
            q_alpha: scalar quantile over boundary/refine-band nonconformity.
        """
        if not self._cfg_bool("sam_use_conformal"):
            self._reset_state(device)
            return self
        if model is None or labeled_loader is None or sam_backend is None:
            self._reset_state(device)
            return self

        target_device = torch.device(device)
        was_training = getattr(model, "training", False)
        nonconformity_chunks = []
        total_pixels = 0
        max_samples = self._cfg_int("sam_conformal_max_samples")

        try:
            if hasattr(model, "eval"):
                model.eval()
            for batch_idx, batch in enumerate(labeled_loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                inputs, gt = self._extract_batch_inputs_gt(batch, target_device)
                if inputs is None or gt is None:
                    continue

                teacher_prob, aux_t = self._forward_teacher_prob_aux(model, inputs, memory)
                if teacher_prob is None:
                    continue
                teacher_prob = self._as_b1hw(teacher_prob, inputs, "teacher_prob").clamp(0.0, 1.0)
                retrieval_aux = build_retrieval_aux_from_cbm_aux(aux_t or {})
                prompt_pack = self.prompt_generator(teacher_prob, retrieval_aux)

                sam_out = sam_backend.predict(
                    inputs,
                    teacher_prob,
                    prompt_pack=prompt_pack,
                    epoch=None,
                    step=batch_idx,
                )
                sam_mask = self._select_sam_mask(sam_out, teacher_prob)
                gt_resized = self._prepare_gt(gt, sam_mask)
                refine_band = self._as_b1hw(
                    prompt_pack.get("refine_band"),
                    sam_mask,
                    "refine_band",
                    mode="nearest",
                    fallback=sam_mask.new_zeros(sam_mask.shape),
                ).bool()

                nonconf = (sam_mask - gt_resized).abs().detach()
                selected = nonconf[refine_band.expand_as(nonconf)]
                if selected.numel() == 0:
                    selected = nonconf.flatten()
                if selected.numel() == 0:
                    continue
                selected = selected.float().detach().flatten().cpu()
                nonconformity_chunks.append(selected)
                total_pixels += int(selected.numel())
                if total_pixels >= max_samples:
                    break
        finally:
            if was_training and hasattr(model, "train"):
                model.train()

        if not nonconformity_chunks:
            self._reset_state(target_device)
            return self

        values = torch.cat(nonconformity_chunks, dim=0)
        if values.numel() > max_samples:
            values = values[:max_samples]
        quantile_level = max(0.0, min(1.0, 1.0 - self._cfg_float("sam_conformal_alpha")))
        q_alpha = torch.quantile(values, quantile_level).to(device=target_device, dtype=torch.float32)
        self.q_alpha.data = q_alpha.reshape(())
        self.num_calibration_pixels.data = torch.tensor(int(values.numel()), device=target_device, dtype=torch.long)
        return self

    @torch.no_grad()
    def estimate_reliability(self, teacher_prob, sam_mask, prompt_pack):
        """Estimate conformal reliability for unlabeled SAM refinement.

        Shape:
            teacher_prob: [B, 1, H, W]
            sam_mask: [B, 1, H, W]
            return: R_conformal [B, 1, H, W]
        """
        p_t = self._as_teacher_prob(teacher_prob)
        if not self._cfg_bool("sam_use_conformal") or not self.is_fitted():
            return p_t.new_zeros(p_t.shape)

        sam = self._as_b1hw(sam_mask, p_t, "sam_mask").clamp(0.0, 1.0)
        pack = prompt_pack if isinstance(prompt_pack, Mapping) else {}
        refine_band = self._as_b1hw(
            pack.get("refine_band"),
            p_t,
            "refine_band",
            mode="nearest",
            fallback=p_t.new_zeros(p_t.shape),
        ).clamp(0.0, 1.0)

        estimated_nonconformity = (sam - p_t).abs()
        denom = self.q_alpha.to(device=p_t.device, dtype=p_t.dtype).clamp_min(self._cfg_float("sam_conformal_eps"))
        reliability_band = 1.0 - (estimated_nonconformity / denom).clamp(0.0, 1.0)
        r_conformal = reliability_band * refine_band + (1.0 - refine_band)
        return r_conformal.clamp(0.0, 1.0).detach()

    def is_fitted(self) -> bool:
        return bool(torch.isfinite(self.q_alpha).item() and self.num_calibration_pixels.item() > 0)

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "q_alpha": self.q_alpha.detach().cpu(),
            "num_calibration_pixels": self.num_calibration_pixels.detach().cpu(),
        }

    def load_calibrator_state(self, state: Mapping[str, Any], device=None):
        target_device = torch.device(device) if device is not None else self.q_alpha.device
        q_alpha = state.get("q_alpha", torch.tensor(float("nan")))
        num_pixels = state.get("num_calibration_pixels", torch.zeros((), dtype=torch.long))
        self.q_alpha.data = torch.as_tensor(q_alpha, device=target_device, dtype=torch.float32).reshape(())
        self.num_calibration_pixels.data = torch.as_tensor(num_pixels, device=target_device, dtype=torch.long).reshape(())
        return self

    def save_state(self, path: str | Path) -> None:
        torch.save(self.to_state_dict(), str(path))

    def load_state(self, path: str | Path, map_location=None):
        state = torch.load(str(path), map_location=map_location)
        return self.load_calibrator_state(state, device=map_location)

    def _forward_teacher_prob_aux(self, model, inputs: torch.Tensor, memory) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        forward_attempts = (
            {"ema": True, "use_memory": True, "cbm": memory, "return_aux": True},
            {"ema": True, "use_memory": True, "return_aux": True},
            {"use_memory": True, "cbm": memory, "return_aux": True},
            {"use_memory": True, "return_aux": True},
            {"ema": True, "use_memory": True},
            {},
        )
        last_error: Optional[Exception] = None
        for kwargs in forward_attempts:
            try:
                output = model(inputs, **kwargs)
                preds, aux = self._unpack_model_output(output)
                prob = self._prediction_to_prob(preds, aux)
                if prob is not None:
                    return prob.detach(), aux
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            return None, {"fallback_reason": "calibrator_forward_failed: {}".format(last_error)}
        return None, {"fallback_reason": "calibrator_forward_failed"}

    @staticmethod
    def _unpack_model_output(output) -> Tuple[Any, Dict[str, Any]]:
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], Mapping):
            return output[0], dict(output[1])
        return output, {}

    @staticmethod
    def _prediction_to_prob(preds, aux: Mapping[str, Any]) -> Optional[torch.Tensor]:
        p_final = aux.get("p_final") if isinstance(aux, Mapping) else None
        if torch.is_tensor(p_final):
            return p_final
        if isinstance(preds, tuple) and len(preds) == 2:
            preds = preds[1]
        if isinstance(preds, (list, tuple)) and preds:
            pred = preds[-1]
        else:
            pred = preds
        if not torch.is_tensor(pred):
            return None
        if pred.min().detach().item() < 0.0 or pred.max().detach().item() > 1.0:
            return pred.sigmoid()
        return pred

    @staticmethod
    def _extract_batch_inputs_gt(batch, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            return None, None
        inputs, gt = batch[0], batch[1]
        if not torch.is_tensor(inputs) or not torch.is_tensor(gt):
            return None, None
        return inputs.to(device), gt.to(device)

    @staticmethod
    def _select_sam_mask(sam_out: Mapping[str, Any], ref: torch.Tensor) -> torch.Tensor:
        masks = sam_out.get("masks") if isinstance(sam_out, Mapping) else None
        if not torch.is_tensor(masks) or masks.numel() == 0:
            return ref.detach().clamp(0.0, 1.0)
        masks = masks.detach().to(device=ref.device, dtype=ref.dtype)
        if masks.dim() == 2:
            masks = masks.reshape(1, 1, *masks.shape[-2:])
        elif masks.dim() == 3:
            if masks.size(0) == ref.size(0):
                masks = masks.unsqueeze(1)
            else:
                masks = masks.unsqueeze(0)
        elif masks.dim() != 4:
            return ref.detach().clamp(0.0, 1.0)
        if masks.size(0) != ref.size(0):
            if masks.size(0) == 1:
                masks = masks.expand(ref.size(0), -1, -1, -1)
            else:
                return ref.detach().clamp(0.0, 1.0)
        if tuple(masks.shape[-2:]) != tuple(ref.shape[-2:]):
            masks = resize_like(masks, ref, mode="bilinear")

        scores = sam_out.get("scores") if isinstance(sam_out, Mapping) else None
        if torch.is_tensor(scores) and scores.numel() > 0:
            scores = scores.detach().to(device=ref.device).reshape(scores.size(0), -1)
            if scores.size(0) == 1 and masks.size(0) > 1:
                scores = scores.expand(masks.size(0), -1)
            if scores.size(0) == masks.size(0) and scores.size(1) >= masks.size(1):
                best = scores[:, : masks.size(1)].argmax(dim=1)
                batch_idx = torch.arange(masks.size(0), device=ref.device)
                return masks[batch_idx, best].unsqueeze(1).clamp(0.0, 1.0)
        return masks[:, :1].clamp(0.0, 1.0)

    @staticmethod
    def _prepare_gt(gt: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        elif gt.dim() == 2:
            gt = gt.unsqueeze(0).unsqueeze(0)
        gt = gt.detach().to(device=ref.device, dtype=ref.dtype)
        if gt.size(1) != 1:
            gt = gt[:, :1]
        if gt.size(0) != ref.size(0):
            if gt.size(0) == 1:
                gt = gt.expand(ref.size(0), -1, -1, -1)
            else:
                raise ValueError("GT batch size must match SAM mask batch size")
        if tuple(gt.shape[-2:]) != tuple(ref.shape[-2:]):
            gt = F.interpolate(gt.float(), size=ref.shape[-2:], mode="nearest").to(dtype=ref.dtype)
        return (gt >= 0.5).to(dtype=ref.dtype)

    @staticmethod
    def _as_teacher_prob(teacher_prob: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(teacher_prob):
            raise TypeError("teacher_prob must be a torch.Tensor")
        if teacher_prob.dim() != 4 or teacher_prob.size(1) != 1:
            raise ValueError("teacher_prob must have shape [B,1,H,W]")
        return teacher_prob.detach().clamp(0.0, 1.0)

    @staticmethod
    def _as_b1hw(value, ref: torch.Tensor, name: str, mode: str = "bilinear", fallback=None) -> torch.Tensor:
        if value is None or not torch.is_tensor(value):
            if fallback is not None:
                return fallback.detach().to(device=ref.device, dtype=ref.dtype)
            raise TypeError("{} must be a torch.Tensor".format(name))
        x = value.detach().to(device=ref.device, dtype=ref.dtype)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            if x.size(1) != 1:
                x = x[:, :1]
        else:
            if fallback is not None:
                return fallback.detach().to(device=ref.device, dtype=ref.dtype)
            raise ValueError("{} must have shape [H,W], [B,H,W], or [B,C,H,W]".format(name))
        if x.size(0) != ref.size(0):
            if x.size(0) == 1:
                x = x.expand(ref.size(0), -1, -1, -1)
            elif fallback is not None:
                return fallback.detach().to(device=ref.device, dtype=ref.dtype)
            else:
                raise ValueError("{} batch size must match reference".format(name))
        if tuple(x.shape[-2:]) != tuple(ref.shape[-2:]):
            x = resize_like(x, ref, mode=mode)
        return x.to(device=ref.device, dtype=ref.dtype)

    def _reset_state(self, device) -> None:
        target_device = torch.device(device)
        self.q_alpha.data = torch.tensor(float("nan"), device=target_device, dtype=torch.float32)
        self.num_calibration_pixels.data = torch.zeros((), device=target_device, dtype=torch.long)

    def _cfg(self, name: str) -> Any:
        if self.cfg is not None and hasattr(self.cfg, name):
            return getattr(self.cfg, name)
        return self.DEFAULTS[name]

    def _cfg_bool(self, name: str) -> bool:
        return bool(self._cfg(name))

    def _cfg_float(self, name: str) -> float:
        return float(self._cfg(name))

    def _cfg_int(self, name: str) -> int:
        return int(self._cfg(name))


__all__ = ["ConformalSAMCalibrator"]
