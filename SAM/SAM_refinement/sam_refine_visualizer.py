from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from .svb_cache import SVBPLRCache
except ImportError:
    from SAM.SAM_refinement.svb_cache import SVBPLRCache


LOGGER = logging.getLogger(__name__)


class SamRefineVisualizer:
    """Save SVB-PLR 3x4 diagnostic panels.

    Shape:
        images: [B, 3, H_img, W_img]
        teacher_prob/p_ref/conf_ref: [B, 1, H, W]
        sam_aux maps: [B, 1, H, W] when present
    """

    PANEL_NAMES = (
        "input",
        "teacher",
        "sam_mask",
        "p_ref",
        "S_fg",
        "S_bg",
        "M_bd",
        "refine_band",
        "points",
        "R_sam",
        "conf_ref",
        "diff_or_student",
    )

    def __init__(self, cfg, logger=None) -> None:
        self.cfg = cfg
        self.logger = logger
        self.enabled = bool(getattr(cfg, "vis_sam_refinement", True))
        self.interval = max(1, int(getattr(cfg, "vis_sam_refine_interval", 200)))
        self.max_samples = max(1, int(getattr(cfg, "vis_sam_refine_max_samples", 2)))
        self.save_dir = Path(getattr(cfg, "sam_refine_vis_dir", "outputs/svb_plr_visualization"))

    def save(
        self,
        images,
        teacher_prob: torch.Tensor,
        sam_mask: torch.Tensor,
        p_ref: torch.Tensor,
        conf_ref: torch.Tensor,
        sam_aux: Dict[str, Any],
        image_ids=None,
        epoch=None,
        step=None,
        student_pred=None,
    ) -> None:
        if not self.enabled:
            return
        if step is not None and int(step) % self.interval != 0:
            return
        try:
            from PIL import Image, ImageDraw

            ref = self._as_b1hw(teacher_prob, teacher_prob)
            ids = SVBPLRCache.normalize_ids(image_ids, ref.size(0))
            if ids is None:
                ids = ["sample{}".format(idx) for idx in range(ref.size(0))]

            save_root = self.save_dir / "epoch_{}".format(self._safe(epoch, width=3)) / "iter_{}".format(self._safe(step, width=6))
            save_root.mkdir(parents=True, exist_ok=True)

            for idx, image_id in enumerate(ids[: min(len(ids), self.max_samples)]):
                panels = self._build_panels(
                    idx=idx,
                    images=images,
                    teacher_prob=ref,
                    sam_mask=sam_mask,
                    p_ref=p_ref,
                    conf_ref=conf_ref,
                    sam_aux=sam_aux or {},
                    student_pred=student_pred,
                )
                panel_image = self._compose_grid(panels, Image, ImageDraw)
                panel_image.save(str(save_root / "{}.png".format(self._safe_name(image_id))))
        except Exception as exc:
            self._warn("[SVB-PLR] visualization save failed: {}".format(exc))

    def _build_panels(
        self,
        idx: int,
        images,
        teacher_prob: torch.Tensor,
        sam_mask,
        p_ref,
        conf_ref,
        sam_aux: Mapping[str, Any],
        student_pred=None,
    ) -> List[Tuple[str, Any]]:
        prompt_pack = sam_aux.get("prompt_pack", {}) if isinstance(sam_aux, Mapping) else {}
        evidence = prompt_pack.get("evidence", {}) if isinstance(prompt_pack, Mapping) else {}
        ref = teacher_prob

        image_rgb = self._image_panel(images, idx, ref)
        teacher_rgb = self._map_panel(ref, idx, ref)
        sam_rgb = self._map_panel(sam_mask, idx, ref)
        pref_rgb = self._map_panel(p_ref, idx, ref)
        sfg_rgb = self._map_panel(self._map_from(evidence, "S_fg_up", ref), idx, ref)
        sbg_rgb = self._map_panel(self._map_from(evidence, "S_bg_up", ref), idx, ref)
        mbd_rgb = self._signed_map_panel(self._map_from(evidence, "M_bd_up", ref), idx, ref)
        band_rgb = self._map_panel(sam_aux.get("refine_band"), idx, ref)
        points_rgb = self._points_panel(image_rgb, prompt_pack, idx, ref)
        rsam_rgb = self._map_panel(sam_aux.get("R_sam"), idx, ref)
        conf_rgb = self._map_panel(conf_ref, idx, ref)
        diff_rgb = self._diff_or_student_panel(student_pred, p_ref, ref, idx)

        return list(
            zip(
                self.PANEL_NAMES,
                (
                    image_rgb,
                    teacher_rgb,
                    sam_rgb,
                    pref_rgb,
                    sfg_rgb,
                    sbg_rgb,
                    mbd_rgb,
                    band_rgb,
                    points_rgb,
                    rsam_rgb,
                    conf_rgb,
                    diff_rgb,
                ),
            )
        )

    def _image_panel(self, images, idx: int, ref: torch.Tensor):
        if not torch.is_tensor(images):
            return self._gray_to_rgb(ref.new_zeros(ref.shape[-2:]))
        image = images.detach().to(device=ref.device, dtype=ref.dtype)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4 or image.size(0) <= idx:
            return self._gray_to_rgb(ref.new_zeros(ref.shape[-2:]))
        image = image[idx : idx + 1]
        if tuple(image.shape[-2:]) != tuple(ref.shape[-2:]):
            image = F.interpolate(image.float(), size=ref.shape[-2:], mode="bilinear", align_corners=False).to(dtype=ref.dtype)
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image = (image * std + mean).clamp(0.0, 1.0)
        return self._rgb_tensor_to_uint8(image[0])

    def _map_panel(self, value, idx: int, ref: torch.Tensor):
        value4 = self._optional_map(value, ref)
        if value4.size(0) <= idx:
            return self._gray_to_rgb(ref.new_zeros(ref.shape[-2:]))
        return self._gray_to_rgb(value4[idx, 0])

    def _signed_map_panel(self, value, idx: int, ref: torch.Tensor):
        value4 = self._optional_map(value, ref)
        if value4.size(0) <= idx:
            return self._gray_to_rgb(ref.new_zeros(ref.shape[-2:]))
        x = value4[idx, 0].detach().float().cpu()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        red = x.clamp_min(0.0)
        blue = (-x).clamp_min(0.0)
        green = 1.0 - (red + blue).clamp(0.0, 1.0)
        return self._rgb_tensor_to_uint8(torch.stack((red, green, blue), dim=0))

    def _points_panel(self, image_rgb, prompt_pack, idx: int, ref: torch.Tensor):
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return image_rgb
        image = Image.fromarray(image_rgb.copy())
        draw = ImageDraw.Draw(image)
        radius = max(2, min(ref.shape[-2:]) // 80)
        self._draw_point_list(draw, self._points_for(prompt_pack, "pos_points", idx), radius, (0, 255, 0))
        self._draw_point_list(draw, self._points_for(prompt_pack, "neg_points", idx), radius, (255, 0, 0))
        self._draw_point_list(draw, self._points_for(prompt_pack, "boundary_points", idx), radius, (255, 255, 0))
        return self._pil_to_array(image)

    def _diff_or_student_panel(self, student_pred, p_ref, teacher_prob: torch.Tensor, idx: int):
        if torch.is_tensor(student_pred):
            pred = student_pred.detach().to(device=teacher_prob.device, dtype=teacher_prob.dtype)
            if pred.min().detach().item() < 0.0 or pred.max().detach().item() > 1.0:
                pred = pred.sigmoid()
            return self._map_panel(pred, idx, teacher_prob)
        pref = self._optional_map(p_ref, teacher_prob)
        diff = pref - teacher_prob
        return self._signed_map_panel(diff, idx, teacher_prob)

    @staticmethod
    def _draw_point_list(draw, points: torch.Tensor, radius: int, color) -> None:
        if not torch.is_tensor(points) or points.numel() == 0:
            return
        for x, y in points.detach().cpu().float().reshape(-1, 2).tolist():
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=max(1, radius // 2))

    @staticmethod
    def _points_for(prompt_pack, key: str, idx: int) -> torch.Tensor:
        if not isinstance(prompt_pack, Mapping):
            return torch.empty(0, 2)
        value = prompt_pack.get(key)
        if isinstance(value, Sequence) and not torch.is_tensor(value) and len(value) > idx and torch.is_tensor(value[idx]):
            return value[idx]
        return torch.empty(0, 2)

    @staticmethod
    def _map_from(mapping, key: str, ref: torch.Tensor):
        if isinstance(mapping, Mapping) and torch.is_tensor(mapping.get(key)):
            return mapping.get(key)
        return ref.new_zeros(ref.shape)

    @staticmethod
    def _optional_map(value, ref: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(value):
            return ref.new_zeros(ref.shape)
        x = value.detach().to(device=ref.device, dtype=ref.dtype)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.size(1) != 1:
            x = x[:, :1]
        if x.dim() != 4:
            return ref.new_zeros(ref.shape)
        if x.size(0) != ref.size(0):
            if x.size(0) == 1:
                x = x.expand(ref.size(0), -1, -1, -1)
            else:
                return ref.new_zeros(ref.shape)
        if tuple(x.shape[-2:]) != tuple(ref.shape[-2:]):
            x = F.interpolate(x.float(), size=ref.shape[-2:], mode="bilinear", align_corners=False).to(dtype=ref.dtype)
        return x

    @staticmethod
    def _as_b1hw(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        x = value.detach()
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.size(1) != 1:
            x = x[:, :1]
        return x.to(device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _gray_to_rgb(value: torch.Tensor):
        x = value.detach().float().cpu()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        denom = (x.max() - x.min()).clamp_min(1e-6)
        x = ((x - x.min()) / denom).clamp(0.0, 1.0)
        rgb = torch.stack((x, x, x), dim=0)
        return SamRefineVisualizer._rgb_tensor_to_uint8(rgb)

    @staticmethod
    def _rgb_tensor_to_uint8(value: torch.Tensor):
        x = value.detach().float().cpu()
        if x.dim() == 4:
            x = x[0]
        if x.size(0) != 3:
            x = x[:1].expand(3, -1, -1)
        x = x.permute(1, 2, 0).clamp(0.0, 1.0)
        return (x.numpy() * 255.0).round().clip(0, 255).astype("uint8")

    @staticmethod
    def _pil_to_array(image):
        import numpy as np

        return np.asarray(image)

    @staticmethod
    def _compose_grid(panels: List[Tuple[str, Any]], image_mod, draw_mod):
        panel_images = [image_mod.fromarray(array) for _, array in panels]
        width, height = panel_images[0].size
        title_h = 18
        canvas = image_mod.new("RGB", (4 * width, 3 * (height + title_h)), color=(255, 255, 255))
        draw = draw_mod.Draw(canvas)
        for idx, ((title, _), panel) in enumerate(zip(panels, panel_images)):
            row, col = divmod(idx, 4)
            x = col * width
            y = row * (height + title_h)
            draw.text((x + 4, y + 2), title, fill=(0, 0, 0))
            canvas.paste(panel, (x, y + title_h))
        return canvas

    @staticmethod
    def _safe(value, width: int) -> str:
        if value is None:
            return "na"
        try:
            return str(int(value)).zfill(width)
        except (TypeError, ValueError):
            return SamRefineVisualizer._safe_name(value)

    @staticmethod
    def _safe_name(value: Any) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))

    def _log_enabled(self) -> bool:
        return bool(getattr(self.cfg, "svb_plr_log_enable", True))

    def _warn(self, message: str) -> None:
        if not self._log_enabled():
            return
        if self.logger is not None:
            method = getattr(self.logger, "warn_info", None) or getattr(self.logger, "warning", None) or getattr(self.logger, "info", None)
            if callable(method):
                method(message)
                return
        LOGGER.warning(message)


__all__ = ["SamRefineVisualizer"]
