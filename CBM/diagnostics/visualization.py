from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from utils.log_control import log_enabled


def collect_visualization_tensors(aux):
    aux = aux or {}
    keys = ("prob3", "B_query", "boundary_mask", "Y_map", "U_map", "cons_map", "gate3", "z_mem3", "p_main", "p_final")
    return {key: aux[key] for key in keys if aux.get(key) is not None}


def save_memory_selection_visualizations(memory, snapshots, epoch: int, split, config) -> List[str]:
    """Save seven-panel diagnostics for labeled core-memory selection."""
    base_dir = getattr(config, "cbm_memory_vis_dir", None)
    if not base_dir:
        base_dir = os.path.join(str(getattr(config, "ckpt_dir", ".")), "cbm_memory_selection_vis")
    split_name = _safe_name(split if split is not None else "manual")
    output_dir = os.path.join(base_dir, split_name, f"epoch_{int(epoch):03d}")
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    region_colors = {
        "fg_core": (0, 200, 0),
        "fg_boundary": (255, 0, 0),
        "bg_near": (255, 215, 0),
        "bg_far": (0, 100, 255),
    }
    for image_id in sorted(snapshots):
        image_tensor, gt_tensor = snapshots[image_id]
        image = _denormalize_input_image(image_tensor)
        gt = _mask_to_rgb(gt_tensor, image.size)
        panels = [("original", image), ("GT", gt)]
        for region in ("fg_core", "fg_boundary", "bg_near", "bg_far"):
            panels.append(
                (region, _memory_token_overlay(image, memory.meta.get(region, []), image_id, region_colors))
            )
        all_meta = []
        for region in ("fg_core", "fg_boundary", "bg_near", "bg_far"):
            all_meta.extend(memory.meta.get(region, []))
        panels.append(("all", _memory_token_overlay(image, all_meta, image_id, region_colors)))
        canvas = _compose_labeled_panels(panels)
        path = os.path.join(output_dir, f"{_safe_name(image_id)}.png")
        canvas.save(path)
        saved.append(path)
    return saved


def _denormalize_input_image(value: torch.Tensor) -> Image.Image:
    value = value.detach().cpu().float()
    if value.dim() == 4:
        value = value[0]
    if value.dim() != 3:
        raise ValueError(f"memory visualization image must be CHW, got {tuple(value.shape)}")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    value = (value[:3] * std + mean).clamp(0.0, 1.0)
    array = (value.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _mask_to_rgb(value: torch.Tensor, size) -> Image.Image:
    value = value.detach().cpu().float().squeeze()
    array = (value.clamp(0.0, 1.0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="L").resize(size, resample=Image.Resampling.NEAREST).convert("RGB")


def _memory_token_overlay(image: Image.Image, metas, image_id: str, colors) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for item in metas:
        if str(item.get("image_id")) != str(image_id):
            continue
        height = max(1, int(item.get("height", 1)))
        width = max(1, int(item.get("width", 1)))
        h, w = item.get("coord", (0, 0))
        x = (float(w) + 0.5) * overlay.width / width
        y = (float(h) + 0.5) * overlay.height / height
        radius = max(2, int(round(min(overlay.size) / 160)))
        color = colors.get(str(item.get("region")), (255, 255, 255))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0))
    return overlay


def _compose_labeled_panels(panels) -> Image.Image:
    width, height = panels[0][1].size
    header = 24
    canvas = Image.new("RGB", (width * len(panels), height + header), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        x = index * width
        canvas.paste(panel, (x, header))
        draw.text((x + 4, 4), str(label), fill=(0, 0, 0))
    return canvas


def save_pfi_binary_visualizations_v42(
    aux,
    batch,
    epoch: int,
    iteration: int,
    config,
    logger=None,
    branch_name: str = "Sup",
) -> List[str]:
    if not bool(getattr(config, "cbm_vis_enable", False)):
        return []
    if bool(getattr(config, "cbm_vis_labeled_only", True)) and branch_name != "Sup":
        return []
    interval = max(1, int(getattr(config, "cbm_vis_interval", 200)))
    if int(iteration) % interval != 0:
        return []
    aux = aux or {}
    if not aux.get("cbm_used", False):
        return []
    if aux.get("p_main") is None or aux.get("p_final") is None:
        return []

    try:
        return _save_visualizations(aux, batch, epoch, iteration, config, branch_name)
    except Exception as exc:
        if log_enabled(config):
            _warn(logger, f"[CBM] pfi_binary_visualizations_v42 save failed: {exc}")
        return []


def _save_visualizations(aux, batch, epoch: int, iteration: int, config, branch_name: str) -> List[str]:
    save_dir = getattr(config, "cbm_vis_dir", None)
    if save_dir is None:
        save_dir = os.path.join(getattr(config, "ckpt_dir", "."), "pfi_binary_visualizations_v42")
    os.makedirs(save_dir, exist_ok=True)

    p_main = _as_4d(aux["p_main"], "p_main").detach()
    p_final = _as_4d(aux["p_final"], "p_final").detach()
    target_size = p_final.shape[-2:]
    maps = _collect_v42_maps(aux, p_main, p_final)
    if not maps:
        return []

    image_ids = _extract_image_ids(batch, p_final.size(0))
    max_images = max(1, int(getattr(config, "cbm_vis_max_images", 2)))
    num_images = min(max_images, p_final.size(0))
    saved_paths: List[str] = []

    for sample_idx in range(num_images):
        image_id = _safe_name(image_ids[sample_idx])
        for map_name, value in maps.items():
            value = _prepare_map(value, target_size)
            single = value[sample_idx : sample_idx + 1]
            path = os.path.join(
                save_dir,
                f"epoch{int(epoch):03d}_iter{int(iteration):06d}_{_safe_name(branch_name)}_{image_id}_{map_name}.png",
            )
            _save_single_channel(single, path)
            saved_paths.append(path)
    return saved_paths


def _collect_v42_maps(aux: Dict[str, Any], p_main: torch.Tensor, p_final: torch.Tensor) -> Dict[str, torch.Tensor]:
    y_map = aux.get("Y_map")
    maps = {
        "m3_prob": aux.get("prob3"),
        "B_query": aux.get("B_query"),
        "U_map": aux.get("U_map"),
        "cons_map": aux.get("cons_map"),
        "gate3": aux.get("gate3"),
        "p_main": p_main,
        "p_final": p_final,
        "p_final_minus_p_main": p_final - _resize_to(p_main, p_final.shape[-2:]),
    }
    if isinstance(y_map, torch.Tensor) and y_map.dim() == 4 and y_map.size(1) >= 3:
        maps["fg_boundary_score"] = y_map[:, 1:2]
        maps["bg_near_score"] = y_map[:, 2:3]
        maps["M_bd"] = y_map[:, 1:2] - y_map[:, 2:3]
    return {name: value for name, value in maps.items() if isinstance(value, torch.Tensor)}


def _prepare_map(value: torch.Tensor, target_size) -> torch.Tensor:
    value = _as_4d(value, "value").detach().float().cpu()
    if value.size(1) != 1:
        value = value[:, :1]
    if tuple(value.shape[-2:]) != tuple(target_size):
        value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
    return value


def _save_single_channel(value: torch.Tensor, path: str) -> None:
    value = value.squeeze(0)
    if value.dim() == 3:
        value = value[0]
    value = _normalize_to_unit(value)
    array = (value.numpy() * 255.0).round().clip(0, 255).astype("uint8")
    Image.fromarray(array, mode="L").save(path)


def _normalize_to_unit(value: torch.Tensor) -> torch.Tensor:
    value = torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = value.min()
    max_value = value.max()
    denom = (max_value - min_value).clamp_min(1e-6)
    return ((value - min_value) / denom).clamp(0.0, 1.0)


def _resize_to(value: torch.Tensor, target_size) -> torch.Tensor:
    if tuple(value.shape[-2:]) == tuple(target_size):
        return value
    return F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)


def _as_4d(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.dim() == 3:
        value = value.unsqueeze(1)
    if value.dim() != 4:
        raise ValueError(f"{name} must be 4D or 3D tensor, got {tuple(value.shape)}")
    return value


def _extract_image_ids(batch, batch_size: int) -> List[str]:
    if isinstance(batch, (list, tuple)) and len(batch) > 2:
        raw_ids = batch[2]
        if isinstance(raw_ids, torch.Tensor):
            return [str(item) for item in raw_ids.detach().cpu().reshape(-1).tolist()]
        if isinstance(raw_ids, (list, tuple)):
            return [str(item) for item in raw_ids]
        return [str(raw_ids)] * batch_size
    return [f"sample{idx}" for idx in range(batch_size)]


def _safe_name(value: Any) -> str:
    text = str(value)
    text = text.replace("\\", "/").split("/")[-1]
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "unknown"


def _warn(logger, message: str) -> None:
    if logger is None:
        print(message)
        return
    log_fn = getattr(logger, "warn_info", None) or getattr(logger, "warning", None) or getattr(logger, "info", None)
    if log_fn is not None:
        log_fn(message)
