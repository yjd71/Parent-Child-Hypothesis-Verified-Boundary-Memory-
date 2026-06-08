from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from PIL import Image


def collect_visualization_tensors(aux):
    aux = aux or {}
    keys = ("prob3", "B_query", "boundary_mask", "Y_map", "U_map", "cons_map", "gate3", "z_mem3", "p_main", "p_final")
    return {key: aux[key] for key in keys if aux.get(key) is not None}


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
