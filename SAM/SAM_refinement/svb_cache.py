from __future__ import annotations

import hashlib
import io
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


LOGGER = logging.getLogger(__name__)


class SVBPLRCache:
    """Disk cache for detached SVB-PLR outputs.

    Cache key:
        image_id + epoch + backend + prompt_mode + teacher_version/hash

    Cached tensors are stored on CPU and moved back to the current
    teacher_prob device/dtype on read.
    """

    TENSOR_KEYS = ("sam_mask", "refine_band", "R_sam", "beta")

    def __init__(self, cfg, logger=None) -> None:
        self.cfg = cfg
        self.logger = logger
        self.enabled = bool(getattr(cfg, "use_sam_cache", True))
        self.cache_refined_masks = bool(getattr(cfg, "cache_refined_masks", True))
        self.cache_prompt_debug = bool(getattr(cfg, "cache_prompt_debug", True))
        self.cache_dir = Path(getattr(cfg, "sam_cache_dir", "./cache/sam_refined_pseudo"))

    def read(
        self,
        image_ids,
        teacher_prob: torch.Tensor,
        epoch=None,
        backend=None,
        prompt_mode=None,
        teacher_version=None,
    ):
        """Return cached (p_ref, conf_ref, sam_aux) or None on miss.

        Shape:
            teacher_prob: [B, 1, H, W]
            p_ref/conf_ref: [B, 1, H, W]
        """
        key_items = self._key_items(
            image_ids=image_ids,
            teacher_prob=teacher_prob,
            epoch=epoch,
            backend=backend,
            prompt_mode=prompt_mode,
            teacher_version=teacher_version,
        )
        if not self.enabled or not self.cache_refined_masks or key_items is None:
            return None

        loaded: List[Dict[str, Any]] = []
        for item in key_items:
            path = self._path_for_key(item["cache_key"])
            if not path.is_file():
                return None
            try:
                payload = self._torch_load(path)
            except Exception:
                return None
            if not isinstance(payload, dict) or payload.get("cache_key") != item["cache_key"]:
                return None
            if not self._payload_shape_matches(payload, teacher_prob):
                return None
            loaded.append(payload)

        try:
            p_ref = torch.cat(
                [self._as_b1hw(payload["p_ref"], teacher_prob) for payload in loaded],
                dim=0,
            )
            conf_ref = torch.cat(
                [self._as_b1hw(payload["conf_ref"], teacher_prob) for payload in loaded],
                dim=0,
            )
        except Exception:
            return None

        sam_aux: Dict[str, Any] = {
            "used_sam": True,
            "cache_hit": True,
            "cache_backend": str(backend if backend is not None else self._default_backend()),
            "svb_ablation_mode": str(prompt_mode if prompt_mode is not None else self._default_prompt_mode()),
            "cache_items": [payload.get("cache_meta", {}) for payload in loaded],
        }
        for key in self.TENSOR_KEYS:
            stacked = self._stack_optional_tensor(key, loaded, teacher_prob)
            if stacked is not None:
                sam_aux[key] = stacked
        prompt_debug = [payload.get("prompt_debug") for payload in loaded if payload.get("prompt_debug") is not None]
        if prompt_debug:
            sam_aux["prompt_debug"] = prompt_debug
        return p_ref.detach(), conf_ref.detach(), sam_aux

    def write(
        self,
        image_ids,
        p_ref: torch.Tensor,
        conf_ref: torch.Tensor,
        sam_aux: Dict[str, Any],
        teacher_prob: Optional[torch.Tensor] = None,
        epoch=None,
        backend=None,
        prompt_mode=None,
        teacher_version=None,
    ) -> None:
        """Persist detached CPU tensors. Failures are logged and ignored."""
        ref = teacher_prob if torch.is_tensor(teacher_prob) else p_ref
        key_items = self._key_items(
            image_ids=image_ids,
            teacher_prob=ref,
            epoch=epoch,
            backend=backend,
            prompt_mode=prompt_mode,
            teacher_version=teacher_version,
        )
        if not self.enabled or not self.cache_refined_masks or key_items is None:
            return
        if not torch.is_tensor(p_ref) or not torch.is_tensor(conf_ref):
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for idx, item in enumerate(key_items):
                payload = {
                    "cache_key": item["cache_key"],
                    "shape": tuple(p_ref[idx : idx + 1].shape),
                    "p_ref": p_ref[idx : idx + 1].detach().cpu(),
                    "conf_ref": conf_ref[idx : idx + 1].detach().cpu(),
                    "cache_meta": {
                        "image_id": item["image_id"],
                        "epoch": item["epoch"],
                        "backend": item["backend"],
                        "prompt_mode": item["prompt_mode"],
                        "teacher_version": item["teacher_version"],
                    },
                }
                for key in self.TENSOR_KEYS:
                    value = sam_aux.get(key) if isinstance(sam_aux, dict) else None
                    if torch.is_tensor(value):
                        payload[key] = self._slice_tensor(value, idx).detach().cpu()
                if self.cache_prompt_debug and isinstance(sam_aux, dict):
                    payload["prompt_debug"] = self._prompt_debug(sam_aux, idx)
                torch.save(payload, str(self._path_for_key(item["cache_key"])))
        except Exception as exc:
            self._warn("[SVB-PLR] cache write failed: {}".format(exc))

    def _key_items(
        self,
        image_ids,
        teacher_prob: torch.Tensor,
        epoch=None,
        backend=None,
        prompt_mode=None,
        teacher_version=None,
    ) -> Optional[List[Dict[str, str]]]:
        if not torch.is_tensor(teacher_prob) or teacher_prob.dim() != 4:
            return None
        ids = self.normalize_ids(image_ids, teacher_prob.size(0))
        if ids is None:
            return None
        versions = self._teacher_versions(teacher_prob, teacher_version)
        if versions is None or len(versions) != len(ids):
            return None

        epoch_text = self._text(epoch if epoch is not None else "none")
        backend_text = self._text(backend if backend is not None else self._default_backend())
        prompt_text = self._text(prompt_mode if prompt_mode is not None else self._default_prompt_mode())
        items: List[Dict[str, str]] = []
        for image_id, version in zip(ids, versions):
            raw = "|".join((str(image_id), epoch_text, backend_text, prompt_text, str(version)))
            cache_key = hashlib.sha1(raw.encode("utf-8")).hexdigest()
            items.append(
                {
                    "image_id": str(image_id),
                    "epoch": epoch_text,
                    "backend": backend_text,
                    "prompt_mode": prompt_text,
                    "teacher_version": str(version),
                    "cache_key": cache_key,
                }
            )
        return items

    def _teacher_versions(self, teacher_prob: torch.Tensor, teacher_version) -> Optional[List[str]]:
        batch_size = int(teacher_prob.size(0))
        if teacher_version is None:
            return [self._hash_tensor(teacher_prob[idx : idx + 1]) for idx in range(batch_size)]
        if torch.is_tensor(teacher_version):
            flat = teacher_version.detach().cpu().reshape(-1).tolist()
            return [str(item) for item in flat] if len(flat) == batch_size else None
        if isinstance(teacher_version, (list, tuple)):
            return [str(item) for item in teacher_version] if len(teacher_version) == batch_size else None
        return [str(teacher_version)] * batch_size

    @staticmethod
    def _hash_tensor(value: torch.Tensor) -> str:
        tensor = value.detach().cpu().contiguous().to(torch.float32)
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return hashlib.sha1(buffer.getvalue()).hexdigest()[:20]

    def _path_for_key(self, cache_key: str) -> Path:
        return self.cache_dir / "{}.pt".format(self._safe_name(cache_key))

    @staticmethod
    def normalize_ids(image_ids, batch_size: int) -> Optional[List[str]]:
        if image_ids is None:
            return None
        if torch.is_tensor(image_ids):
            ids = [str(item) for item in image_ids.detach().cpu().reshape(-1).tolist()]
        elif isinstance(image_ids, (list, tuple)):
            ids = [str(item) for item in image_ids]
        else:
            ids = [str(image_ids)]
        if len(ids) != int(batch_size):
            return None
        return ids

    @staticmethod
    def _payload_shape_matches(payload: Dict[str, Any], ref: torch.Tensor) -> bool:
        p_ref = payload.get("p_ref")
        conf_ref = payload.get("conf_ref")
        if not torch.is_tensor(p_ref) or not torch.is_tensor(conf_ref):
            return False
        expected = (1, 1, int(ref.shape[-2]), int(ref.shape[-1]))
        return tuple(SVBPLRCache._shape_b1hw(p_ref)) == expected and tuple(SVBPLRCache._shape_b1hw(conf_ref)) == expected

    @staticmethod
    def _shape_b1hw(value: torch.Tensor) -> Tuple[int, int, int, int]:
        if value.dim() == 2:
            return 1, 1, int(value.shape[-2]), int(value.shape[-1])
        if value.dim() == 3:
            return int(value.shape[0]), 1, int(value.shape[-2]), int(value.shape[-1])
        if value.dim() == 4:
            return int(value.shape[0]), int(value.shape[1]), int(value.shape[-2]), int(value.shape[-1])
        return 0, 0, 0, 0

    @staticmethod
    def _as_b1hw(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        out = value.detach().to(device=ref.device, dtype=ref.dtype)
        if out.dim() == 2:
            out = out.unsqueeze(0).unsqueeze(0)
        elif out.dim() == 3:
            out = out.unsqueeze(1)
        elif out.dim() == 4 and out.size(1) != 1:
            out = out[:, :1]
        if tuple(out.shape[-2:]) != tuple(ref.shape[-2:]):
            return ref.new_empty((0, 1, *ref.shape[-2:]))
        return out

    def _stack_optional_tensor(self, key: str, payloads: Sequence[Dict[str, Any]], ref: torch.Tensor) -> Optional[torch.Tensor]:
        values: List[torch.Tensor] = []
        for payload in payloads:
            value = payload.get(key)
            if not torch.is_tensor(value):
                return None
            current = self._as_b1hw(value, ref)
            if current.numel() == 0:
                return None
            values.append(current)
        return torch.cat(values, dim=0).detach() if values else None

    @staticmethod
    def _slice_tensor(value: torch.Tensor, idx: int) -> torch.Tensor:
        if value.dim() > 0 and value.size(0) > idx:
            return value[idx : idx + 1]
        return value

    @staticmethod
    def _prompt_debug(sam_aux: Dict[str, Any], idx: int) -> Dict[str, Any]:
        selector_aux = sam_aux.get("selector_aux", {}) if isinstance(sam_aux, dict) else {}
        prompt_pack = sam_aux.get("prompt_pack", {}) if isinstance(sam_aux, dict) else {}
        debug: Dict[str, Any] = {}
        if isinstance(selector_aux, dict):
            best_expert = selector_aux.get("best_expert")
            if isinstance(best_expert, (list, tuple)) and len(best_expert) > idx:
                debug["best_expert"] = str(best_expert[idx])
            best_idx = selector_aux.get("best_candidate_index")
            if torch.is_tensor(best_idx) and best_idx.numel() > idx:
                debug["best_candidate_index"] = int(best_idx.detach().cpu().reshape(-1)[idx].item())
        if isinstance(prompt_pack, dict):
            debug["has_boxes"] = "boxes" in prompt_pack
            debug["has_points"] = "point_coords" in prompt_pack
            debug["has_mask_prompt"] = "mask_prompt" in prompt_pack or "mask_inputs" in prompt_pack
        return debug

    @staticmethod
    def _torch_load(path: Path):
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")

    def _default_backend(self) -> str:
        return str(getattr(self.cfg, "sam_pseudo_backend", "sam1"))

    def _default_prompt_mode(self) -> str:
        return str(getattr(self.cfg, "svb_ablation_mode", "full"))

    @staticmethod
    def _text(value) -> str:
        return str(value).replace("\\", "_").replace("/", "_")

    @staticmethod
    def _safe_name(value: Any) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))

    def _warn(self, message: str) -> None:
        if self.logger is not None:
            method = getattr(self.logger, "warn_info", None) or getattr(self.logger, "warning", None) or getattr(self.logger, "info", None)
            if callable(method):
                method(message)
                return
        LOGGER.warning(message)

__all__ = ["SVBPLRCache"]
