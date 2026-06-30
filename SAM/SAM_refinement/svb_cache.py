from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from utils.log_control import log_enabled


LOGGER = logging.getLogger(__name__)


class SAMImageEmbeddingCache:
    """Bounded two-level cache for frozen SAM image-encoder embeddings.

    The key depends only on the exact image content and the SAM model/cache
    version.  It deliberately excludes epoch and teacher predictions because
    neither changes a frozen image encoder output.  CPU memory is an LRU front
    cache; the optional disk layer makes exact augmented views reusable across
    epochs and process restarts.
    """

    def __init__(
        self,
        cfg=None,
        backend_tag: str = "",
        model_tag: str = "",
        enabled: Optional[bool] = None,
        max_items: Optional[int] = None,
    ) -> None:
        if enabled is None:
            legacy_enabled = bool(getattr(cfg, "use_sam_cache", True)) if cfg is not None else True
            enabled = bool(getattr(cfg, "use_sam_embedding_cache", legacy_enabled)) if cfg is not None else True
        self.enabled = bool(enabled)
        if max_items is None:
            max_items = getattr(cfg, "sam_image_embedding_cache_size", 128) if cfg is not None else 128
        self.max_items = max(1, int(max_items))
        self.backend_tag = str(backend_tag)
        self.model_tag = str(model_tag)
        self.cache_version = (
            str(getattr(cfg, "sam_embedding_cache_version", "v2_float_embed"))
            if cfg is not None
            else "v2_float_embed"
        )

        self.disk_enabled = bool(getattr(cfg, "sam_embedding_cache_disk", False)) if cfg is not None else False
        cache_dir = getattr(cfg, "sam_embedding_cache_dir", "./cache/sam_image_embeddings") if cfg is not None else "./cache/sam_image_embeddings"
        self.cache_dir = Path(cache_dir)
        max_disk_gb = float(getattr(cfg, "sam_embedding_cache_max_gb", 32.0)) if cfg is not None else 32.0
        self.max_disk_bytes = max(0, int(max_disk_gb * (1024 ** 3)))
        self.prune_interval = max(1, int(getattr(cfg, "sam_embedding_cache_prune_interval", 256))) if cfg is not None else 256
        store_dtype = getattr(cfg, "sam_embedding_cache_store_dtype", "float16") if cfg is not None else "float16"
        self.store_dtype = self._parse_store_dtype(store_dtype)

        self._entries: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.hits = 0
        self.memory_hits = 0
        self.disk_hits = 0
        self.misses = 0
        self.disk_writes = 0
        self.disk_errors = 0
        self._disk_pruned_once = False

    def get_or_compute(
        self,
        image,
        compute_fn: Callable[[], torch.Tensor],
        device=None,
        dtype=None,
        extra_tag: str = "",
    ) -> Tuple[torch.Tensor, bool]:
        """Return an embedding and whether it came from memory or disk cache."""
        if not self.enabled:
            value = self._ensure_tensor(compute_fn())
            return self._move(value, device=device, dtype=dtype), False

        key = self.make_key(image, extra_tag=extra_tag)
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            self.memory_hits += 1
            self._entries.move_to_end(key)
            return self._move(cached, device=device, dtype=dtype), True

        cached = self._load_disk(key) if self.disk_enabled else None
        if cached is not None:
            self.hits += 1
            self.disk_hits += 1
            self._remember(key, cached)
            return self._move(cached, device=device, dtype=dtype), True

        self.misses += 1
        value = self._ensure_tensor(compute_fn())
        stored = self._for_storage(value)
        self._remember(key, stored)
        if self.disk_enabled:
            self._write_disk(key, stored)
        # Return the stored representation on the first miss as well.  When
        # fp16 storage is selected this keeps miss/hit decoder inputs identical.
        return self._move(stored, device=device, dtype=dtype), False

    def make_key(self, image, extra_tag: str = "") -> str:
        array = self._as_contiguous_array(image)
        digest = hashlib.sha1(array.tobytes()).hexdigest()
        return "|".join(
            (
                self.cache_version,
                self.backend_tag,
                self.model_tag,
                str(extra_tag),
                str(array.dtype),
                str(array.shape),
                digest,
            )
        )

    def clear(self, clear_disk: bool = False) -> None:
        self._entries.clear()
        self.hits = 0
        self.memory_hits = 0
        self.disk_hits = 0
        self.misses = 0
        if clear_disk and self.cache_dir.is_dir():
            for path in self.cache_dir.rglob("*.pt"):
                try:
                    path.unlink()
                except OSError:
                    self.disk_errors += 1

    def cache_info(self) -> dict:
        return {
            "enabled": self.enabled,
            "size": len(self._entries),
            "max_items": self.max_items,
            "memory_size": len(self._entries),
            "memory_max_items": self.max_items,
            "disk_enabled": self.disk_enabled,
            "disk_dir": str(self.cache_dir),
            "disk_max_bytes": self.max_disk_bytes,
            "hits": self.hits,
            "memory_hits": self.memory_hits,
            "disk_hits": self.disk_hits,
            "misses": self.misses,
            "disk_writes": self.disk_writes,
            "disk_errors": self.disk_errors,
        }

    def _remember(self, key: str, value: torch.Tensor) -> None:
        self._entries[key] = value.detach().cpu()
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_items:
            self._entries.popitem(last=False)

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / "{}.pt".format(digest)

    def _load_disk(self, key: str) -> Optional[torch.Tensor]:
        path = self._path_for_key(key)
        if not path.is_file():
            return None
        try:
            payload = self._torch_load(path)
            if not isinstance(payload, dict) or payload.get("cache_key") != key:
                return None
            value = payload.get("embedding")
            if not torch.is_tensor(value):
                return None
            try:
                path.touch()
            except OSError:
                pass
            return value.detach().cpu()
        except Exception:
            self.disk_errors += 1
            return None

    def _write_disk(self, key: str, value: torch.Tensor) -> None:
        path = self._path_for_key(key)
        if path.is_file():
            return
        tmp_path = path.with_name(".{}.{}.{}.tmp".format(path.name, os.getpid(), threading.get_ident()))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"cache_key": key, "embedding": value.detach().cpu()}, str(tmp_path))
            os.replace(str(tmp_path), str(path))
            self.disk_writes += 1
            if not self._disk_pruned_once or self.disk_writes % self.prune_interval == 0:
                self._prune_disk()
                self._disk_pruned_once = True
        except Exception:
            self.disk_errors += 1
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    def _prune_disk(self) -> None:
        if self.max_disk_bytes <= 0 or not self.cache_dir.is_dir():
            return
        try:
            entries = []
            total_bytes = 0
            for path in self.cache_dir.rglob("*.pt"):
                stat = path.stat()
                total_bytes += stat.st_size
                entries.append((stat.st_mtime_ns, stat.st_size, path))
            if total_bytes <= self.max_disk_bytes:
                return
            entries.sort(key=lambda item: item[0])
            for _, size, path in entries:
                try:
                    path.unlink()
                    total_bytes -= size
                except OSError:
                    self.disk_errors += 1
                if total_bytes <= self.max_disk_bytes:
                    break
        except OSError:
            self.disk_errors += 1

    def _for_storage(self, value: torch.Tensor) -> torch.Tensor:
        stored = value.detach().cpu().contiguous()
        if self.store_dtype is not None and stored.is_floating_point():
            stored = stored.to(dtype=self.store_dtype)
        return stored

    @staticmethod
    def _parse_store_dtype(value) -> Optional[torch.dtype]:
        text = str(value or "native").strip().lower()
        if text in ("float16", "fp16", "half"):
            return torch.float16
        if text in ("bfloat16", "bf16"):
            return torch.bfloat16
        if text in ("float32", "fp32"):
            return torch.float32
        if text in ("native", "none", "same"):
            return None
        raise ValueError("Unsupported sam_embedding_cache_store_dtype '{}'".format(value))

    @staticmethod
    def _ensure_tensor(value) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError("SAM image embedding cache only supports torch.Tensor values")
        return value.detach()

    @staticmethod
    def _move(value: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
        if dtype is not None and value.is_floating_point():
            requested_dtype = torch.empty((), dtype=dtype).dtype
            if not torch.empty((), dtype=requested_dtype).is_floating_point():
                raise TypeError(
                    "SAM image embeddings must remain floating point; requested dtype is {}".format(
                        requested_dtype
                    )
                )
        if device is None and dtype is None:
            return value
        return value.to(
            device=device if device is not None else value.device,
            dtype=dtype if dtype is not None else value.dtype,
        )

    @staticmethod
    def _as_contiguous_array(image) -> np.ndarray:
        if torch.is_tensor(image):
            return image.detach().cpu().contiguous().numpy()
        array = np.asarray(image)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return array

    @staticmethod
    def _torch_load(path: Path):
        try:
            return torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(str(path), map_location="cpu")


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
        legacy_enabled = bool(getattr(cfg, "use_sam_cache", False))
        self.enabled = bool(getattr(cfg, "use_svb_output_cache", legacy_enabled))
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

    def _log_enabled(self) -> bool:
        return log_enabled(self.cfg)

    def _warn(self, message: str) -> None:
        if not self._log_enabled():
            return
        if self.logger is not None:
            method = getattr(self.logger, "warn_info", None) or getattr(self.logger, "warning", None) or getattr(self.logger, "info", None)
            if callable(method):
                method(message)
                return
        LOGGER.warning(message)

__all__ = ["SVBPLRCache"]
