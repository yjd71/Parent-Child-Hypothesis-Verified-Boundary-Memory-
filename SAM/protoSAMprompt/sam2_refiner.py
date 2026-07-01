import hashlib
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.utils2 import extract_points


class Sam2PromptRefiner:
    CACHE_SCHEMA = "sam2_predictor_state_v1"

    def __init__(
        self,
        checkpoint,
        model_cfg,
        device,
        multimask_output=True,
        use_bfloat16=True,
        embedding_cache=None,
    ):
        self.checkpoint = Path(checkpoint)
        self.model_cfg = model_cfg
        self.model_cfg_tag = self._model_config_tag(model_cfg)
        self.device = self._normalize_device(device)
        self.multimask_output = bool(multimask_output)
        self.use_bfloat16 = bool(use_bfloat16)
        self.embedding_cache = embedding_cache

        self._ensure_sam2_importable()
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.model = build_sam2(
            self.model_cfg,
            str(self.checkpoint),
            device=str(self.device),
            mode="eval",
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.predictor = SAM2ImagePredictor(self.model)

    def set_image(self, image):
        """Set an image using a complete, validated SAM2 predictor-state cache."""
        cache_before = self.embedding_cache.cache_info() if self.embedding_cache is not None else {}

        def compute_state():
            with torch.inference_mode():
                self.predictor.set_image(image)
            return self._export_predictor_state()

        if self.embedding_cache is None:
            state = compute_state()
            self._validate_predictor_state(state)
            cache_hit = False
        else:
            state, cache_hit = self.embedding_cache.get_or_compute(
                image,
                compute_state,
                device=self.device,
                dtype=torch.float32,
                extra_tag=self._cache_extra_tag(),
                validator=self._validate_predictor_state,
            )

        self._restore_predictor_state(state)
        stats = self._state_stats(state)
        cache_after = self.embedding_cache.cache_info() if self.embedding_cache is not None else {}
        stats.update(
            {
                "cache_hit": bool(cache_hit),
                "cache_source": self._cache_source(cache_before, cache_after, cache_hit),
                "cache_info": cache_after if self.embedding_cache is not None else None,
            }
        )
        return stats

    def _export_predictor_state(self):
        features = getattr(self.predictor, "_features", None)
        orig_hw = getattr(self.predictor, "_orig_hw", None)
        if not isinstance(features, dict):
            raise RuntimeError("SAM2 predictor did not produce a feature dictionary")
        return {
            "image_embed": features.get("image_embed"),
            "high_res_feats": tuple(features.get("high_res_feats", ())),
            "orig_hw": tuple(tuple(int(value) for value in hw) for hw in (orig_hw or ())),
        }

    def _restore_predictor_state(self, state):
        self._validate_predictor_state(state)
        if hasattr(self.predictor, "reset_predictor"):
            self.predictor.reset_predictor()
        self.predictor._features = {
            "image_embed": state["image_embed"],
            "high_res_feats": list(state["high_res_feats"]),
        }
        self.predictor._orig_hw = [tuple(int(value) for value in hw) for hw in state["orig_hw"]]
        self.predictor._is_image_set = True
        self.predictor._is_batch = False

    def _validate_predictor_state(self, state):
        if not isinstance(state, dict):
            raise TypeError("SAM2 cached predictor state must be a dict")
        missing = {"image_embed", "high_res_feats", "orig_hw"}.difference(state)
        if missing:
            raise ValueError("SAM2 cached predictor state is missing fields: {}".format(sorted(missing)))
        image_embed = state["image_embed"]
        high_res_feats = state["high_res_feats"]
        orig_hw = state["orig_hw"]
        if not torch.is_tensor(image_embed):
            raise TypeError("SAM2 cached image_embed must be a tensor")
        if not isinstance(high_res_feats, (list, tuple)) or not high_res_feats:
            raise ValueError("SAM2 cached high_res_feats must be a non-empty list or tuple")
        tensors = [image_embed, *high_res_feats]
        for name, tensor in zip(
            ["image_embed", *["high_res_feats[{}]".format(i) for i in range(len(high_res_feats))]],
            tensors,
        ):
            if not torch.is_tensor(tensor):
                raise TypeError("SAM2 cached {} must be a tensor".format(name))
            if not tensor.is_floating_point():
                raise TypeError("SAM2 cached {} must be floating point, got {}".format(name, tensor.dtype))
            if tensor.numel() == 0 or tensor.ndim != 4 or tensor.shape[0] != 1:
                raise ValueError("SAM2 cached {} has invalid shape {}".format(name, tuple(tensor.shape)))
            if not torch.isfinite(tensor).all().item():
                raise ValueError("SAM2 cached {} contains non-finite values".format(name))
            if not torch.count_nonzero(tensor).item():
                raise ValueError("SAM2 cached {} is all zero".format(name))
        expected_sizes = getattr(self.predictor, "_bb_feat_sizes", None)
        ordered_features = [*high_res_feats, image_embed]
        if expected_sizes is not None:
            if len(expected_sizes) != len(ordered_features):
                raise ValueError(
                    "SAM2 cached feature count {} does not match predictor feature sizes {}".format(
                        len(ordered_features), len(expected_sizes)
                    )
                )
            for index, (tensor, expected_hw) in enumerate(zip(ordered_features, expected_sizes)):
                actual_hw = tuple(int(value) for value in tensor.shape[-2:])
                expected_hw = tuple(int(value) for value in expected_hw)
                if actual_hw != expected_hw:
                    raise ValueError(
                        "SAM2 cached feature {} has spatial shape {}, expected {}".format(
                            index, actual_hw, expected_hw
                        )
                    )
        if not isinstance(orig_hw, (list, tuple)) or len(orig_hw) != 1:
            raise ValueError("SAM2 cached orig_hw must contain exactly one image size")
        hw = orig_hw[0]
        if not isinstance(hw, (list, tuple)) or len(hw) != 2 or any(int(value) <= 0 for value in hw):
            raise ValueError("SAM2 cached orig_hw is invalid: {}".format(orig_hw))
        return True

    def _cache_extra_tag(self):
        image_size = getattr(self.model, "image_size", "unknown")
        no_mem_embed = getattr(self.model, "directly_add_no_mem_embed", "unknown")
        return "{}|model_cfg={}|image_size={}|direct_no_mem={}".format(
            self.CACHE_SCHEMA,
            getattr(self, "model_cfg_tag", self._model_config_tag(self.model_cfg)),
            image_size,
            no_mem_embed,
        )

    @staticmethod
    def _model_config_tag(model_cfg):
        config_text = str(model_cfg)
        this_dir = Path(__file__).resolve().parent
        candidates = (
            Path(config_text),
            this_dir.parent / "sam2" / "sam2" / config_text,
            this_dir.parent / "sam2" / config_text,
        )
        for candidate in candidates:
            if candidate.is_file():
                try:
                    digest = hashlib.sha1(candidate.read_bytes()).hexdigest()
                    return "{}:{}".format(candidate.resolve(), digest)
                except OSError:
                    break
        return config_text

    @staticmethod
    def _cache_source(before, after, cache_hit):
        if not cache_hit:
            return "miss"
        if int(after.get("memory_hits", 0)) > int(before.get("memory_hits", 0)):
            return "memory"
        if int(after.get("disk_hits", 0)) > int(before.get("disk_hits", 0)):
            return "disk"
        return "cache"

    @staticmethod
    def _state_stats(state):
        named_tensors = [("image_embed", state["image_embed"])] + [
            ("high_res_feats[{}]".format(index), tensor)
            for index, tensor in enumerate(state["high_res_feats"])
        ]
        per_feature = []
        total_count = 0
        total_sum = 0.0
        total_square_sum = 0.0
        feature_min = float("inf")
        feature_max = float("-inf")
        for name, tensor in named_tensors:
            values = tensor.detach().float()
            count = values.numel()
            total_count += count
            total_sum += float(values.sum().item())
            total_square_sum += float(values.square().sum().item())
            feature_min = min(feature_min, float(values.min().item()))
            feature_max = max(feature_max, float(values.max().item()))
            per_feature.append(
                {
                    "name": name,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "min": float(values.min().item()),
                    "max": float(values.max().item()),
                    "mean": float(values.mean().item()),
                    "std": float(values.std(unbiased=False).item()),
                }
            )
        feature_mean = total_sum / total_count
        feature_variance = max(0.0, total_square_sum / total_count - feature_mean ** 2)
        return {
            "feature_shapes": {
                "image_embed": tuple(state["image_embed"].shape),
                "high_res_feats": [tuple(tensor.shape) for tensor in state["high_res_feats"]],
            },
            "feature_dtype": str(state["image_embed"].dtype),
            "feature_min": feature_min,
            "feature_max": feature_max,
            "feature_mean": feature_mean,
            "feature_std": feature_variance ** 0.5,
            "feature_stats": per_feature,
            "orig_hw": [tuple(hw) for hw in state["orig_hw"]],
        }

    @staticmethod
    def _normalize_device(device):
        if isinstance(device, torch.device):
            return device
        if isinstance(device, int):
            return torch.device("cuda:{}".format(device))
        return torch.device(device)

    @staticmethod
    def _ensure_sam2_importable():
        if importlib.util.find_spec("sam2") is not None:
            return
        this_dir = Path(__file__).resolve().parent
        sam2_repo = this_dir.parent / "sam2"
        if sam2_repo.is_dir():
            sys.path.insert(0, str(sam2_repo))

    def _autocast_context(self):
        if self.device.type != "cuda" or not self.use_bfloat16:
            return nullcontext()
        is_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
        if not is_supported():
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def _prepare_masks(self, coarse_masks, coarse_threshold):
        if isinstance(coarse_masks, list):
            coarse_masks = np.stack(coarse_masks, axis=0)
        if coarse_masks.ndim == 2:
            coarse_masks = coarse_masks[None, ...]

        masks = torch.as_tensor(coarse_masks, device=self.device)
        if masks.dtype == torch.bool:
            masks = masks.to(torch.uint8)
        elif masks.is_floating_point():
            masks = (masks > coarse_threshold).to(torch.uint8)
        else:
            masks = (masks > 0).to(torch.uint8)
        if masks.ndim != 3:
            raise AssertionError("coarse mask dim must be (n, h, w), but got {}".format(masks.shape))
        return masks

    @staticmethod
    def _mask_to_box(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None

        h, w = mask.shape
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        if x1 == x2:
            if x2 < w - 1:
                x2 += 1
            elif x1 > 0:
                x1 -= 1
        if y1 == y2:
            if y2 < h - 1:
                y2 += 1
            elif y1 > 0:
                y1 -= 1
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _mask_to_logits(mask, strength):
        logits = np.where(mask > 0, float(strength), -float(strength)).astype(np.float32)
        logits = cv2.resize(logits, (256, 256), interpolation=cv2.INTER_LINEAR)
        return logits[None, ...].astype(np.float32)

    @staticmethod
    def _points_from_mask(mask_tensor, add_neg, gamma):
        point_coords, point_labels, _ = extract_points(
            mask_tensor,
            add_neg=add_neg,
            use_mask=False,
            gamma=gamma,
        )
        coords = point_coords[0].detach().cpu().numpy().astype(np.float32)
        labels = point_labels[0].detach().cpu().numpy().astype(np.int32)
        return coords, labels

    def __call__(
        self,
        image,
        coarse_masks,
        use_point=True,
        use_box=True,
        use_mask=True,
        add_neg=True,
        iters=1,
        gamma=4.0,
        strength=30,
        coarse_threshold=0.5,
    ):
        masks_tensor = self._prepare_masks(coarse_masks, coarse_threshold)
        pred_masks = masks_tensor.detach().cpu().numpy().astype(np.uint8)
        low_res_logits = [None for _ in range(len(pred_masks))]

        self.set_image(image)
        with torch.inference_mode():
            for _ in range(max(1, int(iters))):
                next_masks = []
                next_logits = []
                current_tensor = torch.as_tensor(pred_masks, device=self.device, dtype=torch.uint8)

                for idx, pred_mask in enumerate(pred_masks):
                    box = self._mask_to_box(pred_mask) if use_box else None
                    if use_point:
                        point_coords, point_labels = self._points_from_mask(
                            current_tensor[idx:idx + 1],
                            add_neg=add_neg,
                            gamma=gamma,
                        )
                    else:
                        point_coords, point_labels = None, None

                    if use_mask:
                        mask_input = low_res_logits[idx]
                        if mask_input is None:
                            mask_input = self._mask_to_logits(pred_mask, strength)
                    else:
                        mask_input = None

                    if box is None and point_coords is None and mask_input is None:
                        next_masks.append(pred_mask.astype(np.uint8))
                        next_logits.append(None)
                        continue

                    with self._autocast_context():
                        masks, scores, logits = self.predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            box=box,
                            mask_input=mask_input,
                            multimask_output=self.multimask_output,
                            return_logits=False,
                            normalize_coords=True,
                        )

                    best_idx = int(np.argmax(scores))
                    next_masks.append((masks[best_idx] > 0).astype(np.uint8))
                    next_logits.append(logits[best_idx:best_idx + 1].astype(np.float32))

                pred_masks = np.stack(next_masks, axis=0).astype(np.uint8)
                low_res_logits = next_logits

        return pred_masks, low_res_logits
