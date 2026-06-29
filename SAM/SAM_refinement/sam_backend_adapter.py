from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from SAM.protoSAMprompt.train_pseudo_refiner import (
    Sam1PseudoLabelRefiner,
    Sam2PseudoLabelRefiner,
    build_sam_pseudo_label_refiner,
)

try:
    from .svb_utils import SAMInferenceError
except ImportError:
    from SAM.SAM_refinement.svb_utils import SAMInferenceError


class ExistingSAMBackendAdapter(nn.Module):
    """Adapter around the existing SAM/protoSAMprompt backends.

    Public predict input:
        images: [B, 3, H, W], normalized model input.
        teacher_prob: [B, 1, Ht, Wt], teacher pseudo probability.

    Public predict output:
        masks: [B, K, Ht, Wt]
        scores: [B, K]
        valid_candidates: bool [B, K]
        logits: optional [B, K, ...]
        backend_aux: dict
    """

    def __init__(self, config, device=None, logger=None) -> None:
        super().__init__()
        self.config = config
        self.device = self._normalize_device(device)
        self.logger = logger
        self.backend = str(getattr(config, "sam_pseudo_backend", "sam1")).lower()
        self.threshold = float(getattr(config, "sam_pseudo_threshold", 0.5))
        self.iters = int(getattr(config, "sam_pseudo_iters", 1))
        self.multimask_output = True
        self.build_sam_pseudo_label_refiner = build_sam_pseudo_label_refiner

        if self.backend in ("sam1", "sam", "v1"):
            self.backend_name = "sam1"
            self.refiner = Sam1PseudoLabelRefiner(config=config, device=self.device, logger=logger)
        elif self.backend in ("sam2", "v2"):
            self.backend_name = "sam2"
            self.refiner = Sam2PseudoLabelRefiner(config=config, device=self.device, logger=logger)
        else:
            raise ValueError("Unsupported sam_pseudo_backend '{}'. Expected 'sam1' or 'sam2'.".format(self.backend))

        self._freeze_backend()

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        teacher_prob: torch.Tensor,
        boxes=None,
        point_coords=None,
        point_labels=None,
        mask_inputs=None,
        prompt_pack: Optional[Dict[str, Any]] = None,
        epoch=None,
        step=None,
    ) -> Dict[str, Any]:
        """Predict SAM masks with optional external prompts.

        Shape:
            images: [B, 3, H, W]
            teacher_prob: [B, 1, Ht, Wt]
            boxes: optional per-sample xyxy boxes
            point_coords: optional per-sample xy point coords
            point_labels: optional per-sample point labels
            mask_inputs: optional mask prompts

        Every returned mask slot has a corresponding valid_candidates entry.
        Failed samples and padding slots are zero-filled and marked invalid.
        """
        try:
            images, teacher_prob = self._validate_inputs(images, teacher_prob)
            boxes = self._extract_prompt(prompt_pack, boxes, "boxes")
            point_coords = self._extract_prompt(prompt_pack, point_coords, "point_coords")
            point_labels = self._extract_prompt(prompt_pack, point_labels, "point_labels")
            mask_inputs = self._extract_prompt(prompt_pack, mask_inputs, "mask_inputs", fallback_key="mask_prompt")
            has_external_prompt = any(value is not None for value in (boxes, point_coords, point_labels, mask_inputs))

            if self.backend_name == "sam1":
                result = (
                    self._predict_sam1_external_prompt(images, teacher_prob, boxes, point_coords, point_labels, mask_inputs)
                    if has_external_prompt
                    else self._predict_auto_prompt(images, teacher_prob)
                )
            else:
                result = (
                    self._predict_sam2_external_prompt(images, teacher_prob, boxes, point_coords, point_labels, mask_inputs)
                    if has_external_prompt
                    else self._predict_auto_prompt(images, teacher_prob)
                )
            result["backend_aux"].update(
                {
                    "backend": self.backend_name,
                    "used_external_prompt": has_external_prompt,
                    "epoch": epoch,
                    "step": step,
                }
            )
            return result
        except SAMInferenceError:
            raise
        except Exception as exc:
            raise SAMInferenceError(
                "SAM backend prediction failed",
                epoch=epoch,
                step=step,
                failures=[{"backend": getattr(self, "backend_name", "unknown"), "error": str(exc)}],
            ) from exc

    def _predict_auto_prompt(self, images: torch.Tensor, teacher_prob: torch.Tensor) -> Dict[str, Any]:
        image_np_list = self._denormalize_batch(images)
        pseudo_for_sam = self._resize_prob_to_image(teacher_prob, images)
        sample_masks: List[torch.Tensor] = []
        sample_scores: List[torch.Tensor] = []
        failed: List[Tuple[int, str]] = []

        for idx, image_np in enumerate(image_np_list):
            try:
                coarse_mask = (pseudo_for_sam[idx, 0] > self.threshold).to(torch.uint8)
                if coarse_mask.sum().item() == 0:
                    raise ValueError("empty_coarse_mask")
                refined = self.refiner._refine_one(image_np, coarse_mask.detach().cpu().numpy())
                refined = torch.as_tensor(refined, device=teacher_prob.device, dtype=teacher_prob.dtype).reshape(1, *images.shape[-2:])
                refined = self._resize_candidate_masks(refined, teacher_prob)
                sample_masks.append(refined.clamp(0, 1))
                sample_scores.append(teacher_prob.new_full((refined.size(0),), 0.5))
            except Exception as exc:
                failed.append((idx, str(exc)))
                sample_masks.append(teacher_prob.new_empty((0, *teacher_prob.shape[-2:])))
                sample_scores.append(teacher_prob.new_empty((0,)))

        masks, scores, valid_candidates = self._pad_candidates(sample_masks, sample_scores, teacher_prob)
        return {
            "masks": masks,
            "scores": scores,
            "valid_candidates": valid_candidates,
            "logits": None,
            "backend_aux": {
                "used_fallback": bool(failed),
                "fallback_samples": failed,
                "path": "auto_prompt",
            },
        }

    def _predict_sam1_external_prompt(
        self,
        images: torch.Tensor,
        teacher_prob: torch.Tensor,
        boxes,
        point_coords,
        point_labels,
        mask_inputs,
    ) -> Dict[str, Any]:
        try:
            from SAM.segment_anything.utils.transforms import ResizeLongestSide
        except ImportError:
            from ..segment_anything.utils.transforms import ResizeLongestSide

        sam = self.refiner.sam
        resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
        image_np_list = self._denormalize_batch(images)
        sample_masks: List[torch.Tensor] = []
        sample_scores: List[torch.Tensor] = []
        sample_logits: List[Optional[torch.Tensor]] = []
        failed: List[Tuple[int, str]] = []
        embedding_cache_hits = 0
        embedding_cache_misses = 0
        embedding_cache = getattr(self.refiner, "embedding_cache", None)

        for idx, image_np in enumerate(image_np_list):
            try:
                prompt_dict = self._prepare_sam1_prompt_dict(
                    image_np=image_np,
                    image_tensor=images[idx],
                    teacher_prob=teacher_prob[idx : idx + 1],
                    resize_transform=resize_transform,
                    boxes=self._sample_boxes(boxes, idx, images.size(0)),
                    point_coords=self._sample_points(point_coords, idx, images.size(0)),
                    point_labels=self._sample_labels(point_labels, idx, images.size(0)),
                    mask_inputs=self._sample_mask_inputs(mask_inputs, idx, images.size(0)),
                )
                if not any(key in prompt_dict for key in ("boxes", "point_coords", "mask_inputs")):
                    raise ValueError("empty_external_prompt")

                if embedding_cache is not None:
                    image_embeddings, cache_hit = embedding_cache.get_or_compute(
                        image_np,
                        lambda: sam.image_encoder(torch.stack([sam.preprocess(prompt_dict["image"])], dim=0)),
                        device=sam.device,
                        dtype=prompt_dict["image"].dtype,
                        extra_tag="sam1_preprocess_v1",
                    )
                    if cache_hit:
                        embedding_cache_hits += 1
                    else:
                        embedding_cache_misses += 1
                else:
                    input_image = torch.stack([sam.preprocess(prompt_dict["image"])], dim=0)
                    image_embeddings = sam.image_encoder(input_image)
                sam_output = sam.forward_with_image_embeddings(
                    image_embeddings,
                    [prompt_dict],
                    multimask_output=self.multimask_output,
                )[0]
                masks = self._sam1_mask_logits_to_prob(sam_output["masks"])
                scores = sam_output.get("iou_predictions")
                logits = sam_output.get("low_res_logits")
                masks = self._resize_candidate_masks(masks.to(device=teacher_prob.device, dtype=teacher_prob.dtype), teacher_prob)
                scores = self._flatten_scores(scores, masks.size(0), teacher_prob.device, default=0.5)
                sample_masks.append(masks.clamp(0, 1))
                sample_scores.append(scores)
                sample_logits.append(self._flatten_logits(logits, teacher_prob.device))
            except Exception as exc:
                failed.append((idx, str(exc)))
                sample_masks.append(teacher_prob.new_empty((0, *teacher_prob.shape[-2:])))
                sample_scores.append(teacher_prob.new_empty((0,)))
                sample_logits.append(None)

        masks, scores, valid_candidates = self._pad_candidates(sample_masks, sample_scores, teacher_prob)
        logits = self._pad_logits(sample_logits, masks.size(1), teacher_prob)
        return {
            "masks": masks,
            "scores": scores,
            "valid_candidates": valid_candidates,
            "logits": logits,
            "backend_aux": {
                "used_fallback": bool(failed),
                "fallback_samples": failed,
                "path": "sam1_external_prompt",
                "embedding_cache_hits": embedding_cache_hits,
                "embedding_cache_misses": embedding_cache_misses,
                "embedding_cache_info": embedding_cache.cache_info() if embedding_cache is not None else None,
            },
        }

    def _predict_sam2_external_prompt(
        self,
        images: torch.Tensor,
        teacher_prob: torch.Tensor,
        boxes,
        point_coords,
        point_labels,
        mask_inputs,
    ) -> Dict[str, Any]:
        image_np_list = self._denormalize_batch(images)
        sam2_refiner = self.refiner.sam2_refiner
        sample_masks: List[torch.Tensor] = []
        sample_scores: List[torch.Tensor] = []
        sample_logits: List[Optional[torch.Tensor]] = []
        failed: List[Tuple[int, str]] = []

        for idx, image_np in enumerate(image_np_list):
            try:
                prompt_args = self._prepare_sam2_prompt_args(
                    boxes=self._sample_boxes(boxes, idx, images.size(0)),
                    point_coords=self._sample_points(point_coords, idx, images.size(0)),
                    point_labels=self._sample_labels(point_labels, idx, images.size(0)),
                    mask_inputs=self._sample_mask_inputs(mask_inputs, idx, images.size(0)),
                )
                if not any(value is not None for value in prompt_args.values()):
                    raise ValueError("empty_external_prompt")

                sam2_refiner.predictor.set_image(image_np)
                prompt_calls = self._sam2_prompt_calls(prompt_args)
                if not prompt_calls:
                    raise ValueError("empty_external_prompt")

                masks_collected: List[torch.Tensor] = []
                scores_collected: List[torch.Tensor] = []
                logits_collected: List[torch.Tensor] = []
                with torch.inference_mode():
                    for call_args in prompt_calls:
                        with sam2_refiner._autocast_context():
                            masks_np, scores_np, logits_np = sam2_refiner.predictor.predict(
                                point_coords=call_args.get("point_coords"),
                                point_labels=call_args.get("point_labels"),
                                box=call_args.get("box"),
                                mask_input=call_args.get("mask_input"),
                                multimask_output=sam2_refiner.multimask_output,
                                return_logits=False,
                                normalize_coords=True,
                            )
                        masks_collected.append(torch.as_tensor(masks_np, device=teacher_prob.device, dtype=teacher_prob.dtype))
                        scores_collected.append(torch.as_tensor(scores_np, device=teacher_prob.device, dtype=teacher_prob.dtype).reshape(-1))
                        logits_collected.append(torch.as_tensor(logits_np, device=teacher_prob.device, dtype=teacher_prob.dtype))

                masks = torch.cat([self._ensure_khw(mask) for mask in masks_collected], dim=0)
                scores = torch.cat(scores_collected, dim=0) if scores_collected else teacher_prob.new_zeros((masks.size(0),))
                logits = torch.cat([self._ensure_khw(logit) for logit in logits_collected], dim=0) if logits_collected else None
                masks = self._resize_candidate_masks(masks, teacher_prob)
                sample_masks.append(masks.clamp(0, 1))
                sample_scores.append(scores[: masks.size(0)] if scores.numel() else teacher_prob.new_full((masks.size(0),), 0.5))
                sample_logits.append(logits)
            except Exception as exc:
                failed.append((idx, str(exc)))
                sample_masks.append(teacher_prob.new_empty((0, *teacher_prob.shape[-2:])))
                sample_scores.append(teacher_prob.new_empty((0,)))
                sample_logits.append(None)

        masks, scores, valid_candidates = self._pad_candidates(sample_masks, sample_scores, teacher_prob)
        logits = self._pad_logits(sample_logits, masks.size(1), teacher_prob)
        return {
            "masks": masks,
            "scores": scores,
            "valid_candidates": valid_candidates,
            "logits": logits,
            "backend_aux": {
                "used_fallback": bool(failed),
                "fallback_samples": failed,
                "path": "sam2_external_prompt",
            },
        }

    def _prepare_sam1_prompt_dict(
        self,
        image_np,
        image_tensor: torch.Tensor,
        teacher_prob: torch.Tensor,
        resize_transform,
        boxes,
        point_coords,
        point_labels,
        mask_inputs,
    ) -> Dict[str, torch.Tensor]:
        from SAM.protoSAMprompt.sam_refiner import prepare_image

        sam = self.refiner.sam
        image = prepare_image(image_np, resize_transform, sam.device)
        original_size = tuple(image_np.shape[:2])
        prompt_dict: Dict[str, torch.Tensor] = {
            "image": image,
            "original_size": original_size,
        }

        box_tensor = self._tensor_or_none(boxes, device=sam.device, dtype=torch.float32)
        if box_tensor is not None and box_tensor.numel() > 0:
            box_tensor = box_tensor.reshape(-1, 4)
        else:
            box_tensor = None

        coords_tensor = self._tensor_or_none(point_coords, device=sam.device, dtype=torch.float32)
        labels_tensor = self._tensor_or_none(point_labels, device=sam.device, dtype=torch.long)
        if coords_tensor is not None and labels_tensor is not None and coords_tensor.numel() > 0 and labels_tensor.numel() > 0:
            coords_tensor = coords_tensor.reshape(1, -1, 2) if coords_tensor.dim() == 2 else coords_tensor.reshape(-1, coords_tensor.size(-2), 2)
            labels_tensor = labels_tensor.reshape(1, -1) if labels_tensor.dim() == 1 else labels_tensor.reshape(labels_tensor.size(0), -1)
            if labels_tensor.size(1) != coords_tensor.size(1):
                raise ValueError(
                    "point label count {} does not match coordinate count {}".format(
                        labels_tensor.size(1), coords_tensor.size(1)
                    )
                )
        else:
            coords_tensor = None
            labels_tensor = None

        mask_tensor = self._prepare_mask_input_tensor(mask_inputs, device=sam.device, dtype=torch.float32)
        if mask_tensor is not None and mask_tensor.numel() == 0:
            mask_tensor = None

        box_tensor, coords_tensor, labels_tensor, mask_tensor = self._broadcast_sam1_prompt_batches(
            box_tensor,
            coords_tensor,
            labels_tensor,
            mask_tensor,
        )
        if box_tensor is not None:
            prompt_dict["boxes"] = resize_transform.apply_boxes_torch(box_tensor, original_size)
        if coords_tensor is not None and labels_tensor is not None:
            prompt_dict["point_coords"] = resize_transform.apply_coords_torch(coords_tensor, original_size)
            prompt_dict["point_labels"] = labels_tensor
        if mask_tensor is not None:
            prompt_dict["mask_inputs"] = mask_tensor

        return prompt_dict

    def _prepare_sam2_prompt_args(self, boxes, point_coords, point_labels, mask_inputs) -> Dict[str, Optional[np.ndarray]]:
        box = self._to_numpy_or_none(boxes, dtype=np.float32)
        coords = self._to_numpy_or_none(point_coords, dtype=np.float32)
        labels = self._to_numpy_or_none(point_labels, dtype=np.int32)
        mask_input = self._to_numpy_or_none(mask_inputs, dtype=np.float32)

        if box is not None:
            box = box.reshape(-1, 4)
        if coords is not None:
            coords = coords.reshape(-1, coords.shape[-2], 2) if coords.ndim == 3 else coords.reshape(1, -1, 2)
        if labels is not None:
            labels = labels.reshape(coords.shape[0], -1) if coords is not None else labels.reshape(1, -1)
        if mask_input is not None:
            mask_input = self._prepare_sam2_mask_input(mask_input)
        return {
            "boxes": box,
            "point_coords": coords,
            "point_labels": labels,
            "mask_inputs": mask_input,
        }

    def _sam2_prompt_calls(self, prompt_args: Dict[str, Optional[np.ndarray]]) -> List[Dict[str, Any]]:
        boxes = prompt_args.get("boxes")
        coords = prompt_args.get("point_coords")
        labels = prompt_args.get("point_labels")
        masks = prompt_args.get("mask_inputs")
        counts = [value.shape[0] for value in (boxes, coords, labels, masks) if value is not None and value.ndim > 0]
        prompt_count = max(counts) if counts else 0
        calls: List[Dict[str, Any]] = []
        for idx in range(prompt_count):
            call: Dict[str, Any] = {}
            if boxes is not None:
                call["box"] = boxes[min(idx, boxes.shape[0] - 1)]
            if coords is not None and labels is not None:
                call["point_coords"] = coords[min(idx, coords.shape[0] - 1)]
                call["point_labels"] = labels[min(idx, labels.shape[0] - 1)]
            if masks is not None:
                call["mask_input"] = masks[min(idx, masks.shape[0] - 1)]
            calls.append(call)
        return calls

    def _fallback_output(self, teacher_prob: torch.Tensor, reason: str) -> Dict[str, Any]:
        raise SAMInferenceError(
            reason,
            sample_indices=list(range(teacher_prob.size(0))) if torch.is_tensor(teacher_prob) and teacher_prob.dim() else [],
            failures=[{"backend": getattr(self, "backend_name", "unknown"), "error": reason}],
        )

    @staticmethod
    def _extract_prompt(prompt_pack: Optional[Dict[str, Any]], explicit_value, key: str, fallback_key: Optional[str] = None):
        if explicit_value is not None:
            return explicit_value
        if not prompt_pack:
            return None
        if key in prompt_pack:
            return prompt_pack[key]
        if fallback_key is not None:
            return prompt_pack.get(fallback_key)
        return None

    def _denormalize_batch(self, images: torch.Tensor) -> List[np.ndarray]:
        return [self.refiner._denormalize_image(images[idx]) for idx in range(images.size(0))]

    def _validate_inputs(self, images: torch.Tensor, teacher_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(images) or images.dim() != 4 or images.size(1) != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        if not torch.is_tensor(teacher_prob) or teacher_prob.dim() != 4 or teacher_prob.size(1) != 1:
            raise ValueError("teacher_prob must have shape [B,1,H,W]")
        if images.size(0) != teacher_prob.size(0):
            raise ValueError("images and teacher_prob batch sizes must match")
        if self.device is not None:
            images = images.to(self.device)
            teacher_prob = teacher_prob.to(self.device)
        return images, teacher_prob

    @staticmethod
    def _normalize_device(device):
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, torch.device):
            return device
        if isinstance(device, int):
            return torch.device("cuda:{}".format(device))
        return torch.device(device)

    def _freeze_backend(self) -> None:
        modules = []
        if self.backend_name == "sam1" and hasattr(self.refiner, "sam"):
            modules.append(self.refiner.sam)
        if self.backend_name == "sam2" and hasattr(self.refiner, "sam2_refiner"):
            modules.append(self.refiner.sam2_refiner.model)
        for module in modules:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def _resize_prob_to_image(teacher_prob: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        if tuple(teacher_prob.shape[-2:]) == tuple(images.shape[-2:]):
            return teacher_prob.detach()
        return F.interpolate(teacher_prob.detach().float(), size=images.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def _resize_candidate_masks(masks: torch.Tensor, teacher_prob: torch.Tensor) -> torch.Tensor:
        masks = ExistingSAMBackendAdapter._ensure_khw(masks)
        if tuple(masks.shape[-2:]) == tuple(teacher_prob.shape[-2:]):
            return masks
        return F.interpolate(
            masks.unsqueeze(1).float(),
            size=teacher_prob.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]

    @staticmethod
    def _ensure_khw(masks: torch.Tensor) -> torch.Tensor:
        if masks.dim() == 2:
            return masks.unsqueeze(0)
        if masks.dim() == 3:
            return masks
        if masks.dim() == 4:
            return masks.reshape(-1, *masks.shape[-2:])
        raise ValueError("mask tensor must have shape [H,W], [K,H,W], or [N,K,H,W]")

    @staticmethod
    def _flatten_sam1_masks(masks: torch.Tensor) -> torch.Tensor:
        return ExistingSAMBackendAdapter._ensure_khw(masks)

    @staticmethod
    def _sam1_mask_logits_to_prob(masks: torch.Tensor) -> torch.Tensor:
        return ExistingSAMBackendAdapter._flatten_sam1_masks(masks).float().sigmoid()

    @staticmethod
    def _flatten_scores(scores: Optional[torch.Tensor], count: int, device, default: float = 0.5) -> torch.Tensor:
        if scores is None:
            return torch.full((count,), float(default), device=device)
        flat = scores.detach().to(device=device).reshape(-1)
        if flat.numel() == 0:
            return torch.full((count,), float(default), device=device)
        if flat.numel() < count:
            flat = torch.cat((flat, torch.full((count - flat.numel(),), float(default), device=device, dtype=flat.dtype)))
        return flat[:count]

    @staticmethod
    def _flatten_logits(logits: Optional[torch.Tensor], device) -> Optional[torch.Tensor]:
        if logits is None:
            return None
        return ExistingSAMBackendAdapter._ensure_khw(logits.detach().to(device=device))

    @staticmethod
    def _pad_candidates(
        sample_masks: Sequence[torch.Tensor],
        sample_scores: Sequence[torch.Tensor],
        teacher_prob: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(sample_masks)
        max_k = max((mask.size(0) for mask in sample_masks), default=1)
        max_k = max(1, max_k)
        height, width = teacher_prob.shape[-2:]
        masks_out = teacher_prob.new_zeros((batch_size, max_k, height, width))
        scores_out = teacher_prob.new_zeros((batch_size, max_k))
        valid_out = torch.zeros((batch_size, max_k), device=teacher_prob.device, dtype=torch.bool)
        for idx, masks in enumerate(sample_masks):
            if masks.numel() == 0:
                continue
            masks = ExistingSAMBackendAdapter._resize_candidate_masks(masks.to(device=teacher_prob.device, dtype=teacher_prob.dtype), teacher_prob)
            k = masks.size(0)
            masks_out[idx, :k] = masks
            valid_out[idx, :k] = True
            scores = sample_scores[idx].to(device=teacher_prob.device, dtype=teacher_prob.dtype).reshape(-1)
            scores_out[idx, : min(k, scores.numel())] = scores[: min(k, scores.numel())]
        return masks_out.clamp(0, 1), scores_out, valid_out

    @staticmethod
    def _broadcast_sam1_prompt_batches(boxes, point_coords, point_labels, mask_inputs):
        named = {
            "boxes": boxes,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_inputs": mask_inputs,
        }
        batch_sizes = [value.size(0) for value in named.values() if torch.is_tensor(value)]
        target_batch = max(batch_sizes, default=1)
        broadcasted = {}
        for name, value in named.items():
            if value is None:
                broadcasted[name] = None
            elif value.size(0) == target_batch:
                broadcasted[name] = value
            elif value.size(0) == 1:
                broadcasted[name] = value.expand(target_batch, *value.shape[1:])
            else:
                raise ValueError(
                    "SAM1 prompt batch mismatch: {} has batch {}, expected 1 or {}".format(
                        name, value.size(0), target_batch
                    )
                )
        return (
            broadcasted["boxes"],
            broadcasted["point_coords"],
            broadcasted["point_labels"],
            broadcasted["mask_inputs"],
        )

    @staticmethod
    def _pad_logits(sample_logits: Sequence[Optional[torch.Tensor]], max_k: int, teacher_prob: torch.Tensor):
        valid_logits = [logit for logit in sample_logits if logit is not None and logit.numel() > 0]
        if not valid_logits:
            return None
        logit_shape = valid_logits[0].shape[-2:]
        logits_out = teacher_prob.new_zeros((len(sample_logits), max_k, *logit_shape))
        for idx, logits in enumerate(sample_logits):
            if logits is None or logits.numel() == 0:
                continue
            logits = ExistingSAMBackendAdapter._ensure_khw(logits.to(device=teacher_prob.device, dtype=teacher_prob.dtype))
            k = min(max_k, logits.size(0))
            if tuple(logits.shape[-2:]) != tuple(logit_shape):
                logits = F.interpolate(logits.unsqueeze(1), size=logit_shape, mode="bilinear", align_corners=False)[:, 0]
            logits_out[idx, :k] = logits[:k]
        return logits_out

    @staticmethod
    def _tensor_or_none(value, device, dtype):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _to_numpy_or_none(value, dtype):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu().numpy().astype(dtype)
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def _prepare_mask_input_tensor(mask_inputs, device, dtype) -> Optional[torch.Tensor]:
        mask_tensor = ExistingSAMBackendAdapter._tensor_or_none(mask_inputs, device=device, dtype=dtype)
        if mask_tensor is None:
            return None
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        elif mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(1)
        elif mask_tensor.dim() == 4:
            pass
        else:
            raise ValueError("mask_inputs must have shape [H,W], [N,H,W], or [N,1,H,W]")
        if tuple(mask_tensor.shape[-2:]) != (256, 256):
            mask_tensor = F.interpolate(mask_tensor.float(), size=(256, 256), mode="bilinear", align_corners=False)
        return mask_tensor

    @staticmethod
    def _prepare_sam2_mask_input(mask_input: np.ndarray) -> np.ndarray:
        mask_input = np.asarray(mask_input, dtype=np.float32)
        if mask_input.ndim == 2:
            mask_input = mask_input[None, None]
        elif mask_input.ndim == 3:
            mask_input = mask_input[:, None]
        elif mask_input.ndim != 4:
            raise ValueError("mask_inputs must have shape [H,W], [N,H,W], or [N,1,H,W]")
        if mask_input.shape[-2:] != (256, 256):
            tensors = torch.as_tensor(mask_input, dtype=torch.float32)
            mask_input = F.interpolate(tensors, size=(256, 256), mode="bilinear", align_corners=False).cpu().numpy()
        return mask_input

    @staticmethod
    def _sample_boxes(value, batch_idx: int, batch_size: int):
        return ExistingSAMBackendAdapter._sample_prompt(value, batch_idx, batch_size, trailing_dim=4)

    @staticmethod
    def _sample_points(value, batch_idx: int, batch_size: int):
        return ExistingSAMBackendAdapter._sample_prompt(value, batch_idx, batch_size, trailing_dim=2)

    @staticmethod
    def _sample_labels(value, batch_idx: int, batch_size: int):
        return ExistingSAMBackendAdapter._sample_prompt(value, batch_idx, batch_size, trailing_dim=None)

    @staticmethod
    def _sample_mask_inputs(value, batch_idx: int, batch_size: int):
        return ExistingSAMBackendAdapter._sample_prompt(value, batch_idx, batch_size, trailing_dim=None, is_mask=True)

    @staticmethod
    def _sample_prompt(value, batch_idx: int, batch_size: int, trailing_dim: Optional[int], is_mask: bool = False):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            if len(value) == batch_size:
                return value[batch_idx]
            return value[0] if batch_size == 1 else value[min(batch_idx, len(value) - 1)]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if value.dim() == 0:
            return value
        if is_mask and value.dim() >= 4 and value.size(0) == batch_size:
            return value[batch_idx]
        if trailing_dim is not None and value.shape[-1] == trailing_dim:
            if value.dim() >= 3 and value.size(0) == batch_size:
                return value[batch_idx]
            if value.dim() == 2 and value.size(0) == batch_size and batch_size > 1:
                return value[batch_idx : batch_idx + 1]
            return value
        if value.size(0) == batch_size and batch_size > 1:
            return value[batch_idx]
        return value


__all__ = ["ExistingSAMBackendAdapter", "build_sam_pseudo_label_refiner"]
