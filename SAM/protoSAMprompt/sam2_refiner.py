from contextlib import nullcontext
from pathlib import Path
import importlib.util
import sys

import cv2
import numpy as np
import torch

from utils.utils2 import extract_points


class Sam2PromptRefiner:
    def __init__(self, checkpoint, model_cfg, device, multimask_output=True, use_bfloat16=True):
        self.checkpoint = Path(checkpoint)
        self.model_cfg = model_cfg
        self.device = self._normalize_device(device)
        self.multimask_output = bool(multimask_output)
        self.use_bfloat16 = bool(use_bfloat16)

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

        self.predictor.set_image(image)
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
