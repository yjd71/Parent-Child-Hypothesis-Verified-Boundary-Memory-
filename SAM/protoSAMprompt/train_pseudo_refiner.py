import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from ..SAM_refinement.sam_image_cache import SAMImageEmbeddingCache
    from ..SAM_refinement.sam_refine_visualizer import SamRefineVisualizer
except ImportError:
    from SAM.SAM_refinement.sam_image_cache import SAMImageEmbeddingCache
    from SAM.SAM_refinement.sam_refine_visualizer import SamRefineVisualizer

from utils.sam_pseudo_logging import SamPseudoRefineLogger


class _NoOpPseudoLabelRefiner:
    def __call__(self, images, pseudo_probs, epoch=None, step=None, image_ids=None):
        return pseudo_probs


class _BaseSamPseudoLabelRefiner:
    def __init__(self, config, device, logger, backend, model_type, checkpoint):
        self.config = config
        self.device = self._normalize_device(device)
        self.logger = logger
        self.backend = backend

        self.model_type = model_type
        self.log_model_type = "{}:{}".format(backend, model_type)
        self.checkpoint = self._resolve_path(checkpoint)
        self.threshold = float(getattr(config, "sam_pseudo_threshold", 0.5))
        self.fusion_alpha = float(getattr(config, "sam_pseudo_fusion_alpha", 0.5))
        self.iters = int(getattr(config, "sam_pseudo_iters", 1))
        self.use_point = bool(getattr(config, "sam_pseudo_use_point", True))
        self.use_box = bool(getattr(config, "sam_pseudo_use_box", True))
        self.use_mask = bool(getattr(config, "sam_pseudo_use_mask", True))
        self.add_neg = bool(getattr(config, "sam_pseudo_add_neg", True))
        self.margin = float(getattr(config, "sam_pseudo_margin", 0.0))
        self.gamma = float(getattr(config, "sam_pseudo_gamma", 4.0))
        self.strength = float(getattr(config, "sam_pseudo_strength", 30))
        self.embedding_cache = SAMImageEmbeddingCache(
            config,
            backend_tag=backend,
            model_tag=self._embedding_model_tag(),
        )
        self.refine_logger = SamPseudoRefineLogger(
            logger=logger,
            enabled=getattr(config, "log_enable", True),
            interval=getattr(config, "log_interval", 200),
        )
        self.visualizer = None
        mode = str(getattr(config, "sam_refine_mode", "off")).strip().lower()
        if mode == "legacy_auto" and bool(getattr(config, "vis_sam_refinement", True)):
            self.visualizer = SamRefineVisualizer(config, logger=logger)
            legacy_vis_dir = getattr(config, "legacy_sam_refine_vis_dir", None)
            if not legacy_vis_dir:
                legacy_vis_dir = Path(getattr(config, "ckpt_dir", ".")) / "legacy_sam_refinement_vis"
            self.visualizer.save_dir = Path(legacy_vis_dir)

        self.total_refined = 0
        self.total_skipped = 0

    def _embedding_model_tag(self):
        """Invalidate embeddings when the checkpoint file changes."""
        try:
            stat = self.checkpoint.stat()
            checkpoint_tag = "{}:{}:{}".format(self.checkpoint, stat.st_size, stat.st_mtime_ns)
        except OSError:
            checkpoint_tag = str(self.checkpoint)
        return "{}:{}".format(self.log_model_type, checkpoint_tag)

    @staticmethod
    def _normalize_device(device):
        if isinstance(device, torch.device):
            return device
        if isinstance(device, int):
            return torch.device("cuda:{}".format(device))
        return torch.device(device)

    def _log_init(self):
        self.refine_logger.log_init(
            model_type=self.log_model_type,
            checkpoint=self.checkpoint,
            device=self.device,
            threshold=self.threshold,
            fusion_alpha=self.fusion_alpha,
            iters=self.iters,
            use_point=self.use_point,
            use_box=self.use_box,
            use_mask=self.use_mask,
            add_neg=self.add_neg,
            margin=self.margin,
            gamma=self.gamma,
            strength=self.strength,
        )

    @staticmethod
    def _resolve_path(path_like):
        path = Path(path_like)
        if path.is_absolute():
            return path

        this_dir = Path(__file__).resolve().parent
        repo_root = this_dir.parents[1]
        sam_root = this_dir.parent
        candidates = [
            Path.cwd() / path,
            repo_root / path,
            sam_root / path,
            this_dir / path,
            this_dir / path.name,
            sam_root / path.name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    @staticmethod
    def _ensure_repo_on_path():
        this_dir = Path(__file__).resolve().parent
        repo_root = this_dir.parents[1]
        for path in (repo_root, this_dir):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    @staticmethod
    def _denormalize_image(image):
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        image_cpu = image.detach().float().cpu()
        image_cpu = (image_cpu * std + mean).clamp(0, 1)
        image_cpu = image_cpu.permute(1, 2, 0).mul(255).round().byte()
        return image_cpu.numpy()

    def _refine_one(self, image_np, coarse_mask):
        raise NotImplementedError

    def _refine_one_with_debug(self, image_np, coarse_mask):
        return self._refine_one(image_np, coarse_mask), {}

    def __call__(self, images, pseudo_probs, epoch=None, step=None, image_ids=None):
        if images is None or pseudo_probs is None:
            return pseudo_probs
        if pseudo_probs.ndim != 4 or pseudo_probs.shape[1] != 1:
            self.refine_logger.warn_once("shape", "[!] SAM pseudo refine skipped: expected pseudo_probs shape [B,1,H,W].")
            return pseudo_probs
        if images.ndim != 4 or images.shape[1] != 3:
            self.refine_logger.warn_once("image_shape", "[!] SAM pseudo refine skipped: expected images shape [B,3,H,W].")
            return pseudo_probs

        alpha = min(max(self.fusion_alpha, 0.0), 1.0)
        original_dtype = pseudo_probs.dtype
        original_device = pseudo_probs.device
        output = pseudo_probs.detach().clone()
        pseudo_for_sam = pseudo_probs.detach()
        image_hw = images.shape[-2:]
        pseudo_hw = pseudo_probs.shape[-2:]
        batch_stats = self.refine_logger.new_batch_stats(pseudo_probs.shape[0], image_hw, pseudo_hw)
        change_sums = self.refine_logger.new_change_sums()
        sam_masks_for_vis = pseudo_probs.detach().clone()
        prompt_debug_for_vis = [{} for _ in range(pseudo_probs.shape[0])]

        if pseudo_hw != image_hw:
            pseudo_for_sam = F.interpolate(
                pseudo_for_sam.float(),
                size=image_hw,
                mode="bilinear",
                align_corners=False,
            )

        with torch.no_grad():
            for idx in range(pseudo_probs.shape[0]):
                coarse_prob = pseudo_for_sam[idx, 0].detach().float()
                coarse_mask = (coarse_prob > self.threshold).to(torch.uint8)
                if coarse_mask.sum().item() == 0:
                    self.total_skipped += 1
                    batch_stats["skipped_empty"] += 1
                    continue
                batch_stats["non_empty"] += 1

                try:
                    image_np = self._denormalize_image(images[idx])
                    sam_mask, prompt_debug = self._refine_one_with_debug(
                        image_np, coarse_mask.cpu().numpy()
                    )
                    sam_mask = sam_mask.to(device=original_device, dtype=output.dtype).view(1, 1, *image_hw)
                    if pseudo_hw != image_hw:
                        sam_mask = F.interpolate(
                            sam_mask.float(),
                            size=pseudo_hw,
                            mode="nearest",
                        ).to(dtype=output.dtype)
                    sam_masks_for_vis[idx:idx + 1] = sam_mask
                    prompt_debug_for_vis[idx] = self._normalize_prompt_debug(
                        prompt_debug, image_hw, pseudo_hw
                    )
                    teacher_prob = pseudo_probs[idx:idx + 1].detach()
                    fused = alpha * sam_mask + (1.0 - alpha) * teacher_prob
                    output[idx:idx + 1] = fused.clamp(0, 1).to(dtype=output.dtype)
                    self.total_refined += 1
                    batch_stats["refined"] += 1
                    metrics = self.refine_logger.mask_change_metrics(teacher_prob, sam_mask, fused, self.threshold)
                    self.refine_logger.add_change_metrics(change_sums, metrics)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.total_skipped += 1
                    batch_stats["skipped_error"] += 1
                    self.refine_logger.warn_once("runtime", "[!] SAM pseudo refine skipped for at least one sample: {}".format(exc))
                except Exception as exc:
                    self.total_skipped += 1
                    batch_stats["skipped_error"] += 1
                    self.refine_logger.warn_once("exception", "[!] SAM pseudo refine skipped for at least one sample: {}".format(exc))

        self.refine_logger.log_batch(epoch, step, batch_stats, change_sums, self.total_refined, self.total_skipped)
        self._save_visualization(
            images=images,
            teacher_prob=pseudo_probs,
            sam_mask=sam_masks_for_vis,
            p_ref=output,
            prompt_debug=prompt_debug_for_vis,
            image_ids=image_ids,
            epoch=epoch,
            step=step,
        )
        return output.to(device=original_device, dtype=original_dtype)

    @staticmethod
    def _normalize_prompt_debug(prompt_debug, source_hw, target_hw):
        if not isinstance(prompt_debug, dict):
            return {}
        scale_y = float(target_hw[0]) / max(1.0, float(source_hw[0]))
        scale_x = float(target_hw[1]) / max(1.0, float(source_hw[1]))

        def _scaled(value, width):
            if value is None:
                return torch.empty((0, width), dtype=torch.float32)
            tensor = torch.as_tensor(value).detach().cpu().float().reshape(-1, width)
            if tensor.numel() == 0:
                return tensor
            tensor[:, 0::2] *= scale_x
            tensor[:, 1::2] *= scale_y
            return tensor

        coords = _scaled(prompt_debug.get("point_coords"), 2)
        labels_value = prompt_debug.get("point_labels")
        labels = (
            torch.as_tensor(labels_value).detach().cpu().reshape(-1)
            if labels_value is not None
            else torch.empty(0, dtype=torch.int64)
        )
        count = min(coords.size(0), labels.numel())
        coords = coords[:count]
        labels = labels[:count]
        return {
            "pos_points": coords[labels > 0],
            "neg_points": coords[labels <= 0],
            "boxes": _scaled(prompt_debug.get("boxes"), 4),
        }

    def _save_visualization(
        self,
        images,
        teacher_prob,
        sam_mask,
        p_ref,
        prompt_debug,
        image_ids,
        epoch,
        step,
    ):
        if self.visualizer is None:
            return
        empty = [torch.empty(0, 2) for _ in prompt_debug]
        prompt_pack = {
            "pos_points": [item.get("pos_points", empty[idx]) for idx, item in enumerate(prompt_debug)],
            "neg_points": [item.get("neg_points", empty[idx]) for idx, item in enumerate(prompt_debug)],
            "boundary_points": empty,
            "boxes": [item.get("boxes", torch.empty(0, 4)) for item in prompt_debug],
        }
        self.visualizer.save(
            images=images,
            teacher_prob=teacher_prob,
            sam_mask=sam_mask,
            p_ref=p_ref,
            conf_ref=None,
            sam_aux={"prompt_pack": prompt_pack},
            image_ids=image_ids,
            epoch=epoch,
            step=step,
        )


class Sam1PseudoLabelRefiner(_BaseSamPseudoLabelRefiner):
    def __init__(self, config, device, logger=None):
        super().__init__(
            config=config,
            device=device,
            logger=logger,
            backend="sam1",
            model_type=getattr(config, "sam_pseudo_model_type", "vit_h"),
            checkpoint=getattr(config, "sam_pseudo_checkpoint", "SAM/sam_vit_h_4b8939.pth"),
        )

        self.sam_model_registry, self.sam_refiner = self._import_sam()
        if self.model_type not in self.sam_model_registry:
            raise ValueError("Unsupported SAM v1 model_type '{}'. Available: {}".format(
                self.model_type,
                sorted(self.sam_model_registry.keys()),
            ))
        self.sam = self.sam_model_registry[self.model_type](checkpoint=str(self.checkpoint)).to(self.device)
        self.sam.eval()
        for param in self.sam.parameters():
            param.requires_grad = False
        self._log_init()

    @staticmethod
    def _import_sam():
        _BaseSamPseudoLabelRefiner._ensure_repo_on_path()

        try:
            from ..segment_anything import sam_model_registry
            from .sam_refiner import sam_refiner
        except ImportError:
            from SAM.segment_anything import sam_model_registry
            from sam_refiner import sam_refiner
        return sam_model_registry, sam_refiner

    def _refine_one(self, image_np, coarse_mask):
        refined, _ = self._refine_one_with_debug(image_np, coarse_mask)
        return refined

    def _refine_one_with_debug(self, image_np, coarse_mask):
        if coarse_mask.ndim == 2:
            coarse_mask = coarse_mask[None, ...]
        refined_masks, _, prompt_debug = self.sam_refiner(
            image_np,
            coarse_mask,
            self.sam,
            use_point=self.use_point,
            use_box=self.use_box,
            use_mask=self.use_mask,
            add_neg=self.add_neg,
            iters=self.iters,
            margin=self.margin,
            gamma=self.gamma,
            strength=self.strength,
            use_samhq=False,
            ddp=False,
            is_train=False,
            coarse_threshold=self.threshold,
            embedding_cache=self.embedding_cache,
            return_prompt_debug=True,
        )
        boxes = prompt_debug.get("boxes")
        sample_debug = {
            "boxes": (
                boxes[0:1]
                if torch.is_tensor(boxes) and boxes.numel() > 0
                else torch.empty(0, 4)
            ),
            "point_coords": prompt_debug.get("point_coords", torch.empty(1, 0, 2))[0],
            "point_labels": prompt_debug.get("point_labels", torch.empty(1, 0))[0],
        }
        return torch.from_numpy(refined_masks[0]).float(), sample_debug


class Sam2PseudoLabelRefiner(_BaseSamPseudoLabelRefiner):
    def __init__(self, config, device, logger=None):
        model_cfg = getattr(config, "sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")
        super().__init__(
            config=config,
            device=device,
            logger=logger,
            backend="sam2",
            model_type=self._model_type_from_cfg(model_cfg),
            checkpoint=getattr(config, "sam2_checkpoint", "SAM/sam2.1_hiera_large.pt"),
        )

        try:
            from .sam2_refiner import Sam2PromptRefiner
        except ImportError:
            from sam2_refiner import Sam2PromptRefiner

        self.sam2_refiner = Sam2PromptRefiner(
            checkpoint=self.checkpoint,
            model_cfg=model_cfg,
            device=self.device,
            multimask_output=bool(getattr(config, "sam2_multimask_output", True)),
            use_bfloat16=bool(getattr(config, "sam2_use_bfloat16", True)),
            embedding_cache=self.embedding_cache,
        )
        self._log_init()

    @staticmethod
    def _model_type_from_cfg(model_cfg):
        name = Path(str(model_cfg)).stem
        if name.startswith("sam2.1_"):
            name = name[len("sam2.1_"):]
        elif name.startswith("sam2_"):
            name = name[len("sam2_"):]
        return name

    def _refine_one(self, image_np, coarse_mask):
        refined, _ = self._refine_one_with_debug(image_np, coarse_mask)
        return refined

    def _refine_one_with_debug(self, image_np, coarse_mask):
        if coarse_mask.ndim == 2:
            coarse_mask = coarse_mask[None, ...]
        refined_masks, _, prompt_debug = self.sam2_refiner(
            image_np,
            coarse_mask,
            use_point=self.use_point,
            use_box=self.use_box,
            use_mask=self.use_mask,
            add_neg=self.add_neg,
            iters=self.iters,
            gamma=self.gamma,
            strength=self.strength,
            coarse_threshold=self.threshold,
            return_prompt_debug=True,
        )
        sample_debug = prompt_debug[0] if prompt_debug else {}
        return torch.from_numpy(refined_masks[0]).float(), sample_debug


class SamPseudoLabelRefiner:
    def __init__(self, config, device, logger=None):
        self.refiner = _build_enabled_refiner(config=config, device=device, logger=logger)

    def __call__(self, images, pseudo_probs, epoch=None, step=None, image_ids=None):
        return self.refiner(
            images,
            pseudo_probs,
            epoch=epoch,
            step=step,
            image_ids=image_ids,
        )


def _build_enabled_refiner(config, device, logger=None):
    backend = str(getattr(config, "sam_pseudo_backend", "sam1")).lower()
    if backend in ("sam1", "sam", "v1"):
        return Sam1PseudoLabelRefiner(config=config, device=device, logger=logger)
    if backend in ("sam2", "v2"):
        return Sam2PseudoLabelRefiner(config=config, device=device, logger=logger)
    raise ValueError("Unsupported sam_pseudo_backend '{}'. Expected 'sam1' or 'sam2'.".format(backend))


def build_sam_pseudo_label_refiner(config, device, logger=None):
    return SamPseudoLabelRefiner(config=config, device=device, logger=logger)
