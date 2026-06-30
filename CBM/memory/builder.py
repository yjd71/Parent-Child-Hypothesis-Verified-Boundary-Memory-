from __future__ import annotations

import random
from typing import Dict, Iterable, Sequence, Tuple

import torch

from CBM.diagnostics.visualization import save_memory_selection_visualizations


class LabeledMemoryBuilder:
    """Build labeled-only dense memory from the current teacher/model."""

    def __init__(self, memory, config=None, logger=None) -> None:
        self.memory = memory
        self.config = config
        self.logger = logger

    def prepare_epoch(self, model, labeled_loader, epoch):
        if model is None or labeled_loader is None:
            return self.memory

        target_model = self._unwrap_model(model)
        if not hasattr(target_model, "extract_cbm_memory_features"):
            raise AttributeError("model must expose extract_cbm_memory_features(inputs, ema=True)")

        was_training = target_model.training if hasattr(target_model, "training") else False
        device = self._infer_device(target_model)
        self.memory.clear()
        target_model.eval()
        last_dtype = torch.float32
        vis_samples: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        vis_seen = 0
        vis_limit = max(0, int(getattr(self.config, "cbm_memory_vis_max_images", 5)))
        vis_rng = random.Random(int(getattr(self.config, "cbm_memory_vis_seed", 0)) + int(epoch))
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(labeled_loader):
                    raw_inputs, raw_gt = batch[0], batch[1]
                    inputs = raw_inputs.to(device, non_blocking=True)
                    gt = raw_gt.to(device, non_blocking=True)
                    features = target_model.extract_cbm_memory_features(inputs, ema=True)
                    x3, p3 = self._unpack_features(features)
                    img_ids = self._extract_img_ids(batch, batch_idx, inputs.size(0))
                    if bool(getattr(self.config, "cbm_memory_vis_enable", True)) and vis_limit > 0:
                        for local_idx, image_id in enumerate(img_ids):
                            vis_seen += 1
                            snapshot = (
                                raw_inputs[local_idx].detach().cpu().clone(),
                                raw_gt[local_idx].detach().cpu().clone(),
                            )
                            if len(vis_samples) < vis_limit:
                                vis_samples[str(image_id)] = snapshot
                            else:
                                replace_at = vis_rng.randrange(vis_seen)
                                if replace_at < vis_limit:
                                    old_id = list(vis_samples.keys())[replace_at]
                                    del vis_samples[old_id]
                                    vis_samples[str(image_id)] = snapshot
                    last_dtype = p3.dtype
                    self.memory.append_batch(
                        x3=x3.detach(),
                        p3=p3.detach(),
                        gt=gt.detach(),
                        img_ids=img_ids,
                    )
            self.memory.finalize(device=device, dtype=last_dtype)
            for line in self.memory.distribution_log_lines():
                self._log(line)
            for line in self.memory.diversity_log_lines():
                self._log(line)
            if (
                bool(getattr(self.config, "cbm_memory_vis_enable", True))
                and vis_samples
                and self._is_main_process()
            ):
                paths = save_memory_selection_visualizations(
                    memory=self.memory,
                    snapshots=vis_samples,
                    epoch=int(epoch),
                    split=self.memory.selection_config.split,
                    config=self.config,
                )
                self._log(f"[CBM_MEM_VIS] saved={len(paths)} first={paths[0] if paths else 'none'}")
        finally:
            if was_training:
                target_model.train()
        return self.memory

    def _log(self, message: str) -> None:
        if self.logger is None:
            print(message)
            return
        for name in ("info", "key_info", "success_info"):
            log_fn = getattr(self.logger, name, None)
            if callable(log_fn):
                log_fn(message)
                return
        print(message)

    def _is_main_process(self) -> bool:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def _unwrap_model(self, model):
        return model.module if hasattr(model, "module") else model

    def _infer_device(self, model) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _unpack_features(self, features):
        if isinstance(features, dict):
            return features["x3"], features["p3"]
        if isinstance(features, (list, tuple)) and len(features) >= 2:
            return features[0], features[1]
        raise ValueError("extract_cbm_memory_features must return {'x3': ..., 'p3': ...} or (x3, p3)")

    def _extract_img_ids(self, batch: Sequence[object], batch_idx: int, batch_size: int) -> Iterable[str]:
        if len(batch) > 2:
            raw_ids = batch[2]
            if isinstance(raw_ids, torch.Tensor):
                return [str(item) for item in raw_ids.detach().cpu().reshape(-1).tolist()]
            if isinstance(raw_ids, (list, tuple)):
                return [str(item) for item in raw_ids]
            return [str(raw_ids)] * batch_size
        return [f"epoch_mem_b{batch_idx}_i{idx}" for idx in range(batch_size)]
