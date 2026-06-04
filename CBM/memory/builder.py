from __future__ import annotations

from typing import Iterable, Sequence

import torch


class LabeledMemoryBuilder:
    """Build labeled-only dense memory from the current teacher/model."""

    def __init__(self, memory, logger=None) -> None:
        self.memory = memory
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
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(labeled_loader):
                    inputs, gt = batch[0].to(device), batch[1].to(device)
                    features = target_model.extract_cbm_memory_features(inputs, ema=True)
                    x3, p3 = self._unpack_features(features)
                    img_ids = self._extract_img_ids(batch, batch_idx, inputs.size(0))
                    last_dtype = p3.dtype
                    self.memory.append_batch(
                        x3=x3.detach(),
                        p3=p3.detach(),
                        gt=gt.detach(),
                        img_ids=img_ids,
                    )
            self.memory.finalize(device=device, dtype=last_dtype)
        finally:
            if was_training:
                target_model.train()
        return self.memory

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
