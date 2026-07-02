from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from CBM.memory.labels import REGION_NAMES, REGION_TO_ID, VALUE_LAYOUT
from CBM.sv_ume.config_contract import validate_sv_ume_profile_contract
from CBM.sv_ume.sam_refined_region_builder import build_sam_refined_regions
from CBM.sv_ume.ume_reliability import (
    DEFAULT_CBM_LOGIT_SCALE,
    compute_image_consistency,
    compute_region_consistency,
    compute_token_reliability,
)
from CBM.sv_ume.unlabeled_dense_memory import UnlabeledMemoryToken
from SAM.SAM_refinement.cbm_aux_adapter import build_retrieval_aux_from_cbm_aux


REJECTION_REASONS = (
    "sam_not_used",
    "image_invalid_evidence",
    "image_below_threshold",
    "region_empty",
    "region_below_threshold",
    "token_structural_invalid",
    "token_cbm_invalid",
    "token_below_threshold",
)


class TokenCandidate(UnlabeledMemoryToken):
    """Memory-compatible candidate with typed access to diagnostic metadata."""

    def _meta_value(self, name: str):
        if name not in self.meta:
            raise AttributeError(f"candidate metadata is missing {name!r}")
        return self.meta[name]

    @property
    def region(self) -> str:
        return str(self._meta_value("region"))

    @property
    def region_id(self) -> int:
        return int(self._meta_value("region_id"))

    @property
    def image_id(self) -> str:
        return str(self._meta_value("image_id"))

    @property
    def coord(self) -> Tuple[int, int]:
        value = self._meta_value("coord")
        return int(value[0]), int(value[1])

    @property
    def r_token(self) -> float:
        return float(self.reliability)

    @property
    def C_img(self) -> float:
        return float(self._meta_value("C_img"))

    @property
    def C_region(self) -> float:
        return float(self._meta_value("C_region"))

    @property
    def r_teacher(self) -> float:
        return float(self._meta_value("r_teacher"))

    @property
    def r_sam(self) -> float:
        return float(self._meta_value("r_sam"))

    @property
    def r_cbm(self) -> float:
        return float(self._meta_value("r_cbm"))

    @property
    def r_context(self) -> float:
        return float(self._meta_value("r_context"))

    @property
    def r_density(self) -> float:
        return float(self._meta_value("r_density"))

    @property
    def r_temporal(self) -> float:
        return float(self._meta_value("r_temporal"))

    @property
    def diversity_score(self) -> float:
        return float(self.diversity)

    @property
    def global_type(self) -> str:
        return str(self._meta_value("global_type"))

    @property
    def nearest_labeled_id(self) -> Optional[str]:
        value = self.meta.get("nearest_labeled_id")
        return None if value is None else str(value)

    @property
    def sim_to_labeled(self) -> float:
        return float(self._meta_value("sim_to_labeled"))

    @property
    def source(self) -> str:
        return str(self._meta_value("source"))

    @property
    def epoch_added(self) -> int:
        return int(self._meta_value("epoch_added"))

    @property
    def step_added(self) -> int:
        return int(self._meta_value("step_added"))


class SAMRefinedCandidateBuilder:
    """Build verified teacher-p3 candidates without writing retrieval memory."""

    def __init__(self, cfg, logger=None) -> None:
        self.cfg = cfg
        self.logger = logger
        validate_sv_ume_profile_contract(cfg)
        raw_regions = getattr(cfg, "sv_ume_regions", REGION_NAMES)
        if isinstance(raw_regions, (str, bytes)) or not isinstance(raw_regions, Sequence):
            raise TypeError("sv_ume_regions must be a non-empty sequence of region names")
        unknown_regions = set(raw_regions) - set(REGION_NAMES)
        if unknown_regions:
            raise ValueError(f"sv_ume_regions contains unknown regions: {sorted(unknown_regions)}")
        self.enabled_regions = tuple(region for region in REGION_NAMES if region in raw_regions)
        if not self.enabled_regions:
            raise ValueError("sv_ume_regions must enable at least one region")
        self.score_mode = str(getattr(cfg, "sv_ume_token_score_mode", "product")).strip().lower()
        self.diagnostics_interval = max(1, int(getattr(cfg, "sv_ume_diagnostics_interval", 20)))
        self.cbm_logit_scale = float(
            getattr(cfg, "sv_ume_cbm_logit_scale", DEFAULT_CBM_LOGIT_SCALE)
        )
        if not torch.isfinite(torch.tensor(self.cbm_logit_scale)) or self.cbm_logit_scale <= 0.0:
            raise ValueError("sv_ume_cbm_logit_scale must be finite and positive")
        self.last_result: Optional[Dict[str, Any]] = None
        self._step_epoch: Optional[int] = None
        self._batch_step = 0

    @torch.no_grad()
    def build(
        self,
        *,
        img,
        img_id,
        x3,
        p3,
        p_raw,
        p_ref,
        conf_ref,
        sam_aux,
        retrieval_aux,
        labeled_memory,
        prev_unlabeled_memory=None,
        epoch,
        step,
    ) -> Dict[str, Any]:
        epoch = self._non_negative_int(epoch, "epoch")
        step = self._non_negative_int(step, "step")
        batch_size = self._validate_feature_inputs(img, x3, p3)
        image_ids = self._normalize_image_ids(img_id, batch_size)
        rejected = {reason: 0 for reason in REJECTION_REASONS}
        stats = self._base_stats(batch_size, epoch, step)
        stats["enabled_regions"] = list(self.enabled_regions)
        stats["disabled_regions"] = [
            region for region in REGION_NAMES if region not in self.enabled_regions
        ]
        stats["token_score_mode"] = self.score_mode

        p_sam = sam_aux.get("sam_mask") if isinstance(sam_aux, Mapping) else None
        used_sam = bool(sam_aux.get("used_sam", p_sam is not None)) if isinstance(sam_aux, Mapping) else False
        if not used_sam or not torch.is_tensor(p_sam):
            rejected["sam_not_used"] = batch_size
            stats["sam_used"] = False
            result = self._result(self._empty_pools(), rejected, stats)
            self.last_result = result
            if (step + 1) % self.diagnostics_interval == 0:
                self._log_diagnostics(result)
            return result

        target_size = tuple(p3.shape[-2:])
        region_pack = build_sam_refined_regions(
            p_ref,
            conf_ref,
            target_size=target_size,
            p_raw=p_raw,
            p_sam=p_sam,
            retrieval_aux=retrieval_aux,
        )
        previous_p_ref = self._previous_p_ref_map(
            prev_unlabeled_memory,
            image_ids,
            region_pack["p_ref3"],
        )

        tau_image = float(getattr(self.cfg, "tau_image", 0.80))
        tau_region = getattr(self.cfg, "tau_region", None)
        tau_token = getattr(self.cfg, "tau_token", None)
        image_result = compute_image_consistency(
            p_raw,
            p_ref,
            p_sam,
            sam_aux,
            retrieval_aux,
            x3,
            labeled_memory,
            tau_image=tau_image,
            cbm_logit_scale=self.cbm_logit_scale,
        )
        region_result = compute_region_consistency(
            p_raw,
            p_sam,
            region_pack,
            sam_aux,
            retrieval_aux,
            p3,
            labeled_memory,
            thresholds=tau_region,
            cbm_logit_scale=self.cbm_logit_scale,
        )
        token_result = compute_token_reliability(
            p_raw,
            region_pack,
            p3,
            retrieval_aux,
            labeled_memory,
            p_ref_previous=previous_p_ref,
            thresholds=tau_token,
            cbm_logit_scale=self.cbm_logit_scale,
            score_mode=self.score_mode,
            context_floor=float(getattr(self.cfg, "sv_ume_context_floor", 0.30)),
            non_boundary_context=float(getattr(self.cfg, "sv_ume_non_boundary_context", 0.80)),
        )

        memory_dim, value_dim = self._memory_dimensions(labeled_memory)
        if value_dim != len(VALUE_LAYOUT):
            raise ValueError(
                f"candidate value_dim must match labeled value layout {len(VALUE_LAYOUT)}, got {value_dim}"
            )
        p3_keys = self._fit_channels(p3.detach(), memory_dim)
        global_keys = F.adaptive_avg_pool2d(x3.detach(), 1).flatten(1)
        global_keys = self._fit_last_dim(global_keys, memory_dim)

        pools = self._empty_pools()
        global_types = image_result["global_metadata"]["global_type"]
        nearest_ids = image_result["global_metadata"]["nearest_labeled_id"]
        similarities = image_result["global_metadata"]["sim_max"]
        region_masks = region_pack["regions"]
        structural_valid = region_pack["valid"] > 0.5
        cbm_valid = token_result["cbm_valid"]
        batch_valid_map = token_result["batch_valid_map"]
        token_components = token_result["components"]

        stats["sam_used"] = True
        stats["target_size"] = target_size
        stats["global_type_counts"] = dict(Counter(global_types))
        stats["enabled_regions"] = list(self.enabled_regions)
        stats["disabled_regions"] = [
            region for region in REGION_NAMES if region not in self.enabled_regions
        ]
        stats["token_score_mode"] = token_result["score_mode"]
        stats.update(self._summarize_image_admission(image_result))
        stats["region_pixel_counts"] = {
            region: int((region_masks[region] > 0.5).sum().item())
            for region in REGION_NAMES
        }
        stats["region_image_counts"] = {
            region: int(
                (region_masks[region] > 0.5)
                .reshape(batch_size, -1)
                .any(dim=1)
                .sum()
                .item()
            )
            for region in REGION_NAMES
        }
        stats["image_score_mean"] = float(image_result["score"].mean().item())
        stats["image_score_min"] = float(image_result["score"].min().item())
        stats["image_score_max"] = float(image_result["score"].max().item())
        stats["region_score_mean"] = {
            region: float(region_result["score"][region].mean().item())
            for region in REGION_NAMES
        }
        stats["cbm_valid_ratio"] = {}
        stats["token_score_quantiles"] = {}
        diagnostic_mask = torch.zeros_like(structural_valid)
        for region in self.enabled_regions:
            region_valid = (
                (region_masks[region] > 0.5)
                & structural_valid
                & batch_valid_map
            )
            diagnostic_mask |= region_valid
            stats["token_score_quantiles"][region] = self._summarize_tensor(
                token_result["score"][region_valid]
            )
            stats["cbm_valid_ratio"][region] = (
                float(cbm_valid[region_valid].float().mean().item())
                if bool(region_valid.any())
                else 0.0
            )
        stats["token_component_quantiles"] = {
            name: self._summarize_tensor(value[diagnostic_mask])
            for name, value in token_components.items()
        }

        token_score_sums = {region: 0.0 for region in REGION_NAMES}
        token_score_counts = {region: 0 for region in REGION_NAMES}
        for batch_index, image_id in enumerate(image_ids):
            if not bool(image_result["evidence_valid"][batch_index]):
                rejected["image_invalid_evidence"] += 1
                continue
            if not bool(image_result["allow_image"][batch_index]):
                rejected["image_below_threshold"] += 1
                continue

            global_type = str(global_types[batch_index])
            nearest_id = nearest_ids[batch_index]
            similarity = float(similarities[batch_index].item())
            c_img = float(image_result["score"][batch_index].item())
            for region in self.enabled_regions:
                mask = region_masks[region][batch_index, 0] > 0.5
                pixel_count = int(mask.sum().item())
                if pixel_count == 0:
                    rejected["region_empty"] += 1
                    continue
                if not bool(region_result["allow"][region][batch_index]):
                    rejected["region_below_threshold"] += 1
                    continue

                c_region = float(region_result["score"][region][batch_index].item())
                diversity = float(
                    region_result["components"]["region_diversity"][region][batch_index].item()
                )
                coords = mask.nonzero(as_tuple=False)
                for coordinate in coords:
                    row, col = int(coordinate[0].item()), int(coordinate[1].item())
                    if not bool(structural_valid[batch_index, 0, row, col]):
                        rejected["token_structural_invalid"] += 1
                        continue
                    if not bool(batch_valid_map[batch_index, 0, row, col]):
                        rejected["token_cbm_invalid"] += 1
                        continue
                    if region in ("fg_boundary", "bg_near") and not bool(
                        cbm_valid[batch_index, 0, row, col]
                    ):
                        rejected["token_cbm_invalid"] += 1
                        continue

                    reliability = float(token_result["score"][batch_index, 0, row, col].item())
                    token_score_sums[region] += reliability
                    token_score_counts[region] += 1
                    if reliability <= float(token_result["thresholds"][region]):
                        rejected["token_below_threshold"] += 1
                        continue

                    factor_values = {
                        name: float(value[batch_index, 0, row, col].item())
                        for name, value in token_components.items()
                    }
                    sdf = float(region_pack["sdf"][batch_index, 0, row, col].item())
                    p_ref_value = float(region_pack["p_ref3"][batch_index, 0, row, col].item())
                    conf_ref_value = float(region_pack["conf_ref3"][batch_index, 0, row, col].item())
                    meta = {
                        "image_id": image_id,
                        "coord": (row, col),
                        "region": region,
                        "region_id": int(REGION_TO_ID[region]),
                        "reliability": reliability,
                        "r_token": reliability,
                        "C_img": c_img,
                        "C_region": c_region,
                        "r_teacher": factor_values["r_teacher"],
                        "r_sam": factor_values["r_sam"],
                        "r_cbm": factor_values["r_cbm"],
                        "r_context": factor_values["r_context"],
                        "r_density": factor_values["r_density"],
                        "r_temporal": factor_values["r_temporal"],
                        "r_diversity_local": factor_values["r_diversity_local"],
                        "diversity_score": diversity,
                        "global_type": global_type,
                        "novel_activated": False,
                        "nearest_labeled_id": nearest_id,
                        "sim_to_labeled": similarity,
                        "source": "unlabeled_sam_refined",
                        "epoch_added": epoch,
                        "step_added": step,
                        "p_ref_value": p_ref_value,
                        "conf_ref_value": conf_ref_value,
                        "sdf": sdf,
                    }
                    global_meta = {
                        "image_id": image_id,
                        "global_type": global_type,
                        "novel_activated": False,
                        "nearest_labeled_id": nearest_id,
                        "sim_to_labeled": similarity,
                        "source": "unlabeled_sam_refined",
                        "epoch_added": epoch,
                    }
                    value = self._build_value(
                        region,
                        sdf,
                        reliability,
                        device=p3.device,
                        dtype=p3.dtype,
                    )
                    candidate = TokenCandidate(
                        key=p3_keys[batch_index, :, row, col].detach().cpu().clone(),
                        value=value.detach().cpu().clone(),
                        global_key=global_keys[batch_index].detach().cpu().clone(),
                        meta=meta,
                        reliability=reliability,
                        diversity=diversity,
                        global_meta=global_meta,
                    )
                    pools[region].append(candidate)
                    if global_type == "novel_pending":
                        stats["novel_pending_candidates"] += 1

        stats["candidate_counts"] = {region: len(pools[region]) for region in REGION_NAMES}
        stats["accepted_total"] = sum(stats["candidate_counts"].values())
        stats["token_score_mean"] = {
            region: (
                token_score_sums[region] / token_score_counts[region]
                if token_score_counts[region] > 0
                else 0.0
            )
            for region in REGION_NAMES
        }
        stats["rejected_total"] = int(sum(rejected.values()))
        result = self._result(pools, rejected, stats)
        self.last_result = result
        if (step + 1) % self.diagnostics_interval == 0:
            self._log_diagnostics(result)
        return result

    @torch.no_grad()
    def build_batch(
        self,
        *,
        teacher,
        sam_refiner,
        batch,
        labeled_memory,
        memory_for_retrieval,
        epoch,
        device,
    ) -> Mapping[str, Sequence[TokenCandidate]]:
        epoch = self._non_negative_int(epoch, "epoch")
        step = self._next_batch_step(epoch)
        image, image_ids = self._extract_batch_inputs(batch)
        if not torch.is_tensor(image):
            raise TypeError("unlabeled batch must contain an image Tensor")
        image = image.to(device)

        if isinstance(batch, Mapping) and self._is_precomputed_batch(batch):
            result = self.build(
                img=image,
                img_id=image_ids,
                x3=batch["x3"].to(device),
                p3=batch["p3"].to(device),
                p_raw=batch["p_raw"].to(device),
                p_ref=batch["p_ref"].to(device),
                conf_ref=batch["conf_ref"].to(device),
                sam_aux=batch["sam_aux"],
                retrieval_aux=batch["retrieval_aux"],
                labeled_memory=labeled_memory,
                prev_unlabeled_memory=batch.get(
                    "prev_unlabeled_memory",
                    self._resolve_previous_memory(memory_for_retrieval),
                ),
                epoch=epoch,
                step=int(batch.get("step", step)),
            )
            return result["candidate_pools"]

        if teacher is None or sam_refiner is None:
            raise ValueError("teacher and sam_refiner are required for raw build_batch input")
        features = self._extract_teacher_features(teacher, image)
        teacher_predictions, teacher_aux = self._teacher_forward(
            teacher,
            image,
            memory_for_retrieval,
        )
        p_raw = self._teacher_probability(teacher_predictions, teacher_aux)
        retrieval_aux = build_retrieval_aux_from_cbm_aux(teacher_aux)
        normalized_ids = self._normalize_image_ids(image_ids, image.size(0))
        refine = getattr(sam_refiner, "refine", None)
        if not callable(refine):
            raise TypeError("sam_refiner must expose refine()")
        p_ref, conf_ref, sam_aux = refine(
            images=image,
            teacher_prob=p_raw,
            retrieval_aux=retrieval_aux,
            image_ids=normalized_ids,
            epoch=epoch,
            step=step,
        )
        result = self.build(
            img=image,
            img_id=normalized_ids,
            x3=features["x3"],
            p3=features["p3"],
            p_raw=p_raw,
            p_ref=p_ref,
            conf_ref=conf_ref,
            sam_aux=sam_aux,
            retrieval_aux=retrieval_aux,
            labeled_memory=labeled_memory,
            prev_unlabeled_memory=self._resolve_previous_memory(memory_for_retrieval),
            epoch=epoch,
            step=step,
        )
        return result["candidate_pools"]

    def _extract_teacher_features(self, teacher, image: torch.Tensor) -> Mapping[str, torch.Tensor]:
        extractor = getattr(teacher, "extract_cbm_memory_features", None)
        if not callable(extractor):
            raise TypeError("teacher must expose extract_cbm_memory_features()")
        if hasattr(teacher, "teacher"):
            features = extractor(image, ema=True)
        else:
            features = extractor(image)
        if not isinstance(features, Mapping) or not all(key in features for key in ("x3", "p3")):
            raise TypeError("extract_cbm_memory_features() must return {'x3', 'p3'}")
        if not torch.is_tensor(features["x3"]) or not torch.is_tensor(features["p3"]):
            raise TypeError("teacher x3 and p3 features must be tensors")
        return features

    def _teacher_forward(self, teacher, image: torch.Tensor, memory_for_retrieval):
        if not callable(teacher):
            raise TypeError("teacher must be callable")
        kwargs: Dict[str, Any] = {"use_memory": True, "return_aux": True}
        if hasattr(teacher, "teacher"):
            kwargs["ema"] = True
        if isinstance(memory_for_retrieval, Mapping):
            has_labeled = any(
                key in memory_for_retrieval
                for key in ("labeled_memory", "L_t")
            )
            has_unlabeled = any(
                key in memory_for_retrieval
                for key in ("unlabeled_memory", "U_prev")
            )
            if has_labeled or has_unlabeled:
                kwargs["memory_t"] = memory_for_retrieval
        elif callable(getattr(memory_for_retrieval, "apply_p3_hook", None)):
            kwargs["cbm"] = memory_for_retrieval
        output = teacher(image, **kwargs)
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise TypeError("teacher return_aux forward must return (predictions, aux)")
        predictions, aux = output
        if not isinstance(aux, Mapping):
            raise TypeError("teacher auxiliary output must be a mapping")
        return predictions, aux

    @staticmethod
    def _teacher_probability(predictions, aux: Mapping[str, Any]) -> torch.Tensor:
        value = aux.get("p_final")
        if torch.is_tensor(value):
            return value.detach().clamp(0.0, 1.0)
        outputs = predictions
        if isinstance(outputs, tuple) and len(outputs) == 2:
            outputs = outputs[1]
        if not isinstance(outputs, (tuple, list)) or not outputs:
            raise TypeError("teacher predictions must contain a final logit tensor")
        value = outputs[-1]
        if not torch.is_tensor(value):
            raise TypeError("teacher final prediction must be a Tensor")
        return value.sigmoid().detach()

    @staticmethod
    def _extract_batch_inputs(batch) -> Tuple[Any, Any]:
        if isinstance(batch, Mapping):
            image = SAMRefinedCandidateBuilder._first_present(
                batch, ("img_u_w", "image_w", "weak", "image", "img")
            )
            image_ids = SAMRefinedCandidateBuilder._first_present(
                batch, ("image_ids", "image_id", "ids", "id")
            )
            return image, image_ids
        if not isinstance(batch, (tuple, list)) or not batch:
            raise TypeError("unlabeled batch must be a mapping or non-empty tuple/list")
        image_ids = batch[2] if len(batch) > 2 else None
        return batch[0], image_ids

    @staticmethod
    def _first_present(mapping: Mapping[str, Any], names: Sequence[str]):
        for name in names:
            if name in mapping and mapping[name] is not None:
                return mapping[name]
        return None

    @staticmethod
    def _is_precomputed_batch(batch: Mapping[str, Any]) -> bool:
        return all(
            name in batch
            for name in ("x3", "p3", "p_raw", "p_ref", "conf_ref", "sam_aux", "retrieval_aux")
        )

    @staticmethod
    def _resolve_previous_memory(memory_for_retrieval):
        if memory_for_retrieval is None:
            return None
        meta = getattr(memory_for_retrieval, "meta", None)
        keys = getattr(memory_for_retrieval, "keys", None)
        if isinstance(meta, Mapping) and isinstance(keys, Mapping):
            return memory_for_retrieval
        for name in ("U_prev", "unlabeled_memory", "aux_memory"):
            value = getattr(memory_for_retrieval, name, None)
            if value is not None:
                return value
        if isinstance(memory_for_retrieval, Mapping):
            for name in ("U_prev", "unlabeled_memory", "aux_memory"):
                if memory_for_retrieval.get(name) is not None:
                    return memory_for_retrieval[name]
        return None

    @staticmethod
    def _previous_p_ref_map(previous_memory, image_ids, current: torch.Tensor):
        if previous_memory is None:
            return None
        meta = getattr(previous_memory, "meta", None)
        if not isinstance(meta, Mapping):
            return None
        image_to_index = {str(image_id): index for index, image_id in enumerate(image_ids)}
        previous = current.detach().clone()
        found = False
        for region in REGION_NAMES:
            entries = meta.get(region, ())
            if not isinstance(entries, Sequence):
                continue
            for item in entries:
                if not isinstance(item, Mapping):
                    continue
                image_id = str(item.get("image_id"))
                if image_id not in image_to_index or "p_ref_value" not in item:
                    continue
                coord = item.get("coord")
                if not isinstance(coord, Sequence) or len(coord) != 2:
                    continue
                row, col = int(coord[0]), int(coord[1])
                if not (0 <= row < current.size(-2) and 0 <= col < current.size(-1)):
                    continue
                value = float(item["p_ref_value"])
                if not torch.isfinite(torch.tensor(value)):
                    continue
                previous[image_to_index[image_id], 0, row, col] = max(0.0, min(1.0, value))
                found = True
        return previous if found else None

    def _next_batch_step(self, epoch: int) -> int:
        if self._step_epoch != epoch:
            self._step_epoch = epoch
            self._batch_step = 0
        step = self._batch_step
        self._batch_step += 1
        return step

    @staticmethod
    def _validate_feature_inputs(img, x3, p3) -> int:
        for name, value in (("img", img), ("x3", x3), ("p3", p3)):
            if not torch.is_tensor(value) or value.dim() != 4:
                shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
                raise ValueError(f"{name} must be a 4D Tensor, got {shape}")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be a finite floating-point Tensor")
        batch_size = img.size(0)
        if batch_size < 1 or x3.size(0) != batch_size or p3.size(0) != batch_size:
            raise ValueError("img, x3, and p3 must have the same non-empty batch size")
        return batch_size

    @staticmethod
    def _normalize_image_ids(value, batch_size: int) -> List[str]:
        if value is None:
            raise ValueError("real image IDs are required for SV-UME candidates")
        if isinstance(value, (str, bytes)):
            values = [value]
        elif torch.is_tensor(value):
            values = value.detach().cpu().reshape(-1).tolist()
        elif hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
            raw = value.tolist()
            values = raw if isinstance(raw, list) else [raw]
        elif isinstance(value, Sequence):
            values = list(value)
        else:
            values = [value]
        if len(values) != batch_size:
            raise ValueError(f"image ID count must be {batch_size}, got {len(values)}")
        return [str(item) for item in values]

    def _memory_dimensions(self, labeled_memory) -> Tuple[int, int]:
        if labeled_memory is None:
            raise ValueError("labeled_memory is required")
        memory_dim = int(getattr(labeled_memory, "mem_dim", getattr(self.cfg, "cbm_memory_dim", 128)))
        value_dim = int(getattr(labeled_memory, "value_dim", getattr(self.cfg, "cbm_value_dim", len(VALUE_LAYOUT))))
        if memory_dim <= 0 or value_dim <= 0:
            raise ValueError("labeled memory dimensions must be positive")
        return memory_dim, value_dim

    @staticmethod
    def _fit_channels(value: torch.Tensor, channels: int) -> torch.Tensor:
        if value.size(1) == channels:
            return value
        if value.size(1) > channels:
            return value[:, :channels]
        return F.pad(value, (0, 0, 0, 0, 0, channels - value.size(1)))

    @staticmethod
    def _fit_last_dim(value: torch.Tensor, width: int) -> torch.Tensor:
        if value.size(-1) == width:
            return value
        if value.size(-1) > width:
            return value[..., :width]
        return F.pad(value, (0, width - value.size(-1)))

    @staticmethod
    def _build_value(region: str, sdf: float, reliability: float, *, device, dtype):
        value = torch.zeros(len(VALUE_LAYOUT), device=device, dtype=dtype)
        value[REGION_TO_ID[region]] = 1.0
        is_foreground = region in ("fg_core", "fg_boundary")
        value[4] = 0.0 if is_foreground else 1.0
        value[5] = 1.0 if is_foreground else 0.0
        value[6] = float(sdf)
        value[7] = float(reliability)
        return value

    @staticmethod
    def _empty_pools() -> Dict[str, List[TokenCandidate]]:
        return {region: [] for region in REGION_NAMES}

    @staticmethod
    def _base_stats(batch_size: int, epoch: int, step: int) -> Dict[str, Any]:
        return {
            "batch_size": int(batch_size),
            "epoch": int(epoch),
            "step": int(step),
            "sam_used": False,
            "target_size": None,
            "region_pixel_counts": {region: 0 for region in REGION_NAMES},
            "region_image_counts": {region: 0 for region in REGION_NAMES},
            "candidate_counts": {region: 0 for region in REGION_NAMES},
            "image_score_mean": 0.0,
            "image_score_min": 0.0,
            "image_score_max": 0.0,
            "image_threshold": None,
            "image_evidence_valid_count": 0,
            "image_above_threshold_count": 0,
            "image_allowed_count": 0,
            "image_score_quantiles": {},
            "image_component_quantiles": {},
            "region_score_mean": {region: 0.0 for region in REGION_NAMES},
            "token_score_mean": {region: 0.0 for region in REGION_NAMES},
            "token_score_quantiles": {region: {} for region in REGION_NAMES},
            "token_component_quantiles": {},
            "cbm_valid_ratio": {region: 0.0 for region in REGION_NAMES},
            "enabled_regions": list(REGION_NAMES),
            "disabled_regions": [],
            "token_score_mode": "product",
            "accepted_total": 0,
            "rejected_total": 0,
            "novel_pending_candidates": 0,
            "global_type_counts": {},
            "cbm_logit_scale": None,
            "rejection_units": {
                "image": ("sam_not_used", "image_invalid_evidence", "image_below_threshold"),
                "image_region": ("region_empty", "region_below_threshold"),
                "token": (
                    "token_structural_invalid",
                    "token_cbm_invalid",
                    "token_below_threshold",
                ),
            },
        }

    @staticmethod
    def _summarize_tensor(value: torch.Tensor) -> Dict[str, float]:
        value = value.detach().float().reshape(-1)
        value = value[torch.isfinite(value)]
        if value.numel() == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p99": 0.0,
                "max": 0.0,
            }
        quantiles = torch.quantile(
            value,
            torch.tensor([0.50, 0.90, 0.99], device=value.device),
        )
        return {
            "count": int(value.numel()),
            "mean": float(value.mean().item()),
            "p50": float(quantiles[0].item()),
            "p90": float(quantiles[1].item()),
            "p99": float(quantiles[2].item()),
            "max": float(value.max().item()),
        }

    @classmethod
    def _summarize_image_admission(cls, image_result: Mapping[str, Any]) -> Dict[str, Any]:
        scores = image_result["score"]
        evidence_valid = image_result["evidence_valid"].bool()
        threshold = float(image_result["threshold"])
        return {
            "image_threshold": threshold,
            "image_evidence_valid_count": int(evidence_valid.sum().item()),
            "image_above_threshold_count": int((scores > threshold).sum().item()),
            "image_allowed_count": int(image_result["allow_image"].sum().item()),
            "image_score_quantiles": cls._summarize_tensor(scores[evidence_valid]),
            "image_component_quantiles": {
                name: cls._summarize_tensor(value[evidence_valid])
                for name, value in image_result["components"].items()
            },
        }

    def _log_diagnostics(self, result: Mapping[str, Any]) -> None:
        if self.logger is None:
            return
        stats = result.get("stats", {})
        rejected = result.get("rejected", {})
        lines = (
            f"[SV-UME][reject][step={stats.get('step')}] {rejected}",
            f"[SV-UME][stats][step={stats.get('step')}] "
            f"image_score=({stats.get('image_score_min', 0.0):.4f},"
            f"{stats.get('image_score_mean', 0.0):.4f},"
            f"{stats.get('image_score_max', 0.0):.4f}) "
            f"image_threshold={stats.get('image_threshold')} "
            f"image_valid={stats.get('image_evidence_valid_count')} "
            f"image_above={stats.get('image_above_threshold_count')} "
            f"image_allowed={stats.get('image_allowed_count')} "
            f"region_pixels={stats.get('region_pixel_counts')} "
            f"region_images={stats.get('region_image_counts')} "
            f"region_scores={stats.get('region_score_mean')} "
            f"cbm_valid_ratio={stats.get('cbm_valid_ratio')} "
            f"candidate_counts={stats.get('candidate_counts')} "
            f"enabled={stats.get('enabled_regions')} disabled={stats.get('disabled_regions')}",
            f"[SV-UME][image][step={stats.get('step')}] "
            f"score={stats.get('image_score_quantiles')} "
            f"components={stats.get('image_component_quantiles')}",
            f"[SV-UME][token][step={stats.get('step')}] "
            f"score={stats.get('token_score_quantiles')} "
            f"components={stats.get('token_component_quantiles')}",
        )
        for line in lines:
            for name in ("info", "key_info", "success_info"):
                log_fn = getattr(self.logger, name, None)
                if callable(log_fn):
                    log_fn(line)
                    break

    def _result(self, pools, rejected, stats) -> Dict[str, Any]:
        stats["cbm_logit_scale"] = float(self.cbm_logit_scale)
        stats["rejected_total"] = int(sum(rejected.values()))
        return {
            "candidate_pools": pools,
            "rejected": dict(rejected),
            "stats": stats,
        }

    @staticmethod
    def _non_negative_int(value, name: str) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        normalized = int(value)
        if normalized < 0 or normalized != value:
            raise ValueError(f"{name} must be a non-negative integer")
        return normalized


__all__ = ["TokenCandidate", "SAMRefinedCandidateBuilder", "REJECTION_REASONS"]
