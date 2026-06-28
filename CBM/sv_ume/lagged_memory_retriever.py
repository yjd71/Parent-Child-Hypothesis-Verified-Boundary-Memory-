from __future__ import annotations

import math
import weakref
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from CBM.memory.labels import VALUE_LAYOUT
from CBM.sv_ume.quality_adaptive_fusion import QualityAdaptiveSourceFusion


class LaggedLabeledUnlabeledRetriever(nn.Module):
    """Retrieve labeled and lagged-unlabeled memories without concatenating them."""

    def __init__(
        self,
        cfg,
        pointwise_retriever: nn.Module,
        global_router: nn.Module,
        source_fusion: Optional[nn.Module] = None,
        register_backends: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(pointwise_retriever, nn.Module):
            raise TypeError("pointwise_retriever must be an nn.Module")
        if not isinstance(global_router, nn.Module):
            raise TypeError("global_router must be an nn.Module")
        if not hasattr(pointwise_retriever, "q_proj"):
            raise TypeError("pointwise_retriever must expose q_proj")

        self.cfg = cfg
        self.register_backends = bool(register_backends)
        if self.register_backends:
            self.pointwise_retriever = pointwise_retriever
            self.global_router = global_router
        else:
            # The engine already owns these modules. Weak proxies keep the wrapper
            # callable without registering duplicate state_dict paths.
            object.__setattr__(
                self,
                "pointwise_retriever",
                weakref.proxy(pointwise_retriever),
            )
            object.__setattr__(
                self,
                "global_router",
                weakref.proxy(global_router),
            )
        self.source_fusion = (
            QualityAdaptiveSourceFusion(cfg)
            if source_fusion is None
            else source_fusion
        )
        if not isinstance(self.source_fusion, nn.Module):
            raise TypeError("source_fusion must be an nn.Module")
        if not callable(getattr(self.source_fusion, "compute_score", None)):
            raise TypeError("source_fusion must expose compute_score(retrieval, reference)")

        self.enabled = bool(getattr(cfg, "use_sv_ume", False))
        self.top_img_k = int(getattr(cfg, "cbm_top_img_k", 8))
        self.topk_token = int(getattr(cfg, "cbm_topk_token", 16))
        self.eps = float(getattr(pointwise_retriever, "eps", 1.0e-6))
        if self.top_img_k <= 0:
            raise ValueError("cbm_top_img_k must be positive")
        if self.topk_token <= 0:
            raise ValueError("cbm_topk_token must be positive")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("pointwise_retriever.eps must be finite and positive")

        try:
            self.reliability_index = tuple(VALUE_LAYOUT).index("reliability")
        except ValueError as exc:
            raise ValueError("VALUE_LAYOUT must contain reliability") from exc
        value_dim = int(
            getattr(pointwise_retriever, "value_dim", len(VALUE_LAYOUT))
        )
        if self.reliability_index >= value_dim:
            raise ValueError("pointwise_retriever value_dim has no reliability field")

    def forward(
        self,
        *,
        p3: torch.Tensor,
        B_query: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
        x3: torch.Tensor,
        labeled_memory,
        unlabeled_memory=None,
    ) -> Dict[str, Any]:
        self._validate_inputs(p3, B_query, x3)
        self._validate_memory_protocol(labeled_memory, "labeled_memory")

        ret_l = self._retrieve_source(
            p3=p3,
            B_query=B_query,
            boundary_mask=boundary_mask,
            x3=x3,
            memory=labeled_memory,
            source="labeled",
        )
        query_map = self._query_map(p3)
        ret_l = self._add_quality_maps(ret_l, query_map, "ret_l")

        if not self.enabled or not self._memory_ready(unlabeled_memory):
            return self._labeled_only_output(ret_l, ret_u=None)

        self._validate_memory_protocol(unlabeled_memory, "unlabeled_memory")
        self._validate_unlabeled_frozen(unlabeled_memory)
        ret_u = self._retrieve_source(
            p3=p3,
            B_query=B_query,
            boundary_mask=boundary_mask,
            x3=x3,
            memory=unlabeled_memory,
            source="unlabeled_lagged",
        )
        ret_u = self._add_quality_maps(ret_u, query_map, "ret_u")
        self._validate_source_shapes(ret_l, ret_u)

        used_unlabeled = bool(ret_u["valid_map"].detach().bool().any().item())
        if not used_unlabeled:
            return self._labeled_only_output(ret_l, ret_u=ret_u)

        fused = dict(self.source_fusion(ret_l, ret_u))
        self._validate_fused_output(fused, ret_l)
        fused.update(
            {
                "ret_l": ret_l,
                "ret_u": ret_u,
                "used_unlabeled_memory": True,
            }
        )
        return fused

    def _retrieve_source(
        self,
        *,
        p3: torch.Tensor,
        B_query: torch.Tensor,
        boundary_mask: Optional[torch.Tensor],
        x3: torch.Tensor,
        memory,
        source: str,
    ) -> Dict[str, Any]:
        top_img_ids, img_scores = self.global_router(
            x3,
            memory,
            top_img_k=self.top_img_k,
        )
        sub_memory = memory.get_sub_memory(
            top_img_ids,
            device=p3.device,
            dtype=p3.dtype,
        )
        if not isinstance(sub_memory, tuple) or len(sub_memory) != 3:
            raise TypeError("memory.get_sub_memory() must return (keys, values, meta)")
        keys, values, memory_meta = sub_memory
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("memory sub-memory keys and values must be tensors")
        if keys.dim() != 2 or values.dim() != 2 or keys.size(0) != values.size(0):
            raise ValueError("memory sub-memory keys/values must be aligned 2D tensors")

        retrieval = self.pointwise_retriever(
            p3,
            B_query=B_query,
            boundary_mask=boundary_mask,
            K_mem=keys,
            V_mem=values,
            topk_token=self.topk_token,
        )
        if not isinstance(retrieval, Mapping):
            raise TypeError("pointwise_retriever must return a mapping")
        output = dict(retrieval)
        output.update(
            {
                "top_img_ids": top_img_ids,
                "img_scores": img_scores,
                "memory_meta": memory_meta,
                "routed_token_count": int(keys.size(0)),
                "source": source,
            }
        )
        return output

    def _add_quality_maps(
        self,
        retrieval: Mapping[str, Any],
        query_map: torch.Tensor,
        name: str,
    ) -> Dict[str, Any]:
        output = dict(retrieval)
        y_map, r_map, _, valid_map = self._retrieval_maps(output, name)
        if query_map.shape != r_map.shape:
            raise ValueError(
                f"{name} query/R shapes must match, got "
                f"{tuple(query_map.shape)} and {tuple(r_map.shape)}"
            )
        if y_map.size(1) <= self.reliability_index:
            raise ValueError(f"{name} Y_map has no reliability channel")

        valid = (valid_map > 0.5).to(dtype=y_map.dtype)
        sim_mean = (query_map * r_map).sum(dim=1, keepdim=True)
        sim_mean = sim_mean.clamp(-1.0, 1.0) * valid

        region_evidence = y_map[:, :4].clamp_min(0.0)
        region_distribution = region_evidence / region_evidence.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps)
        topk_consistency = region_distribution.max(dim=1, keepdim=True).values
        topk_consistency = topk_consistency.clamp(0.0, 1.0) * valid

        memory_reliability = y_map[
            :,
            self.reliability_index : self.reliability_index + 1,
        ]
        memory_reliability = memory_reliability.clamp(0.0, 1.0) * valid

        for field, tensor in (
            ("sim_mean", sim_mean),
            ("topk_consistency", topk_consistency),
            ("memory_reliability", memory_reliability),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name}.{field} contains NaN or Inf")
            output[field] = tensor
        output["consistency_mode"] = "soft_region_evidence_mass"
        output["similarity_mode"] = "attention_weighted_query_key"
        return output

    def _labeled_only_output(
        self,
        ret_l: Mapping[str, Any],
        ret_u: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        y_map, r_map, u_map, valid_map = self._retrieval_maps(ret_l, "ret_l")
        score_l = self.source_fusion.compute_score(ret_l, y_map)
        score_u = (
            torch.zeros_like(score_l)
            if ret_u is None
            else self.source_fusion.compute_score(ret_u, y_map)
        )
        w_l = torch.ones_like(score_l)
        w_u = torch.zeros_like(score_l)
        source_entropy = torch.zeros_like(score_l)

        output = dict(ret_l)
        output.update(
            {
                "Y_map": y_map,
                "Y": y_map,
                "R_map": r_map,
                "R": r_map,
                "U_map": u_map,
                "U": u_map,
                "uncertainty": u_map,
                "valid_map": valid_map,
                "w_l_map": w_l,
                "w_l": w_l,
                "w_u_map": w_u,
                "w_u": w_u,
                "score_l": score_l,
                "score_u": score_u,
                "source_entropy": source_entropy,
                "ret_l": ret_l,
                "ret_u": ret_u,
                "used_unlabeled_memory": False,
            }
        )
        return output

    def _query_map(self, p3: torch.Tensor) -> torch.Tensor:
        query_map = self.pointwise_retriever.q_proj(p3)
        query_map = F.normalize(query_map, dim=1, eps=self.eps)
        if not torch.isfinite(query_map).all():
            raise ValueError("projected pointwise query contains NaN or Inf")
        return query_map

    @staticmethod
    def _retrieval_maps(
        retrieval: Mapping[str, Any],
        name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fields = []
        for field in ("Y_map", "R_map", "U_map", "valid_map"):
            value = retrieval.get(field)
            if not isinstance(value, torch.Tensor) or value.dim() != 4:
                raise ValueError(f"{name}.{field} must have shape [B,C,H,W]")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name}.{field} contains NaN or Inf")
            fields.append(value)
        y_map, r_map, u_map, valid_map = fields
        if y_map.size(1) < 4:
            raise ValueError(f"{name}.Y_map must have at least four channels")
        if u_map.size(1) != 1 or valid_map.size(1) != 1:
            raise ValueError(f"{name} U_map and valid_map must be single-channel")
        base = (y_map.size(0), y_map.size(2), y_map.size(3))
        for field, tensor in (
            ("R_map", r_map),
            ("U_map", u_map),
            ("valid_map", valid_map),
        ):
            current = (tensor.size(0), tensor.size(2), tensor.size(3))
            if current != base:
                raise ValueError(f"{name}.{field} batch/spatial shape does not match Y_map")
            if tensor.device != y_map.device or tensor.dtype != y_map.dtype:
                raise ValueError(f"{name}.{field} device/dtype does not match Y_map")
        return y_map, r_map, u_map, valid_map

    @classmethod
    def _validate_source_shapes(
        cls,
        ret_l: Mapping[str, Any],
        ret_u: Mapping[str, Any],
    ) -> None:
        maps_l = cls._retrieval_maps(ret_l, "ret_l")
        maps_u = cls._retrieval_maps(ret_u, "ret_u")
        names = ("Y_map", "R_map", "U_map", "valid_map")
        for field, tensor_l, tensor_u in zip(names, maps_l, maps_u):
            if tensor_l.shape != tensor_u.shape:
                raise ValueError(f"ret_l/ret_u {field} shapes must match")
            if tensor_l.device != tensor_u.device or tensor_l.dtype != tensor_u.dtype:
                raise ValueError(f"ret_l/ret_u {field} device/dtype must match")

    @classmethod
    def _validate_fused_output(
        cls,
        fused: Mapping[str, Any],
        ret_l: Mapping[str, Any],
    ) -> None:
        fused_maps = cls._retrieval_maps(fused, "fused")
        labeled_maps = cls._retrieval_maps(ret_l, "ret_l")
        for name, fused_map, labeled_map in zip(
            ("Y_map", "R_map", "U_map", "valid_map"),
            fused_maps,
            labeled_maps,
        ):
            if fused_map.shape != labeled_map.shape:
                raise ValueError(f"fused {name} shape must match ret_l")
        for name in ("w_l", "w_u", "score_l", "score_u", "source_entropy"):
            value = fused.get(name)
            if not isinstance(value, torch.Tensor):
                raise KeyError(f"fused output is missing tensor {name!r}")

    @staticmethod
    def _memory_ready(memory) -> bool:
        if memory is None:
            return False
        ready = getattr(memory, "is_ready", None)
        return bool(callable(ready) and ready())

    @staticmethod
    def _validate_memory_protocol(memory, name: str) -> None:
        if memory is None:
            raise TypeError(f"{name} must not be None")
        missing = [
            method
            for method in ("is_ready", "get_image_keys", "get_sub_memory")
            if not callable(getattr(memory, method, None))
        ]
        if missing:
            raise TypeError(f"{name} is missing methods: {missing}")

    @staticmethod
    def _validate_unlabeled_frozen(memory) -> None:
        if hasattr(memory, "_frozen") and not bool(memory._frozen):
            raise ValueError("unlabeled_memory must be frozen before retrieval")

    @staticmethod
    def _validate_inputs(
        p3: torch.Tensor,
        B_query: torch.Tensor,
        x3: torch.Tensor,
    ) -> None:
        for name, tensor in (("p3", p3), ("x3", x3)):
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
                raise ValueError(f"{name} must have shape [B,C,H,W]")
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite floating point")
        if p3.size(0) != x3.size(0):
            raise ValueError("p3 and x3 batch sizes must match")
        if not isinstance(B_query, torch.Tensor) or B_query.dim() not in (3, 4):
            raise ValueError("B_query must have shape [B,H,W] or [B,1,H,W]")
        if B_query.dim() == 4 and B_query.size(1) != 1:
            raise ValueError("B_query must be single-channel")
        if B_query.size(0) != p3.size(0):
            raise ValueError("B_query and p3 batch sizes must match")
        if not torch.isfinite(B_query).all():
            raise ValueError("B_query contains NaN or Inf")


__all__ = ["LaggedLabeledUnlabeledRetriever"]
