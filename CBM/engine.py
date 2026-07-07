from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import torch
import torch.nn as nn

from CBM.boundary.query import build_pred_boundary
from CBM.config.defaults import apply_cbm_defaults
from CBM.config.labeled_memory import resolve_labeled_memory_profile
from CBM.config.schedule import cbm_enabled_for_epoch, cbm_stage_epoch, cbm_stage_name
from CBM.context.aggregator import ContextualBoundaryAggregator
from CBM.core.outputs import build_fallback_aux, build_used_aux
from CBM.core.state import CBMState
from CBM.correction.logit_fusion import BoundaryLogitFusion
from CBM.correction.p3_correction import BoundaryCorrectionHead
from CBM.losses.total import compute_cbm_losses
from CBM.memory.bank import DenseBoundaryMemory
from CBM.memory.builder import LabeledMemoryBuilder
from CBM.retrieval.global_router import GlobalMemoryRouter
from CBM.retrieval.pointwise import PointwiseBoundaryRetriever
from utils.log_control import log_enabled


def _normalize_device(device: Optional[torch.device]) -> Optional[torch.device]:
    if device is None:
        return None
    if isinstance(device, int):
        return torch.device("cuda", device) if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device)


class CBMPFIEngine(nn.Module):
    """Single stable orchestration object for PLAN_V4.2 CBM-PFI."""

    def __init__(self, config, device: Optional[torch.device] = None, logger=None) -> None:
        super().__init__()
        self.config = apply_cbm_defaults(config)
        self.device = _normalize_device(device)
        self.logger = logger
        self.state = CBMState()
        self.labeled_memory_profile = resolve_labeled_memory_profile(self.config)
        setattr(self.config, "cbm_top_img_k", int(self.labeled_memory_profile.top_img_k))
        self.memory = DenseBoundaryMemory(
            mem_dim=int(getattr(self.config, "cbm_memory_dim", 128)),
            value_dim=int(getattr(self.config, "cbm_value_dim", 8)),
            selection_config=self.labeled_memory_profile,
        )
        self.memory.set_compat_meta(self._build_memory_compat_meta())
        self.builder = LabeledMemoryBuilder(self.memory, config=self.config, logger=logger)

        self.router: Optional[GlobalMemoryRouter] = None
        self.retriever: Optional[PointwiseBoundaryRetriever] = None
        self.context: Optional[ContextualBoundaryAggregator] = None
        self.correction: Optional[BoundaryCorrectionHead] = None
        self.logit_fusion = BoundaryLogitFusion(lambda_logit=float(getattr(self.config, "cbm_lambda_logit", 0.5)))
        self._x3_channels: Optional[int] = None
        self._p3_channels: Optional[int] = None
        if self.device is not None:
            self.to(self.device)

    def prepare_epoch(self, model, labeled_loader, epoch: int) -> None:
        self.state.epoch = int(epoch)
        self.state.stage_epoch = cbm_stage_epoch(self.config, epoch)
        self.state.stage_name = cbm_stage_name(self.config, epoch)
        self.state.memory_build_failed = False
        self.state.memory_build_error = None
        try:
            self.builder.prepare_epoch(model, labeled_loader, epoch)
        except Exception as exc:
            self.memory.clear()
            self.state.memory_build_failed = True
            self.state.memory_build_error = str(exc)
            self._warn(f"[CBM] memory build failed at epoch {epoch}: {exc}. Fallback to baseline.")
        self.state.memory_ready = self.memory.is_ready()
        self._info(self.memory.diagnostic_string())

    def enabled_for_epoch(self, epoch: Optional[int] = None, memory_t=None) -> bool:
        current_epoch = self.state.epoch if epoch is None else int(epoch)
        self.state.memory_ready = self.memory_ready(memory_t)
        return cbm_enabled_for_epoch(self.config, current_epoch, self.state.memory_ready)

    def memory_ready(self, memory_t=None) -> bool:
        labeled_memory = self._resolve_forward_memory(memory_t)
        ready = getattr(labeled_memory, "is_ready", None)
        return bool(callable(ready) and ready())

    def initialize_modules(
        self,
        x3_channels: Optional[int] = None,
        p3_channels: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        channels = getattr(self.config, "lateral_channels_in_collection", None)
        if x3_channels is None:
            if channels is None or len(channels) < 2:
                raise ValueError("x3_channels is required when config.lateral_channels_in_collection is unavailable")
            x3_channels = int(channels[1])
        if p3_channels is None:
            if channels is None or len(channels) < 3:
                raise ValueError("p3_channels is required when config.lateral_channels_in_collection is unavailable")
            p3_channels = int(channels[2])
        target_device = _normalize_device(device) or self.device or torch.device("cpu")
        ref = torch.empty(0, device=target_device, dtype=dtype)
        self._ensure_modules(int(x3_channels), int(p3_channels), ref)
        return self

    def apply_p3_hook(
        self,
        *,
        x: torch.Tensor,
        x3: torch.Tensor,
        p3: torch.Tensor,
        m3: Optional[torch.Tensor],
        training: bool = False,
        memory_t=None,
    ):
        del x, training
        if m3 is None:
            aux = build_fallback_aux("m3_none", p3=p3)
            self.state.last_aux = aux
            return p3, aux
        if not self.enabled_for_epoch(memory_t=memory_t):
            aux = build_fallback_aux("cbm_disabled_or_memory_not_ready", p3=p3)
            self.state.last_aux = aux
            return p3, aux

        self._ensure_modules(x3_channels=x3.size(1), p3_channels=p3.size(1), ref=p3)
        prob3 = torch.sigmoid(m3)
        B_query, boundary_mask = build_pred_boundary(
            prob3,
            kernel=int(getattr(self.config, "cbm_boundary_kernel", 3)),
            alpha_unc=float(getattr(self.config, "cbm_boundary_alpha_unc", 0.5)),
            alpha_grad=float(getattr(self.config, "cbm_boundary_alpha_grad", 0.5)),
            theta=float(getattr(self.config, "cbm_boundary_theta", 0.2)),
        )

        labeled_memory = self._resolve_forward_memory(memory_t)
        top_img_ids, img_scores = self.router(
            x3,
            labeled_memory,
            top_img_k=int(self.labeled_memory_profile.top_img_k),
        )
        K_mem, V_mem, meta = labeled_memory.get_sub_memory(
            top_img_ids,
            device=p3.device,
            dtype=p3.dtype,
        )
        retrieval = self.retriever(
            p3,
            B_query=B_query,
            boundary_mask=boundary_mask,
            K_mem=K_mem,
            V_mem=V_mem,
            topk_token=int(getattr(self.config, "cbm_topk_token", 16)),
        )
        Y_ctx, R_ctx, cons_map = self.context(
            p3,
            prob3,
            retrieval["Y_map"],
            retrieval["R_map"],
            retrieval["valid_map"],
        )
        p3_corr, z_mem3, gate3 = self.correction(
            p3,
            m3,
            B_query,
            retrieval["Y_map"],
            Y_ctx,
            R_ctx,
            retrieval["U_map"],
            cons_map,
            retrieval["valid_map"],
        )
        aux = build_used_aux(
            top_img_ids=top_img_ids,
            img_scores=img_scores,
            K_mem=K_mem,
            B_query=B_query,
            boundary_mask=boundary_mask,
            gate3=gate3,
            z_mem3=z_mem3,
            Y_map=retrieval["Y_map"],
            Y_ctx=Y_ctx,
            R_map=retrieval["R_map"],
            R_ctx=R_ctx,
            cons_map=cons_map,
            U_map=retrieval["U_map"],
            valid_map=retrieval["valid_map"],
            prob3=prob3,
            meta=meta,
        )
        self.state.last_aux = aux
        return p3_corr, aux

    def apply_final_fusion(self, p1_out: torch.Tensor, aux):
        if not aux or not aux.get("cbm_used", False):
            return p1_out
        z_mem3 = aux.get("z_mem3")
        B_query = aux.get("B_query")
        gate3 = aux.get("gate3")
        if z_mem3 is None or B_query is None or gate3 is None:
            return p1_out
        aux["p_main"] = torch.sigmoid(p1_out).detach()
        z_final = self.logit_fusion(p1_out, z_mem3, B_query, gate3)
        aux["p_final"] = torch.sigmoid(z_final).detach()
        retrieval = aux.get("retrieval")
        if isinstance(retrieval, dict):
            retrieval["p_main"] = aux["p_main"]
            retrieval["p_final"] = aux["p_final"]
        self.state.last_aux = aux
        return z_final

    def compute_losses(self, aux, gt: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss, loss_dict = compute_cbm_losses(aux, gt, self.config)
        self.state.loss_dict = {key: float(value.detach().item()) for key, value in loss_dict.items()}
        return loss

    def memory_state_dict(self):
        self.memory.set_compat_meta(self._build_memory_compat_meta())
        return self.memory.to_state_dict()

    def load_memory_state_dict(
        self,
        state,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        target_device = _normalize_device(device) or self.device or torch.device("cpu")
        self._check_memory_state_compatible(state)
        self.memory.load_state_dict(state, device=target_device, dtype=dtype)
        self.state.memory_ready = self.memory.is_ready()

    def _build_memory_compat_meta(self):
        return {
            "backbone": str(getattr(self.config, "backbone", "")),
            "img_size": int(getattr(self.config, "img_size", 0)),
            "lateral_channels_in_collection": [
                int(channel) for channel in getattr(self.config, "lateral_channels_in_collection", [])
            ],
            "pc_dim": int(getattr(self.config, "cbm_memory_dim", 128)),
            "value_dim": int(getattr(self.config, "cbm_value_dim", 8)),
            "feature_version": str(getattr(self.config, "cbm_memory_feature_version", "swin_l_pc_hbm_v1")),
        }

    def _check_memory_state_compatible(self, state) -> None:
        if not state:
            return
        if not isinstance(state, Mapping):
            raise RuntimeError("Memory checkpoint must be a mapping. Rebuild labelled memory.")
        memory_meta = state.get("compat_meta")
        if not memory_meta:
            raise RuntimeError("Memory checkpoint has no compat_meta. Rebuild labelled memory.")
        expected = self._build_memory_compat_meta()
        for key, expected_value in expected.items():
            if memory_meta.get(key) != expected_value:
                raise RuntimeError(
                    f"Memory incompatible: {key}: memory={memory_meta.get(key)} "
                    f"expected={expected_value}. Rebuild labelled memory."
                )

    def _ensure_modules(self, x3_channels: int, p3_channels: int, ref: torch.Tensor) -> None:
        if self.router is not None and self._x3_channels != int(x3_channels):
            raise ValueError(
                f"CBM x3 channel mismatch: initialized with {self._x3_channels}, got {int(x3_channels)}. "
                "Rebuilding learnable CBM modules after optimizer creation would leave parameters unoptimized."
            )
        if self.router is None:
            self.router = GlobalMemoryRouter(
                x3_channels=int(x3_channels),
                memory_dim=int(getattr(self.config, "cbm_memory_dim", 128)),
                top_img_k=int(self.labeled_memory_profile.top_img_k),
            )
            self._x3_channels = int(x3_channels)
        if (self.retriever is not None or self.correction is not None) and self._p3_channels != int(p3_channels):
            raise ValueError(
                f"CBM p3 channel mismatch: initialized with {self._p3_channels}, got {int(p3_channels)}. "
                "Rebuilding learnable CBM modules after optimizer creation would leave parameters unoptimized."
            )
        if self.retriever is None or self.correction is None:
            self.retriever = PointwiseBoundaryRetriever(
                p3_channels=int(p3_channels),
                memory_dim=int(getattr(self.config, "cbm_memory_dim", 128)),
                value_dim=int(getattr(self.config, "cbm_value_dim", 8)),
                topk_token=int(getattr(self.config, "cbm_topk_token", 16)),
            )
            self.correction = BoundaryCorrectionHead(
                p3_channels=int(p3_channels),
                memory_dim=int(getattr(self.config, "cbm_memory_dim", 128)),
                value_dim=int(getattr(self.config, "cbm_value_dim", 8)),
                lambda_feat=float(getattr(self.config, "cbm_lambda_feat", 0.1)),
            )
            self._p3_channels = int(p3_channels)
        if self.context is None:
            self.context = ContextualBoundaryAggregator(
                kernel_size=int(getattr(self.config, "cbm_context_kernel_size", 3)),
                tau_feat=float(getattr(self.config, "cbm_context_tau_feat", 0.1)),
                tau_prob=float(getattr(self.config, "cbm_context_tau_prob", 0.2)),
                tau_evi=float(getattr(self.config, "cbm_context_tau_evi", 0.2)),
            )
        self.to(device=ref.device, dtype=ref.dtype)

    def _resolve_forward_memory(self, memory_t=None):
        if memory_t is None:
            return self.memory
        if not isinstance(memory_t, Mapping):
            raise TypeError("memory_t must be a mapping or None")

        labeled_memory = self._memory_entry(
            memory_t,
            canonical="labeled_memory",
            alias="L_t",
            default=self.memory,
        )
        if labeled_memory is None:
            labeled_memory = self.memory
        return labeled_memory

    @staticmethod
    def _memory_entry(memory_t, *, canonical, alias, default):
        has_canonical = canonical in memory_t
        has_alias = alias in memory_t
        if has_canonical and has_alias:
            canonical_value = memory_t[canonical]
            alias_value = memory_t[alias]
            if canonical_value is not alias_value:
                raise ValueError(
                    f"memory_t contains conflicting {canonical!r} and {alias!r} values"
                )
        if has_canonical:
            return memory_t[canonical]
        if has_alias:
            return memory_t[alias]
        return default

    def _info(self, message: str) -> None:
        if not log_enabled(self.config):
            return
        if self.logger is None:
            print(message)
            return
        log_fn = getattr(self.logger, "info", None) or getattr(self.logger, "key_info", None)
        if log_fn is not None:
            log_fn(message)

    def _warn(self, message: str) -> None:
        if not log_enabled(self.config):
            return
        if self.logger is None:
            print(message)
            return
        log_fn = getattr(self.logger, "warn_info", None) or getattr(self.logger, "warning", None) or getattr(self.logger, "info", None)
        if log_fn is not None:
            log_fn(message)
