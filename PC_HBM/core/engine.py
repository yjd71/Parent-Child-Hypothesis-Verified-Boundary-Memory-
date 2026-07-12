"""PC-HBM orchestration engine for TALNet.

The engine owns PC-HBM modules and labelled-only memory.  It reuses TALNet's
existing encoder/decoder split functions and only activates when the
``use_pc_hbm``/``pc_hbm_enable`` gate is true and memory is ready.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.channel_spec import build_talnet_channel_spec

from ..debug.diagnostics import collect_pc_hbm_diagnostics
from ..fusion.hypothesis_token_builder import HypothesisTokenBuilder
from ..fusion.p3_gated_residual import P3GatedResidual
from ..fusion.pc_hca import PCHCA
from ..fusion.pc_scatter import pc_scatter
from ..fusion.pc_token_decoder import PCTokenDecoder
from ..fusion.query_state_builder import QueryStateBuilder
from ..fusion.structured_gate_mlp import StructuredGateMLP
from ..memory.pc_memory import PCHBMMemory, parent_values_from_region
from ..memory.pc_region_builder import build_pc_regions
from ..memory.sampling_policy import sample_region_indices
from ..refinement.adaptive_mixture_head import AdaptiveMixtureHead
from ..refinement.boundary_query_head import BoundaryQueryHead3
from ..refinement.p1_pixel_refinement_attention import P1PixelRefinementAttention
from ..refinement.p2_boundary_retarget_attention import P2BoundaryRetargetAttention
from ..retrieval.child_query_builder import ChildQueryBuilder
from ..retrieval.child_verifier_v2 import ChildVerifierV2
from ..routing.camouflage_context_router import CamouflageContextRouter
from .pc_config import apply_pc_hbm_defaults, pc_hbm_enabled, pc_hbm_should_rebuild_memory
from ..common.utils import boundary_features_from_logits, finite_or_zero, gather_tokens


def _normalize_device(device: Optional[torch.device]) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if isinstance(device, int):
        return torch.device("cuda", device) if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device)


class PCHBMEngine(nn.Module):
    """Complete PC-HBM engine attached to ``ModelEMA``."""

    def __init__(self, config: Any, device: Optional[torch.device] = None, logger=None) -> None:
        super().__init__()
        self.config = apply_pc_hbm_defaults(config)
        self.logger = logger
        self.device_hint = _normalize_device(device)
        self.channel_spec = build_talnet_channel_spec(self.config)
        self.dim = int(getattr(self.config, "cbm_memory_dim", self.channel_spec.pc_dim))
        self.value_dim = int(getattr(self.config, "cbm_value_dim", self.channel_spec.value_dim))
        self.geometry_dim = int(getattr(self.config, "geometry_dim", 6))
        self.memory = PCHBMMemory(self.dim, self.value_dim, self.geometry_dim, config=self.config)
        self.memory.compat_meta = self._build_memory_compat_meta()
        self.boundary3 = BoundaryQueryHead3(
            top_ratio=0.25,
            max_tokens=getattr(config, "p3_boundary_max_tokens", 600),
                                            )
        self.router = CamouflageContextRouter(self.channel_spec.x3, dim=self.dim, top_img_k=int(getattr(self.config, "cbm_top_img_k", 32)))
        self.parent_retriever = ParentRetrieverProxy(self.channel_spec.p3, self.dim, int(getattr(self.config, "parent_topk", 64)), float(getattr(self.config, "cbm_tau_parent", 0.07)))
        self.child_query = ChildQueryBuilder(self.channel_spec.p2, dim=self.dim, window=int(getattr(self.config, "child_window_size", 5)))
        self.child_verifier = ChildVerifierV2(dim=self.dim, value_dim=self.value_dim, geometry_dim=self.geometry_dim)
        self.hyp_builder = HypothesisTokenBuilder(dim=self.dim, value_dim=self.value_dim, geometry_dim=self.geometry_dim)
        self.query_state = QueryStateBuilder(dim=self.dim)
        self.hca = PCHCA(dim=self.dim, num_heads=int(getattr(self.config, "attn_num_heads", 8)), head_dim=int(getattr(self.config, "attn_head_dim", 64)), tau=float(getattr(self.config, "cbm_tau_hca", 0.10)))
        self.token_decoder = PCTokenDecoder(dim=self.dim)
        self.gate_mlp = StructuredGateMLP()
        self.p3_residual = P3GatedResidual(dim=self.dim, p3_ch=self.channel_spec.p3)
        self.p2_bra = P2BoundaryRetargetAttention(
            self.channel_spec.p2,
            dim=self.dim,
            window=int(getattr(self.config, "p2_bra_local_window", 3)),
            tau=float(getattr(self.config, "cbm_tau_bra", 0.10)),
            top_ratio=float(getattr(self.config, "p2_boundary_top_ratio", 0.25)),
            detach_refs=bool(getattr(self.config, "pc_hbm_detach_refs", True)),
            max_tokens=getattr(config, "p2_boundary_max_tokens", 1200),
        )
        self.p1_pra = P1PixelRefinementAttention(
            self.channel_spec.p1,
            dim=self.dim,
            window=int(getattr(self.config, "p1_pra_local_window", 3)),
            tau=float(getattr(self.config, "cbm_tau_pra", 0.10)),
            top_ratio=float(getattr(self.config, "p1_boundary_top_ratio", 0.20)),
            detach_refs=bool(getattr(self.config, "pc_hbm_detach_refs", True)),
            max_tokens=getattr(config, "p1_boundary_max_tokens", 2500),
        )
        self.mixture = AdaptiveMixtureHead(
            r_max=float(getattr(self.config, "r_max", 2.0)),
            max_offset=float(getattr(self.config, "max_offset", 3.0)),
            mask_corr_epsilon=float(getattr(self.config, "mask_corr_epsilon", 0.10)),
            init_bias=getattr(self.config, "mixture_init_bias", [1.0, -0.5, -0.5, -0.5]),
            use_branch_quality=bool(getattr(self.config, "use_branch_quality_head", True)),
            use_branch_dropout=bool(getattr(self.config, "use_branch_dropout", True)),
        )
        self.loss_dict: Dict[str, float] = {}
        self.last_aux: Optional[Dict[str, Any]] = None
        self.memory_build_failed = False
        self.memory_build_error: Optional[str] = None
        self.to(self.device_hint)

    def enabled_for_epoch(self, epoch: Optional[int] = None, memory_t=None) -> bool:
        if not pc_hbm_enabled(self.config):
            return False
        memory = self._resolve_forward_memory(memory_t)
        if memory is None or not memory.is_ready():
            return False
        if epoch is None:
            return True
        return int(epoch) >= int(getattr(self.config, "parent_start_epoch", 6))

    def memory_ready(self, memory_t=None) -> bool:
        memory = self._resolve_forward_memory(memory_t)
        return bool(memory is not None and memory.is_ready())

    @torch.no_grad()
    def prepare_epoch(self, model, labeled_loader, epoch: int) -> None:
        """Rebuild labelled-only memory with the EMA teacher."""

        if not pc_hbm_should_rebuild_memory(self.config, epoch):
            return
        if bool(getattr(self.config, "use_unlabeled_memory_update", False)):
            self._warn("[PC-HBM] use_unlabeled_memory_update=True ignored; memory remains labelled-only.")
            setattr(self.config, "use_unlabeled_memory_update", False)
        self.memory_build_failed = False
        self.memory_build_error = None
        self.memory.clear()
        self.memory.compat_meta = self._build_memory_compat_meta()
        if labeled_loader is None:
            self.memory_build_failed = True
            self.memory_build_error = "labeled_loader_none"
            self._warn("[PC-HBM] labelled memory rebuild skipped: labeled_loader is None.")
            return
        target_model = model.module if hasattr(model, "module") else model
        was_training = target_model.training if hasattr(target_model, "training") else False
        target_model.eval()
        device = self._infer_device(target_model)
        try:
            for batch_idx, batch in enumerate(labeled_loader):
                img = batch[0].to(device, non_blocking=True)
                gt = batch[1].to(device, non_blocking=True)
                ids = self._extract_img_ids(batch, batch_idx, img.size(0))
                features = target_model.forward_return_pc_hbm_features(img, ema=True)
                self._append_memory_batch(features, gt, ids)
            self.memory.finalize(device=torch.device("cpu"), dtype=torch.float32)
            self._info(self.memory.diagnostic_string())
        except Exception as exc:
            self.memory.clear()
            self.memory_build_failed = True
            self.memory_build_error = str(exc)
            self._warn(f"[PC-HBM] memory rebuild failed at epoch {epoch}: {exc}. Fallback will be used.")
        finally:
            if was_training:
                target_model.train()

    def forward_talnet(
        self,
        talnet,
        img: torch.Tensor,
        memory=None,
        use_memory: bool = True,
        return_all_logits: bool = True,
        epoch: int | None = None,
        forward_mode: str = "full",
        need_p1_pra: bool | None = None,
        need_final_mixture: bool | None = None,
        return_debug_aux: bool | None = None,
        store_last_aux: bool | None = None,
    ):
        """Run TALNet plus full PC-HBM and return ``(outputs, aux)``."""

        (
            forward_mode,
            need_p1_pra,
            need_final_mixture,
            return_debug_aux,
            store_last_aux,
        ) = self._resolve_forward_policy(
            forward_mode,
            need_p1_pra,
            need_final_mixture,
            return_debug_aux,
            store_last_aux,
        )
        x1, x2, x3, x4, features = talnet._build_decoder_features(img)
        state, p3, m3 = talnet.decoder.forward_to_p3(features)
        state2_pre, p2_pre, m2_pre = talnet.decoder.forward_p2_from_p3(state, p3)
        memory_obj = self._resolve_forward_memory(memory)
        fallback = self._fallback_reason(memory_obj, use_memory, m3, epoch)
        if fallback is not None:
            scaled_preds, p1, z_main = talnet.decoder.forward_p1_from_p2(state2_pre, p2_pre)
            outputs = self._outputs_from_state(state, m3, m2_pre, z_main)
            aux = self._fallback_aux(fallback, x1, x2, x3, x4, p3, p2_pre, p1, z_main, outputs, forward_mode=forward_mode)
            return outputs, self._finalize_return_aux(aux, forward_mode, return_debug_aux, store_last_aux)
        prob3 = torch.sigmoid(m3)
        b3_input = boundary_features_from_logits(m3)
        B3, boundary3 = self.boundary3(b3_input)
        batch_ids3 = boundary3["batch_ids"]
        flat_indices3 = boundary3["flat_indices"]
        route = self.router(x3, prob3, memory_obj, top_img_k=int(getattr(self.config, "cbm_top_img_k", 32)))
        parent_subbank = memory_obj.get_parent_subbank(route["top_img_ids"], device=img.device, dtype=img.dtype)
        parent_ret = self.parent_retriever(p3, batch_ids3, flat_indices3, parent_subbank)
        if parent_ret["top_parent_keys"].numel() == 0:
            scaled_preds, p1, z_main = talnet.decoder.forward_p1_from_p2(state2_pre, p2_pre)
            outputs = self._outputs_from_state(state, m3, m2_pre, z_main)
            aux = self._fallback_aux("parent_subbank_empty", x1, x2, x3, x4, p3, p2_pre, p1, z_main, outputs, forward_mode=forward_mode)
            return outputs, self._finalize_return_aux(aux, forward_mode, return_debug_aux, store_last_aux)
        if epoch is not None and int(epoch) < int(getattr(self.config, "child_start_epoch", 11)):
            scaled_preds, p1, z_main = talnet.decoder.forward_p1_from_p2(state2_pre, p2_pre)
            outputs = self._outputs_from_state(state, m3, m2_pre, z_main)
            aux = self._fallback_aux("stage_parent_only", x1, x2, x3, x4, p3, p2_pre, p1, z_main, outputs, forward_mode=forward_mode)
            aux["pc_hbm"].update(
                {
                    "B3": B3,
                    "boundary_indices3": boundary3,
                    "batch_ids3": batch_ids3,
                    "flat_indices3": flat_indices3,
                    "route_entropy": route["route_entropy"],
                    "route_entropy_norm": route["route_entropy_norm"],
                    "top_img_ids": route["top_img_ids"],
                    "top_img_scores": route["top_img_scores"],
                    "top_parent_keys": parent_ret["top_parent_keys"],
                    "top_parent_values": parent_ret["top_parent_values"],
                    "top_parent_geo": parent_ret["top_parent_geo"],
                    "top_parent_region_ids": parent_ret["top_parent_region_ids"],
                    "top_parent_reliability": parent_ret["top_parent_reliability"],
                    "top_parent_scores": parent_ret["top_parent_scores"],
                    "A_parent": parent_ret["A_parent"],
                    "P3_group": parent_ret["P3_group"],
                    "parent_entropy": parent_ret["parent_entropy"],
                }
            )
            return outputs, self._finalize_return_aux(aux, forward_mode, return_debug_aux, store_last_aux)
        child_query = self.child_query(p2_pre, batch_ids3, flat_indices3, p3.shape[-2:])
        child_bank = memory_obj.get_child_by_ptr(parent_ret["top_child_ptrs"], device=img.device, dtype=img.dtype)
        child_ver = self.child_verifier(child_query["q_child"], child_query["G2_query"], parent_ret, child_bank)
        h_tokens = self.hyp_builder(parent_ret, child_ver)
        route_context = route["route_context"].index_select(0, batch_ids3.long())
        q_state = self.query_state(parent_ret["q3"], child_query["q_child"], route_context, child_ver["C23_token"], parent_ret["parent_entropy"])
        q3_new, hca_attn = self.hca(q_state, h_tokens, child_ver["prior_bias"], route_context)
        token_aux = self.token_decoder(q3_new, hca_attn, parent_ret, child_ver)
        confidence = boundary3["token_scores"].unsqueeze(1).to(dtype=img.dtype)
        u_token = (1.0 - confidence).clamp(0.0, 1.0)
        gate_pc = self.gate_mlp(
            confidence,
            child_ver["C23_token"],
            u_token,
            parent_ret["parent_entropy"],
            child_ver["child_entropy"],
            child_ver["S_child"],
            child_ver["S_geo"],
        )
        pc_maps = pc_scatter(img.size(0), p3.size(2), p3.size(3), batch_ids3, flat_indices3, token_aux, gate_pc, child_ver["C23_token"])
        p3_corr, delta3 = self.p3_residual(p3, batch_ids3, flat_indices3, token_aux["Z3_token"], confidence, gate_pc)
        state2, p2, m2 = talnet.decoder.forward_p2_from_p3(state, p3_corr)
        p2_aux = self.p2_bra(p2, torch.sigmoid(m2), pc_maps)
        scaled_preds, p1, z_main = talnet.decoder.forward_p1_from_p2(state2, p2_aux["p2_refined"])
        z_nomix = z_main
        if need_p1_pra and need_final_mixture:
            p1_aux = self.p1_pra(p1, z_main, p2_aux)
            mix_aux = self.mixture(
                z_main,
                p1_aux,
                pc_maps,
                epoch=epoch,
                temperature=self._mixture_temperature(epoch),
                eps_floor=self._mixture_eps(epoch),
            )
            z_final = mix_aux["z_final"]
            p_final = mix_aux["p_final"]
            mixture_skipped = False
        else:
            p1_aux = {}
            mix_aux = {}
            z_final = z_main
            p_final = torch.sigmoid(z_main)
            mixture_skipped = True
        outputs = self._outputs_from_state(state, m3, m2, z_main)
        pc_aux = {
            **pc_maps,
            "B3": B3,
            "boundary_indices3": boundary3,
            "batch_ids3": batch_ids3,
            "flat_indices3": flat_indices3,
            "route_entropy": route["route_entropy"],
            "route_entropy_norm": route["route_entropy_norm"],
            "top_img_ids": route["top_img_ids"],
            "top_img_scores": route["top_img_scores"],
            "route_context": route["route_context"],
            "top_parent_keys": parent_ret["top_parent_keys"],
            "top_parent_values": parent_ret["top_parent_values"],
            "top_parent_geo": parent_ret["top_parent_geo"],
            "top_parent_region_ids": parent_ret["top_parent_region_ids"],
            "top_parent_reliability": parent_ret["top_parent_reliability"],
            "top_parent_scores": parent_ret["top_parent_scores"],
            "A_parent": parent_ret["A_parent"],
            "P3_group": parent_ret["P3_group"],
            "parent_entropy": parent_ret["parent_entropy"],
            "K_child_top": child_ver["K_child_top"],
            "G2_child_top": child_ver["G2_child_top"],
            "S_child": child_ver["S_child"],
            "S_geo": child_ver["S_geo"],
            "prior_bias": child_ver["prior_bias"],
            "S_hyp": child_ver["S_hyp"],
            "P_pc_group": child_ver["P_pc_group"],
            "C23_token": child_ver["C23_token"],
            "H_tokens": h_tokens,
            "q3_new": q3_new,
            "E_attn": token_aux["E_attn"],
            "G_attn": token_aux["G_attn"],
            "G_child_attn": token_aux["G_child_attn"],
            "M_pc_token": token_aux["M_pc_token"],
            "M_pc_evidence": token_aux["M_pc_evidence"],
            "M_pc_residual": token_aux["M_pc_residual"],
            "O_pc_token": token_aux["O_pc_token"],
            "gate_pc_token": gate_pc,
            "delta3_p3": delta3,
        }
        aux = {
            "m4": outputs[0],
            "m3": outputs[1],
            "m2": outputs[2],
            "p3": p3,
            "p3_corr": p3_corr,
            "p2_pre": p2_pre,
            "p2": p2,
            "p2_refined": p2_aux["p2_refined"],
            "p1": p1,
            "z_main": z_main,
            "z_nomix": z_nomix,
            "z_final": z_final,
            "p_final": p_final,
            "pc_hbm": pc_aux,
            "p2_bra": p2_aux,
            "p1_pra": p1_aux,
            "mixture": mix_aux,
            "features": {"x1": x1, "x2": x2, "x3": x3, "x4": x4},
            "p_main": torch.sigmoid(z_main).detach(),
            "pc_hbm_used": True,
            "forward_mode": forward_mode,
            "mixture_skipped": mixture_skipped,
        }
        return outputs, self._finalize_return_aux(aux, forward_mode, return_debug_aux, store_last_aux)

    def _resolve_forward_policy(
        self,
        forward_mode: str,
        need_p1_pra: bool | None,
        need_final_mixture: bool | None,
        return_debug_aux: bool | None,
        store_last_aux: bool | None,
    ) -> tuple[str, bool, bool, bool, bool]:
        forward_mode = str(forward_mode or "full")
        if forward_mode not in {"full", "teacher_pseudo", "student_core"}:
            raise ValueError(f"Unsupported PC-HBM forward_mode: {forward_mode}")
        if forward_mode == "teacher_pseudo":
            need_p1_pra = True
            need_final_mixture = True
        elif forward_mode == "student_core":
            need_p1_pra = False
            need_final_mixture = False
        else:
            need_p1_pra = True if need_p1_pra is None else bool(need_p1_pra)
            need_final_mixture = True if need_final_mixture is None else bool(need_final_mixture)

        if return_debug_aux is None:
            key = "pc_hbm_return_debug_aux_train" if self.training else "pc_hbm_return_debug_aux_eval"
            return_debug_aux = bool(getattr(self.config, key, False))
        if store_last_aux is None:
            key = "pc_hbm_store_last_aux_train" if self.training else "pc_hbm_store_last_aux_eval"
            store_last_aux = bool(getattr(self.config, key, False))
        return forward_mode, bool(need_p1_pra), bool(need_final_mixture), bool(return_debug_aux), bool(store_last_aux)

    def _finalize_return_aux(self, aux: Dict[str, Any], forward_mode: str, return_debug_aux: bool, store_last_aux: bool) -> Dict[str, Any]:
        if return_debug_aux and "diagnostics" not in aux:
            aux["diagnostics"] = collect_pc_hbm_diagnostics(aux)
        self._store_last_aux(aux, store_last_aux)
        if forward_mode == "teacher_pseudo" and bool(getattr(self.config, "pc_hbm_slim_teacher_aux", True)):
            return self._slim_teacher_pseudo_aux(aux)
        if forward_mode == "student_core" and bool(getattr(self.config, "pc_hbm_slim_student_aux", True)):
            return self._slim_student_unsup_aux(aux)
        return aux

    def _store_last_aux(self, aux: Dict[str, Any], store_last_aux: bool) -> None:
        if store_last_aux:
            self.last_aux = self._slim_last_aux(aux)
        else:
            self.last_aux = None

    def _slim_teacher_pseudo_aux(self, aux: Dict[str, Any]) -> Dict[str, Any]:
        pc = aux.get("pc_hbm", {}) or {}
        mix = aux.get("mixture", {}) or {}
        out = {
            "z_main": aux.get("z_main"),
            "z_nomix": aux.get("z_nomix", aux.get("z_main")),
            "z_final": aux.get("z_final", aux.get("z_main")),
            "p_final": aux.get("p_final"),
            "p_main": aux.get("p_main"),
            "pc_hbm": {
                "C23_map": pc.get("C23_map"),
                "route_entropy": pc.get("route_entropy"),
                "route_entropy_norm": pc.get("route_entropy_norm"),
            },
            "mixture": {
                "pi": mix.get("pi"),
                "B_pix": mix.get("B_pix"),
                "Mask_corr": mix.get("Mask_corr"),
            },
            "pc_hbm_used": aux.get("pc_hbm_used", False),
            "fallback_reason": aux.get("fallback_reason"),
            "forward_mode": aux.get("forward_mode"),
            "mixture_skipped": aux.get("mixture_skipped", False),
        }
        if "diagnostics" in aux:
            out["diagnostics"] = self._detach_debug_value(aux["diagnostics"])
        return out

    def _slim_student_unsup_aux(self, aux: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "z_main": aux.get("z_main"),
            "z_nomix": aux.get("z_nomix", aux.get("z_main")),
            "z_final": aux.get("z_final"),
            "pc_hbm_used": aux.get("pc_hbm_used", False),
            "fallback_reason": aux.get("fallback_reason"),
            "forward_mode": aux.get("forward_mode"),
            "mixture_skipped": aux.get("mixture_skipped", False),
        }
        if "diagnostics" in aux:
            out["diagnostics"] = self._detach_debug_value(aux["diagnostics"])
        return out

    def _slim_last_aux(self, aux: Dict[str, Any]) -> Dict[str, Any]:
        diagnostics = aux.get("diagnostics")
        if diagnostics is None:
            diagnostics = collect_pc_hbm_diagnostics(aux)
        return {
            "pc_hbm_used": aux.get("pc_hbm_used", False),
            "fallback_reason": aux.get("fallback_reason"),
            "forward_mode": aux.get("forward_mode"),
            "mixture_skipped": aux.get("mixture_skipped", False),
            "diagnostics": self._detach_debug_value(diagnostics),
        }

    def _detach_debug_value(self, value):
        if torch.is_tensor(value):
            tensor = value.detach().float().cpu()
            return tensor.reshape(()) if tensor.numel() == 1 else tensor.mean().reshape(())
        if isinstance(value, dict):
            return {key: self._detach_debug_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._detach_debug_value(item) for item in value)
        return value

    def compute_losses(self, outputs, aux, gt):
        from ..training.pc_losses import compute_pc_hbm_labeled_loss

        loss, log = compute_pc_hbm_labeled_loss(outputs, aux, gt, self.config)
        diagnostics = collect_pc_hbm_diagnostics(aux, gt)
        for key, value in diagnostics.items():
            log.setdefault(key, value.detach())
        self.loss_dict = {key: float(value.detach().item()) for key, value in log.items()}
        return loss

    def memory_state_dict(self):
        self.memory.compat_meta = self._build_memory_compat_meta()
        return self.memory.to_state_dict()

    def load_memory_state_dict(self, state, device=None, dtype=None) -> None:
        self._check_memory_state_compatible(state)
        self.memory.load_state_dict(state, device=device or torch.device("cpu"), dtype=dtype)

    def _append_memory_batch(self, features: Dict[str, torch.Tensor], gt: torch.Tensor, img_ids) -> None:
        x3 = features["x3"].to(device=self.device_hint)
        p3 = features["p3"].to(device=self.device_hint)
        p2 = features["p2"].to(device=self.device_hint)
        m3 = features.get("m3")
        prob3 = torch.sigmoid(m3.to(device=self.device_hint)) if isinstance(m3, torch.Tensor) else None
        route_tokens = self.router.encode_route_tokens(x3, prob3)
        self.memory.append_route(
            x3_global=route_tokens["x3_global"],
            x3_boundary=route_tokens["x3_boundary"],
            x3_uncertain=route_tokens["x3_uncertain"],
            x3_bg_near=route_tokens["x3_bg_near"],
            x3_environment=route_tokens["x3_environment"],
            route_embed=route_tokens["route_embed"],
            img_ids=img_ids,
        )
        q3_map = self.parent_retriever.encode_q_map(p3)
        regions3 = build_pc_regions(gt.to(device=self.device_hint), p3.shape[-2:])
        regions2 = build_pc_regions(gt.to(device=self.device_hint), p2.shape[-2:])
        for b, img_id in enumerate(img_ids):
            parent_keys = []
            parent_values = []
            parent_geo = []
            parent_meta = []
            child_keys = []
            child_geo = []
            child_meta = []
            child_ptr_chunks = []
            for region in ("fg_core", "fg_boundary", "bg_near", "bg_far"):
                mask = regions3[region][b, 0].bool()
                rel = regions3["geometry"][b, 5]
                flat3 = sample_region_indices(mask, rel, region).to(device=self.device_hint)
                if flat3.numel() == 0:
                    continue
                b_ids = torch.full_like(flat3, b)
                p_keys = gather_tokens(q3_map, b_ids, flat3)
                sdf = regions3["geometry"][b, 0].flatten().index_select(0, flat3)
                reliability = regions3["geometry"][b, 5].flatten().index_select(0, flat3)
                values = parent_values_from_region(region, sdf, reliability)
                geo = regions3["geometry"][b].flatten(1).transpose(0, 1).index_select(0, flat3)
                cq = self.child_query(p2[b : b + 1], torch.zeros_like(flat3), flat3, p3.shape[-2:])
                c_keys = cq["q_child"]
                flat2 = cq["flat_indices2_from_p3"]
                c_geo = regions2["geometry"][b].flatten(1).transpose(0, 1).index_select(0, flat2.clamp(0, p2.size(2) * p2.size(3) - 1))
                child_start = sum(item.size(0) for item in child_keys)
                child_ptr_chunks.append(torch.arange(child_start, child_start + c_keys.size(0), device=self.device_hint, dtype=torch.long))
                height, width = p3.shape[-2:]
                coords_y = torch.div(flat3, width, rounding_mode="floor")
                coords_x = flat3.remainder(width)
                for idx in range(flat3.numel()):
                    parent_meta.append(
                        {
                            "image_id": str(img_id),
                            "region": region,
                            "region_id": int(values[idx, :4].argmax().item()),
                            "flat_index": int(flat3[idx].item()),
                            "coord": (int(coords_y[idx].item()), int(coords_x[idx].item())),
                            "reliability": float(reliability[idx].item()),
                        }
                    )
                    child_meta.append({"image_id": str(img_id), "region": region, "parent_flat_index": int(flat3[idx].item())})
                parent_keys.append(p_keys)
                parent_values.append(values)
                parent_geo.append(geo)
                child_keys.append(c_keys)
                child_geo.append(c_geo)
            if parent_keys:
                child_ptr = self.memory.append_child(torch.cat(child_keys, dim=0), torch.cat(child_geo, dim=0), child_meta)
                local_ptr = torch.cat(child_ptr_chunks, dim=0)
                global_ptr = child_ptr.index_select(0, local_ptr.cpu().to(child_ptr.device))
                self.memory.append_parent(torch.cat(parent_keys, dim=0), torch.cat(parent_values, dim=0), torch.cat(parent_geo, dim=0), global_ptr, parent_meta)

    def _fallback_reason(self, memory, use_memory: bool, m3: torch.Tensor | None, epoch: int | None) -> str | None:
        if not pc_hbm_enabled(self.config):
            return "pc_hbm_disabled"
        if not use_memory:
            return "use_memory_false"
        if m3 is None:
            return "m3_none"
        if memory is None or not memory.is_ready():
            return "memory_not_ready"
        if epoch is not None and int(epoch) < int(getattr(self.config, "parent_start_epoch", 6)):
            return "stage_memory_disabled"
        return None

    def _fallback_aux(self, reason, x1, x2, x3, x4, p3, p2, p1, z_main, outputs, forward_mode="full"):
        z_final = z_main
        zeros3 = p3.new_zeros(p3.size(0), 1, p3.size(2), p3.size(3))
        aux = {
            "m4": outputs[0],
            "m3": outputs[1],
            "m2": outputs[2],
            "p3": p3,
            "p3_corr": p3,
            "p2_pre": p2,
            "p2": p2,
            "p2_refined": p2,
            "p1": p1,
            "z_main": z_main,
            "z_nomix": z_main,
            "z_final": z_final,
            "p_final": torch.sigmoid(z_final),
            "pc_hbm": {
                "fallback_reason": reason,
                "M_pc_map": zeros3,
                "O_pc_map": p3.new_zeros(p3.size(0), 2, p3.size(2), p3.size(3)),
                "gate_pc_map": zeros3,
                "C23_map": zeros3,
                "Z3_map": p3.new_zeros(p3.size(0), self.dim, p3.size(2), p3.size(3)),
                "E_attn_map": p3.new_zeros(p3.size(0), self.value_dim, p3.size(2), p3.size(3)),
                "G_attn_map": p3.new_zeros(p3.size(0), self.geometry_dim, p3.size(2), p3.size(3)),
                "valid3_map": zeros3,
            },
            "p2_bra": {},
            "p1_pra": {},
            "mixture": {},
            "features": {"x1": x1, "x2": x2, "x3": x3, "x4": x4},
            "fallback_reason": reason,
            "pc_hbm_used": False,
            "forward_mode": forward_mode,
            "mixture_skipped": True,
        }
        return aux

    def _outputs_from_state(self, state, m3, m2, z_main):
        m4 = state.get("m4")
        if m4 is None:
            m4 = F.interpolate(z_main, scale_factor=1 / 32, mode="bilinear", align_corners=False)
        if m3 is None:
            m3 = F.interpolate(z_main, size=(z_main.size(2) // 16, z_main.size(3) // 16), mode="bilinear", align_corners=False)
        if m2 is None:
            m2 = F.interpolate(z_main, size=(z_main.size(2) // 8, z_main.size(3) // 8), mode="bilinear", align_corners=False)
        return [m4, m3, m2, z_main]

    def _resolve_forward_memory(self, memory_t=None):
        if memory_t is None:
            return self.memory
        if isinstance(memory_t, PCHBMMemory):
            return memory_t
        if isinstance(memory_t, Mapping):
            return memory_t.get("pc_hbm_memory", memory_t.get("labeled_memory", memory_t.get("L_t", self.memory)))
        return self.memory

    def _mixture_temperature(self, epoch: int | None) -> float:
        if epoch is None:
            return float(getattr(self.config, "mixture_temperature_end", 0.8))
        start = float(getattr(self.config, "mixture_temperature_start", 1.5))
        end = float(getattr(self.config, "mixture_temperature_end", 0.8))
        decay = max(1, int(getattr(self.config, "mixture_eps_decay_epoch", 10)))
        t = min(1.0, max(0.0, float(epoch) / decay))
        return start + (end - start) * t

    def _mixture_eps(self, epoch: int | None) -> float:
        if epoch is None:
            return float(getattr(self.config, "mixture_eps_end", 0.0))
        start = float(getattr(self.config, "mixture_eps_start", 0.10))
        end = float(getattr(self.config, "mixture_eps_end", 0.0))
        decay = max(1, int(getattr(self.config, "mixture_eps_decay_epoch", 10)))
        t = min(1.0, max(0.0, float(epoch) / decay))
        return start + (end - start) * t

    def _build_memory_compat_meta(self):
        return {
            "backbone": str(getattr(self.config, "backbone", "")),
            "lateral_channels_in_collection": [int(v) for v in getattr(self.config, "lateral_channels_in_collection", [])],
            "pc_dim": self.dim,
            "value_dim": self.value_dim,
            "value_schema_version": "fg4_bg5_v2",
            "geometry_dim": self.geometry_dim,
            "feature_version": str(getattr(self.config, "cbm_memory_feature_version", "swin_l_pc_hbm_v1")),
        }

    def _check_memory_state_compatible(self, state) -> None:
        if not state:
            return
        meta = state.get("compat_meta", {}) if isinstance(state, Mapping) else {}
        if not meta:
            return
        expected = self._build_memory_compat_meta()
        for key, value in expected.items():
            if meta.get(key) != value:
                raise RuntimeError(f"PC-HBM memory incompatible: {key}: memory={meta.get(key)} expected={value}. Rebuild labelled memory.")

    def _infer_device(self, model) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return self.device_hint

    def _extract_img_ids(self, batch, batch_idx: int, batch_size: int):
        if len(batch) > 2:
            raw = batch[2]
            if isinstance(raw, torch.Tensor):
                return [str(item) for item in raw.detach().cpu().reshape(-1).tolist()]
            if isinstance(raw, (list, tuple)):
                return [str(item) for item in raw]
            return [str(raw)] * batch_size
        return [f"pc_hbm_epoch_mem_b{batch_idx}_i{idx}" for idx in range(batch_size)]

    def _info(self, message: str) -> None:
        if self.logger is None:
            print(message)
            return
        fn = getattr(self.logger, "info", None) or getattr(self.logger, "key_info", None)
        if callable(fn):
            fn(message)

    def _warn(self, message: str) -> None:
        if self.logger is None:
            print(message)
            return
        fn = getattr(self.logger, "warn_info", None) or getattr(self.logger, "warning", None) or getattr(self.logger, "info", None)
        if callable(fn):
            fn(message)


from ..retrieval.parent_retriever import ParentRetriever as ParentRetrieverProxy


def build_pc_hbm(config, device=None, logger=None) -> PCHBMEngine:
    """Factory used by ``models.build_model``."""

    apply_pc_hbm_defaults(config)
    return PCHBMEngine(config=config, device=device, logger=logger)
