"""Configuration defaults and schedule helpers for PC-HBM."""

from __future__ import annotations

from typing import Any


PC_HBM_DEFAULTS = {
    "use_pc_hbm": False,
    "pc_hbm_enable": False,
    "memory_source": "labeled_only",
    "memory_rebuild_interval": 1,
    "use_unlabeled_memory_update": False,
    "cbm_memory_dim": 512,
    "cbm_value_dim": 8,
    "geometry_dim": 6,
    "cbm_top_img_k": 32,
    "parent_topk": 64,
    "cbm_tau_parent": 0.07,
    "cbm_tau_child": 0.10,
    "cbm_tau_hca": 0.10,
    "cbm_tau_bra": 0.10,
    "cbm_tau_pra": 0.10,
    "use_camouflage_context_router": True,
    "use_multi_hypothesis_parent": True,
    "parent_group_mode": "four_region",
    "use_child_verifier": True,
    "child_window_size": 5,
    "use_geometry_value": True,
    "use_structured_prior_bias_net": True,
    "use_hyp_score_net": True,
    "use_structured_gate_mlp": True,
    "use_boundary_query_head": True,
    "use_pc_hca": True,
    "use_p2_bra": True,
    "use_p1_pra": True,
    "attn_num_heads": 8,
    "attn_head_dim": 64,
    "p2_bra_local_window": 3,
    "p1_pra_local_window": 3,
    "p2_boundary_top_ratio": 0.25,
    "p1_boundary_top_ratio": 0.20,
    "use_adaptive_mixture": True,
    "mixture_branches": ["keep", "residual", "deformation", "suppress"],
    "mixture_init_bias": [1.0, -0.5, -0.5, -0.5],
    "mixture_eps_start": 0.10,
    "mixture_eps_end": 0.00,
    "mixture_eps_decay_epoch": 10,
    "mixture_temperature_start": 1.5,
    "mixture_temperature_end": 0.8,
    "use_branch_oracle_supervision": True,
    "use_branch_quality_head": True,
    "use_conditional_usage_loss": True,
    "use_branch_dropout": True,
    "use_suppress_head": True,
    "use_mask_corr_head": False,
    "mask_corr_epsilon": 0.10,
    "r_max": 2.0,
    "max_offset": 3.0,
    "lambda_final": 1.0,
    "lambda_main": 1.0,
    "lambda_nomix": 0.5,
    "lambda_mem": 0.2,
    "lambda_boundary_aux": 0.2,
    "lambda_mix_oracle": 0.2,
    "lambda_branch": 0.2,
    "lambda_quality": 0.05,
    "lambda_usage": 0.02,
    "lambda_reg": 0.05,
    "lambda_u": 1.0,
    "warmup_epoch": 5,
    "parent_start_epoch": 6,
    "child_start_epoch": 11,
    "attention_refine_start_epoch": 11,
    "unlabeled_start_epoch": 16,
    "pc_hbm_detach_refs": True,
    "pc_hbm_unsup_final_consistency_weight": 0.1,
    "pc_hbm_tau_oracle": 0.5,
}


def apply_pc_hbm_defaults(config: Any) -> Any:
    """Apply missing PC-HBM defaults in-place and return config."""

    for key, value in PC_HBM_DEFAULTS.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    setattr(config, "cbm_memory_dim", int(getattr(config, "cbm_memory_dim", 512)))
    setattr(config, "cbm_value_dim", int(getattr(config, "cbm_value_dim", 8)))
    setattr(config, "geometry_dim", int(getattr(config, "geometry_dim", 6)))
    return config


def pc_hbm_enabled(config: Any) -> bool:
    """Return True when the PC-HBM gate is enabled."""

    return bool(getattr(config, "use_pc_hbm", False) or getattr(config, "pc_hbm_enable", False))


def pc_hbm_stage(config: Any, epoch: int | None) -> int:
    """Map epoch to PC-HBM stage: 1 warmup, 2 parent, 3 full labeled, 4 semi."""

    if epoch is None:
        return 4
    one_based = int(epoch)
    if one_based <= int(getattr(config, "warmup_epoch", 5)):
        return 1
    if one_based < int(getattr(config, "child_start_epoch", 11)):
        return 2
    if one_based < int(getattr(config, "unlabeled_start_epoch", 16)):
        return 3
    return 4


def pc_hbm_should_rebuild_memory(config: Any, epoch: int | None) -> bool:
    """Labelled-only memory rebuild schedule."""

    if not pc_hbm_enabled(config):
        return False
    if str(getattr(config, "memory_source", "labeled_only")) != "labeled_only":
        return False
    interval = max(1, int(getattr(config, "memory_rebuild_interval", 1)))
    return epoch is None or int(epoch) >= int(getattr(config, "parent_start_epoch", 6)) and int(epoch) % interval == 0


def pc_hbm_unlabeled_enabled(config: Any, epoch: int | None) -> bool:
    """Return True when Stage 4 semi-supervised PC-HBM branch is enabled."""

    if not pc_hbm_enabled(config):
        return False
    if epoch is None:
        return False
    return int(epoch) >= int(getattr(config, "unlabeled_start_epoch", 16))
