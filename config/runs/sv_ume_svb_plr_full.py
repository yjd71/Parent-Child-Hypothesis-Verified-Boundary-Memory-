import os

# Reuse the baseline CBM-PFI run config first, then override only the
# experiment output path and SVB-PLR / SV-UME switches below.
with open(os.path.join("config", "runs", "run.py"), "r", encoding="utf-8") as _base_cfg:
    exec(_base_cfg.read())
del _base_cfg


# experiment settings
ckpt_dir = "/home/zhangqing/YJD/SCOD/CBM-PFI-SAM/CBM-PFI/works/sv_ume_svb_plr_full"
pred_save_root = ckpt_dir.rstrip("/\\") + "/training_preds"

load_all = False

# SVB-PLR main switches
use_svb_plr = True
use_sam_refine_unlabeled = True
svb_ablation_mode = "full"  # off | teacher_sam_full | boundary_only | cbm_points | reliability | prompt_expert | conformal | full
sv_training_log_enable = False
sv_training_log_interval = 600
svb_plr_start_epoch = 16
sam_start_epoch = svb_plr_start_epoch
sam_refine_interval = 1


# SV-UME: Lagged Quality-Adaptive SAM-refined Unlabeled Memory Expansion.
# Disabled by default so existing CBM-PFI and SVB-PLR behavior is unchanged.

# Main
use_sv_ume = False
sv_ume_require_svb_plr = True

# Stage
sv_ume_start_epoch = 21
use_lagged_unlabeled_memory = True
build_unlabeled_memory_after_epoch = True
use_unlabeled_memory_during_current_epoch = False

# Labeled / unlabeled memory policy
rebuild_labeled_memory_each_epoch = True
do_not_update_labeled_memory_with_unlabeled = True
unlabeled_memory_source = "sam_refined_pseudo_label"
unlabeled_memory_feature_source = "teacher_p3"
use_sam_embedding_as_memory_key = False

# Capacity
unlabeled_to_labeled_ratio = 1.0
region_capacity_ratio = {
    "fg_core": 1.0,
    "fg_boundary": 1.0,
    "bg_near": 1.0,
    "bg_far": 1.0,
}

# Sampling, same as labeled memory
sample_per_image_unlabeled = {
    "fg_core": 128,
    "fg_boundary": 384,
    "bg_near": 384,
    "bg_far": 128,
}

# Image / region / token thresholds
tau_image = 0.80
tau_region = {
    "fg_core": 0.85,
    "fg_boundary": 0.92,
    "bg_near": 0.94,
    "bg_far": 0.85,
}
tau_token = {
    "fg_core": 0.85,
    "fg_boundary": 0.92,
    "bg_near": 0.94,
    "bg_far": 0.85,
}

# Diversity
use_diversity_selection = True
lambda_diversity = 0.2
spatial_nms_distance = 2
feature_dup_sim_threshold = 0.95

# Global type metadata
use_global_type_metadata = True
tau_match = 0.70
tau_low = 0.55
use_fixed_matched_novel_ratio = False
use_matched_expanded_novel_as_metadata_only = True

# Novel pending activation
use_novel_pending = True
novel_cluster_min_size = 3
novel_cluster_min_sim = 0.75
novel_min_reliability = 0.90
novel_min_temporal_stability = 0.85

# Retrieval fusion
retrieve_labeled_and_unlabeled_separately = True
use_aux_evidence_fusion = True
use_aux_feature_fusion = True
aux_fusion_mode = "quality_adaptive_symmetric"
gamma_max_final = 1.0
use_aux_source_penalty = False
allow_aux_dominate = True

# Quality score weights
fusion_score_sim_weight = 1.0
fusion_score_cons_weight = 1.0
fusion_score_rel_weight = 1.0
fusion_score_unc_weight = 0.5

# Memory update
use_unlabeled_memory_snapshot_build = True
use_unlabeled_memory_ema_refresh = False
unlabeled_memory_momentum = 0.99

# Losses
use_ume_evidence_loss = False
use_source_consistency_loss = False
lambda_ume_evi = 0.05
lambda_source_cons = 0.02
source_consistency_tau = 0.70

# Diagnostics
sv_ume_save_memory_state = True
sv_ume_checkpoint_candidate_pool = False


# SAM backend selector
sam_pseudo_backend = "sam1"  # sam1 | sam2
# sam_pseudo_checkpoint = "/home/zhangqing/YJD/SCOD/Prototype_Feature_Interaction/SAM/sam_hq_vit_h.pth"
sam_pseudo_checkpoint = "/home/zhangqing/YJD/SCOD/Prototype_Feature_Interaction/SAM/sam_vit_h_4b8939.pth"
sam_pseudo_model_type = "vit_h"
sam_pseudo_threshold = 0.5
sam_pseudo_iters = 1
sam_pseudo_use_point = True
sam_pseudo_use_box = True
sam_pseudo_use_mask = True
sam_pseudo_add_neg = True
sam_pseudo_margin = 0.0
sam_pseudo_gamma = 4.0
sam_pseudo_strength = 30

sam2_checkpoint = "SAM/sam2.1_hiera_large.pt"
sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_multimask_output = True
sam2_use_bfloat16 = True


# Prompt generation
sam_use_box_prompt = True
sam_use_point_prompt = True
sam_use_mask_prompt = True
sam_use_boundary_points = True
sam_num_pos_points = 8
sam_num_neg_points = 8
sam_num_boundary_points = 12
sam_box_expand_ratio = 0.05
sam_prompt_min_area = 32


# Boundary/refinement band
sam_refine_boundary_only = True
sam_refine_theta = 0.25
sam_unc_weight = 0.5
sam_grad_weight = 0.5
sam_cbm_boundary_weight = 1.0
sam_cons_weight = 0.5
sam_gate_weight = 0.5


# SAM-CBM reliability filter
sam_use_teacher_agreement = True
sam_use_cbm_agreement = True
sam_use_stability = True
sam_use_conformal = True
sam_min_reliability = 0.3
sam_teacher_agree_weight = 0.25
sam_cbm_agree_weight = 0.20
sam_stability_weight = 0.45
sam_conformal_weight = 0.10


# Soft pseudo-label fusion
sam_beta_max = 1
sam_lambda_start = 1.0
sam_lambda_end = 0.3
sam_lambda_decay = False


# Prompt expert selector
use_prompt_expert = False
sam_prompt_experts = ["box", "box_point", "mask", "boundary"]
sam_prompt_select_tau = 0.1



# Cache
# The legacy switch stays off.  Output pseudo labels depend on the changing
# teacher, so caching one full-resolution payload per image and epoch has very
# low reuse and unbounded disk growth.
use_sam_cache = False
use_svb_output_cache = False
sam_cache_dir = "./cache/sam_refined_pseudo/sv_ume_svb_plr_full"
cache_refined_masks = False
cache_prompt_debug = False

# Frozen SAM image-encoder embeddings are teacher-independent.  Keep a small
# CPU LRU and a bounded persistent disk layer for exact augmented-image views.
use_sam_embedding_cache = True
sam_image_embedding_cache_size = 64
sam_embedding_cache_disk = True
sam_embedding_cache_dir = "./cache/sam_image_embeddings/sam1_vit_h"
sam_embedding_cache_max_gb = 32
sam_embedding_cache_store_dtype = "float16"
sam_embedding_cache_prune_interval = 256
sam_embedding_cache_version = "sam1_vit_h_4b8939_v1"


# Visualization
vis_sam_refinement = True
vis_sam_refine_interval = 20
vis_sam_refine_max_samples = 100
sam_refine_vis_dir = ckpt_dir.rstrip("/\\") + "/sv_ume_svb_plr_visualization"


# Weighted unsupervised loss
use_svb_weighted_unsup_loss = True
sam_boundary_loss_boost = 0.5


# wandb metadata
ModelName = "PrototypeNet_SV_UME_SVB_PLR_Full"
others = {
    "use_svb_plr": use_svb_plr,
    "use_sam_refine_unlabeled": use_sam_refine_unlabeled,
    "use_sv_ume": use_sv_ume,
    "svb_ablation_mode": svb_ablation_mode,
    "svb_plr_start_epoch": svb_plr_start_epoch,
    "sv_ume_require_svb_plr": sv_ume_require_svb_plr,
    "sv_ume_start_epoch": sv_ume_start_epoch,
    "sam_backend": sam_pseudo_backend,
    "sam_start_epoch": sam_start_epoch,
    "sam_use_conformal": sam_use_conformal,
    "use_prompt_expert": use_prompt_expert,
    "use_svb_weighted_unsup_loss": use_svb_weighted_unsup_loss,
}


# python -m scripts.train --config config/runs/sv_ume_svb_plr_full.py
