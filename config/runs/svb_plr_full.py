import os

# Reuse the baseline CBM-PFI run config first, then override only the
# experiment output path and SVB-PLR switches/parameters below.
with open(os.path.join("config", "runs", "run.py"), "r", encoding="utf-8") as _base_cfg:
    exec(_base_cfg.read())
del _base_cfg


# experiment settings
ckpt_dir = "/home/zhangqing/YJD/SCOD/CBM-PFI-SAM/CBM-PFI/works/test"
pred_save_root = ckpt_dir.rstrip("/\\") + "/training_preds"


# SVB-PLR main switches
use_svb_plr = True
use_sam_refine_unlabeled = True
svb_ablation_mode = "full"  # off | teacher_sam_full | boundary_only | cbm_points | reliability | prompt_expert | conformal | full
sam_start_epoch = 16
sam_refine_interval = 1

# SAM backend selector
sam_pseudo_backend = "sam1"  # sam1 | sam2
# sam_pseudo_checkpoint = "/home/zhangqing/YJD/SCOD/Prototype_Feature_Interaction/SAM/sam_hq_vit_h.pth"
sam_pseudo_checkpoint = "/home/zhangqing/YJD/SCOD/Prototype_Feature_Interaction/SAM/sam_vit_h_4b8939.pth"


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
sam_cache_dir = "./cache/sam_refined_pseudo/finetune_27_svb_plr_aggressive"
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
vis_sam_refine_interval = 200
vis_sam_refine_max_samples = 2
sam_refine_vis_dir = ckpt_dir.rstrip("/\\") + "/svb_plr_visualization"


# Weighted unsupervised loss
use_svb_weighted_unsup_loss = True
sam_boundary_loss_boost = 0.5


# wandb metadata
ModelName = "PrototypeNet_SVB_PLR_Full"
others = {

    "use_svb_plr": use_svb_plr,
    "svb_ablation_mode": svb_ablation_mode,
    "sam_backend": sam_pseudo_backend,
    "sam_start_epoch": sam_start_epoch,
    "sam_use_conformal": sam_use_conformal,
    "use_prompt_expert": use_prompt_expert,
    "use_svb_weighted_unsup_loss": use_svb_weighted_unsup_loss,
}


# python -m scripts.train --config config/runs/svb_plr_full.py
