import os

# Reuse the epoch-27 fine-tune config, then override the settings for a
# weights-only restart + aggressive SVB-PLR run.
with open(os.path.join("config", "runs", "finetune_27.py"), "r", encoding="utf-8") as _base_cfg:
    exec(_base_cfg.read())
del _base_cfg


# experiment settings
ckpt_dir = "/home/zhangqing/YJD/SCOD/CBM-PFI-SAM/CBM-PFI/works/test"
pred_save_root = os.path.join(ckpt_dir, "training_preds")

load_all = True


# training settings
# weights-only checkpoint restarts from epoch_st=0; this keeps the existing
# save/eval boundaries while letting the optimizer start fresh.
tot_epochs = 30
eval_epoch = 23
eval_step = 1
save_step = 1
save_last = 7

# Re-state the inherited supervised warmup threshold so static analyzers see it.
sup_only_train_epoch = 5

optimizer = "AdamW"
lr = 5e-5

# Keep the fine-tune loss schedule from finetune_27, but shift the tail to
# structure-heavy optimization sooner.
IoU_finetune_last_epochs = -8

# learning-rate schedule
# Cosine is the aggressive default; multistep remains as a fallback if needed.
scheduler_type = "cosine"
scheduler_warmup_epochs = 1
scheduler_warmup_start_factor = 0.2
scheduler_t_max = tot_epochs + 1
scheduler_eta_min = 1e-6
lr_decay_epochs = [8, 16, 24]
lr_decay_rate = 0.5


# SVB-PLR main switches
use_svb_plr = True
use_sam_refine_unlabeled = True
svb_ablation_mode = "full"
sam_start_epoch = 8
sam_refine_interval = 1


# SAM backend selector
sam_pseudo_backend = "sam1"  # sam1 | sam2
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

# SAM2 backend parameters. These are kept here so the backend can be switched
# by changing only sam_pseudo_backend = "sam2".
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
use_sam_cache = False
sam_cache_dir = "./cache/sam_refined_pseudo/finetune_27_svb_plr_aggressive"
cache_refined_masks = True
cache_prompt_debug = False


# Visualization
vis_sam_refinement = False
vis_sam_refine_interval = 200
vis_sam_refine_max_samples = 10
sam_refine_vis_dir = ckpt_dir.rstrip("/\\") + "/svb_plr_visualization"


# Weighted unsupervised loss
use_svb_weighted_unsup_loss = True
sam_boundary_loss_boost = 0.5


# wandb metadata
ModelName = "PrototypeNet_Finetune27_SVB_PLR_Aggressive"
others = {
    "total_epoch": tot_epochs,
    "sup_epoch": sup_only_train_epoch,
    "optimizer": optimizer,
    "lr": lr,
    "scheduler_type": scheduler_type,
    "scheduler_t_max": scheduler_t_max,
    "scheduler_warmup_epochs": scheduler_warmup_epochs,
    "scheduler_warmup_start_factor": scheduler_warmup_start_factor,
    "scheduler_eta_min": scheduler_eta_min,
    "use_svb_plr": use_svb_plr,
    "svb_ablation_mode": svb_ablation_mode,
    "sam_backend": sam_pseudo_backend,
    "sam_start_epoch": sam_start_epoch,
    "sam_use_conformal": sam_use_conformal,
    "use_prompt_expert": use_prompt_expert,
    "use_svb_weighted_unsup_loss": use_svb_weighted_unsup_loss,
    "resume_mode": "weights_only",
}


# python -m scripts.train --config config/runs/finetune_27_svb_plr.py --resume /home/zhangqing/YJD/SCOD/CBM-PFI/works/CBM_prot_labled_split_topk_64/split0.05_model_27_weights_only.pth
