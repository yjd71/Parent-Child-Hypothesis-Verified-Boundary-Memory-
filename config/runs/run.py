import os

# training settings
ckpt_dir = "/home/zhangqing/YJD/Prototype_final/CBM-PFI_withoutSAM_onePrompt/works/pc_hbm_full_384"

tot_epochs = 30

sup_only_train_epoch = 18
distributed_train = False
device_map = {
    'model': '*'
}  # Only available for non distributed training
rand_seed = 7
lr = 1e-4
scheduler_type = "multistep"
lr_decay_epochs = [23, 27]
lr_decay_rate = 0.2

IoU_finetune_last_epochs = [0, -3][1]

# model settings
compile_model = False
precisionHigh = True
img_size = 384

backbone = 'swin_v1_l'
lateral_channels_in_collection = [3072, 1536, 768, 384]
cxt_num = 3
cxt = [384, 768, 1536]

# data settings
load_all = True
batch_size = 16
batch_size_valid = 16
data_split = [0.05]  # [0.01, 0.05, 0.1]

# Generate this file first with scripts/generate_random_indices.py.
data_split_indices_file_format = "data/cache/labeled_indices/split{}_labeled_indices_stratified.pt"
# data_split_indices_file_format = "data/cache/labeled_indices/split{}_labeled_indices_random.pt"
# data_split_indices_file_format = "data/cache/labeled_indices/split{}_labeled_indices.pt"

task = "COD"
training_set = "TR-COD10K+TR-CAMO"
# testing_sets = "TE-COD10K+TE-CAMO"
# testing_sets = "TE-CAMO+CHAMELEON+TE-COD10K+NC4K"
testing_sets = "TE-CAMO+CHAMELEON"

# evaluate settings
pred_save_root = os.path.join(ckpt_dir, 'training_preds')

# eval
eval_epoch = 1
eval_step = 1
# save model_checkpoint
save_step = 1
save_last = 13

# PC-HBM core
use_pc_hbm = True
pc_hbm_enable = True
memory_source = "labeled_only"
memory_rebuild_interval = 1
use_unlabeled_memory_update = False
cbm_memory_dim = 512
cbm_value_dim = 8
cbm_top_img_k = 32
parent_topk = 64
p2_boundary_top_ratio = 0.25
p1_boundary_top_ratio = 0.20
pc_hbm_checkpoint_memory = True

# PC-HBM stage schedule
warmup_epoch = 5
parent_start_epoch = 6
child_start_epoch = 11
attention_refine_start_epoch = 11
unlabeled_start_epoch = 18

# PC-HBM loss weights
lambda_final = 1.0
lambda_main = 1.0
lambda_nomix = 0.5
lambda_mem = 0.2
lambda_boundary_aux = 0.2
lambda_mix_oracle = 0.2
lambda_branch = 0.2
lambda_quality = 0.05
lambda_usage = 0.02
lambda_reg = 0.05
lambda_u = 0.1

# PC-HBM memory/speed optimization without precision changes
pc_hbm_unsup_student_core_only = True
pc_hbm_unsup_student_need_p1_pra = False
pc_hbm_unsup_student_need_final_mixture = False
pc_hbm_unsup_final_consistency_weight = 0.0

pc_hbm_store_last_aux_train = False
pc_hbm_store_last_aux_eval = False
pc_hbm_return_debug_aux_train = False
pc_hbm_return_debug_aux_eval = False
pc_hbm_slim_teacher_aux = True
pc_hbm_slim_student_aux = True

pc_hbm_memory_build_img_index = True
pc_hbm_memory_gpu_cache = True
pc_hbm_memory_cache_device = "cuda"
pc_hbm_memory_cache_dtype = "float32"
pc_hbm_memory_cache_min_free_gb = 1.5
pc_hbm_memory_cache_route = True
pc_hbm_memory_cache_parent = True
pc_hbm_memory_cache_child = True

# wandb
ModelName = 'PrototypeNet'
others = {
    'sup_epoch': sup_only_train_epoch,
    'total_epoch': tot_epochs,
}

# python -m scripts.train --config config/runs/run.py
