import os

# training settings
ckpt_dir = "/home/zhangqing/YJD/SCOD/PC-HBM/works/pc_hbm_full"

tot_epochs = 30

sup_only_train_epoch = 15
distributed_train = False
device_map = {
    'model': '*'
}  # Only available for non distributed training
rand_seed = 7
lr = 1e-4

# Disable BCE in the last epochs and emphasize structure-aware losses.
IoU_finetune_last_epochs = [0, -3][1]

# model settings
compile_model = False
precisionHigh = True
img_size = 640

backbone = 'swin_v1_l'
lateral_channels_in_collection = [3072, 1536, 768, 384]
cxt_num = 3
cxt = [384, 768, 1536]

# data settings
load_all = True
batch_size = 6
batch_size_valid = 6
data_split = [0.05]  # [0.01, 0.05, 0.1]

# Generate this file first with scripts/generate_random_indices.py.
# data_split_indices_file_format = "data/cache/labeled_indices/split{}_labeled_indices_stratified.pt"
data_split_indices_file_format = "data/cache/labeled_indices/split{}_labeled_indices_random.pt"
# data_split_indices_file_format = "data/cache/labeled_indices/split{}_labeled_indices.pt"

task = "COD"
training_set = "TR-COD10K+TR-CAMO"
# testing_sets = "TE-COD10K"
testing_sets = "TE-COD10K+TE-CAMO"

# evaluate settings
pred_save_root = os.path.join(ckpt_dir, 'training_preds')

# eval
eval_epoch = 23
eval_step = 1
# save model_checkpoint
save_step = 1
save_last = 7

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
unlabeled_start_epoch = 16

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
lambda_u = 1.0

# wandb
ModelName = 'PrototypeNet'
others = {
    'sup_epoch': sup_only_train_epoch,
    'total_epoch': tot_epochs,
}

# python -m scripts.train --config config/runs/run.py
