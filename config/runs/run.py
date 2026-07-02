import os

# training settings

ckpt_dir = "/home/zhangqing/YJD/SCOD/CBM-PFI/works/CBM_topk_64_2"


tot_epochs = 30

sup_only_train_epoch = 15
distributed_train = False
sam_refine_mode = "off"

device_map = {
    'model': '*'
}  # Only available for non distributed training
rand_seed = 7
lr = 1e-4

# 后期关闭 BCE，转向结构型损失（SSIM 为主，IoU 逐步减弱）。
IoU_finetune_last_epochs = [0, -3][1]

# model settings
compile_model = False
precisionHigh = True
img_size = 640

backbone = [
    'vgg16', 'vgg16bn', 'resnet50',         # 0, 1, 2
    'pvt_v2_b2', 'pvt_v2_b5',               # 3-bs10, 4-bs5
    'swin_v1_b', 'swin_v1_l',               # 5-bs9, 6-bs6
    'swin_v1_t', 'swin_v1_s',               # 7, 8
    'pvt_v2_b0', 'pvt_v2_b1',               # 9, 10
][5]
lateral_channels_in_collection = {
    'vgg16': [512, 256, 128, 64], 'vgg16bn': [512, 256, 128, 64], 'resnet50': [1024, 512, 256, 64],
    'pvt_v2_b2': [512, 320, 128, 64], 'pvt_v2_b5': [512, 320, 128, 64],
    'swin_v1_b': [1024, 512, 256, 128], 'swin_v1_l': [1536, 768, 384, 192],
    'swin_v1_t': [768, 384, 192, 96], 'swin_v1_s': [768, 384, 192, 96],
    'pvt_v2_b0': [256, 160, 64, 32], 'pvt_v2_b1': [512, 320, 128, 64],
}[backbone]
lateral_channels_in_collection = [channel * 2 for channel in lateral_channels_in_collection]
cxt_num = [0, 3][1]
cxt = lateral_channels_in_collection[1:][::-1][-cxt_num:] if cxt_num else []

# data settings
load_all = True
batch_size = 6
batch_size_valid = 6
data_split = [0.05]  # [0.01, 0.05, 0.1]

# ⚠️ 重要：使用随机生成的索引文件
# 需要先运行 scripts/generate_random_indices.py 生成随机索引

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


# CBM-PFI
cbm_pfi_enable = True

# CBM stage schedule: epoch 从 0 开始时，stage_epoch = epoch + 1
cbm_stage_epoch_offset = 1
cbm_stage1_end = 5          # stage 1: baseline warmup, 不用 memory
cbm_stage2_end = 15         # stage 2: labeled CBM
cbm_unlabeled_start_epoch = 16  # stage 3: labeled + unlabeled CBM

# CBM memory/retrieval
cbm_memory_dim = 512
cbm_value_dim = 8
cbm_top_img_k = 32
cbm_topk_token = 128

# CBM correction strength
cbm_lambda_feat = 0.1
cbm_lambda_logit = 0.5

# CBM losses
cbm_lambda_mem = 0.2
cbm_lambda_bd = 0.2
cbm_lambda_ctx = 0.05
cbm_lambda_aff = 0.05
cbm_lambda_gate_sparse = 0.01
cbm_lambda_gate_boundary = 0.05

# CBM visualization
cbm_vis_enable = False
cbm_vis_interval = 20
cbm_vis_max_images = 5
# cbm_vis_dir = "/home/zhangqing/YJD/SCOD/CBM-PFI/works/CBM_topk_64_logit_0.1/cbm_vis_debug"
cbm_vis_labeled_only = True


# CBM checkpoint / eval
cbm_checkpoint_memory = True

# wandb
ModelName = 'PrototypeNet'
others = {
    'sup_epoch': sup_only_train_epoch,
    'total_epoch': tot_epochs,
}

# python -m scripts.train --config config/runs/run.py
