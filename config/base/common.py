import os
import math

rand_seed = 7
save_last = 50
save_step = 10
distillation_epochs = 50
# common training settings
precisionHigh = True
compile = False
verbose_eval = True

# Module diagnostics logging only. Baseline training progress keeps its
# fixed 20-batch cadence.
log_enable = True
log_interval = 200

lambdas_pix_last = {
    # not 0 means opening this loss
    # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
    'bce': 30 * 1,          # high performance
    'iou': 0.5 * 1,         # 0 / 255
    'iou_patch': 0.5 * 0,   # 0 / 255, win_size = (64, 64)
    'mse': 150 * 0,         # can smooth the saliency map
    'triplet': 3 * 0,
    'reg': 100 * 0,
    'ssim': 10 * 1,          # help contours,
    'cnt': 5 * 0,          # help contours
}

IoU_finetune_last_epochs = [0, -20][1]

# dataset settings
load_all = True
img_size = 1024
batch_size = 2
batch_size_valid = 1

sys_home_dir = "/home/zhangqing/YJD/SCOD/LFGM/SCOUT-main_NO_TFM"

data_root_dir = os.path.join(sys_home_dir, 'dataset')
task = "COD"
training_set = "TR-COD10K+TR-CAMO"
# preproc_methods = ['flip', 'enhance', 'rotate', 'pepper', 'crop'][:4]
preproc_methods = ['flip', 'crop', 'rotate', 'enhance', 'gaussian', 'pepper']

num_workers = 5
optimizer = ['Adam', 'AdamW'][0]
lr = 1e-5 # * math.sqrt(batch_size / 5)  # adapt the lr linearly
lr_decay_epochs = [1e4]    # Set to negative N to decay the lr in the last N-th epoch.
lr_decay_rate = 0.5
only_S_MAE = False
SDPA_enabled = False  

# Checkpoint migration defaults for backbone/channel changes.
checkpoint_load_strategy = "shape_matched"
resume_training_state = False

# backbone weights settings
weights_root_dir = os.path.join(sys_home_dir, 'weights')
weights = {
    'pvt_v2_b2': os.path.join(weights_root_dir, 'pvt_v2_b2.pth'),
    'pvt_v2_b5': os.path.join(weights_root_dir, ['pvt_v2_b5.pth', 'pvt_v2_b5_22k.pth'][0]),
    'swin_v1_b': os.path.join(weights_root_dir, ['swin_base_patch4_window12_384_22kto1k.pth', 'swin_base_patch4_window12_384_22k.pth'][0]),
    'swin_v1_l': os.path.join(weights_root_dir, ['swin_large_patch4_window12_384_22kto1k.pth', 'swin_large_patch4_window12_384_22k.pth'][0]),
    'swin_v1_t': os.path.join(weights_root_dir, ['swin_tiny_patch4_window7_224_22kto1k_finetune.pth'][0]),
    'swin_v1_s': os.path.join(weights_root_dir, ['swin_small_patch4_window7_224_22kto1k_finetune.pth'][0]),
    'pvt_v2_b0': os.path.join(weights_root_dir, ['pvt_v2_b0.pth'][0]),
    'pvt_v2_b1': os.path.join(weights_root_dir, ['pvt_v2_b1.pth'][0]),
}

# evaluate settings

