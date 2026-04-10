import os

# prototype-guided semi-supervised COD experiment

ckpt_dir = "/home/zhangqing/YJD/SCOD/LFGM/works/pbg_proto_swin_v1_b_bs6"

tot_epochs = 30
sup_only_train_epoch = 15

distributed_train = True
device_map = {
    'model': '*'
}  # Only available for non distributed training

rand_seed = 7
lr = 1e-4

# late-stage loss adjustment
IoU_finetune_last_epochs = [0, -6][1]

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
load_all = False
batch_size = 6
data_split = [0.05]

# random labeled split file
data_split_indices_file_format = "data/cache/labeled_indices/split{}_labeled_indices_random.pt"

task = "COD"
training_set = "TR-COD10K+TR-CAMO"
testing_sets = "CHAMELEON+TE-COD10K+TE-CAMO+NC4K"

# evaluation settings
pred_save_root = os.path.join(ckpt_dir, 'training_preds')
eval_epoch = 20
eval_step = 1
save_step = 1

# prototype experiment switches and key knobs
prototype_enable = True
prototype_feature_level = "p3"
prototype_bank_policy = "per_image_masked_pool_dynamic"
prototype_topk = 16
prototype_sim_temperature = 0.05
prototype_tau = 0.07
prototype_mu_init = 0.5
prototype_loss_weight_h = 0.3

# wandb
ModelName = 'BPGNet-Proto'
others = {
    'sup_epoch': sup_only_train_epoch,
    'total_epoch': tot_epochs,
    'prototype_feature_level': prototype_feature_level,
    'prototype_bank_policy': prototype_bank_policy,
    'prototype_topk': prototype_topk,
    'prototype_sim_temperature': prototype_sim_temperature,
    'prototype_tau': prototype_tau,
    'prototype_mu_init': prototype_mu_init,
}
