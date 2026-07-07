model_name = 'Default'
backbone = 'swin_v1_l'
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

ms_supervision = True
out_ref = ms_supervision and True
dec_ipt = True
dec_ipt_split = True
mul_scl_ipt = ['', 'add', 'cat'][2]

dec_att = ['', 'ASPP', 'ASPPDeformable'][2]
squeeze_block = ['', 'BasicDecBlk_x1', 'ResBlk_x4', 'ASPP_x3', 'ASPPDeformable_x3'][1]
dec_blk = ['BasicDecBlk', 'ResBlk', 'HierarAttDecBlk'][0]

freeze_bb = False
compile_model = False
precisionHigh = False
