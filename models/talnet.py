import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from kornia.filters import laplacian

from models.backbones.build_backbone import build_backbone
from models.modules.decoder_blocks import BasicDecBlk, ResBlk, HierarAttDecBlk
from models.modules.lateral_blocks import BasicLatBlk
from models.modules.aspp import ASPP, ASPPDeformable
from models.modules.ing import *
from models.refinement.refiner import Refiner, RefinerPVTInChannels4, RefUNet
from models.refinement.stem_layer import StemLayer

class ModelEMA(nn.Module):
    def __init__(self, config, bb_pretrained=True, alpha=0.999):
        super(ModelEMA, self).__init__()
        self.alpha = alpha
        self.student = TalNet(config=config, bb_pretrained=bb_pretrained)
        self.teacher = TalNet(config=config, bb_pretrained=bb_pretrained, ema=True)
        self.cbm = None
        self._init_teacher_params()

    def set_cbm(self, cbm):
        self.cbm = cbm
        return self

    def extract_cbm_memory_features(self, x, ema=True):
        if ema:
            return self.teacher.extract_cbm_memory_features(x)
        return self.student.extract_cbm_memory_features(x)

    def forward(self, x, ema=False, use_memory=None, cbm=None, return_aux=False, memory_t=None):
        active_cbm = self.cbm if cbm is None else cbm
        if ema:
            return self.teacher(
                x,
                use_memory=use_memory,
                cbm=active_cbm,
                memory_t=memory_t,
                return_aux=return_aux,
            )
        else:
            return self.student(
                x,
                use_memory=use_memory,
                cbm=active_cbm,
                memory_t=memory_t,
                return_aux=return_aux,
            )
        
    def ema_update(self, global_step, alpha=None):
        with torch.no_grad():
            if alpha == None:
                alpha = min(1 - 1 / (global_step + 1), self.alpha)
            for ema_param, param in zip(self.teacher.parameters(), self.student.parameters()):
                ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
            for ema_buff, buff in zip(self.teacher.buffers(), self.student.buffers()):
                ema_buff = ema_buff.float()
                buff = buff.float()
                ema_buff.data.mul_(alpha).add_(buff.data, alpha=1 - alpha)

    def sync(self):
        for param in self.teacher.parameters():
            dist.all_reduce(param.data) 
            param.data /= dist.get_world_size()  

    def _init_teacher_params(self):
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data.copy_(param_s.data)
    def train(self, mode=True):
        super().train(mode)
        self.student.train(mode)
        self.teacher.eval()  

    def eval(self):
        super().eval()
        self.student.eval()
        self.teacher.eval()

class TalNet(nn.Module):
    def __init__(self, config, bb_pretrained=True, ema=False):
        super(TalNet, self).__init__()
        self.config = config
        self.epoch = 1
        self.ema = ema
        self.cbm = None
        self.bb = build_backbone(self.config.backbone, config=config, pretrained=bb_pretrained)
        channels = self.config.lateral_channels_in_collection
        if self.config.squeeze_block:
            self.squeeze_module = nn.Sequential(*[
                eval(self.config.squeeze_block.split('_x')[0])(config, channels[0]+sum(self.config.cxt), channels[0])
                for _ in range(eval(self.config.squeeze_block.split('_x')[1]))
            ])
        
        self.decoder = Decoder(config, channels)

        if ema:
            for param in self.parameters():
                param.requires_grad=False

    def freeze(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    def set_cbm(self, cbm):
        self.cbm = cbm
        return self

    def forward_enc(self, x):
        if self.config.backbone in ['vgg16', 'vgg16bn', 'resnet50']:
            x1 = self.bb.conv1(x); x2 = self.bb.conv2(x1); x3 = self.bb.conv3(x2); x4 = self.bb.conv4(x3)
        else:
            x1, x2, x3, x4 = self.bb(x)
            B, C, H, W = x.shape
            x1_, x2_, x3_, x4_ = self.bb(F.interpolate(x, size=(H//2, W//2), mode='bilinear', align_corners=True))
            x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)], dim=1)
            x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)], dim=1)
            x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)], dim=1)
            x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        
        if self.config.cxt:
            x4 = torch.cat(
                (
                    *[
                        F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=True),
                        F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True),
                        F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True),
                    ][-len(self.config.cxt):],
                    x4
                ),
                dim=1
            )
        return (x1, x2, x3, x4)

    def _build_decoder_features(self, x):
        ########## Encoder ##########
        x1, x2, x3, x4 = self.forward_enc(x)
        if self.config.squeeze_block:
            x4 = self.squeeze_module(x4)
        ########## Decoder ##########
        features = [x, x1, x2, x3, x4]
        if self.training and self.config.out_ref:
            features.append(laplacian(torch.mean(x, dim=1).unsqueeze(1), kernel_size=5))
        return x1, x2, x3, x4, features

    def forward_ori(self, x):
        *_, features = self._build_decoder_features(x)
        return self.decoder(features)

    def extract_cbm_memory_features(self, x):
        was_training = self.training
        self.eval()
        try:
            x1, x2, x3, x4, features = self._build_decoder_features(x)
            del x1, x2, x4
            _, p3, _ = self.decoder.forward_to_p3(features)
            return {"x3": x3.detach(), "p3": p3.detach()}
        finally:
            if was_training:
                self.train()

    def forward_cbm_pfi(self, x, cbm=None, return_aux=False, memory_t=None):
        active_cbm = self.cbm if cbm is None else cbm
        reason = self._cbm_fallback_reason(active_cbm, memory_t=memory_t)
        if reason is not None:
            return self._return_with_optional_aux(self.forward_ori(x), reason, return_aux)

        x1, x2, x3, x4, features = self._build_decoder_features(x)
        del x1, x2, x4
        state, p3, m3 = self.decoder.forward_to_p3(features)
        if m3 is None:
            scaled_preds = self.decoder.forward_from_p3(state, p3)
            return self._return_with_optional_aux(scaled_preds, "m3_none", return_aux, p3=p3)

        hook_kwargs = {
            "x": x,
            "x3": x3,
            "p3": p3,
            "m3": m3,
            "training": self.training,
        }
        if memory_t is not None:
            hook_kwargs["memory_t"] = memory_t
        p3_corr, aux = active_cbm.apply_p3_hook(**hook_kwargs)
        if not aux or not aux.get("cbm_used", False):
            scaled_preds = self.decoder.forward_from_p3(state, p3)
            reason = "cbm_hook_fallback" if not aux else aux.get("fallback_reason", "cbm_hook_fallback")
            return self._return_scaled_preds(scaled_preds, aux or self._make_fallback_aux(reason, p3=p3), return_aux)

        scaled_preds = self.decoder.forward_from_p3(state, p3_corr)
        scaled_preds = self._apply_cbm_final_fusion(scaled_preds, active_cbm, aux)
        return self._return_scaled_preds(scaled_preds, aux, return_aux)

    def forward(self, x, use_memory=None, cbm=None, return_aux=False, memory_t=None):
        active_cbm = self.cbm if cbm is None else cbm
        if not self._should_use_cbm(use_memory, active_cbm):
            return self._return_with_optional_aux(
                self.forward_ori(x),
                self._disabled_forward_reason(use_memory, active_cbm, memory_t=memory_t),
                return_aux,
            )
        return self.forward_cbm_pfi(
            x,
            cbm=active_cbm,
            memory_t=memory_t,
            return_aux=return_aux,
        )

    def _should_use_cbm(self, use_memory, cbm):
        if use_memory is False:
            return False
        if cbm is None:
            return False
        if use_memory is True:
            return True
        return (not self.training) and self._cbm_is_enabled(cbm)

    def _cbm_fallback_reason(self, cbm, memory_t=None):
        if cbm is None:
            return "cbm_none"
        if not self._cbm_memory_ready(cbm, memory_t=memory_t):
            return "memory_not_ready"
        if not self._cbm_is_enabled(cbm, memory_t=memory_t):
            return "cbm_disabled"
        return None

    def _disabled_forward_reason(self, use_memory, cbm, memory_t=None):
        if use_memory is False:
            return "use_memory_false"
        if cbm is None:
            return "cbm_none"
        if self.training and use_memory is None:
            return "train_default_baseline"
        return self._cbm_fallback_reason(cbm, memory_t=memory_t) or "cbm_not_requested"

    def _cbm_memory_ready(self, cbm, memory_t=None):
        memory_ready = getattr(cbm, "memory_ready", None)
        if callable(memory_ready):
            return bool(memory_ready(memory_t))
        if memory_t is not None:
            memory = memory_t.get(
                "labeled_memory",
                memory_t.get("L_t"),
            )
            if memory is not None:
                return bool(memory.is_ready())
        memory = getattr(cbm, "memory", None)
        return bool(memory is not None and memory.is_ready())

    def _cbm_is_enabled(self, cbm, memory_t=None):
        enabled_for_epoch = getattr(cbm, "enabled_for_epoch", None)
        if enabled_for_epoch is None:
            return self._cbm_memory_ready(cbm, memory_t=memory_t)
        if memory_t is None:
            return bool(enabled_for_epoch())
        return bool(enabled_for_epoch(memory_t=memory_t))

    def _apply_cbm_final_fusion(self, scaled_preds, cbm, aux):
        if self.config.out_ref and self.training:
            gdt_outputs, outs = scaled_preds
            outs = list(outs)
            outs[-1] = cbm.apply_final_fusion(outs[-1], aux)
            return gdt_outputs, outs
        outs = list(scaled_preds)
        outs[-1] = cbm.apply_final_fusion(outs[-1], aux)
        return outs

    def _return_with_optional_aux(self, scaled_preds, reason, return_aux, p3=None):
        return self._return_scaled_preds(scaled_preds, self._make_fallback_aux(reason, p3=p3), return_aux)

    def _return_scaled_preds(self, scaled_preds, aux, return_aux):
        if return_aux:
            return scaled_preds, aux
        return scaled_preds

    def _make_fallback_aux(self, reason, p3=None):
        aux = {
            "cbm_used": False,
            "fallback_reason": reason,
            "top_img_ids": [],
            "img_scores": None,
            "num_memory_tokens": 0,
            "num_valid_boundary_tokens": 0,
            "valid_ratio": 0.0,
            "B3_mean": 0.0,
            "gate_mean": 0.0,
            "cons_mean": 0.0,
            "u_mean": 0.0,
            "p_final": None,
            "p_main": None,
            "B_query": None,
            "boundary_mask": None,
            "z_mem3": None,
            "gate3": None,
        }
        if p3 is not None:
            aux["p3_shape"] = tuple(p3.shape)
        return aux


class Decoder(nn.Module):
    def __init__(self, config, channels):
        super(Decoder, self).__init__()
        self.config = config
        DecoderBlock = eval(self.config.dec_blk)
        LateralBlock = BasicLatBlk

        if self.config.dec_ipt:
            self.split = self.config.dec_ipt_split
            N_dec_ipt = 64
            DBlock = SimpleConvs
            ic = 64
            ipt_cha_opt = 1
            self.ipt_blk4 = DBlock(2**8*3 if self.split else 3, [N_dec_ipt, channels[0]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk3 = DBlock(2**6*3 if self.split else 3, [N_dec_ipt, channels[1]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk2 = DBlock(2**4*3 if self.split else 3, [N_dec_ipt, channels[2]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk1 = DBlock(2**0*3 if self.split else 3, [N_dec_ipt, channels[3]//8][ipt_cha_opt], inter_channels=ic)
        else:
            self.split = None

        self.decoder_block4 = DecoderBlock(config,channels[0], channels[1])
        self.decoder_block3 = DecoderBlock(config,channels[1]+([N_dec_ipt, channels[0]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[2])
        self.decoder_block2 = DecoderBlock(config,channels[2]+([N_dec_ipt, channels[1]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[3])
        self.decoder_block1 = DecoderBlock(config,channels[3]+([N_dec_ipt, channels[2]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[3]//2)
        self.conv_out1 = nn.Sequential(nn.Conv2d(channels[3]//2+([N_dec_ipt, channels[3]//8][ipt_cha_opt] if self.config.dec_ipt else 0), 1, 1, 1, 0))

        self.lateral_block4 = LateralBlock(channels[1], channels[1])
        self.lateral_block3 = LateralBlock(channels[2], channels[2])
        self.lateral_block2 = LateralBlock(channels[3], channels[3])

        if self.config.ms_supervision:
            self.conv_ms_spvn_4 = nn.Conv2d(channels[1], 1, 1, 1, 0)
            self.conv_ms_spvn_3 = nn.Conv2d(channels[2], 1, 1, 1, 0)
            self.conv_ms_spvn_2 = nn.Conv2d(channels[3], 1, 1, 1, 0)

            if self.config.out_ref:
                _N = 16
                # self.gdt_convs_4 = nn.Sequential(nn.Conv2d(channels[1], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
                self.gdt_convs_3 = nn.Sequential(nn.Conv2d(channels[2], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
                self.gdt_convs_2 = nn.Sequential(nn.Conv2d(channels[3], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))

                # self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                
                # self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))


    def get_patches_batch(self, x, p):
        _size_h, _size_w = p.shape[2:]
        patches_batch = []
        for idx in range(x.shape[0]):
            columns_x = torch.split(x[idx], split_size_or_sections=_size_w, dim=-1)
            patches_x = []
            for column_x in columns_x:
                patches_x += [p.unsqueeze(0) for p in torch.split(column_x, split_size_or_sections=_size_h, dim=-2)]
            patch_sample = torch.cat(patches_x, dim=1)
            patches_batch.append(patch_sample)
        return torch.cat(patches_batch, dim=0)

    def forward_to_p3(self, features):
        if self.training and self.config.out_ref:
            outs_gdt_pred = []
            outs_gdt_label = []
            x, x1, x2, x3, x4, gdt_gt = features
        else:
            outs_gdt_pred = []
            outs_gdt_label = []
            gdt_gt = None
            x, x1, x2, x3, x4 = features
        outs = []
        p4 = self.decoder_block4(x4)
        m4 = self.conv_ms_spvn_4(p4) if self.config.ms_supervision else None
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)
        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p3) if self.split else x
            _p3 = torch.cat((_p3, self.ipt_blk4(F.interpolate(patches_batch, size=x3.shape[2:], mode='bilinear', align_corners=True))), 1)

        p3 = self.decoder_block3(_p3)
        m3 = self.conv_ms_spvn_3(p3) if self.config.ms_supervision else None
        if self.config.out_ref:
            p3_gdt = self.gdt_convs_3(p3)
            if self.training:
                # >> GT:
                # m3 --dilation--> m3_dia
                # G_3^gt * m3_dia --> G_3^m, which is the label of gradient
                m3_dia = m3
                gdt_label_main_3 = gdt_gt * F.interpolate(m3_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True)
                outs_gdt_label.append(gdt_label_main_3)
                # >> Pred:
                # p3 --conv--BN--> F_3^G, where F_3^G predicts the \hat{G_3} with xx
                # F_3^G --sigmoid--> A_3^G
                gdt_pred_3 = self.gdt_convs_pred_3(p3_gdt)
                outs_gdt_pred.append(gdt_pred_3)
            gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
            # >> Finally:
            # p3 = p3 * A_3^G
            p3 = p3 * gdt_attn_3

        state = {
            "x": x,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "gdt_gt": gdt_gt,
            "outs": outs,
            "outs_gdt_pred": outs_gdt_pred,
            "outs_gdt_label": outs_gdt_label,
            "m4": m4,
            "m3": m3,
        }
        return state, p3, m3

    def forward_from_p3(self, state, p3_override):
        x = state["x"]
        x1 = state["x1"]
        x2 = state["x2"]
        gdt_gt = state["gdt_gt"]
        outs = list(state["outs"])
        outs_gdt_pred = list(state["outs_gdt_pred"])
        outs_gdt_label = list(state["outs_gdt_label"])
        m4 = state["m4"]
        m3 = state["m3"]

        p3 = p3_override
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)
        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p2) if self.split else x
            _p2 = torch.cat((_p2, self.ipt_blk3(F.interpolate(patches_batch, size=x2.shape[2:], mode='bilinear', align_corners=True))), 1)

        p2 = self.decoder_block2(_p2)
        m2 = self.conv_ms_spvn_2(p2) if self.config.ms_supervision else None
        if self.config.out_ref:
            p2_gdt = self.gdt_convs_2(p2)
            if self.training:
                # >> GT:
                m2_dia = m2
                gdt_label_main_2 = gdt_gt * F.interpolate(m2_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True)
                outs_gdt_label.append(gdt_label_main_2)
                # >> Pred:
                gdt_pred_2 = self.gdt_convs_pred_2(p2_gdt)
                outs_gdt_pred.append(gdt_pred_2)
            gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
            # >> Finally:
            p2 = p2 * gdt_attn_2
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)
        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p1) if self.split else x
            _p1 = torch.cat((_p1, self.ipt_blk2(F.interpolate(patches_batch, size=x1.shape[2:], mode='bilinear', align_corners=True))), 1)

        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode='bilinear', align_corners=True)
        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p1) if self.split else x
            _p1 = torch.cat((_p1, self.ipt_blk1(F.interpolate(patches_batch, size=x.shape[2:], mode='bilinear', align_corners=True))), 1)
        p1_out = self.conv_out1(_p1)

        if self.config.ms_supervision:
            outs.append(m4)
            outs.append(m3)
            outs.append(m2)
        outs.append(p1_out)
        scaled_preds = outs if not (self.config.out_ref and self.training) else ([outs_gdt_pred, outs_gdt_label], outs)
        return scaled_preds

    def forward(self, features):
        state, p3, _ = self.forward_to_p3(features)
        return self.forward_from_p3(state, p3)


class SimpleConvs(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, inter_channels=64
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))
