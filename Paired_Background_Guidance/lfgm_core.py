"""
LFGM Core Module
协调四大子模块：Quad-GDPB、BBR、FPCI、SMM
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from .prototype_bank import QuadGDPB
from .background_reconstructor import BBRCore
from .feature_interaction import FPCICore
from .consistency import SMMCore
from .losses import LFGMLoss


class LFGMCore(nn.Module):
    """
    LFGM v3.0 核心模块
    
    功能：
    1. 管理四元原型库（Quad-GDPB）
    2. 背景偏置重构（BBR v3.0）
    3. 双流特征交互（Dual-Stream FPCI）
    4. 协同混合一致性（SMM v3.0）
    
    Args:
        config: LFGM配置字典
        dinov3_backbone: 冻结的DINOv3骨干网络
    """
    
    def __init__(self, config: Dict, dinov3_backbone: nn.Module):
        super(LFGMCore, self).__init__()
        
        self.config = config
        self.dinov3 = dinov3_backbone
        
        # 初始化四大核心模块
        self.prototype_bank = QuadGDPB(config['prototype_bank'])
        self.bbr = BBRCore(config['bbr'])
        self.fpci = FPCICore(config['fpci'])
        self.smm = SMMCore(config['smm'])
        
        # 损失函数
        self.loss_fn = LFGMLoss(config['loss'])
        
    def forward(
        self,
        student_features: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
        update_prototypes: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        前向传播

        Args:
            student_features: Student模型的特征 [B, C, H, W]
            gt_mask: Ground truth mask（仅训练时需要）[B, 1, H, W]
            update_prototypes: 是否更新原型库（仅在处理有标签数据时为True）

        Returns:
            refined_features: 精炼后的特征 [B, C, H, W]
            lfgm_outputs: LFGM输出字典（训练时包含损失和SMM信息，推理时为None）
                - 'recon_loss': 重构损失
                - 'smm_info': SMM混合信息（用于计算一致性损失）
                    - 'mixed_features': 混合特征 [B, C, H, W]
                    - 'nearest_prototypes': 最近原型特征 [B, C, H, W]
                    - 'alpha': 混合系数
        """

        # 训练模式：更新原型库
        if self.training and update_prototypes and gt_mask is not None:
            with torch.no_grad():
                # 使用DINOv3提取权威特征并更新原型库
                dinov3_features = self.dinov3.extract_features(student_features)
                self.prototype_bank.update(dinov3_features, gt_mask)

        # 1. BBR: 背景重构与异常提取
        anomaly_features, recon_loss = self.bbr(
            student_features,
            self.prototype_bank.get_background_prototypes()
        )

        # 2. FPCI: 双流特征交互
        refined_features = self.fpci(
            anomaly_features,
            texture_prototypes=self.prototype_bank.get_texture_prototypes(),
            foreground_prototypes=self.prototype_bank.get_foreground_prototypes(),
            edge_prototypes=self.prototype_bank.get_edge_prototypes(),
            background_prototypes=self.prototype_bank.get_background_prototypes()
        )

        # 3. SMM: 协同混合一致性（仅训练时）
        # 注意：SMM返回混合特征信息，一致性损失需要在主模型中计算
        # 因为需要Decoder的预测结果
        smm_info = None
        if self.training:
            mixed_features, nearest_prototypes, alpha = self.smm(
                refined_features,
                self.prototype_bank.get_foreground_prototypes()
            )
            smm_info = {
                'mixed_features': mixed_features,
                'nearest_prototypes': nearest_prototypes,
                'alpha': alpha
            }

        # 返回结果
        if self.training:
            lfgm_outputs = {
                'recon_loss': recon_loss,
                'smm_info': smm_info
            }
            return refined_features, lfgm_outputs
        else:
            return refined_features, None
    
    def update_prototypes_from_labeled(
        self, 
        labeled_images: torch.Tensor, 
        labeled_masks: torch.Tensor
    ):
        """
        从有标签数据更新原型库
        
        Args:
            labeled_images: 有标签图像 [B, 3, H, W]
            labeled_masks: 对应的GT mask [B, 1, H, W]
        """
        with torch.no_grad():
            # 提取DINOv3特征
            dinov3_features = self.dinov3.extract_features(labeled_images)
            # 更新原型库
            self.prototype_bank.update(dinov3_features, labeled_masks)
    
    def save_prototypes(self, path: str):
        """保存原型库到checkpoint"""
        self.prototype_bank.save(path)
    
    def load_prototypes(self, path: str):
        """从checkpoint加载原型库"""
        self.prototype_bank.load(path)
