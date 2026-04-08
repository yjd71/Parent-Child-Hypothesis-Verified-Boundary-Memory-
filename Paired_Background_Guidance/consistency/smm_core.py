"""
SMM Core Module
协同混合一致性核心模块
"""

import torch
import torch.nn as nn
from .mixup import FeatureMixer
from .consistency_constraint import ConsistencyConstraint


class SMMCore(nn.Module):
    """
    协同混合一致性 v3.0

    利用特征混合（MixUp）约束流形的平滑性

    流程：
    1. 配对：找到最相似的前景原型
    2. 混合：生成混合特征
    3. 预测：分别预测三个特征的输出
    4. 一致性约束：M_mix ≈ α * M_u + (1-α) * M_p
    """

    def __init__(self, config):
        super(SMMCore, self).__init__()

        self.config = config
        self.alpha_min = config.get('alpha_min', 0.3)
        self.alpha_max = config.get('alpha_max', 0.7)

        # 特征混合器
        self.mixer = FeatureMixer(self.alpha_min, self.alpha_max)

        # 一致性约束
        self.consistency = ConsistencyConstraint()

    def forward(self, unlabeled_features, foreground_prototypes):
        """
        前向传播

        Args:
            unlabeled_features: 无标签特征 [B, D, H, W]
            foreground_prototypes: 前景原型 [N, D]

        Returns:
            consistency_loss: 一致性损失
        """

        # 1. 找到最相似的前景原型
        nearest_prototypes = self._find_nearest_prototypes(
            unlabeled_features,
            foreground_prototypes
        )  # [B, D, H, W]

        # 2. 特征混合
        mixed_features, alpha = self.mixer(
            unlabeled_features,
            nearest_prototypes
        )  # [B, D, H, W], scalar

        # 3. 计算一致性损失
        # 注意：这里需要Decoder来生成预测，但Decoder在主模型中
        # 所以这个损失需要在主模型的forward中计算
        # 这里只返回混合特征和alpha供外部使用

        return mixed_features, nearest_prototypes, alpha

    def _find_nearest_prototypes(self, features, prototypes):
        """
        为每个特征位置找到最相似的原型

        Args:
            features: 特征 [B, D, H, W]
            prototypes: 原型 [N, D]

        Returns:
            nearest_prototypes: 最近的原型 [B, D, H, W]
        """
        B, D, H, W = features.shape
        N = prototypes.shape[0]

        # 重塑特征
        feat_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, D)  # [B*H*W, D]

        # 归一化
        feat_norm = torch.nn.functional.normalize(feat_flat, dim=-1)
        proto_norm = torch.nn.functional.normalize(prototypes, dim=-1)

        # 计算相似度
        similarity = torch.matmul(feat_norm, proto_norm.T)  # [B*H*W, N]

        # 找到最相似的原型
        max_indices = similarity.argmax(dim=-1)  # [B*H*W]
        nearest = prototypes[max_indices]  # [B*H*W, D]

        # 重塑回原始形状
        nearest_prototypes = nearest.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]

        return nearest_prototypes
