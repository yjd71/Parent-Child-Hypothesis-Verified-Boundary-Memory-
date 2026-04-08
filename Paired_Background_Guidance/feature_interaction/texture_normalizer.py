"""
Texture Normalizer
浅层纹理归一化（AdaIN）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureNormalizer(nn.Module):
    """
    纹理归一化器

    使用AdaIN（Adaptive Instance Normalization）
    消除无标签数据与有标签数据在光照、色温上的分布差异
    """

    def __init__(self):
        super(TextureNormalizer, self).__init__()

    def forward(self, features, target_mean, target_std):
        """
        前向传播

        Args:
            features: 输入特征 [B, D, H, W]
            target_mean: 目标均值 [D] or [1, D]
            target_std: 目标标准差 [D] or [1, D]

        Returns:
            normalized_features: 归一化后的特征 [B, D, H, W]
        """
        # 计算当前特征的均值和标准差
        B, D, H, W = features.shape

        # 沿空间维度计算统计量
        feat_mean = features.mean(dim=[2, 3], keepdim=True)  # [B, D, 1, 1]
        feat_std = features.std(dim=[2, 3], keepdim=True) + 1e-8  # [B, D, 1, 1]

        # 归一化到标准分布
        normalized = (features - feat_mean) / feat_std

        # 调整到目标分布
        if target_mean.dim() == 1:
            target_mean = target_mean.view(1, -1, 1, 1)
            target_std = target_std.view(1, -1, 1, 1)
        elif target_mean.dim() == 2:
            target_mean = target_mean.view(1, -1, 1, 1)
            target_std = target_std.view(1, -1, 1, 1)

        # AdaIN: y = σ_target * ((x - μ_x) / σ_x) + μ_target
        normalized_features = normalized * target_std + target_mean

        return normalized_features
