"""
Feature Operations
特征操作工具函数
"""

import torch
import torch.nn.functional as F


class FeatureOps:
    """特征操作工具类"""

    @staticmethod
    def cosine_similarity(features_a, features_b, dim=-1):
        """
        计算余弦相似度

        Args:
            features_a: 特征A
            features_b: 特征B
            dim: 计算维度

        Returns:
            similarity: 余弦相似度
        """
        return F.cosine_similarity(features_a, features_b, dim=dim)

    @staticmethod
    def normalize(features, dim=-1, eps=1e-8):
        """
        L2归一化

        Args:
            features: 输入特征
            dim: 归一化维度
            eps: 数值稳定性参数

        Returns:
            normalized_features: 归一化后的特征
        """
        return F.normalize(features, p=2, dim=dim, eps=eps)

    @staticmethod
    def euclidean_distance(features_a, features_b):
        """
        计算欧氏距离

        Args:
            features_a: 特征A [B, D]
            features_b: 特征B [N, D]

        Returns:
            distance: 距离矩阵 [B, N]
        """
        # 使用广播计算距离
        diff = features_a.unsqueeze(1) - features_b.unsqueeze(0)  # [B, N, D]
        distance = torch.norm(diff, p=2, dim=-1)  # [B, N]
        return distance

    @staticmethod
    def channel_wise_statistics(features, mask=None):
        """
        计算通道级统计量（均值、标准差）

        Args:
            features: 特征 [B, C, H, W]
            mask: 可选的掩码 [B, 1, H, W]

        Returns:
            mean: 均值 [B, C]
            std: 标准差 [B, C]
        """
        if mask is not None:
            # 在掩码区域内计算统计量
            masked_features = features * mask
            sum_features = masked_features.sum(dim=[2, 3])
            mask_sum = mask.sum(dim=[2, 3]) + 1e-8
            mean = sum_features / mask_sum

            # 计算标准差
            diff = (features - mean.unsqueeze(-1).unsqueeze(-1)) * mask
            var = (diff ** 2).sum(dim=[2, 3]) / mask_sum
            std = torch.sqrt(var + 1e-8)
        else:
            # 在整个特征图上计算
            mean = features.mean(dim=[2, 3])
            std = features.std(dim=[2, 3]) + 1e-8

        return mean, std

    @staticmethod
    def spatial_average_pooling(features, mask):
        """
        空间平均池化（在掩码区域内）

        Args:
            features: 特征 [B, C, H, W]
            mask: 掩码 [B, 1, H, W]

        Returns:
            pooled: 池化后的特征 [B, C]
        """
        masked_features = features * mask
        sum_features = masked_features.sum(dim=[2, 3])
        mask_sum = mask.sum(dim=[2, 3]) + 1e-8
        pooled = sum_features / mask_sum
        return pooled
