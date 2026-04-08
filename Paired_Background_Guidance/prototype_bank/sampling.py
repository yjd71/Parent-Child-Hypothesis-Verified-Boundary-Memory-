"""
Sampling Strategies
采样策略：FPS、MAP、熵选择
"""

import torch
import torch.nn.functional as F


class SamplingStrategies:
    """采样策略集合"""

    @staticmethod
    def farthest_point_sampling(features, mask, k):
        """
            最远点采样（FPS）
            用于背景原型采样，确保多样性

            Args:
                features: 特征 [B, C, H, W]
                mask: 采样区域掩码 [B, 1, H, W]
                k: 采样点数量

            Returns:
                sampled_features: 采样的特征 [k, C]
            """
            # TODO: 实现FPS算法
        pass


    @staticmethod
    def masked_average_pooling(features, mask):
        """
        掩码平均池化（MAP）
        用于前景原型提取

        Args:
            features: 特征 [B, C, H, W]
            mask: 前景掩码 [B, 1, H, W]

        Returns:
            pooled_features: 池化后的特征 [B, C]
        """
        # 在掩码区域内计算平均
        masked_features = features * mask
        sum_features = masked_features.sum(dim=[2, 3])  # [B, C]
        mask_sum = mask.sum(dim=[2, 3]) + 1e-8  # [B, 1]
        pooled_features = sum_features / mask_sum
        return pooled_features

    @staticmethod
    def entropy_based_sampling(features, mask, k):
        """
        基于熵的采样
        用于边界原型采样，选择最不确定的区域

        Args:
            features: 特征 [B, C, H, W]
            mask: 边界掩码 [B, 1, H, W]
            k: 采样点数量

        Returns:
            sampled_features: 采样的特征 [k, C]
        """
        # TODO: 实现熵计算和Top-K选择
        pass

    @staticmethod
    def compute_entropy(features):
        """
        计算特征的熵

        Args:
            features: 特征 [B, C, H, W]

        Returns:
            entropy: 熵图 [B, 1, H, W]
        """
        # 归一化到概率分布
        probs = F.softmax(features, dim=1)
        # 计算熵：H = -Σ p*log(p)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)
        return entropy
