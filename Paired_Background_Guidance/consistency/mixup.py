"""
Feature Mixer
特征混合（MixUp）
"""

import torch
import torch.nn as nn


class FeatureMixer:
    """
    特征混合器

    使用MixUp策略混合无标签特征和原型特征
    """

    def __init__(self, alpha_min=0.3, alpha_max=0.7):
        """
        Args:
            alpha_min: 混合系数最小值
            alpha_max: 混合系数最大值
        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def __call__(self, features_a, features_b):
        """
        混合两个特征

        Args:
            features_a: 特征A [B, D, H, W]
            features_b: 特征B [B, D, H, W]

        Returns:
            mixed_features: 混合特征 [B, D, H, W]
            alpha: 混合系数
        """
        # 从Beta分布采样混合系数
        # 使用均匀分布简化
        alpha = torch.rand(1).item() * (self.alpha_max - self.alpha_min) + self.alpha_min

        # 混合特征
        mixed_features = alpha * features_a + (1 - alpha) * features_b

        return mixed_features, alpha
