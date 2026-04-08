"""
Prototype Builder
从DINOv3特征构建原型
"""

import torch
import torch.nn as nn


class PrototypeBuilder:
    """从DINOv3特征构建原型"""

    @staticmethod
    def build_texture_prototypes(shallow_features, mask_bg):
        """
        构建纹理原型（统计量）

        Args:
            shallow_features: 浅层特征 [B, C, H, W]
            mask_bg: 背景掩码 [B, 1, H, W]

        Returns:
            mean: 通道均值 [B, C]
            std: 通道标准差 [B, C]
        """
        # TODO: 实现纹理统计量提取
        pass

    @staticmethod
    def build_background_prototypes(deep_features, mask_bg, k=512):
        """
        构建背景原型（使用FPS采样）

        Args:
            deep_features: 深层特征 [B, C, H, W]
            mask_bg: 背景掩码 [B, 1, H, W]
            k: 采样数量

        Returns:
            prototypes: 背景原型 [k, C]
        """
        # TODO: 实现FPS采样
        pass

    @staticmethod
    def build_foreground_prototypes(deep_features, mask_fg):
        """
        构建前景原型（使用MAP）

        Args:
            deep_features: 深层特征 [B, C, H, W]
            mask_fg: 前景掩码 [B, 1, H, W]

        Returns:
            prototypes: 前景原型 [B, C]
        """
        # TODO: 实现Masked Average Pooling
        pass

    @staticmethod
    def build_edge_prototypes(deep_features, mask_edge, k=128):
        """
        构建边界原型（使用熵选择）

        Args:
            deep_features: 深层特征 [B, C, H, W]
            mask_edge: 边界掩码 [B, 1, H, W]
            k: 采样数量

        Returns:
            prototypes: 边界原型 [k, C]
        """
        # TODO: 实现熵选择
        pass
