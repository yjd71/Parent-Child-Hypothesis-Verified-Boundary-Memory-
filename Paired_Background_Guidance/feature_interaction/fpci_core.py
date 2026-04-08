"""
FPCI Core Module
双流全原型特征交互器核心模块
"""

import torch
import torch.nn as nn
from typing import Tuple

from .texture_normalizer import TextureNormalizer
from .foreground_enhancer import ForegroundEnhancer
from .background_suppressor import BackgroundSuppressor
from .stream_fusion import StreamFusion


class FPCICore(nn.Module):
    """
    双流全原型特征交互器 v3.0

    两个层级的交互：
    1. 浅层交互：纹理矫正（AdaIN）
    2. 深层交互：双流Cross-Attention
       - Stream 1: 正向语义流（拉向前景）
       - Stream 2: 负向抑制流（推离背景）
    """

    def __init__(self, config):
        super(FPCICore, self).__init__()

        self.config = config
        self.feature_dim = config.get('feature_dim', 768)

        # 浅层纹理矫正
        self.texture_normalizer = TextureNormalizer()

        # 深层双流交互
        self.foreground_enhancer = ForegroundEnhancer(self.feature_dim)
        self.background_suppressor = BackgroundSuppressor(self.feature_dim)

        # 双流融合
        self.stream_fusion = StreamFusion(self.feature_dim)

    def forward(
        self,
        anomaly_features: torch.Tensor,
        texture_prototypes: Tuple[torch.Tensor, torch.Tensor],
        foreground_prototypes: torch.Tensor,
        edge_prototypes: torch.Tensor,
        background_prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            anomaly_features: BBR提取的异常特征 [B, D, H, W]
            texture_prototypes: 纹理原型（均值、标准差）
            foreground_prototypes: 前景原型 [N_fg, D]
            edge_prototypes: 边界原型 [N_edge, D]
            background_prototypes: 背景原型 [N_bg, D]

        Returns:
            refined_features: 精炼后的特征 [B, D, H, W]
        """

        # 1. 浅层纹理矫正（AdaIN）
        texture_mean, texture_std = texture_prototypes
        normalized_features = self.texture_normalizer(
            anomaly_features,
            texture_mean,
            texture_std
        )

        # 2. 深层双流交互
        # Stream 1: 正向语义流（前景+边界）
        fg_edge_prototypes = torch.cat([foreground_prototypes, edge_prototypes], dim=0)
        foreground_stream = self.foreground_enhancer(
            normalized_features,
            fg_edge_prototypes
        )

        # Stream 2: 负向抑制流（背景）
        background_stream = self.background_suppressor(
            normalized_features,
            background_prototypes
        )

        # 3. 双流融合
        refined_features = self.stream_fusion(
            normalized_features,
            foreground_stream,
            background_stream
        )

        return refined_features
