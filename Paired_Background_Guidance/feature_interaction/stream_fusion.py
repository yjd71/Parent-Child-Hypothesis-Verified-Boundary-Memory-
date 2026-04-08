"""
Stream Fusion
双流融合（门控机制）
"""

import torch
import torch.nn as nn


class StreamFusion(nn.Module):
    """
    双流融合器

    使用可学习的门控参数融合前景增强流和背景抑制流
    公式：F_refined = λ * F_fg_stream + (1-λ) * (F_anomaly - F_bg_stream)
    """

    def __init__(self, feature_dim):
        """
        Args:
            feature_dim: 特征维度
        """
        super(StreamFusion, self).__init__()

        # 可学习的门控参数（初始化为0.5）
        self.gate = nn.Parameter(torch.tensor(0.5))

        # 可选：使用自适应门控（根据特征动态调整）
        self.adaptive_gate = nn.Sequential(
            nn.Conv2d(feature_dim * 3, feature_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, original_features, foreground_stream, background_stream):
        """
        前向传播

        Args:
            original_features: 原始异常特征 [B, D, H, W]
            foreground_stream: 前景增强流 [B, D, H, W]
            background_stream: 背景抑制流 [B, D, H, W]

        Returns:
            fused_features: 融合后的特征 [B, D, H, W]
        """

        # 方式1：使用全局门控参数
        # lambda_gate = torch.sigmoid(self.gate)
        # fused = lambda_gate * foreground_stream + (1 - lambda_gate) * (original_features - background_stream)

        # 方式2：使用自适应门控（推荐）
        # 拼接三个特征用于计算门控权重
        concat_features = torch.cat([
            original_features,
            foreground_stream,
            background_stream
        ], dim=1)

        # 计算自适应门控权重 [B, 1, H, W]
        lambda_gate = self.adaptive_gate(concat_features)

        # 融合
        positive_term = foreground_stream
        negative_term = original_features - background_stream

        fused_features = lambda_gate * positive_term + (1 - lambda_gate) * negative_term

        return fused_features
