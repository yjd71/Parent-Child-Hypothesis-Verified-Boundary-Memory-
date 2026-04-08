"""
Background Suppressor
负向背景抑制流
"""

import torch
import torch.nn as nn


class BackgroundSuppressor(nn.Module):
    """
    背景抑制器

    使用Cross-Attention检查异常特征中是否还残留背景噪声
    并给予抑制权重
    """

    def __init__(self, feature_dim, num_heads=8):
        """
        Args:
            feature_dim: 特征维度
            num_heads: 注意力头数
        """
        super(BackgroundSuppressor, self).__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, query_features, background_prototypes):
        """
        前向传播

        Args:
            query_features: 查询特征（异常特征） [B, D, H, W]
            background_prototypes: 背景原型 [N, D]

        Returns:
            suppression_features: 需要抑制的背景成分 [B, D, H, W]
        """
        B, D, H, W = query_features.shape
        N = background_prototypes.shape[0]

        # 重塑为序列格式
        query = query_features.permute(0, 2, 3, 1).reshape(B, H * W, D)  # [B, H*W, D]
        key_value = background_prototypes.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]

        # Cross-Attention
        suppression, attn_weights = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value
        )  # [B, H*W, D]

        # 重塑回原始形状
        suppression_features = suppression.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]

        return suppression_features
