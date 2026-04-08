"""
Foreground Enhancer
正向语义增强流
"""

import torch
import torch.nn as nn


class ForegroundEnhancer(nn.Module):
    """
    前景增强器

    使用Cross-Attention将异常特征拉向前景和边界原型
    实现语义锚定
    """

    def __init__(self, feature_dim, num_heads=8):
        """
        Args:
            feature_dim: 特征维度
            num_heads: 注意力头数
        """
        super(ForegroundEnhancer, self).__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, query_features, fg_edge_prototypes):
        """
        前向传播

        Args:
            query_features: 查询特征（异常特征） [B, D, H, W]
            fg_edge_prototypes: 前景+边界原型 [N, D]

        Returns:
            enhanced_features: 增强后的特征 [B, D, H, W]
        """
        B, D, H, W = query_features.shape
        N = fg_edge_prototypes.shape[0]

        # 重塑为序列格式
        query = query_features.permute(0, 2, 3, 1).reshape(B, H * W, D)  # [B, H*W, D]
        key_value = fg_edge_prototypes.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]

        # Cross-Attention
        enhanced, attn_weights = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value
        )  # [B, H*W, D]

        # 重塑回原始形状
        enhanced_features = enhanced.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]

        return enhanced_features
