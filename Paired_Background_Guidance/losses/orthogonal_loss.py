"""
Orthogonal Loss
正交约束损失
"""

import torch
import torch.nn as nn


class OrthogonalLoss(nn.Module):
    """
    正交约束损失

    确保异常特征与背景特征正交
    """

    def __init__(self):
        super(OrthogonalLoss, self).__init__()

    def forward(self, anomaly_features, background_features):
        """
        计算正交损失

        Args:
            anomaly_features: 异常特征 [B, D, H, W]
            background_features: 背景特征 [B, D, H, W]

        Returns:
            loss: 正交损失
        """
        # 归一化
        anomaly_norm = torch.nn.functional.normalize(anomaly_features, dim=1)
        background_norm = torch.nn.functional.normalize(background_features, dim=1)

        # 计算内积（应该接近0）
        dot_product = (anomaly_norm * background_norm).sum(dim=1)  # [B, H, W]

        # 损失：内积的平方
        loss = (dot_product ** 2).mean()

        return loss
