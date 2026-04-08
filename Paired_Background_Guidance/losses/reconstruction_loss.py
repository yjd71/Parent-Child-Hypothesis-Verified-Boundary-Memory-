"""
Reconstruction Loss
重构损失
"""

import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """
    重构损失

    衡量背景重构的质量
    """

    def __init__(self, loss_type='mse'):
        """
        Args:
            loss_type: 损失类型 ('mse', 'l1', 'smooth_l1')
        """
        super(ReconstructionLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, original, reconstructed, mask=None):
        """
        计算重构损失

        Args:
            original: 原始特征 [B, D, H, W]
            reconstructed: 重构特征 [B, D, H, W]
            mask: 可选的掩码 [B, 1, H, W]

        Returns:
            loss: 重构损失
        """
        if self.loss_type == 'mse':
            loss = nn.functional.mse_loss(reconstructed, original, reduction='none')
        elif self.loss_type == 'l1':
            loss = nn.functional.l1_loss(reconstructed, original, reduction='none')
        elif self.loss_type == 'smooth_l1':
            loss = nn.functional.smooth_l1_loss(reconstructed, original, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # 如果提供了掩码，只在掩码区域计算损失
        if mask is not None:
            loss = loss * mask

        return loss.mean()
