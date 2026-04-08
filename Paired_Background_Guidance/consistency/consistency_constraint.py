"""
Consistency Constraint
一致性约束计算
"""

import torch
import torch.nn as nn


class ConsistencyConstraint:
    """
    一致性约束

    确保混合特征的预测与原始特征预测的线性组合一致
    L_SMM = ||M_mix - (α * M_u + (1-α) * M_p)||²
    """

    def __init__(self):
        pass

    def __call__(self, pred_mixed, pred_unlabeled, pred_prototype, alpha):
        """
        计算一致性损失

        Args:
            pred_mixed: 混合特征的预测 [B, 1, H, W]
            pred_unlabeled: 无标签特征的预测 [B, 1, H, W]
            pred_prototype: 原型特征的预测 [B, 1, H, W]
            alpha: 混合系数

        Returns:
            loss: 一致性损失
        """
        # 计算期望的混合预测
        expected_mixed = alpha * pred_unlabeled + (1 - alpha) * pred_prototype

        # L2损失
        loss = torch.nn.functional.mse_loss(pred_mixed, expected_mixed)

        return loss
