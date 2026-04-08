"""
LFGM Loss
LFGM总损失函数
"""

import torch
import torch.nn as nn


class LFGMLoss(nn.Module):
    """
    LFGM总损失

    整合多个损失项：
    1. 重构损失（BBR）
    2. 正交损失（BBR）
    3. 原型对比损失
    4. 一致性损失（SMM）
    """

    def __init__(self, config):
        super(LFGMLoss, self).__init__()

        self.config = config

        # 损失权重
        self.w_recon = config.get('w_recon', 1.0)
        self.w_ortho = config.get('w_ortho', 0.5)
        self.w_proto = config.get('w_proto', 0.3)
        self.w_consistency = config.get('w_consistency', 0.5)

    def forward(self, loss_dict):
        """
        计算总损失

        Args:
            loss_dict: 损失字典
                - 'recon_loss': 重构损失
                - 'ortho_loss': 正交损失（可选）
                - 'proto_loss': 原型对比损失（可选）
                - 'consistency_loss': 一致性损失

        Returns:
            total_loss: 总损失
        """
        total_loss = 0.0

        # 重构损失
        if 'recon_loss' in loss_dict and loss_dict['recon_loss'] is not None:
            total_loss += self.w_recon * loss_dict['recon_loss']

        # 正交损失
        if 'ortho_loss' in loss_dict and loss_dict['ortho_loss'] is not None:
            total_loss += self.w_ortho * loss_dict['ortho_loss']

        # 原型对比损失
        if 'proto_loss' in loss_dict and loss_dict['proto_loss'] is not None:
            total_loss += self.w_proto * loss_dict['proto_loss']

        # 一致性损失
        if 'consistency_loss' in loss_dict and loss_dict['consistency_loss'] is not None:
            total_loss += self.w_consistency * loss_dict['consistency_loss']

        return total_loss
