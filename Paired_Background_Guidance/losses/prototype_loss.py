"""
Prototype Contrastive Loss
原型对比损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeContrastiveLoss(nn.Module):
    """
    原型对比损失

    拉近前景特征与前景原型，推远前景特征与背景原型
    """

    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: 温度参数
        """
        super(PrototypeContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, fg_prototypes, bg_prototypes, mask):
        """
        计算对比损失

        Args:
            features: 特征 [B, D, H, W]
            fg_prototypes: 前景原型 [N_fg, D]
            bg_prototypes: 背景原型 [N_bg, D]
            mask: 前景掩码 [B, 1, H, W]

        Returns:
            loss: 对比损失
        """
        B, D, H, W = features.shape

        # 提取前景区域的特征
        mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')
        fg_features = features * mask_resized  # [B, D, H, W]

        # 展平
        fg_features_flat = fg_features.permute(0, 2, 3, 1).reshape(-1, D)  # [B*H*W, D]
        mask_flat = mask_resized.reshape(-1)  # [B*H*W]

        # 只保留前景位置
        fg_features_valid = fg_features_flat[mask_flat > 0.5]  # [N_valid, D]

        if fg_features_valid.shape[0] == 0:
            return torch.tensor(0.0, device=features.device)

        # 归一化
        fg_features_norm = F.normalize(fg_features_valid, dim=-1)
        fg_proto_norm = F.normalize(fg_prototypes, dim=-1)
        bg_proto_norm = F.normalize(bg_prototypes, dim=-1)

        # 计算相似度
        sim_fg = torch.matmul(fg_features_norm, fg_proto_norm.T) / self.temperature  # [N_valid, N_fg]
        sim_bg = torch.matmul(fg_features_norm, bg_proto_norm.T) / self.temperature  # [N_valid, N_bg]

        # 对比损失：拉近前景，推远背景
        # 使用InfoNCE loss
        pos_sim = sim_fg.max(dim=-1)[0]  # [N_valid]
        neg_sim = sim_bg  # [N_valid, N_bg]

        # log-sum-exp
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [N_valid, 1+N_bg]
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        return loss
