"""
Mask Processor
掩码形态学处理（腐蚀、边界提取）
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np


class MaskProcessor:
    """掩码处理器"""

    @staticmethod
    def generate_morphed_masks(gt_mask, fg_kernel_size=5, bg_kernel_size=3):
        """
        生成形态学处理后的掩码

        Args:
            gt_mask: Ground truth mask [B, 1, H, W]
            fg_kernel_size: 前景腐蚀核大小
            bg_kernel_size: 背景腐蚀核大小

        Returns:
            mask_fg: 前景核心区 [B, 1, H, W]
            mask_edge: 边界不确定区 [B, 1, H, W]
            mask_bg: 纯净背景区 [B, 1, H, W]
        """
        B, _, H, W = gt_mask.shape
        device = gt_mask.device

        mask_fg_list = []
        mask_edge_list = []
        mask_bg_list = []

        for i in range(B):
            mask = gt_mask[i, 0].cpu().numpy()

            # 前景腐蚀：M_fg = Erode(M, kernel=5)
            kernel_fg = np.ones((fg_kernel_size, fg_kernel_size), np.uint8)
            mask_fg = cv2.erode(mask, kernel_fg, iterations=1)

            # 边界区域：M_edge = M - M_fg
            mask_edge = mask - mask_fg

            # 背景腐蚀：M_bg = (1 - M) ⊙ Erode(1-M, kernel=3)
            mask_inv = 1 - mask
            kernel_bg = np.ones((bg_kernel_size, bg_kernel_size), np.uint8)
            mask_bg = cv2.erode(mask_inv, kernel_bg, iterations=1)

            mask_fg_list.append(torch.from_numpy(mask_fg).unsqueeze(0))
            mask_edge_list.append(torch.from_numpy(mask_edge).unsqueeze(0))
            mask_bg_list.append(torch.from_numpy(mask_bg).unsqueeze(0))

        mask_fg = torch.stack(mask_fg_list, dim=0).unsqueeze(1).to(device)
        mask_edge = torch.stack(mask_edge_list, dim=0).unsqueeze(1).to(device)
        mask_bg = torch.stack(mask_bg_list, dim=0).unsqueeze(1).to(device)

        return mask_fg, mask_edge, mask_bg

    @staticmethod
    def downsample_mask(mask, target_size):
        """
        下采样掩码以匹配DINOv3的patch size

        Args:
            mask: 原始掩码 [B, 1, H, W]
            target_size: 目标尺寸 (H', W')

        Returns:
            downsampled_mask: 下采样后的掩码 [B, 1, H', W']
        """
        return F.interpolate(mask, size=target_size, mode='nearest')
