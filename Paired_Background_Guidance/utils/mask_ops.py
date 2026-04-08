"""
Mask Operations
掩码操作工具函数
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np


class MaskOps:
    """掩码操作工具类"""

    @staticmethod
    def resize_mask(mask, target_size, mode='nearest'):
        """
        调整掩码大小

        Args:
            mask: 输入掩码 [B, 1, H, W]
            target_size: 目标尺寸 (H', W')
            mode: 插值模式

        Returns:
            resized_mask: 调整后的掩码 [B, 1, H', W']
        """
        return F.interpolate(mask, size=target_size, mode=mode)

    @staticmethod
    def binarize_mask(mask, threshold=0.5):
        """
        二值化掩码

        Args:
            mask: 输入掩码 [B, 1, H, W]
            threshold: 阈值

        Returns:
            binary_mask: 二值掩码 [B, 1, H, W]
        """
        return (mask > threshold).float()

    @staticmethod
    def erode_mask(mask, kernel_size=5):
        """
        腐蚀掩码

        Args:
            mask: 输入掩码 [B, 1, H, W]
            kernel_size: 腐蚀核大小

        Returns:
            eroded_mask: 腐蚀后的掩码 [B, 1, H, W]
        """
        B, _, H, W = mask.shape
        device = mask.device

        eroded_list = []
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        for i in range(B):
            mask_np = mask[i, 0].cpu().numpy()
            eroded_np = cv2.erode(mask_np, kernel, iterations=1)
            eroded_list.append(torch.from_numpy(eroded_np).unsqueeze(0))

        eroded_mask = torch.stack(eroded_list, dim=0).unsqueeze(1).to(device)
        return eroded_mask

    @staticmethod
    def dilate_mask(mask, kernel_size=5):
        """
        膨胀掩码

        Args:
            mask: 输入掩码 [B, 1, H, W]
            kernel_size: 膨胀核大小

        Returns:
            dilated_mask: 膨胀后的掩码 [B, 1, H, W]
        """
        B, _, H, W = mask.shape
        device = mask.device

        dilated_list = []
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        for i in range(B):
            mask_np = mask[i, 0].cpu().numpy()
            dilated_np = cv2.dilate(mask_np, kernel, iterations=1)
            dilated_list.append(torch.from_numpy(dilated_np).unsqueeze(0))

        dilated_mask = torch.stack(dilated_list, dim=0).unsqueeze(1).to(device)
        return dilated_mask

    @staticmethod
    def extract_boundary(mask, kernel_size=5):
        """
        提取边界区域

        Args:
            mask: 输入掩码 [B, 1, H, W]
            kernel_size: 腐蚀核大小

        Returns:
            boundary_mask: 边界掩码 [B, 1, H, W]
        """
        eroded = MaskOps.erode_mask(mask, kernel_size)
        boundary = mask - eroded
        return boundary

    @staticmethod
    def compute_mask_area(mask):
        """
        计算掩码面积

        Args:
            mask: 输入掩码 [B, 1, H, W]

        Returns:
            area: 面积 [B]
        """
        return mask.sum(dim=[1, 2, 3])

    @staticmethod
    def mask_iou(mask_a, mask_b):
        """
        计算掩码IoU

        Args:
            mask_a: 掩码A [B, 1, H, W]
            mask_b: 掩码B [B, 1, H, W]

        Returns:
            iou: IoU [B]
        """
        intersection = (mask_a * mask_b).sum(dim=[1, 2, 3])
        union = (mask_a + mask_b - mask_a * mask_b).sum(dim=[1, 2, 3])
        iou = intersection / (union + 1e-8)
        return iou
