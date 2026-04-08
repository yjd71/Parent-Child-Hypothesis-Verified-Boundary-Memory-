"""
LFGM Visualizer
可视化工具
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class LFGMVisualizer:
    """
    LFGM可视化工具

    功能：
    1. 可视化原型分布
    2. 可视化重构误差
    3. 可视化异常图
    4. 可视化注意力权重
    """

    def __init__(self, save_dir='visualizations'):
        """
        Args:
            save_dir: 可视化结果保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def visualize_prototypes(self, prototypes, labels=None, title='Prototype Distribution'):
        """
        可视化原型分布（使用t-SNE或PCA）

        Args:
            prototypes: 原型 [N, D]
            labels: 原型标签 [N]
            title: 图标题
        """
        # TODO: 实现t-SNE可视化
        pass

    def visualize_reconstruction_error(self, original, reconstructed, save_name='recon_error.png'):
        """
        可视化重构误差

        Args:
            original: 原始特征 [B, D, H, W]
            reconstructed: 重构特征 [B, D, H, W]
            save_name: 保存文件名
        """
        # 计算L2误差
        error = torch.norm(original - reconstructed, p=2, dim=1, keepdim=True)  # [B, 1, H, W]

        # 归一化到[0, 1]
        error_min = error.min()
        error_max = error.max()
        error_norm = (error - error_min) / (error_max - error_min + 1e-8)

        # 转换为numpy
        error_np = error_norm[0, 0].cpu().numpy()

        # 绘制热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(error_np, cmap='hot')
        plt.colorbar(label='Reconstruction Error')
        plt.title('Reconstruction Error Map')
        plt.axis('off')

        save_path = self.save_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"Reconstruction error visualization saved to {save_path}")

    def visualize_anomaly_map(self, anomaly_features, save_name='anomaly_map.png'):
        """
        可视化异常图

        Args:
            anomaly_features: 异常特征 [B, D, H, W]
            save_name: 保存文件名
        """
        # 计算异常强度（L2范数）
        anomaly_intensity = torch.norm(anomaly_features, p=2, dim=1, keepdim=True)  # [B, 1, H, W]

        # 归一化
        intensity_min = anomaly_intensity.min()
        intensity_max = anomaly_intensity.max()
        intensity_norm = (anomaly_intensity - intensity_min) / (intensity_max - intensity_min + 1e-8)

        # 转换为numpy
        intensity_np = intensity_norm[0, 0].cpu().numpy()

        # 绘制热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(intensity_np, cmap='viridis')
        plt.colorbar(label='Anomaly Intensity')
        plt.title('Anomaly Detection Map')
        plt.axis('off')

        save_path = self.save_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"Anomaly map visualization saved to {save_path}")

    def visualize_attention_weights(self, attention_weights, save_name='attention.png'):
        """
        可视化注意力权重

        Args:
            attention_weights: 注意力权重 [B, H, W, N]
            save_name: 保存文件名
        """
        # 取第一个样本
        attn = attention_weights[0].cpu().numpy()  # [H, W, N]

        # 对每个位置，显示最大注意力权重
        max_attn = attn.max(axis=-1)  # [H, W]

        # 绘制热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(max_attn, cmap='coolwarm')
        plt.colorbar(label='Max Attention Weight')
        plt.title('Attention Weight Map')
        plt.axis('off')

        save_path = self.save_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"Attention visualization saved to {save_path}")

    def visualize_comparison(self, image, gt_mask, pred_mask, anomaly_map, save_name='comparison.png'):
        """
        可视化对比图（原图、GT、预测、异常图）

        Args:
            image: 原始图像 [3, H, W]
            gt_mask: GT掩码 [1, H, W]
            pred_mask: 预测掩码 [1, H, W]
            anomaly_map: 异常图 [1, H, W]
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 原始图像
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        img_np - img_np.min()) / (img_np.max() - img_np.min())
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # GT掩码
        gt_np = gt_mask[0].cpu().numpy()
        axes[1].imshow(gt_np, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        # 预测掩码
        pred_np = pred_mask[0].cpu().numpy()
        axes[2].imshow(pred_np, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        # 异常图
        anomaly_np = anomaly_map[0].cpu().numpy()
        im = axes[3].imshow(anomaly_np, cmap='hot')
        axes[3].set_title('Anomaly Map')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3])

        save_path = self.save_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"Comparison visualization saved to {save_path}")
