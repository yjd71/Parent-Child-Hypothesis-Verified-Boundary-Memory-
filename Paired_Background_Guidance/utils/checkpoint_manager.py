"""
Prototype Checkpoint Manager
原型库checkpoint管理
"""

import torch
import os
from pathlib import Path


class PrototypeCheckpointManager:
    """
    原型库checkpoint管理器

    功能：
    1. 保存/加载原型库
    2. 支持断点续训
    3. 原型库版本管理
    """

    def __init__(self, checkpoint_dir='checkpoints/prototypes'):
        """
        Args:
            checkpoint_dir: checkpoint保存目录
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, prototype_bank, epoch, step=None, prefix='prototype'):
        """
        保存原型库

        Args:
            prototype_bank: 原型库模块
            epoch: 当前epoch
            step: 当前step（可选）
            prefix: 文件名前缀
        """
        if step is not None:
            filename = f"{prefix}_epoch{epoch}_step{step}.pth"
        else:
            filename = f"{prefix}_epoch{epoch}.pth"

        filepath = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'step': step,
            'P_texture_mean': prototype_bank.P_texture_mean,
            'P_texture_std': prototype_bank.P_texture_std,
            'P_bg': prototype_bank.P_bg,
            'P_fg': prototype_bank.P_fg,
            'P_edge': prototype_bank.P_edge,
            'counts': {
                'texture': prototype_bank.texture_count,
                'bg': prototype_bank.bg_count,
                'fg': prototype_bank.fg_count,
                'edge': prototype_bank.edge_count
            }
        }

        torch.save(checkpoint, filepath)
        print(f"Prototype bank saved to {filepath}")

        # 保存最新的checkpoint链接
        latest_path = self.checkpoint_dir / f"{prefix}_latest.pth"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(filename)

    def load(self, prototype_bank, checkpoint_path=None, load_latest=False, prefix='prototype'):
        """
        加载原型库

        Args:
            prototype_bank: 原型库模块
            checkpoint_path: checkpoint路径（可选）
            load_latest: 是否加载最新的checkpoint
            prefix: 文件名前缀

        Returns:
            epoch: 加载的epoch
            step: 加载的step
        """
        if load_latest:
            checkpoint_path = self.checkpoint_dir / f"{prefix}_latest.pth"
        elif checkpoint_path is None:
            raise ValueError("Must specify checkpoint_path or set load_latest=True")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)

        # 加载原型
        prototype_bank.P_texture_mean.copy_(checkpoint['P_texture_mean'])
        prototype_bank.P_texture_std.copy_(checkpoint['P_texture_std'])
        prototype_bank.P_bg.copy_(checkpoint['P_bg'])
        prototype_bank.P_fg.copy_(checkpoint['P_fg'])
        prototype_bank.P_edge.copy_(checkpoint['P_edge'])

        # 加载计数器
        if 'counts' in checkpoint:
            prototype_bank.texture_count.copy_(checkpoint['counts']['texture'])
            prototype_bank.bg_count.copy_(checkpoint['counts']['bg'])
            prototype_bank.fg_count.copy_(checkpoint['counts']['fg'])
            prototype_bank.edge_count.copy_(checkpoint['counts']['edge'])

        epoch = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', None)

        print(f"Prototype bank loaded from {checkpoint_path} (epoch={epoch}, step={step})")

        return epoch, step

    def list_checkpoints(self, prefix='prototype'):
        """
        列出所有checkpoint

        Args:
            prefix: 文件名前缀

        Returns:
            checkpoints: checkpoint文件列表
        """
        pattern = f"{prefix}_epoch*.pth"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        return checkpoints

    def remove_old_checkpoints(self, keep_last_n=5, prefix='prototype'):
        """
        删除旧的checkpoint，只保留最近的N个

        Args:
            keep_last_n: 保留的checkpoint数量
            prefix: 文件名前缀
        """
        checkpoints = self.list_checkpoints(prefix)

        if len(checkpoints) > keep_last_n:
            for ckpt in checkpoints[:-keep_last_n]:
                ckpt.unlink()
                print(f"Removed old checkpoint: {ckpt}")
