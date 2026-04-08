"""
Quad-GDPB: 四元动态原型库
管理四类原型：纹理、背景、前景、边界
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class QuadGDPB(nn.Module):
    """
    四元动态原型库
    
    四类原型：
    - P_texture: 浅层纹理统计量（均值、标准差）
    - P_bg: 背景环境原型
    - P_fg: 前景语义原型
    - P_edge: 边界不确定原型
    """
    
    def __init__(self, config: Dict):
        super(QuadGDPB, self).__init__()
        
        self.config = config
        
        # 原型库容量
        self.texture_size = config.get('texture_size', 256)
        self.bg_size = config.get('bg_size', 512)
        self.fg_size = config.get('fg_size', 256)
        self.edge_size = config.get('edge_size', 128)
        
        # 特征维度
        self.feature_dim = config.get('feature_dim', 768)
        
        # 初始化原型库（使用nn.Parameter以便保存到checkpoint）
        self.register_buffer('P_texture_mean', torch.zeros(self.texture_size, self.feature_dim))
        self.register_buffer('P_texture_std', torch.ones(self.texture_size, self.feature_dim))
        self.register_buffer('P_bg', torch.randn(self.bg_size, self.feature_dim))
        self.register_buffer('P_fg', torch.randn(self.fg_size, self.feature_dim))
        self.register_buffer('P_edge', torch.randn(self.edge_size, self.feature_dim))
        
        # 原型库计数器（用于跟踪更新次数）
        self.register_buffer('texture_count', torch.zeros(1))
        self.register_buffer('bg_count', torch.zeros(1))
        self.register_buffer('fg_count', torch.zeros(1))
        self.register_buffer('edge_count', torch.zeros(1))
        
    def update(self, dinov3_features: Dict[str, torch.Tensor], gt_mask: torch.Tensor):
        """
        使用DINOv3特征和GT mask更新原型库
        
        Args:
            dinov3_features: DINOv3提取的特征字典
                - 'shallow': 浅层特征 [B, C, H, W]
                - 'deep': 深层特征 [B, C, H, W]
            gt_mask: Ground truth mask [B, 1, H, W]
        """
        # TODO: 实现原型库更新逻辑
        # 1. 掩码形态学处理（腐蚀、边界提取）
        # 2. 使用采样策略提取原型
        # 3. 更新四个原型库
        pass
    
    def get_texture_prototypes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回纹理原型（均值和标准差）"""
        return self.P_texture_mean, self.P_texture_std
    
    def get_background_prototypes(self) -> torch.Tensor:
        """返回背景原型"""
        return self.P_bg
    
    def get_foreground_prototypes(self) -> torch.Tensor:
        """返回前景原型"""
        return self.P_fg
    
    def get_edge_prototypes(self) -> torch.Tensor:
        """返回边界原型"""
        return self.P_edge
    
    def save(self, path: str):
        """保存原型库"""
        torch.save({
            'P_texture_mean': self.P_texture_mean,
            'P_texture_std': self.P_texture_std,
            'P_bg': self.P_bg,
            'P_fg': self.P_fg,
            'P_edge': self.P_edge,
            'counts': {
                'texture': self.texture_count,
                'bg': self.bg_count,
                'fg': self.fg_count,
                'edge': self.edge_count
            }
        }, path)
    
    def load(self, path: str):
        """加载原型库"""
        checkpoint = torch.load(path)
        self.P_texture_mean.copy_(checkpoint['P_texture_mean'])
        self.P_texture_std.copy_(checkpoint['P_texture_std'])
        self.P_bg.copy_(checkpoint['P_bg'])
        self.P_fg.copy_(checkpoint['P_fg'])
        self.P_edge.copy_(checkpoint['P_edge'])
        
        if 'counts' in checkpoint:
            self.texture_count.copy_(checkpoint['counts']['texture'])
            self.bg_count.copy_(checkpoint['counts']['bg'])
            self.fg_count.copy_(checkpoint['counts']['fg'])
            self.edge_count.copy_(checkpoint['counts']['edge'])
