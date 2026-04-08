"""
LFGM v3.0: Label Feature Guidance Module
基于DINOv3权威特征和异常检测范式的半监督伪装目标检测模块
"""

from .lfgm_core import LFGMCore

__version__ = '3.0.0'
__all__ = ['LFGMCore']
