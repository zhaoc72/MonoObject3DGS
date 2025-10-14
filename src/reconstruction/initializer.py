"""
Gaussian Initializer
Gaussian初始化器
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


class GaussianInitializer:
    """Gaussian初始化器"""
    
    def __init__(self):
        print("✓ GaussianInitializer initialized")
    
    def initialize_from_depth(
        self,
        depth_map: np.ndarray,
        image: np.ndarray,
        camera_params: Dict,
        mask: Optional[np.ndarray] = None,
        subsample_factor: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从深度图初始化
        
        Returns:
            points, colors, scales
        """
        H, W = depth_map.shape
        
        # 创建像素网格
        ys, xs = np.meshgrid(
            np.arange(0, H, subsample_factor),
            np.arange(0, W, subsample_factor),
            indexing='ij'
        )
        ys = ys.flatten()
        xs = xs.flatten()
        
        # 应用mask
        if mask is not None:
            mask_sampled = mask[::subsample_factor, ::subsample_factor].flatten()
            valid = mask_sampled
            xs = xs[valid]
            ys = ys[valid]
        
        if len(xs) == 0:
            raise ValueError("No valid points after masking")
        
        # 获取深度和颜色
        depths = depth_map[ys