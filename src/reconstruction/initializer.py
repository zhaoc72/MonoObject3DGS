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
        depths = depth_map[ys, xs]
        colors = image[ys, xs] / 255.0
        
        # 反投影到3D
        fx = camera_params['fx']
        fy = camera_params['fy']
        cx = camera_params['cx']
        cy = camera_params['cy']
        
        points_3d = np.zeros((len(xs), 3), dtype=np.float32)
        points_3d[:, 0] = (xs - cx) * depths / fx
        points_3d[:, 1] = (ys - cy) * depths / fy
        points_3d[:, 2] = depths
        
        # 估计尺度
        scales = self._estimate_point_scales(points_3d, subsample_factor, fx, fy)
        
        # 转换为tensor
        points = torch.from_numpy(points_3d)
        colors = torch.from_numpy(colors.astype(np.float32))
        scales = torch.from_numpy(scales.astype(np.float32))
        
        return points, colors, scales
    
    def _estimate_point_scales(
        self,
        points: np.ndarray,
        subsample_factor: int,
        fx: float,
        fy: float
    ) -> np.ndarray:
        """估计点尺度"""
        # 基于深度和采样因子估计
        depths = points[:, 2]
        
        # 像素间距在3D空间中的投影
        pixel_size_x = depths / fx * subsample_factor
        pixel_size_y = depths / fy * subsample_factor
        
        scales = np.stack([pixel_size_x, pixel_size_y, 
                          (pixel_size_x + pixel_size_y) / 2], axis=1)
        
        # 限制最小尺度
        scales = np.maximum(scales, 0.001)
        
        return scales
    
    def initialize_from_pointcloud(
        self,
        pointcloud: np.ndarray,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从点云初始化
        
        Returns:
            points, colors, scales
        """
        points = torch.from_numpy(pointcloud.astype(np.float32))
        
        if colors is None:
            colors = torch.ones_like(points) * 0.5
        else:
            colors = torch.from_numpy(colors.astype(np.float32))
        
        # 估计尺度（k近邻）
        scales = self._estimate_scales_knn(pointcloud, k=8)
        scales = torch.from_numpy(scales.astype(np.float32))
        
        return points, colors, scales
    
    def _estimate_scales_knn(
        self,
        points: np.ndarray,
        k: int = 8
    ) -> np.ndarray:
        """基于k近邻估计尺度"""
        try:
            from scipy.spatial import cKDTree
            
            tree = cKDTree(points)
            distances, _ = tree.query(points, k=k+1)
            
            # 使用平均距离作为尺度
            mean_distances = distances[:, 1:].mean(axis=1)
            
            scales = np.stack([mean_distances] * 3, axis=1)
            scales = np.maximum(scales, 0.001)
            
            return scales
        
        except ImportError:
            # 回退到固定尺度
            return np.ones((len(points), 3), dtype=np.float32) * 0.01


# 测试
if __name__ == "__main__":
    initializer = GaussianInitializer()
    
    # 测试深度初始化
    H, W = 480, 640
    depth = np.random.rand(H, W) * 5 + 2
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
    camera_params = {
        'fx': 525.0, 'fy': 525.0,
        'cx': 320.0, 'cy': 240.0
    }
    
    points, colors, scales = initializer.initialize_from_depth(
        depth, image, camera_params, subsample_factor=4
    )
    
    print(f"Initialized {len(points)} points")
    print(f"Points range: [{points.min():.2f}, {points.max():.2f}]")
    print(f"Scales range: [{scales.min():.4f}, {scales.max():.4f}]")