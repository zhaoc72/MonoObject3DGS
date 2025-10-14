"""
Depth Refiner
深度图优化和时序一致性
"""

import numpy as np
import cv2
from typing import Optional
from collections import deque


class DepthRefiner:
    """深度图优化器"""
    
    def __init__(
        self,
        bilateral_filter: bool = True,
        edge_preserving: bool = True,
        remove_outliers: bool = True
    ):
        self.bilateral_filter = bilateral_filter
        self.edge_preserving = edge_preserving
        self.remove_outliers = remove_outliers
        
        print("✓ DepthRefiner initialized")
    
    def refine(
        self,
        depth: np.ndarray,
        image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        优化深度图
        
        Args:
            depth: 深度图 (H, W)
            image: RGB图像 (H, W, 3)
            mask: 有效区域mask (H, W)
            
        Returns:
            refined_depth: 优化后的深度
        """
        refined = depth.copy()
        
        # 去除异常值
        if self.remove_outliers:
            refined = self._remove_outliers(refined, mask)
        
        # 双边滤波
        if self.bilateral_filter:
            refined = self._bilateral_filter(refined, image)
        
        # 边缘保持平滑
        if self.edge_preserving and image is not None:
            refined = self._guided_filter(refined, image)
        
        return refined
    
    def _remove_outliers(
        self,
        depth: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """去除异常值"""
        window_size = 5
        local_mean = cv2.blur(depth, (window_size, window_size))
        local_std = np.sqrt(
            cv2.blur(depth**2, (window_size, window_size)) - local_mean**2
        )
        
        outlier_mask = np.abs(depth - local_mean) > 3 * (local_std + 1e-6)
        
        if mask is not None:
            outlier_mask = outlier_mask & mask
        
        refined = depth.copy()
        refined[outlier_mask] = local_mean[outlier_mask]
        
        return refined
    
    def _bilateral_filter(
        self,
        depth: np.ndarray,
        image: Optional[np.ndarray] = None,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75
    ) -> np.ndarray:
        """双边滤波"""
        depth_normalized = (
            (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
        ).astype(np.uint8)
        
        if image is not None:
            # 联合双边滤波
            try:
                filtered = cv2.ximgproc.jointBilateralFilter(
                    image.astype(np.uint8),
                    depth_normalized,
                    d=d,
                    sigmaColor=sigma_color,
                    sigmaSpace=sigma_space
                )
            except:
                # 回退到标准双边滤波
                filtered = cv2.bilateralFilter(
                    depth_normalized,
                    d=d,
                    sigmaColor=sigma_color,
                    sigmaSpace=sigma_space
                )
        else:
            filtered = cv2.bilateralFilter(
                depth_normalized,
                d=d,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )
        
        refined = (
            filtered.astype(np.float32) / 255.0 * 
            (depth.max() - depth.min()) + depth.min()
        )
        
        return refined
    
    def _guided_filter(
        self,
        depth: np.ndarray,
        guide: np.ndarray,
        radius: int = 8,
        eps: float = 1e-4
    ) -> np.ndarray:
        """导向滤波"""
        if len(guide.shape) == 3:
            guide_gray = cv2.cvtColor(
                guide.astype(np.uint8), 
                cv2.COLOR_RGB2GRAY
            ).astype(np.float32) / 255.0
        else:
            guide_gray = guide.astype(np.float32)
        
        mean_I = cv2.boxFilter(guide_gray, cv2.CV_32F, (radius, radius))
        mean_p = cv2.boxFilter(depth, cv2.CV_32F, (radius, radius))
        mean_Ip = cv2.boxFilter(guide_gray * depth, cv2.CV_32F, (radius, radius))
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = cv2.boxFilter(guide_gray * guide_gray, cv2.CV_32F, (radius, radius))
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
        
        refined = mean_a * guide_gray + mean_b
        
        return refined


class DepthConsistencyRefiner:
    """时序深度一致性优化"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.depth_history = deque(maxlen=window_size)
        
    def add_frame(self, depth: np.ndarray):
        """添加新帧"""
        self.depth_history.append(depth)
    
    def refine_temporal(
        self,
        current_depth: np.ndarray,
        flow: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """时序优化"""
        if len(self.depth_history) == 0:
            return current_depth
        
        # 简单平均（可以改进为加权或卡尔曼滤波）
        refined = current_depth.copy()
        
        for prev_depth in list(self.depth_history)[-3:]:
            refined = 0.7 * refined + 0.3 * prev_depth
        
        return refined


# 测试
if __name__ == "__main__":
    refiner = DepthRefiner()
    
    # 测试数据
    H, W = 480, 640
    depth = np.random.rand(H, W) * 5 + 2
    depth += np.random.randn(H, W) * 0.5  # 添加噪声
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
    print(f"Original depth range: [{depth.min():.2f}, {depth.max():.2f}]")
    
    refined = refiner.refine(depth, image)
    
    print(f"Refined depth range: [{refined.min():.2f}, {refined.max():.2f}]")