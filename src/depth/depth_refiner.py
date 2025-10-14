"""
Depth Refiner
深度图优化和后处理
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from scipy.ndimage import median_filter, gaussian_filter


class DepthRefiner:
    """
    深度图优化器
    包含边缘保持滤波、异常值去除等功能
    """
    
    def __init__(
        self,
        bilateral_filter: bool = True,
        edge_preserving: bool = True,
        remove_outliers: bool = True
    ):
        """
        Args:
            bilateral_filter: 是否使用双边滤波
            edge_preserving: 是否进行边缘保持平滑
            remove_outliers: 是否去除异常值
        """
        self.bilateral_filter = bilateral_filter
        self.edge_preserving = edge_preserving
        self.remove_outliers = remove_outliers
        
        print("✓ DepthRefiner初始化完成")
    
    def refine(
        self,
        depth: np.ndarray,
        image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        优化深度图
        
        Args:
            depth: 原始深度图 (H, W)
            image: RGB图像 (H, W, 3)，用于引导滤波
            mask: 有效区域mask (H, W)
            
        Returns:
            refined_depth: 优化后的深度图 (H, W)
        """
        refined = depth.copy()
        
        # 1. 去除异常值
        if self.remove_outliers:
            refined = self._remove_outliers(refined, mask)
        
        # 2. 双边滤波（边缘保持）
        if self.bilateral_filter:
            refined = self._bilateral_filter(refined, image)
        
        # 3. 边缘保持平滑
        if self.edge_preserving and image is not None:
            refined = self._guided_filter(refined, image)
        
        # 4. 填充空洞
        if mask is not None:
            refined = self._fill_holes(refined, mask)
        
        return refined
    
    def _remove_outliers(
        self,
        depth: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        去除深度异常值
        使用统计方法检测和修正异常值
        """
        # 计算局部统计
        window_size = 5
        local_mean = cv2.blur(depth, (window_size, window_size))
        local_std = np.sqrt(cv2.blur(depth**2, (window_size, window_size)) - local_mean**2)
        
        # 检测异常值（超过3倍标准差）
        outlier_mask = np.abs(depth - local_mean) > 3 * (local_std + 1e-6)
        
        if mask is not None:
            outlier_mask = outlier_mask & mask
        
        # 用局部均值替换异常值
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
        """
        双边滤波（保持边缘的平滑）
        
        Args:
            depth: 深度图
            image: RGB图像（如果提供，使用联合双边滤波）
            d: 滤波器直径
            sigma_color: 颜色空间标准差
            sigma_space: 坐标空间标准差
        """
        # 归一化深度到0-255
        depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
        
        if image is not None:
            # 联合双边滤波（使用图像引导）
            filtered = cv2.ximgproc.jointBilateralFilter(
                image.astype(np.uint8),
                depth_normalized,
                d=d,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )
        else:
            # 标准双边滤波
            filtered = cv2.bilateralFilter(
                depth_normalized,
                d=d,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )
        
        # 反归一化
        refined = filtered.astype(np.float32) / 255.0 * (depth.max() - depth.min()) + depth.min()
        
        return refined
    
    def _guided_filter(
        self,
        depth: np.ndarray,
        guide: np.ndarray,
        radius: int = 8,
        eps: float = 1e-4
    ) -> np.ndarray:
        """
        导向滤波（边缘保持）
        
        Args:
            depth: 深度图
            guide: 引导图像（通常是RGB图像）
            radius: 滤波半径
            eps: 正则化参数
        """
        # 转换引导图像为灰度
        if len(guide.shape) == 3:
            guide_gray = cv2.cvtColor(guide.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            guide_gray = guide.astype(np.float32)
        
        # 实现导向滤波
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
    
    def _fill_holes(
        self,
        depth: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        填充深度图中的空洞
        
        Args:
            depth: 深度图
            mask: 有效区域mask
        """
        # 创建无效区域mask
        invalid_mask = ~mask
        
        if not invalid_mask.any():
            return depth
        
        # 使用inpainting填充
        depth_uint8 = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
        filled = cv2.inpaint(
            depth_uint8,
            invalid_mask.astype(np.uint8),
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )
        
        # 反归一化
        filled = filled.astype(np.float32) / 255.0 * (depth.max() - depth.min()) + depth.min()
        
        return filled
    
    def compute_depth_gradients(
        self,
        depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算深度梯度
        
        Args:
            depth: 深度图
            
        Returns:
            grad_x: x方向梯度
            grad_y: y方向梯度
        """
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        
        return grad_x, grad_y
    
    def detect_depth_edges(
        self,
        depth: np.ndarray,
        threshold: float = 0.1
    ) -> np.ndarray:
        """
        检测深度不连续边缘
        
        Args:
            depth: 深度图
            threshold: 梯度阈值
            
        Returns:
            edges: 边缘mask
        """
        grad_x, grad_y = self.compute_depth_gradients(depth)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 归一化
        grad_magnitude = grad_magnitude / (grad_magnitude.max() + 1e-8)
        
        # 阈值化
        edges = grad_magnitude > threshold
        
        return edges
    
    def align_depth_to_edges(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        edge_threshold: float = 0.1
    ) -> np.ndarray:
        """
        将深度边缘对齐到图像边缘
        
        Args:
            depth: 深度图
            image: RGB图像
            edge_threshold: 边缘阈值
            
        Returns:
            aligned_depth: 对齐后的深度图
        """
        # 检测图像边缘
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        image_edges = cv2.Canny(gray, 50, 150)
        
        # 检测深度边缘
        depth_edges = self.detect_depth_edges(depth, edge_threshold)
        
        # 在图像边缘处加强深度不连续
        # 这里使用简单的策略：在图像边缘处增加深度梯度
        aligned = depth.copy()
        
        # 膨胀图像边缘
        kernel = np.ones((3, 3), np.uint8)
        image_edges_dilated = cv2.dilate(image_edges, kernel, iterations=1)
        
        # TODO: 实现更复杂的边缘对齐策略
        
        return aligned


class DepthConsistencyRefiner:
    """
    多帧深度一致性优化器
    利用时序信息优化深度估计
    """
    
    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: 时间窗口大小
        """
        self.window_size = window_size
        self.depth_history = []
        
    def add_frame(self, depth: np.ndarray):
        """添加新的深度帧"""
        self.depth_history.append(depth)
        
        # 保持窗口大小
        if len(self.depth_history) > self.window_size:
            self.depth_history.pop(0)
    
    def refine_temporal(
        self,
        current_depth: np.ndarray,
        flow: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        时序优化当前深度图
        
        Args:
            current_depth: 当前帧深度
            flow: 光流（如果可用）
            
        Returns:
            refined_depth: 优化后的深度
        """
        if len(self.depth_history) == 0:
            return current_depth
        
        # 简单的时序平均（可以改进为加权平均或卡尔曼滤波）
        refined = current_depth.copy()
        
        for prev_depth in self.depth_history[-3:]:  # 使用最近3帧
            # 如果有光流，可以warp前一帧深度
            if flow is not None:
                # TODO: 实现基于光流的深度warp
                pass
            
            # 简单平均
            refined = 0.7 * refined + 0.3 * prev_depth
        
        return refined


if __name__ == "__main__":
    # 测试代码
    print("=== 测试DepthRefiner ===")
    
    # 创建测试数据
    H, W = 480, 640
    
    # 生成带噪声的深度图
    depth = np.random.rand(H, W) * 5 + 2  # 2-7米
    depth += np.random.randn(H, W) * 0.5  # 添加噪声
    
    # 添加一些异常值
    outlier_mask = np.random.rand(H, W) < 0.05
    depth[outlier_mask] = np.random.rand(outlier_mask.sum()) * 20
    
    # 创建测试图像
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
    # 创建mask
    mask = np.ones((H, W), dtype=bool)
    mask[100:150, 200:250] = False  # 创建一些空洞
    
    print(f"原始深度范围: [{depth.min():.2f}, {depth.max():.2f}]")
    
    # 测试优化
    refiner = DepthRefiner(
        bilateral_filter=True,
        edge_preserving=True,
        remove_outliers=True
    )
    
    refined_depth = refiner.refine(depth, image, mask)
    
    print(f"优化后深度范围: [{refined_depth.min():.2f}, {refined_depth.max():.2f}]")
    
    # 测试边缘检测
    edges = refiner.detect_depth_edges(refined_depth)
    print(f"检测到 {edges.sum()} 个边缘像素")
    
    # 测试时序优化
    print("\n测试时序优化:")
    temporal_refiner = DepthConsistencyRefiner(window_size=5)
    
    for i in range(5):
        frame_depth = depth + np.random.randn(H, W) * 0.1
        temporal_refiner.add_frame(frame_depth)
    
    temporal_refined = temporal_refiner.refine_temporal(depth)
    print(f"时序优化后深度范围: [{temporal_refined.min():.2f}, {temporal_refined.max():.2f}]")
    
    print("\n测试完成！")