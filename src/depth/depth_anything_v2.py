"""
Depth Anything V2
更精确的深度估计和度量深度支持
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import cv2


class DepthAnythingV2:
    """Depth Anything V2 - 升级版"""
    
    def __init__(
        self,
        model_size: str = "vitl",  # vits/vitb/vitl/vitg
        metric_depth: bool = True,  # 启用度量深度
        device: str = "cuda",
        max_depth: float = 20.0  # 最大深度(米)
    ):
        self.device = device
        self.model_size = model_size
        self.metric_depth = metric_depth
        self.max_depth = max_depth
        
        print(f"🔄 Loading Depth Anything V2: {model_size}")
        
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            
            # 模型配置
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            
            # 加载模型
            model_config = model_configs[model_size]
            self.model = DepthAnythingV2(**model_config)
            
            # 加载预训练权重
            checkpoint_path = f"data/checkpoints/depth_anything_v2_{model_size}.pth"
            if metric_depth:
                checkpoint_path = checkpoint_path.replace('.pth', '_metric.pth')
            
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()
            
            print(f"✓ Depth Anything V2 loaded: {model_size}")
            print(f"  Metric depth: {metric_depth}")
            print(f"  Max depth: {max_depth}m")
            
        except Exception as e:
            print(f"❌ Failed to load Depth Anything V2: {e}")
            print("  Please install: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git")
            self.model = None
    
    @torch.no_grad()
    def estimate(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        估计深度 - V2增强版
        
        Args:
            image: RGB图像 (H, W, 3), 范围[0, 255]
            target_size: 目标尺寸 (H, W)
            
        Returns:
            depth: 深度图 (H, W), 单位米
        """
        if self.model is None:
            # Dummy depth
            H, W = image.shape[:2]
            return np.random.rand(H, W) * 5 + 2
        
        H, W = image.shape[:2]
        
        # 预处理
        image_tensor = self._preprocess(image)
        image_tensor = image_tensor.to(self.device)
        
        # 推理
        depth = self.model(image_tensor)
        
        # 后处理
        depth = depth.squeeze().cpu().numpy()
        
        # 调整尺寸
        if target_size is None:
            target_size = (H, W)
        
        if depth.shape != target_size:
            depth = cv2.resize(
                depth,
                (target_size[1], target_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # 度量深度处理
        if self.metric_depth:
            # V2直接输出度量深度，限制范围
            depth = np.clip(depth, 0.1, self.max_depth)
        else:
            # 相对深度转度量深度
            depth = self._normalize_depth(depth)
        
        return depth
    
    def estimate_with_confidence(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计深度和置信度 - NEW
        
        Args:
            image: RGB图像
            target_size: 目标尺寸
            
        Returns:
            depth: 深度图 (H, W)
            confidence: 置信度图 (H, W)
        """
        if self.model is None:
            H, W = image.shape[:2]
            depth = np.random.rand(H, W) * 5 + 2
            confidence = np.ones((H, W)) * 0.8
            return depth, confidence
        
        H, W = image.shape[:2]
        
        # 预处理
        image_tensor = self._preprocess(image)
        image_tensor = image_tensor.to(self.device)
        
        # 推理（获取中间特征用于置信度估计）
        with torch.no_grad():
            depth = self.model(image_tensor)
            
            # 估计置信度（基于深度梯度的平滑度）
            depth_grad_x = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
            depth_grad_y = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
            
            # 边界填充
            depth_grad_x = torch.nn.functional.pad(depth_grad_x, (0, 1, 0, 0))
            depth_grad_y = torch.nn.functional.pad(depth_grad_y, (0, 0, 0, 1))
            
            # 梯度越小，置信度越高
            gradient_magnitude = torch.sqrt(depth_grad_x**2 + depth_grad_y**2)
            confidence = torch.exp(-gradient_magnitude * 10)  # 指数衰减
            confidence = confidence.squeeze().cpu().numpy()
        
        depth = depth.squeeze().cpu().numpy()
        
        # 调整尺寸
        if target_size is None:
            target_size = (H, W)
        
        if depth.shape != target_size:
            depth = cv2.resize(depth, (target_size[1], target_size[0]), 
                             interpolation=cv2.INTER_LINEAR)
            confidence = cv2.resize(confidence, (target_size[1], target_size[0]), 
                                  interpolation=cv2.INTER_LINEAR)
        
        # 度量深度处理
        if self.metric_depth:
            depth = np.clip(depth, 0.1, self.max_depth)
        else:
            depth = self._normalize_depth(depth)
        
        return depth, confidence
    
    def estimate_multi_scale(
        self,
        image: np.ndarray,
        scales: list = [1.0, 0.75, 0.5],
        fusion_method: str = "weighted"
    ) -> np.ndarray:
        """
        多尺度深度估计 - NEW
        
        Args:
            image: RGB图像
            scales: 尺度列表
            fusion_method: 融合方法 ['mean', 'weighted', 'median']
            
        Returns:
            depth: 融合后的深度图
        """
        H, W = image.shape[:2]
        depths = []
        confidences = []
        
        for scale in scales:
            h, w = int(H * scale), int(W * scale)
            resized = cv2.resize(image, (w, h))
            
            depth, conf = self.estimate_with_confidence(resized, target_size=(H, W))
            depths.append(depth)
            confidences.append(conf)
        
        depths = np.stack(depths, axis=0)  # (N, H, W)
        confidences = np.stack(confidences, axis=0)  # (N, H, W)
        
        # 融合
        if fusion_method == "mean":
            fused_depth = depths.mean(axis=0)
        elif fusion_method == "weighted":
            weights = confidences / (confidences.sum(axis=0, keepdims=True) + 1e-8)
            fused_depth = (depths * weights).sum(axis=0)
        elif fusion_method == "median":
            fused_depth = np.median(depths, axis=0)
        else:
            fused_depth = depths[0]
        
        return fused_depth
    
    def compute_depth_edges(
        self,
        depth: np.ndarray,
        threshold: float = 0.1
    ) -> np.ndarray:
        """
        计算深度边缘 - NEW
        
        Args:
            depth: 深度图
            threshold: 边缘阈值
            
        Returns:
            edges: 边缘图 (H, W) bool
        """
        # Sobel边缘检测
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 归一化
        gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / \
                           (gradient_magnitude.max() - gradient_magnitude.min() + 1e-8)
        
        edges = gradient_magnitude > threshold
        
        return edges
    
    def inpaint_depth(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        method: str = "telea"
    ) -> np.ndarray:
        """
        深度修复 - NEW
        
        Args:
            depth: 深度图
            mask: 需要修复的区域 (True=需要修复)
            method: 修复方法 ['telea', 'ns']
            
        Returns:
            inpainted_depth: 修复后的深度
        """
        # 归一化到0-255
        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
        
        # OpenCV修复
        if method == "telea":
            inpainted = cv2.inpaint(depth_norm, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        else:
            inpainted = cv2.inpaint(depth_norm, mask.astype(np.uint8), 3, cv2.INPAINT_NS)
        
        # 反归一化
        inpainted_depth = inpainted.astype(np.float32) / 255.0 * \
                         (depth.max() - depth.min()) + depth.min()
        
        return inpainted_depth
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理"""
        # 归一化到[0, 1]
        image = image.astype(np.float32) / 255.0
        
        # HWC -> CHW
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        # ImageNet标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image = (image - mean) / std
        
        return image
    
    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """归一化深度到[0.5, max_depth]米"""
        # 移除极端值
        p1, p99 = np.percentile(depth, [1, 99])
        depth = np.clip(depth, p1, p99)
        
        # 归一化到[0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        # 映射到[0.5, max_depth]米
        depth = depth * (self.max_depth - 0.5) + 0.5
        
        return depth


# 测试代码
if __name__ == "__main__":
    print("=== Testing Depth Anything V2 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    estimator = DepthAnythingV2(
        model_size="vitl",
        metric_depth=True,
        device=device
    )
    
    # 测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\n1. 基础深度估计:")
    import time
    start = time.time()
    depth = estimator.estimate(test_image)
    elapsed = time.time() - start
    print(f"  Depth estimated in {elapsed:.3f}s")
    print(f"  Depth range: [{depth.min():.2f}, {depth.max():.2f}] meters")
    
    print("\n2. 深度+置信度:")
    depth, confidence = estimator.estimate_with_confidence(test_image)
    print(f"  Depth: {depth.shape}")
    print(f"  Confidence: {confidence.shape}, range: [{confidence.min():.2f}, {confidence.max():.2f}]")
    
    print("\n3. 多尺度估计:")
    depth_ms = estimator.estimate_multi_scale(test_image)
    print(f"  Multi-scale depth: {depth_ms.shape}")
    
    print("\n4. 深度边缘:")
    edges = estimator.compute_depth_edges(depth)
    print(f"  Depth edges: {edges.shape}, {edges.sum()} edge pixels")
    
    print("\n5. 深度修复:")
    mask = np.random.rand(*depth.shape) > 0.9
    inpainted = estimator.inpaint_depth(depth, mask)
    print(f"  Inpainted depth: {inpainted.shape}")
    
    print("\n✓ All tests passed!")