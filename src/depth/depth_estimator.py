"""
Depth Estimator
单目深度估计器，支持多种模型
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, Tuple, Dict
from pathlib import Path
import warnings


class DepthEstimator:
    """
    深度估计器基类
    支持多种单目深度估计模型
    """
    
    def __init__(
        self,
        method: str = "depth_anything_v2",
        checkpoint: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Args:
            method: 深度估计方法 (depth_anything_v2, midas, zoedepth)
            checkpoint: 模型检查点路径
            device: 计算设备
        """
        self.method = method
        self.device = device
        self.checkpoint = checkpoint
        
        # 初始化模型
        self.model = self._load_model()
        
        print(f"✓ DepthEstimator初始化: method={method}")
    
    def _load_model(self) -> nn.Module:
        """加载深度估计模型"""
        if self.method == "depth_anything_v2":
            return DepthAnythingV2(self.checkpoint, self.device)
        elif self.method == "midas":
            return MiDaS(self.checkpoint, self.device)
        elif self.method == "zoedepth":
            return ZoeDepth(self.checkpoint, self.device)
        else:
            raise ValueError(f"Unknown depth estimation method: {self.method}")
    
    def estimate(
        self,
        image: np.ndarray,
        return_confidence: bool = False
    ) -> np.ndarray:
        """
        估计深度图
        
        Args:
            image: RGB图像 (H, W, 3), 范围[0, 255]
            return_confidence: 是否返回置信度图
            
        Returns:
            depth: 深度图 (H, W), 相对深度或度量深度
            confidence: (可选) 置信度图 (H, W)
        """
        return self.model.estimate(image, return_confidence)
    
    def estimate_batch(
        self,
        images: list,
        batch_size: int = 4
    ) -> list:
        """
        批量估计深度
        
        Args:
            images: 图像列表
            batch_size: 批大小
            
        Returns:
            depths: 深度图列表
        """
        depths = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_depths = [self.estimate(img) for img in batch]
            depths.extend(batch_depths)
        return depths


class DepthAnythingV2:
    """
    Depth Anything V2 深度估计模型
    提供相对深度估计
    """
    
    def __init__(self, checkpoint: Optional[str] = None, device: str = "cuda"):
        self.device = device
        
        try:
            # 尝试加载Depth Anything V2
            import torch.hub
            
            # 从torch hub加载
            if checkpoint and Path(checkpoint).exists():
                self.model = torch.load(checkpoint, map_location=device)
            else:
                # 使用预训练模型
                warnings.warn("Depth Anything V2模型将从torch hub下载")
                # 这里需要根据实际的Depth Anything V2仓库地址修改
                self.model = torch.hub.load(
                    'LiheYoung/Depth-Anything',
                    'DepthAnything_vits14',
                    pretrained=True
                )
            
            self.model = self.model.to(device)
            self.model.eval()
            
            print("✓ Depth Anything V2模型加载成功")
            
        except Exception as e:
            print(f"警告: 无法加载Depth Anything V2: {e}")
            print("将使用MiDaS作为备选")
            # 回退到MiDaS
            self.model = self._load_midas_fallback()
    
    def _load_midas_fallback(self):
        """加载MiDaS作为备选"""
        import torch.hub
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        model = model.to(self.device)
        model.eval()
        return model
    
    @torch.no_grad()
    def estimate(
        self,
        image: np.ndarray,
        return_confidence: bool = False
    ) -> np.ndarray:
        """
        估计深度
        
        Args:
            image: RGB图像 (H, W, 3)
            return_confidence: 是否返回置信度
            
        Returns:
            depth: 深度图 (H, W)
        """
        H, W = image.shape[:2]
        
        # 预处理
        image_input = self._preprocess(image)
        image_input = image_input.to(self.device)
        
        # 推理
        try:
            depth = self.model(image_input)
        except:
            # 如果失败，尝试不同的输入格式
            depth = self.model.forward(image_input)
        
        # 后处理
        if isinstance(depth, dict):
            depth = depth['depth'] if 'depth' in depth else depth['metric_depth']
        
        depth = depth.squeeze().cpu().numpy()
        
        # 调整到原始尺寸
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # 限制深度范围
        depth = np.clip(depth, 0.1, 20.0)
        
        if return_confidence:
            confidence = np.ones_like(depth) * 0.9  # ZoeDepth通常更准确
            return depth, confidence
        
        return depth


if __name__ == "__main__":
    # 测试代码
    print("=== 测试DepthEstimator ===")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 测试Depth Anything V2
    print("\n1. 测试Depth Anything V2:")
    try:
        estimator = DepthEstimator(method="depth_anything_v2", device="cpu")
        depth = estimator.estimate(test_image)
        print(f"   深度图形状: {depth.shape}")
        print(f"   深度范围: [{depth.min():.2f}, {depth.max():.2f}]")
        
        # 测试置信度
        depth, confidence = estimator.estimate(test_image, return_confidence=True)
        print(f"   置信度范围: [{confidence.min():.2f}, {confidence.max():.2f}]")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 测试MiDaS
    print("\n2. 测试MiDaS:")
    try:
        estimator = DepthEstimator(method="midas", device="cpu")
        depth = estimator.estimate(test_image)
        print(f"   深度图形状: {depth.shape}")
        print(f"   深度范围: [{depth.min():.2f}, {depth.max():.2f}]")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 测试批处理
    print("\n3. 测试批处理:")
    images = [test_image] * 3
    depths = estimator.estimate_batch(images, batch_size=2)
    print(f"   处理了 {len(depths)} 张图像")
    
    print("\n测试完成！")

        
        # 限制深度范围
        depth = np.clip(depth, 0.1, 20.0)
        
        if return_confidence:
            confidence = np.ones_like(depth) * 0.9  # ZoeDepth通常更准确
            return depth, confidence
        
        return depth


if __name__ == "__main__":
    # 测试代码
    print("=== 测试DepthEstimator ===")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 测试Depth Anything V2
    print("\n1. 测试Depth Anything V2:")
    try:
        estimator = DepthEstimator(method="depth_anything_v2", device="cpu")
        depth = estimator.estimate(test_image)
        print(f"   深度图形状: {depth.shape}")
        print(f"   深度范围: [{depth.min():.2f}, {depth.max():.2f}]")
        
        # 测试置信度
        depth, confidence = estimator.estimate(test_image, return_confidence=True)
        print(f"   置信度范围: [{confidence.min():.2f}, {confidence.max():.2f}]")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 测试MiDaS
    print("\n2. 测试MiDaS:")
    try:
        estimator = DepthEstimator(method="midas", device="cpu")
        depth = estimator.estimate(test_image)
        print(f"   深度图形状: {depth.shape}")
        print(f"   深度范围: [{depth.min():.2f}, {depth.max():.2f}]")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 测试批处理
    print("\n3. 测试批处理:")
    images = [test_image] * 3
    depths = estimator.estimate_batch(images, batch_size=2)
    print(f"   处理了 {len(depths)} 张图像")
    
    print("\n测试完成！")
LINEAR)
        
        # 归一化到合理范围
        depth = self._normalize_depth(depth)
        
        if return_confidence:
            # 简单的置信度估计（基于深度图的平滑度）
            confidence = self._estimate_confidence(depth)
            return depth, confidence
        
        return depth
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        # 标准化（ImageNet统计）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image = (image - mean) / std
        
        return image
    
    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """归一化深度图"""
        # 移除异常值
        depth = np.clip(depth, np.percentile(depth, 1), np.percentile(depth, 99))
        
        # 归一化到[0, 1]然后映射到合理的深度范围[0.5, 10]米
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = depth * 9.5 + 0.5  # 映射到[0.5, 10]米
        
        return depth
    
    def _estimate_confidence(self, depth: np.ndarray) -> np.ndarray:
        """估计深度置信度"""
        # 基于局部方差估计置信度
        # 平滑区域 = 高置信度，变化大的区域 = 低置信度
        
        # 计算梯度幅值
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 反转并归一化（梯度小 = 置信度高）
        confidence = 1.0 - (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
        
        return confidence


class MiDaS:
    """
    MiDaS深度估计模型
    """
    
    def __init__(self, checkpoint: Optional[str] = None, device: str = "cuda"):
        self.device = device
        
        import torch.hub
        
        # 加载MiDaS
        if checkpoint and Path(checkpoint).exists():
            self.model = torch.load(checkpoint, map_location=device)
        else:
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # 加载transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform
        
        print("✓ MiDaS模型加载成功")
    
    @torch.no_grad()
    def estimate(
        self,
        image: np.ndarray,
        return_confidence: bool = False
    ) -> np.ndarray:
        """估计深度"""
        H, W = image.shape[:2]
        
        # 预处理
        image_input = self.transform(image).to(self.device)
        
        # 推理
        depth = self.model(image_input)
        
        # 后处理
        depth = depth.squeeze().cpu().numpy()
        
        # 调整尺寸
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_CUBIC)
        
        # 反转（MiDaS输出的是视差）
        depth = 1.0 / (depth + 1e-6)
        
        # 归一化
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = depth * 9.5 + 0.5  # 映射到[0.5, 10]米
        
        if return_confidence:
            confidence = np.ones_like(depth) * 0.8  # MiDaS没有置信度输出
            return depth, confidence
        
        return depth


class ZoeDepth:
    """
    ZoeDepth - 度量深度估计
    """
    
    def __init__(self, checkpoint: Optional[str] = None, device: str = "cuda"):
        self.device = device
        
        try:
            import torch.hub
            
            # 加载ZoeDepth
            self.model = torch.hub.load(
                "isl-org/ZoeDepth",
                "ZoeD_N",
                pretrained=True
            )
            
            self.model = self.model.to(device)
            self.model.eval()
            
            print("✓ ZoeDepth模型加载成功")
            
        except Exception as e:
            print(f"警告: 无法加载ZoeDepth: {e}")
            print("将使用MiDaS作为备选")
            # 回退到MiDaS
            self.model = MiDaS(checkpoint, device).model
    
    @torch.no_grad()
    def estimate(
        self,
        image: np.ndarray,
        return_confidence: bool = False
    ) -> np.ndarray:
        """估计深度（度量深度）"""
        H, W = image.shape[:2]
        
        # ZoeDepth接受PIL Image或numpy array
        depth = self.model.infer_pil(image)
        
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        
        # 调整尺寸
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_