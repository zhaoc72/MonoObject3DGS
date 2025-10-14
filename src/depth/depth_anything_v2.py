"""
Depth Anything V2 - 最先进的单目深度估计
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import cv2


class DepthAnythingV2:
    """Depth Anything V2深度估计器"""
    
    def __init__(
        self,
        model_size: str = "base",  # small/base/large
        device: str = "cuda"
    ):
        self.device = device
        self.model_size = model_size
        
        self.model = self._load_model(model_size)
        self.model.eval()
        
        print(f"✓ DepthAnythingV2 loaded: {model_size}")
    
    def _load_model(self, model_size: str):
        """加载模型"""
        try:
            # 尝试加载Depth Anything V2
            import torch.hub
            
            model_map = {
                'small': 'depth_anything_v2_vits',
                'base': 'depth_anything_v2_vitb',
                'large': 'depth_anything_v2_vitl'
            }
            
            model_name = model_map.get(model_size, 'depth_anything_v2_vitb')
            
            # 从torch hub加载
            model = torch.hub.load(
                'depth-anything/Depth-Anything-V2',
                model_name,
                pretrained=True
            )
            
            return model.to(self.device)
            
        except Exception as e:
            print(f"Warning: Could not load Depth Anything V2: {e}")
            print("Falling back to MiDaS...")
            return self._load_midas_fallback()
    
    def _load_midas_fallback(self):
        """备用MiDaS模型"""
        import torch.hub
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        return model.to(self.device)
    
    @torch.no_grad()
    def estimate(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        估计深度
        
        Args:
            image: RGB图像 (H, W, 3), 范围[0, 255]
            target_size: 目标尺寸 (H, W)
            
        Returns:
            depth: 深度图 (H, W), 单位米
        """
        H, W = image.shape[:2]
        
        # 预处理
        image_tensor = self._preprocess(image)
        image_tensor = image_tensor.to(self.device)
        
        # 推理
        try:
            depth = self.model(image_tensor)
        except:
            # 兼容不同接口
            depth = self.model.forward(image_tensor)
        
        # 后处理
        if isinstance(depth, dict):
            depth = depth.get('metric_depth', depth.get('depth'))
        
        depth = depth.squeeze().cpu().numpy()
        
        # 调整尺寸
        if target_size is None:
            target_size = (H, W)
        
        if depth.shape != target_size:
            depth = cv2.resize(depth, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # 归一化到合理范围 [0.5, 10]米
        depth = self._normalize_depth(depth)
        
        return depth
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
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
        """归一化深度到合理范围"""
        # 移除极端值
        p1, p99 = np.percentile(depth, [1, 99])
        depth = np.clip(depth, p1, p99)
        
        # 归一化到[0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        # 映射到[0.5, 10]米
        depth = depth * 9.5 + 0.5
        
        return depth


# 测试
if __name__ == "__main__":
    estimator = DepthAnythingV2(model_size="base", device="cpu")
    
    # 测试图像
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    import time
    start = time.time()
    depth = estimator.estimate(image)
    elapsed = time.time() - start
    
    print(f"Depth estimated in {elapsed:.3f}s")
    print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}] meters")