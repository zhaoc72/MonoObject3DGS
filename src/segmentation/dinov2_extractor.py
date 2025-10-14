"""
DINOv2 Feature Extractor
提取语义特征用于分割优化
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


class DINOv2Extractor:
    """DINOv2特征提取器"""
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        feature_dim: int = 768,
        device: str = "cuda"
    ):
        self.device = device
        self.feature_dim = feature_dim
        self.model_name = model_name
        
        # 加载模型
        try:
            from transformers import AutoModel, AutoImageProcessor
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model.eval()
            print(f"✓ DINOv2 loaded: {model_name}")
        except Exception as e:
            print(f"Warning: Could not load DINOv2: {e}")
            print("Using dummy feature extractor")
            self.model = None
            self.processor = None
    
    @torch.no_grad()
    def extract_features(
        self,
        image: np.ndarray,
        return_cls: bool = True,
        return_patch: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        提取图像特征
        
        Args:
            image: RGB图像 (H, W, 3), 范围[0, 255]
            return_cls: 是否返回CLS token
            return_patch: 是否返回patch tokens
            
        Returns:
            features: 特征字典
        """
        if self.model is None:
            # Dummy features
            H, W = image.shape[:2]
            patch_h = patch_w = H // 14
            return {
                'cls_token': torch.randn(1, self.feature_dim),
                'patch_tokens': torch.randn(1, patch_h * patch_w, self.feature_dim),
                'patch_h': patch_h,
                'patch_w': patch_w
            }
        
        # 预处理
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 前向传播
        outputs = self.model(**inputs)
        
        features = {}
        
        if return_cls:
            features['cls_token'] = outputs.last_hidden_state[:, 0]
        
        if return_patch:
            patch_tokens = outputs.last_hidden_state[:, 1:]
            features['patch_tokens'] = patch_tokens
            
            num_patches = patch_tokens.shape[1]
            patch_h = patch_w = int(np.sqrt(num_patches))
            features['patch_h'] = patch_h
            features['patch_w'] = patch_w
        
        return features
    
    def get_dense_features(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        获取密集特征图
        
        Args:
            image: RGB图像
            target_size: 目标尺寸 (H, W)
            
        Returns:
            features: (1, D, H, W)
        """
        features = self.extract_features(image, return_patch=True, return_cls=False)
        
        patch_tokens = features['patch_tokens']  # (1, N, D)
        patch_h = features['patch_h']
        patch_w = features['patch_w']
        
        B, N, D = patch_tokens.shape
        
        # 重塑为2D特征图
        feature_map = patch_tokens.reshape(B, patch_h, patch_w, D)
        feature_map = feature_map.permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # 上采样到目标尺寸
        if target_size is not None:
            feature_map = torch.nn.functional.interpolate(
                feature_map,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        
        return feature_map


# 测试代码
if __name__ == "__main__":
    extractor = DINOv2Extractor(device="cpu")
    
    # 测试图像
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 提取特征
    features = extractor.extract_features(image)
    print(f"CLS token: {features['cls_token'].shape}")
    print(f"Patch tokens: {features['patch_tokens'].shape}")
    
    # 密集特征
    dense = extractor.get_dense_features(image, target_size=(480, 640))
    print(f"Dense features: {dense.shape}")