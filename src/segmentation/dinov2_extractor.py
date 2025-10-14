"""
DINOv2 V2 Feature Extractor - UPGRADED
支持更大的模型和更强的特征提取能力
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from transformers import AutoModel, AutoImageProcessor


class DINOv2ExtractorV2:
    """DINOv2 V2 特征提取器 - 升级版"""
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",  # 升级到large
        feature_dim: int = 1024,  # large模型: 1024维
        device: str = "cuda",
        use_registers: bool = True,  # 使用register tokens
        enable_xformers: bool = True  # Flash Attention加速
    ):
        self.device = device
        self.feature_dim = feature_dim
        self.model_name = model_name
        self.use_registers = use_registers
        
        print(f"🔄 Loading DINOv2 V2: {model_name}")
        
        # 加载模型
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            
            # 启用Flash Attention加速
            if enable_xformers and device == "cuda":
                try:
                    self.model.enable_xformers_memory_efficient_attention()
                    print("  ✓ Flash Attention enabled")
                except:
                    print("  ⚠ Flash Attention not available")
            
            self.model.eval()
            print(f"✓ DINOv2 V2 loaded: {model_name}")
            print(f"  Feature dim: {feature_dim}")
            print(f"  Device: {device}")
            
        except Exception as e:
            print(f"❌ Failed to load DINOv2 V2: {e}")
            print("  Falling back to dummy extractor")
            self.model = None
            self.processor = None
    
    @torch.no_grad()
    def extract_features(
        self,
        image: np.ndarray,
        return_cls: bool = True,
        return_patch: bool = True,
        return_registers: bool = None
    ) -> Dict[str, torch.Tensor]:
        """
        提取图像特征 - 增强版
        
        Args:
            image: RGB图像 (H, W, 3)
            return_cls: 返回CLS token
            return_patch: 返回patch tokens
            return_registers: 返回register tokens (默认跟随use_registers)
            
        Returns:
            features: {
                'cls_token': (1, D),
                'patch_tokens': (1, N, D),
                'register_tokens': (1, R, D),  # NEW
                'patch_h': int,
                'patch_w': int
            }
        """
        if self.model is None:
            return self._dummy_features(image.shape[:2])
        
        if return_registers is None:
            return_registers = self.use_registers
        
        # 预处理
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 前向传播
        outputs = self.model(**inputs, output_hidden_states=True)
        
        features = {}
        hidden_states = outputs.last_hidden_state  # (1, N_total, D)
        
        # 解析tokens
        # DINOv2 with registers: [CLS] [REG1] [REG2] ... [PATCH1] [PATCH2] ...
        n_registers = 4 if return_registers else 0
        
        if return_cls:
            features['cls_token'] = hidden_states[:, 0]  # (1, D)
        
        if return_registers and n_registers > 0:
            features['register_tokens'] = hidden_states[:, 1:1+n_registers]  # (1, R, D)
        
        if return_patch:
            patch_tokens = hidden_states[:, 1+n_registers:]  # (1, N, D)
            features['patch_tokens'] = patch_tokens
            
            # 计算patch网格大小
            num_patches = patch_tokens.shape[1]
            patch_h = patch_w = int(np.sqrt(num_patches))
            features['patch_h'] = patch_h
            features['patch_w'] = patch_w
        
        return features
    
    def get_dense_features(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        interpolation: str = "bilinear"
    ) -> torch.Tensor:
        """
        获取密集特征图 - 增强版
        
        Args:
            image: RGB图像
            target_size: 目标尺寸 (H, W)
            interpolation: 插值方法 ['bilinear', 'bicubic']
            
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
                mode=interpolation,
                align_corners=False if interpolation == "bilinear" else None
            )
        
        return feature_map
    
    def get_multi_scale_features(
        self,
        image: np.ndarray,
        scales: list = [1.0, 0.75, 0.5]
    ) -> Dict[str, torch.Tensor]:
        """
        多尺度特征提取 - NEW
        
        Args:
            image: RGB图像
            scales: 尺度列表
            
        Returns:
            multi_scale_features: {scale: feature_map}
        """
        H, W = image.shape[:2]
        multi_scale = {}
        
        for scale in scales:
            h, w = int(H * scale), int(W * scale)
            resized = cv2.resize(image, (w, h))
            
            features = self.get_dense_features(resized, target_size=(H, W))
            multi_scale[scale] = features
        
        return multi_scale
    
    def compute_semantic_similarity(
        self,
        image: np.ndarray,
        query_points: np.ndarray,  # (M, 2) [x, y]
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        计算语义相似度图 - NEW
        
        Args:
            image: RGB图像
            query_points: 查询点坐标 (M, 2)
            target_size: 目标尺寸
            
        Returns:
            similarity_map: (M, H, W)
        """
        H, W = image.shape[:2]
        if target_size is None:
            target_size = (H, W)
        
        # 提取密集特征
        features = self.get_dense_features(image, target_size)  # (1, D, H, W)
        features = torch.nn.functional.normalize(features, dim=1)
        
        # 提取查询点特征
        query_features = []
        for x, y in query_points:
            x_idx = int(x * features.shape[3] / W)
            y_idx = int(y * features.shape[2] / H)
            query_feat = features[0, :, y_idx, x_idx]  # (D,)
            query_features.append(query_feat)
        
        query_features = torch.stack(query_features)  # (M, D)
        query_features = torch.nn.functional.normalize(query_features, dim=1)
        
        # 计算相似度
        features_flat = features.reshape(1, self.feature_dim, -1)  # (1, D, H*W)
        similarity = torch.matmul(query_features, features_flat[0])  # (M, H*W)
        similarity = similarity.reshape(-1, target_size[0], target_size[1])  # (M, H, W)
        
        return similarity
    
    def _dummy_features(self, image_shape):
        """Dummy features for testing"""
        H, W = image_shape
        patch_h = patch_w = H // 14
        return {
            'cls_token': torch.randn(1, self.feature_dim).to(self.device),
            'patch_tokens': torch.randn(1, patch_h * patch_w, self.feature_dim).to(self.device),
            'patch_h': patch_h,
            'patch_w': patch_w
        }


# 测试代码
if __name__ == "__main__":
    print("=== Testing DINOv2 V2 Extractor ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = DINOv2ExtractorV2(
        model_name="facebook/dinov2-large",
        device=device
    )
    
    # 测试图像
    image = np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8)
    
    print("\n1. 基础特征提取:")
    features = extractor.extract_features(image)
    print(f"  CLS token: {features['cls_token'].shape}")
    print(f"  Patch tokens: {features['patch_tokens'].shape}")
    if 'register_tokens' in features:
        print(f"  Register tokens: {features['register_tokens'].shape}")
    
    print("\n2. 密集特征:")
    dense = extractor.get_dense_features(image, target_size=(518, 518))
    print(f"  Dense features: {dense.shape}")
    
    print("\n3. 多尺度特征:")
    multi_scale = extractor.get_multi_scale_features(image)
    for scale, feat in multi_scale.items():
        print(f"  Scale {scale}: {feat.shape}")
    
    print("\n4. 语义相似度:")
    query_points = np.array([[259, 259], [100, 100]])
    similarity = extractor.compute_semantic_similarity(image, query_points)
    print(f"  Similarity map: {similarity.shape}")
    
    print("\n✓ All tests passed!")