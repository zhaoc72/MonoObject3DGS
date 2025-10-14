"""
DINOv2 Feature Extractor
提取图像的语义特征用于分割和语义理解
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from typing import Dict, Optional, Tuple
import numpy as np


class DINOv2Extractor(nn.Module):
    """DINOv2特征提取器"""
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        feature_dim: int = 768,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        
        # 加载模型和处理器
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.eval()
        
        print(f"✓ DINOv2模型加载成功: {model_name}")
        
    @torch.no_grad()
    def extract_features(
        self, 
        image: np.ndarray,
        return_cls: bool = False,
        return_patch: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        提取图像特征
        
        Args:
            image: RGB图像 (H, W, 3)
            return_cls: 是否返回CLS token
            return_patch: 是否返回patch tokens
            
        Returns:
            features字典包含:
                - cls_token: 全局特征 (1, D)
                - patch_tokens: 局部特征 (1, N, D)
                - patch_h, patch_w: patch网格尺寸
        """
        # 预处理
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 前向传播
        outputs = self.model(**inputs)
        
        features = {}
        
        if return_cls:
            # CLS token: 全局特征
            features['cls_token'] = outputs.last_hidden_state[:, 0]
            
        if return_patch:
            # Patch tokens: 局部特征
            patch_tokens = outputs.last_hidden_state[:, 1:]
            features['patch_tokens'] = patch_tokens
            
            # 计算patch网格尺寸
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
        features = self.extract_features(image, return_patch=True)
        
        # 重塑为2D特征图
        patch_tokens = features['patch_tokens']  # (1, N, D)
        patch_h = features['patch_h']
        patch_w = features['patch_w']
        
        B, N, D = patch_tokens.shape
        feature_map = patch_tokens.reshape(B, patch_h, patch_w, D)
        feature_map = feature_map.permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # 上采样到目标尺寸
        if target_size is not None:
            feature_map = nn.functional.interpolate(
                feature_map,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            
        return feature_map
    
    def compute_similarity_map(
        self,
        image: np.ndarray,
        query_point: Tuple[int, int],
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        计算查询点与图像其他位置的相似度
        用于交互式分割
        
        Args:
            image: RGB图像
            query_point: 查询点坐标 (x, y)
            target_size: 输出尺寸
            
        Returns:
            similarity_map: (H, W)
        """
        feature_map = self.get_dense_features(image, target_size)
        B, D, H, W = feature_map.shape
        
        # 归一化特征
        feature_map = nn.functional.normalize(feature_map, dim=1)
        
        # 提取查询点特征
        x, y = query_point
        if target_size:
            x = int(x * H / target_size[0])
            y = int(y * W / target_size[1])
        query_feature = feature_map[0, :, y, x]  # (D,)
        
        # 计算余弦相似度
        feature_map_flat = feature_map.reshape(D, -1)  # (D, H*W)
        similarity = torch.matmul(query_feature, feature_map_flat)  # (H*W,)
        similarity_map = similarity.reshape(H, W)
        
        return similarity_map.cpu().numpy()
    
    def cluster_features(
        self,
        image: np.ndarray,
        num_clusters: int = 10
    ) -> np.ndarray:
        """
        对特征进行聚类以获得语义分割
        
        Args:
            image: RGB图像
            num_clusters: 聚类数量
            
        Returns:
            cluster_map: (H, W) 聚类标签
        """
        from sklearn.cluster import KMeans
        
        feature_map = self.get_dense_features(image)
        B, D, H, W = feature_map.shape
        
        # 重塑为 (H*W, D)
        features = feature_map.squeeze(0).permute(1, 2, 0).reshape(-1, D)
        features = features.cpu().numpy()
        
        # KMeans聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        labels = kmeans.fit_predict(features)
        cluster_map = labels.reshape(H, W)
        
        return cluster_map


if __name__ == "__main__":
    # 测试代码
    import cv2
    
    extractor = DINOv2Extractor()
    
    # 加载测试图像
    image = cv2.imread("test_image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 提取特征
    features = extractor.extract_features(image)
    print(f"CLS token shape: {features['cls_token'].shape}")
    print(f"Patch tokens shape: {features['patch_tokens'].shape}")
    
    # 获取密集特征
    dense_features = extractor.get_dense_features(image, target_size=(512, 512))
    print(f"Dense features shape: {dense_features.shape}")
    
    # 特征聚类
    cluster_map = extractor.cluster_features(image, num_clusters=10)
    print(f"Cluster map shape: {cluster_map.shape}")