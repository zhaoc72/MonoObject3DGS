"""
Semantic Classifier
基于CLIP的零样本语义分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional


class SemanticClassifier:
    """语义分类器"""
    
    def __init__(
        self,
        method: str = "clip",
        model_name: str = "openai/clip-vit-base-patch32",
        categories: Optional[List[str]] = None,
        device: str = "cuda"
    ):
        self.method = method
        self.device = device
        
        if categories is None:
            self.categories = [
                "chair", "table", "sofa", "bed", "desk",
                "cabinet", "shelf", "lamp", "tv", "plant",
                "door", "window"
            ]
        else:
            self.categories = categories
        
        if method == "clip":
            self._init_clip(model_name)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"✓ SemanticClassifier initialized: {method}, {len(self.categories)} categories")
    
    def _init_clip(self, model_name: str):
        """初始化CLIP"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model.eval()
            
            # 预计算文本特征
            with torch.no_grad():
                text_prompts = [f"a photo of a {cat}" for cat in self.categories]
                text_inputs = self.clip_processor(
                    text=text_prompts,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                self.text_features = self.clip_model.get_text_features(**text_inputs)
                self.text_features = F.normalize(self.text_features, dim=-1)
            
            print(f"  CLIP text features: {self.text_features.shape}")
            
        except Exception as e:
            print(f"Warning: Could not load CLIP: {e}")
            print("Using dummy classifier")
            self.clip_model = None
            self.clip_processor = None
            self.text_features = None
    
    @torch.no_grad()
    def classify(
        self,
        image_crop: np.ndarray,
        mask: Optional[np.ndarray] = None,
        top_k: int = 1
    ) -> List[Dict[str, float]]:
        """
        分类物体
        
        Args:
            image_crop: 物体裁剪图像 (H, W, 3)
            mask: 物体mask (H, W)
            top_k: 返回前k个预测
            
        Returns:
            predictions: 预测结果列表
        """
        if self.clip_model is None:
            # Dummy prediction
            return [{
                'category': np.random.choice(self.categories),
                'score': 0.8
            }]
        
        # 应用mask
        if mask is not None:
            image_masked = image_crop.copy()
            image_masked[~mask] = 255  # 白色背景
        else:
            image_masked = image_crop
        
        # 处理图像
        inputs = self.clip_processor(
            images=image_masked,
            return_tensors="pt"
        ).to(self.device)
        
        # 提取图像特征
        image_features = self.clip_model.get_image_features(**inputs)
        image_features = F.normalize(image_features, dim=-1)
        
        # 计算相似度
        similarities = (image_features @ self.text_features.T).squeeze(0)
        scores = torch.softmax(similarities * 100, dim=0)  # 温度缩放
        
        # 获取top-k
        top_scores, top_indices = torch.topk(scores, min(top_k, len(self.categories)))
        
        predictions = []
        for score, idx in zip(top_scores, top_indices):
            predictions.append({
                'category': self.categories[idx.item()],
                'score': score.item()
            })
        
        return predictions


# 测试
if __name__ == "__main__":
    classifier = SemanticClassifier(device="cpu")
    
    # 测试图像
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # 分类
    predictions = classifier.classify(image, top_k=3)
    print("\nPredictions:")
    for pred in predictions:
        print(f"  {pred['category']}: {pred['score']:.3f}")