"""
Semantic Classifier
基于CLIP/DINOv2特征的物体语义分类器
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F


class SemanticClassifier:
    """
    物体语义分类器
    使用CLIP进行零样本分类或使用训练的分类头
    """
    
    def __init__(
        self,
        method: str = "clip",  # "clip" or "learned"
        clip_model_name: str = "openai/clip-vit-base-patch32",
        categories: Optional[List[str]] = None,
        device: str = "cuda"
    ):
        """
        Args:
            method: 分类方法 ("clip" 或 "learned")
            clip_model_name: CLIP模型名称
            categories: 类别列表
            device: 设备
        """
        self.method = method
        self.device = device
        
        # 默认类别
        if categories is None:
            self.categories = [
                "chair", "table", "sofa", "bed", "desk",
                "cabinet", "shelf", "lamp", "tv", "plant",
                "door", "window", "floor", "wall", "ceiling"
            ]
        else:
            self.categories = categories
        
        if method == "clip":
            self._init_clip(clip_model_name)
        elif method == "learned":
            self._init_learned_classifier()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"✓ SemanticClassifier初始化: method={method}, {len(self.categories)} 个类别")
    
    def _init_clip(self, model_name: str):
        """初始化CLIP模型"""
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
        
        print(f"  CLIP文本特征: {self.text_features.shape}")
    
    def _init_learned_classifier(self):
        """初始化学习的分类器"""
        self.classifier = LearnedClassifier(
            input_dim=768,  # DINOv2特征维度
            num_classes=len(self.categories),
            hidden_dims=[512, 256]
        ).to(self.device)
    
    def classify(
        self,
        image_crop: Optional[np.ndarray] = None,
        features: Optional[torch.Tensor] = None,
        mask: Optional[np.ndarray] = None,
        top_k: int = 1
    ) -> List[Dict[str, float]]:
        """
        分类物体
        
        Args:
            image_crop: 物体裁剪图像 (H, W, 3)
            features: DINOv2特征 (D,)
            mask: 物体mask (H, W)
            top_k: 返回前k个预测
            
        Returns:
            predictions: 预测结果列表，每个包含 {'category': str, 'score': float}
        """
        if self.method == "clip":
            return self._classify_clip(image_crop, mask, top_k)
        elif self.method == "learned":
            return self._classify_learned(features, top_k)
    
    def _classify_clip(
        self,
        image_crop: np.ndarray,
        mask: Optional[np.ndarray] = None,
        top_k: int = 1
    ) -> List[Dict[str, float]]:
        """使用CLIP进行分类"""
        # 如果有mask，应用mask到图像
        if mask is not None:
            # 将mask外的区域设为白色背景
            image_masked = image_crop.copy()
            image_masked[~mask] = 255
        else:
            image_masked = image_crop
        
        # 处理图像
        inputs = self.clip_processor(
            images=image_masked,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            # 提取图像特征
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)
            
            # 计算相似度
            similarities = (image_features @ self.text_features.T).squeeze(0)
            scores = torch.softmax(similarities * 100, dim=0)  # 温度缩放
            
            # 获取top-k
            top_scores, top_indices = torch.topk(scores, top_k)
        
        predictions = []
        for score, idx in zip(top_scores, top_indices):
            predictions.append({
                'category': self.categories[idx.item()],
                'score': score.item()
            })
        
        return predictions
    
    def _classify_learned(
        self,
        features: torch.Tensor,
        top_k: int = 1
    ) -> List[Dict[str, float]]:
        """使用学习的分类器"""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.classifier(features)
            scores = torch.softmax(logits, dim=1).squeeze(0)
            
            # 获取top-k
            top_scores, top_indices = torch.topk(scores, top_k)
        
        predictions = []
        for score, idx in zip(top_scores, top_indices):
            predictions.append({
                'category': self.categories[idx.item()],
                'score': score.item()
            })
        
        return predictions
    
    def classify_batch(
        self,
        images: Optional[List[np.ndarray]] = None,
        features: Optional[torch.Tensor] = None,
        masks: Optional[List[np.ndarray]] = None,
        top_k: int = 1
    ) -> List[List[Dict[str, float]]]:
        """批量分类"""
        if self.method == "clip" and images is not None:
            results = []
            for i, img in enumerate(images):
                mask = masks[i] if masks is not None else None
                results.append(self._classify_clip(img, mask, top_k))
            return results
        elif self.method == "learned" and features is not None:
            # 批量处理
            with torch.no_grad():
                logits = self.classifier(features)
                scores = torch.softmax(logits, dim=1)
                
                top_scores, top_indices = torch.topk(scores, top_k, dim=1)
            
            results = []
            for i in range(len(features)):
                predictions = []
                for score, idx in zip(top_scores[i], top_indices[i]):
                    predictions.append({
                        'category': self.categories[idx.item()],
                        'score': score.item()
                    })
                results.append(predictions)
            
            return results
    
    def train_classifier(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        num_epochs: int = 50,
        learning_rate: float = 1e-3
    ):
        """
        训练学习的分类器
        
        Args:
            train_features: 训练特征 (N, D)
            train_labels: 训练标签 (N,)
            val_features: 验证特征
            val_labels: 验证标签
            num_epochs: 训练轮数
            learning_rate: 学习率
        """
        if self.method != "learned":
            print("警告: 只有learned方法需要训练")
            return
        
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # 训练
            self.classifier.train()
            optimizer.zero_grad()
            
            logits = self.classifier(train_features)
            loss = criterion(logits, train_labels)
            
            loss.backward()
            optimizer.step()
            
            # 计算训练准确率
            _, preds = torch.max(logits, 1)
            train_acc = (preds == train_labels).float().mean().item()
            
            # 验证
            if val_features is not None and val_labels is not None:
                self.classifier.eval()
                with torch.no_grad():
                    val_logits = self.classifier(val_features)
                    _, val_preds = torch.max(val_logits, 1)
                    val_acc = (val_preds == val_labels).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Loss={loss.item():.4f}, "
                          f"Train Acc={train_acc:.4f}, "
                          f"Val Acc={val_acc:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Loss={loss.item():.4f}, "
                          f"Train Acc={train_acc:.4f}")
        
        print(f"✓ 训练完成，最佳验证准确率: {best_val_acc:.4f}")
    
    def save_classifier(self, path: str):
        """保存分类器"""
        if self.method == "learned":
            torch.save({
                'state_dict': self.classifier.state_dict(),
                'categories': self.categories
            }, path)
            print(f"✓ 分类器保存到: {path}")
    
    def load_classifier(self, path: str):
        """加载分类器"""
        if self.method == "learned":
            checkpoint = torch.load(path)
            self.classifier.load_state_dict(checkpoint['state_dict'])
            self.categories = checkpoint['categories']
            print(f"✓ 分类器加载自: {path}")


class LearnedClassifier(nn.Module):
    """学习的分类器网络"""
    
    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 15,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CategoryStatistics:
    """类别统计工具，用于分析和可视化分类结果"""
    
    def __init__(self, categories: List[str]):
        self.categories = categories
        self.predictions_history = []
        
    def add_predictions(self, predictions: List[Dict[str, float]]):
        """添加预测结果"""
        self.predictions_history.extend(predictions)
    
    def get_category_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        distribution = {cat: 0 for cat in self.categories}
        
        for pred in self.predictions_history:
            category = pred['category']
            if category in distribution:
                distribution[category] += 1
        
        return distribution
    
    def get_average_confidence(self) -> Dict[str, float]:
        """获取每个类别的平均置信度"""
        category_scores = {cat: [] for cat in self.categories}
        
        for pred in self.predictions_history:
            category = pred['category']
            score = pred['score']
            if category in category_scores:
                category_scores[category].append(score)
        
        avg_confidence = {}
        for cat, scores in category_scores.items():
            if scores:
                avg_confidence[cat] = np.mean(scores)
            else:
                avg_confidence[cat] = 0.0
        
        return avg_confidence
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n=== 分类统计 ===")
        print(f"总预测数: {len(self.predictions_history)}")
        
        print("\n类别分布:")
        distribution = self.get_category_distribution()
        for cat, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {cat}: {count}")
        
        print("\n平均置信度:")
        avg_conf = self.get_average_confidence()
        for cat, conf in sorted(avg_conf.items(), key=lambda x: x[1], reverse=True):
            if conf > 0:
                print(f"  {cat}: {conf:.3f}")


class HierarchicalClassifier:
    """
    层次化分类器
    先粗分类（家具/电器/建筑元素），再细分类
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # 定义层次结构
        self.hierarchy = {
            "furniture": ["chair", "table", "sofa", "bed", "desk", "cabinet", "shelf"],
            "electronics": ["tv", "lamp", "computer", "phone"],
            "architecture": ["door", "window", "wall", "floor", "ceiling"],
            "decoration": ["plant", "picture", "curtain", "rug"]
        }
        
        # 创建粗分类器
        self.coarse_categories = list(self.hierarchy.keys())
        self.coarse_classifier = SemanticClassifier(
            method="clip",
            categories=self.coarse_categories,
            device=device
        )
        
        # 创建细分类器字典
        self.fine_classifiers = {}
        for coarse_cat, fine_cats in self.hierarchy.items():
            self.fine_classifiers[coarse_cat] = SemanticClassifier(
                method="clip",
                categories=fine_cats,
                device=device
            )
        
        print(f"✓ HierarchicalClassifier初始化: {len(self.coarse_categories)} 个粗类别")
    
    def classify(
        self,
        image_crop: np.ndarray,
        mask: Optional[np.ndarray] = None,
        top_k: int = 1
    ) -> List[Dict]:
        """
        层次化分类
        
        Returns:
            predictions: 包含粗分类和细分类结果
        """
        # 第一步：粗分类
        coarse_preds = self.coarse_classifier.classify(
            image_crop=image_crop,
            mask=mask,
            top_k=1
        )
        
        coarse_category = coarse_preds[0]['category']
        coarse_score = coarse_preds[0]['score']
        
        # 第二步：细分类
        fine_classifier = self.fine_classifiers[coarse_category]
        fine_preds = fine_classifier.classify(
            image_crop=image_crop,
            mask=mask,
            top_k=top_k
        )
        
        # 组合结果
        results = []
        for fine_pred in fine_preds:
            results.append({
                'coarse_category': coarse_category,
                'coarse_score': coarse_score,
                'fine_category': fine_pred['category'],
                'fine_score': fine_pred['score'],
                'combined_score': coarse_score * fine_pred['score']
            })
        
        return results


class EnsembleClassifier:
    """
    集成分类器
    结合多个分类方法的结果
    """
    
    def __init__(
        self,
        classifiers: List[SemanticClassifier],
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            classifiers: 分类器列表
            weights: 每个分类器的权重
        """
        self.classifiers = classifiers
        
        if weights is None:
            self.weights = [1.0 / len(classifiers)] * len(classifiers)
        else:
            self.weights = weights
        
        print(f"✓ EnsembleClassifier初始化: {len(classifiers)} 个分类器")
    
    def classify(
        self,
        image_crop: Optional[np.ndarray] = None,
        features: Optional[torch.Tensor] = None,
        mask: Optional[np.ndarray] = None,
        top_k: int = 1
    ) -> List[Dict[str, float]]:
        """
        集成分类
        
        使用加权投票
        """
        # 收集所有分类器的预测
        all_predictions = []
        for classifier, weight in zip(self.classifiers, self.weights):
            preds = classifier.classify(
                image_crop=image_crop,
                features=features,
                mask=mask,
                top_k=len(classifier.categories)  # 获取所有类别的分数
            )
            all_predictions.append((preds, weight))
        
        # 聚合分数
        category_scores = {}
        for preds, weight in all_predictions:
            for pred in preds:
                cat = pred['category']
                score = pred['score'] * weight
                
                if cat in category_scores:
                    category_scores[cat] += score
                else:
                    category_scores[cat] = score
        
        # 归一化
        total_score = sum(category_scores.values())
        if total_score > 0:
            category_scores = {k: v/total_score for k, v in category_scores.items()}
        
        # 排序并返回top-k
        sorted_preds = sorted(
            category_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        results = [
            {'category': cat, 'score': score}
            for cat, score in sorted_preds
        ]
        
        return results


if __name__ == "__main__":
    # 测试代码
    print("=== 测试SemanticClassifier ===")
    
    # 测试CLIP分类器
    print("\n1. CLIP分类器测试:")
    clip_classifier = SemanticClassifier(
        method="clip",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # 分类
    predictions = clip_classifier.classify(image_crop=test_image, top_k=3)
    print("\n预测结果:")
    for pred in predictions:
        print(f"  {pred['category']}: {pred['score']:.3f}")
    
    # 测试学习的分类器
    print("\n2. 学习的分类器测试:")
    learned_classifier = SemanticClassifier(
        method="learned",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 创建模拟训练数据
    num_samples = 100
    train_features = torch.randn(num_samples, 768)
    train_labels = torch.randint(0, len(learned_classifier.categories), (num_samples,))
    
    # 训练
    print("\n训练分类器...")
    learned_classifier.train_classifier(
        train_features,
        train_labels,
        num_epochs=20
    )
    
    # 测试
    test_features = torch.randn(1, 768)
    predictions = learned_classifier.classify(features=test_features, top_k=3)
    print("\n预测结果:")
    for pred in predictions:
        print(f"  {pred['category']}: {pred['score']:.3f}")
    
    # 测试层次化分类器
    print("\n3. 层次化分类器测试:")
    hierarchical = HierarchicalClassifier(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    predictions = hierarchical.classify(test_image, top_k=2)
    print("\n层次化预测结果:")
    for pred in predictions:
        print(f"  粗类别: {pred['coarse_category']} ({pred['coarse_score']:.3f})")
        print(f"  细类别: {pred['fine_category']} ({pred['fine_score']:.3f})")
        print(f"  综合分数: {pred['combined_score']:.3f}")
    
    # 测试统计工具
    print("\n4. 统计工具测试:")
    stats = CategoryStatistics(clip_classifier.categories)
    
    # 添加一些模拟预测
    for _ in range(20):
        preds = [
            {'category': np.random.choice(clip_classifier.categories), 'score': np.random.rand()}
        ]
        stats.add_predictions(preds)
    
    stats.print_statistics()
    
    print("\n所有测试完成！")