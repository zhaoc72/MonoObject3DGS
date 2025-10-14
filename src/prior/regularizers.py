"""
Shape Prior Regularizers
形状先验正则化器
"""

import torch
import numpy as np
from typing import Optional, Dict


class ShapePriorRegularizer:
    """形状先验正则化器"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("✓ ShapePriorRegularizer initialized")
    
    def smoothness_loss(
        self,
        pointcloud: torch.Tensor,
        k: int = 8
    ) -> torch.Tensor:
        """平滑性损失"""
        try:
            # 简化版本：使用全局方差
            center = pointcloud.mean(dim=0)
            variance = ((pointcloud - center) ** 2).mean()
            return variance * 0.1
        except:
            return torch.tensor(0.0, device=pointcloud.device)
    
    def compactness_loss(
        self,
        pointcloud: torch.Tensor
    ) -> torch.Tensor:
        """紧凑性损失"""
        center = pointcloud.mean(dim=0)
        distances = torch.norm(pointcloud - center, dim=1)
        return distances.mean()
    
    def symmetry_loss(
        self,
        pointcloud: torch.Tensor,
        axis: int = 0
    ) -> torch.Tensor:
        """对称性损失"""
        mirrored = pointcloud.clone()
        mirrored[:, axis] = -mirrored[:, axis]
        
        # 简化：直接计算MSE
        return torch.nn.functional.mse_loss(pointcloud, mirrored)
    
    def compute_combined_regularization(
        self,
        pointcloud: torch.Tensor,
        weights: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """计算组合正则化"""
        if weights is None:
            weights = {
                'smoothness': 0.05,
                'compactness': 0.02,
                'symmetry': 0.01
            }
        
        losses = {}
        total_loss = torch.tensor(0.0, device=pointcloud.device)
        
        if 'smoothness' in weights and weights['smoothness'] > 0:
            loss = self.smoothness_loss(pointcloud)
            losses['smoothness'] = loss
            total_loss += weights['smoothness'] * loss
        
        if 'compactness' in weights and weights['compactness'] > 0:
            loss = self.compactness_loss(pointcloud)
            losses['compactness'] = loss
            total_loss += weights['compactness'] * loss
        
        if 'symmetry' in weights and weights['symmetry'] > 0:
            axis = kwargs.get('symmetry_axis', 0)
            loss = self.symmetry_loss(pointcloud, axis=axis)
            losses['symmetry'] = loss
            total_loss += weights['symmetry'] * loss
        
        losses['total'] = total_loss
        
        return losses


# 测试
if __name__ == "__main__":
    reg = ShapePriorRegularizer(device="cpu")
    
    pc = torch.randn(500, 3)
    
    losses = reg.compute_combined_regularization(
        pc,
        weights={'smoothness': 0.05, 'compactness': 0.02, 'symmetry': 0.01}
    )
    
    print("Regularization losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")