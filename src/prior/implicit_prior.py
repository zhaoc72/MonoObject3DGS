"""
Implicit Shape Prior
隐式形状先验（学习的潜空间）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ShapeEncoder(nn.Module):
    """点云编码器"""
    
    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 256,
        hidden_dims: List[int] = [64, 128, 256, 512]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 逐点MLP
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.point_mlp = nn.Sequential(*layers)
        
        # 全局特征
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim)
        )
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) or (N, 3)
        Returns:
            latent: (B, D) or (D,)
        """
        if points.dim() == 2:
            points = points.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, D = points.shape
        
        # 逐点特征
        point_features = self.point_mlp(points.reshape(B * N, D))
        point_features = point_features.reshape(B, N, -1)
        
        # 全局池化
        global_features, _ = torch.max(point_features, dim=1)
        
        # 潜向量
        latent = self.global_mlp(global_features)
        
        if squeeze_output:
            latent = latent.squeeze(0)
        
        return latent


class ShapeDecoder(nn.Module):
    """形状解码器"""
    
    def __init__(
        self,
        latent_dim: int = 256,
        num_points: int = 2048,
        hidden_dims: List[int] = [256, 512, 1024]
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_points = num_points
        
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_points * 3))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, D) or (D,)
        Returns:
            points: (B, N, 3) or (N, 3)
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        output = self.decoder(latent)
        points = output.reshape(-1, self.num_points, 3)
        
        if squeeze_output:
            points = points.squeeze(0)
        
        return points


class ImplicitShapePrior(nn.Module):
    """隐式形状先验"""
    
    def __init__(
        self,
        latent_dim: int = 256,
        encoder_hidden: List[int] = [64, 128, 256, 512],
        decoder_hidden: List[int] = [256, 512, 1024],
        num_output_points: int = 2048
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_output_points = num_output_points
        
        self.encoder = ShapeEncoder(
            input_dim=3,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden
        )
        
        self.decoder = ShapeDecoder(
            latent_dim=latent_dim,
            num_points=num_output_points,
            hidden_dims=decoder_hidden
        )
        
        # 类别原型
        self.category_prototypes = nn.ParameterDict()
        
        print(f"✓ ImplicitShapePrior initialized: latent_dim={latent_dim}")
    
    def add_category_prototype(self, category: str):
        """添加类别原型"""
        if category not in self.category_prototypes:
            prototype = torch.randn(1, self.latent_dim) * 0.01
            self.category_prototypes[category] = nn.Parameter(prototype)
            print(f"  Added category prototype: {category}")
    
    def encode(self, pointcloud: torch.Tensor) -> torch.Tensor:
        """编码点云"""
        return self.encoder(pointcloud)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """解码潜向量"""
        return self.decoder(latent)
    
    def get_category_prior(self, category: str) -> Optional[torch.Tensor]:
        """获取类别先验形状"""
        if category in self.category_prototypes:
            latent = self.category_prototypes[category]
            prior_shape = self.decode(latent)
            return prior_shape.squeeze(0)
        else:
            return None
    
    def compute_prior_loss(
        self,
        pointcloud: torch.Tensor,
        category: str
    ) -> torch.Tensor:
        """计算隐式先验损失"""
        # 编码
        latent = self.encode(pointcloud)
        
        # 重建
        reconstructed = self.decode(latent)
        
        if pointcloud.dim() == 2:
            pointcloud = pointcloud.unsqueeze(0)
        
        # 重建损失（简化的Chamfer）
        recon_loss = self._chamfer_loss(reconstructed, pointcloud)
        
        # 类别一致性损失
        category_loss = torch.tensor(0.0, device=pointcloud.device)
        if category in self.category_prototypes:
            prototype_latent = self.category_prototypes[category]
            category_loss = F.mse_loss(latent, prototype_latent.expand_as(latent))
        
        return recon_loss + 0.1 * category_loss
    
    def _chamfer_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """简化的Chamfer损失"""
        if pred.shape[1] != target.shape[1]:
            min_pts = min(pred.shape[1], target.shape[1])
            pred = pred[:, :min_pts, :]
            target = target[:, :min_pts, :]
        
        return F.mse_loss(pred, target)


# 测试
if __name__ == "__main__":
    prior = ImplicitShapePrior(latent_dim=256)
    
    # 添加类别
    for cat in ["chair", "table"]:
        prior.add_category_prototype(cat)
    
    # 测试编码-解码
    test_pc = torch.randn(100, 3)
    
    latent = prior.encode(test_pc)
    print(f"Latent: {latent.shape}")
    
    reconstructed = prior.decode(latent)
    print(f"Reconstructed: {reconstructed.shape}")
    
    # 测试先验
    prior_shape = prior.get_category_prior("chair")
    if prior_shape is not None:
        print(f"Prior shape: {prior_shape.shape}")
    
    # 测试损失
    loss = prior.compute_prior_loss(test_pc, "chair")
    print(f"Prior loss: {loss.item():.4f}")