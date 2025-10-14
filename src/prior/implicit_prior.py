"""
Implicit Shape Prior
隐式形状先验 - 基于学习的潜空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class ShapeEncoder(nn.Module):
    """
    点云编码器
    将点云编码到潜空间（简化版PointNet）
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 256,
        hidden_dims: List[int] = [64, 128, 256, 512]
    ):
        """
        Args:
            input_dim: 输入维度（通常为3: x,y,z）
            latent_dim: 潜空间维度
            hidden_dims: 隐藏层维度列表
        """
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
        
        # 全局特征提取
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim)
        )
        
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        编码点云
        
        Args:
            points: (B, N, 3) 或 (N, 3)
            
        Returns:
            latent: (B, latent_dim) 或 (latent_dim,)
        """
        if points.dim() == 2:
            points = points.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, D = points.shape
        
        # 逐点特征提取
        point_features = self.point_mlp(points.reshape(B * N, D))
        point_features = point_features.reshape(B, N, -1)
        
        # 全局最大池化
        global_features, _ = torch.max(point_features, dim=1)
        
        # 潜向量
        latent = self.global_mlp(global_features)
        
        if squeeze_output:
            latent = latent.squeeze(0)
        
        return latent


class ShapeDecoder(nn.Module):
    """
    形状解码器
    从潜空间解码出点云
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        num_points: int = 2048,
        hidden_dims: List[int] = [256, 512, 1024]
    ):
        """
        Args:
            latent_dim: 潜空间维度
            num_points: 输出点云数量
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_points = num_points
        
        # 解码MLP
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
        解码潜向量
        
        Args:
            latent: (B, latent_dim) 或 (latent_dim,)
            
        Returns:
            points: (B, num_points, 3) 或 (num_points, 3)
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 解码
        output = self.decoder(latent)
        points = output.reshape(-1, self.num_points, 3)
        
        if squeeze_output:
            points = points.squeeze(0)
        
        return points


class ImplicitShapePrior(nn.Module):
    """
    隐式形状先验
    学习类别级别的形状潜空间
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        encoder_hidden: List[int] = [64, 128, 256, 512],
        decoder_hidden: List[int] = [256, 512, 1024],
        num_output_points: int = 2048
    ):
        """
        Args:
            latent_dim: 潜空间维度
            encoder_hidden: 编码器隐藏层
            decoder_hidden: 解码器隐藏层
            num_output_points: 输出点数
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_output_points = num_output_points
        
        # 编码器和解码器
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
        
        # 类别原型（可学习的类别级潜向量）
        self.category_prototypes = nn.ParameterDict()
        
        print(f"✓ ImplicitShapePrior初始化: latent_dim={latent_dim}")
    
    def add_category_prototype(self, category: str):
        """添加类别原型"""
        if category not in self.category_prototypes:
            prototype = torch.randn(1, self.latent_dim) * 0.01
            self.category_prototypes[category] = nn.Parameter(prototype)
            print(f"  添加类别原型: {category}")
    
    def encode(self, pointcloud: torch.Tensor) -> torch.Tensor:
        """
        编码点云到潜空间
        
        Args:
            pointcloud: (B, N, 3) 或 (N, 3)
            
        Returns:
            latent: (B, latent_dim) 或 (latent_dim,)
        """
        return self.encoder(pointcloud)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        从潜向量解码点云
        
        Args:
            latent: (B, latent_dim) 或 (latent_dim,)
            
        Returns:
            pointcloud: (B, num_points, 3) 或 (num_points, 3)
        """
        return self.decoder(latent)
    
    def get_category_prior(self, category: str) -> Optional[torch.Tensor]:
        """
        获取类别的先验形状
        
        Args:
            category: 类别名称
            
        Returns:
            prior_shape: (num_points, 3) 先验点云
        """
        if category in self.category_prototypes:
            latent = self.category_prototypes[category]
            prior_shape = self.decode(latent)
            return prior_shape.squeeze(0)
        else:
            print(f"警告: 未找到类别 '{category}' 的原型")
            return None
    
    def compute_prior_loss(
        self,
        pointcloud: torch.Tensor,
        category: str
    ) -> torch.Tensor:
        """
        计算隐式先验损失
        
        Args:
            pointcloud: 输入点云 (N, 3) 或 (B, N, 3)
            category: 类别
            
        Returns:
            loss: 损失值
        """
        # 编码
        latent = self.encode(pointcloud)
        
        # 重建损失
        reconstructed = self.decode(latent)
        
        if pointcloud.dim() == 2:
            pointcloud = pointcloud.unsqueeze(0)
        
        # Chamfer距离作为重建损失
        recon_loss = self._chamfer_loss(reconstructed, pointcloud)
        
        # 类别一致性损失
        category_loss = 0.0
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
        # pred: (B, N, 3), target: (B, M, 3)
        
        # 确保维度匹配
        if pred.shape[1] != target.shape[1]:
            # 采样到相同数量
            min_pts = min(pred.shape[1], target.shape[1])
            pred = pred[:, :min_pts, :]
            target = target[:, :min_pts, :]
        
        # 简化为MSE（真实实现应使用Chamfer距离）
        return F.mse_loss(pred, target)
    
    def interpolate_shapes(
        self,
        category1: str,
        category2: str,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        在两个类别之间插值
        
        Args:
            category1: 第一个类别
            category2: 第二个类别
            alpha: 插值系数 [0, 1]
            
        Returns:
            interpolated_shape: 插值后的形状
        """
        if category1 not in self.category_prototypes or category2 not in self.category_prototypes:
            return None
        
        latent1 = self.category_prototypes[category1]
        latent2 = self.category_prototypes[category2]
        
        # 线性插值
        interpolated_latent = alpha * latent1 + (1 - alpha) * latent2
        
        # 解码
        interpolated_shape = self.decode(interpolated_latent)
        
        return interpolated_shape.squeeze(0)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'category_prototypes': self.category_prototypes,
            'config': {
                'latent_dim': self.latent_dim,
                'num_output_points': self.num_output_points
            }
        }, path)
        print(f"✓ 隐式先验模型保存到: {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.category_prototypes = checkpoint['category_prototypes']
        print(f"✓ 隐式先验模型加载自: {path}")


if __name__ == "__main__":
    # 测试代码
    print("=== 测试ImplicitShapePrior ===")
    
    # 初始化
    prior = ImplicitShapePrior(latent_dim=256)
    
    # 添加类别
    categories = ["chair", "table", "sofa"]
    for cat in categories:
        prior.add_category_prototype(cat)
    
    # 测试编码-解码
    print("\n1. 测试编码-解码:")
    test_pc = torch.randn(100, 3)
    
    latent = prior.encode(test_pc)
    print(f"   潜向量: {latent.shape}")
    
    reconstructed = prior.decode(latent)
    print(f"   重建点云: {reconstructed.shape}")
    
    # 测试获取类别先验
    print("\n2. 测试类别先验:")
    for cat in categories:
        prior_shape = prior.get_category_prior(cat)
        if prior_shape is not None:
            print(f"   {cat}先验形状: {prior_shape.shape}")
    
    # 测试先验损失
    print("\n3. 测试先验损失:")
    loss = prior.compute_prior_loss(test_pc, "chair")
    print(f"   损失: {loss.item():.4f}")
    
    # 测试形状插值
    print("\n4. 测试形状插值:")
    interp_shape = prior.interpolate_shapes("chair", "table", alpha=0.5)
    if interp_shape is not None:
        print(f"   插值形状: {interp_shape.shape}")
    
    # 测试保存和加载
    print("\n5. 测试保存和加载:")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        prior.save(f.name)
        
        prior2 = ImplicitShapePrior(latent_dim=256)
        prior2.load(f.name)
        print(f"   成功加载模型")
    
    print("\n测试完成！")