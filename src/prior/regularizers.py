"""
Shape Prior Regularizers
形状先验正则化器
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class ShapePriorRegularizer:
    """
    形状先验正则化器
    包含对称性、平滑性、紧凑性等约束
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: 计算设备
        """
        self.device = device
        print("✓ ShapePriorRegularizer初始化完成")
    
    def symmetry_loss(
        self,
        pointcloud: torch.Tensor,
        axis: int = 0,
        tolerance: float = 0.1
    ) -> torch.Tensor:
        """
        对称性损失
        鼓励点云沿指定轴对称
        
        Args:
            pointcloud: 点云 (N, 3)
            axis: 对称轴 (0=x, 1=y, 2=z)
            tolerance: 容差
            
        Returns:
            loss: 对称性损失
        """
        # 沿对称轴镜像
        mirrored = pointcloud.clone()
        mirrored[:, axis] = -mirrored[:, axis]
        
        # 计算最近邻距离
        try:
            from pytorch3d.ops import knn_points
            
            knn_result = knn_points(
                pointcloud.unsqueeze(0),
                mirrored.unsqueeze(0),
                K=1
            )
            
            distances = knn_result.dists.squeeze()
            loss = distances.mean()
            
        except ImportError:
            # 简化版本：直接计算对称点之间的距离
            loss = torch.nn.functional.mse_loss(pointcloud, mirrored)
        
        return loss
    
    def smoothness_loss(
        self,
        pointcloud: torch.Tensor,
        k: int = 8
    ) -> torch.Tensor:
        """
        平滑性损失
        鼓励局部邻域平滑
        
        Args:
            pointcloud: 点云 (N, 3)
            k: 近邻数量
            
        Returns:
            loss: 平滑性损失
        """
        try:
            from pytorch3d.ops import knn_points, knn_gather
            
            # 计算k近邻
            knn_result = knn_points(
                pointcloud.unsqueeze(0),
                pointcloud.unsqueeze(0),
                K=k+1  # +1 因为包含自己
            )
            
            # 获取近邻点
            neighbor_idx = knn_result.idx[:, :, 1:]  # 排除自己
            neighbor_points = knn_gather(
                pointcloud.unsqueeze(0),
                neighbor_idx
            ).squeeze(0)  # (N, k, 3)
            
            # 计算局部方差
            point_expanded = pointcloud.unsqueeze(1).expand_as(neighbor_points)
            local_variance = ((neighbor_points - point_expanded) ** 2).mean()
            
            return local_variance
            
        except ImportError:
            # 简化版本：使用全局方差
            center = pointcloud.mean(dim=0)
            variance = ((pointcloud - center) ** 2).mean()
            return variance * 0.1
    
    def compactness_loss(
        self,
        pointcloud: torch.Tensor
    ) -> torch.Tensor:
        """
        紧凑性损失
        鼓励点云集中
        
        Args:
            pointcloud: 点云 (N, 3)
            
        Returns:
            loss: 紧凑性损失
        """
        center = pointcloud.mean(dim=0)
        distances = torch.norm(pointcloud - center, dim=1)
        loss = distances.mean()
        
        return loss
    
    def planarity_loss(
        self,
        pointcloud: torch.Tensor,
        plane_normal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        平面性损失
        鼓励点云接近平面（用于桌面等平面物体）
        
        Args:
            pointcloud: 点云 (N, 3)
            plane_normal: 平面法向量 (3,)，None时自动计算
            
        Returns:
            loss: 平面性损失
        """
        # 计算或使用给定的平面法向量
        if plane_normal is None:
            # 使用PCA计算主平面
            centered = pointcloud - pointcloud.mean(dim=0)
            _, _, V = torch.svd(centered.T)
            plane_normal = V[:, -1]  # 最小特征值对应的向量
        
        # 计算点到平面的距离
        center = pointcloud.mean(dim=0)
        distances = torch.abs(torch.matmul(pointcloud - center, plane_normal))
        loss = distances.mean()
        
        return loss
    
    def surface_normal_consistency(
        self,
        pointcloud: torch.Tensor,
        normals: torch.Tensor,
        k: int = 8
    ) -> torch.Tensor:
        """
        表面法向量一致性
        鼓励邻近点的法向量相似
        
        Args:
            pointcloud: 点云 (N, 3)
            normals: 法向量 (N, 3)
            k: 近邻数量
            
        Returns:
            loss: 法向量一致性损失
        """
        try:
            from pytorch3d.ops import knn_points, knn_gather
            
            # 找近邻
            knn_result = knn_points(
                pointcloud.unsqueeze(0),
                pointcloud.unsqueeze(0),
                K=k+1
            )
            
            neighbor_idx = knn_result.idx[:, :, 1:]
            neighbor_normals = knn_gather(
                normals.unsqueeze(0),
                neighbor_idx
            ).squeeze(0)  # (N, k, 3)
            
            # 计算法向量差异
            normal_expanded = normals.unsqueeze(1).expand_as(neighbor_normals)
            cosine_sim = torch.nn.functional.cosine_similarity(
                normal_expanded.reshape(-1, 3),
                neighbor_normals.reshape(-1, 3),
                dim=1
            )
            
            # 一致性损失 = 1 - 平均余弦相似度
            loss = 1.0 - cosine_sim.mean()
            
            return loss
            
        except ImportError:
            # 简化版本
            return torch.tensor(0.0, device=pointcloud.device)
    
    def volume_preservation(
        self,
        pointcloud: torch.Tensor,
        target_volume: float
    ) -> torch.Tensor:
        """
        体积保持
        鼓励点云保持特定体积
        
        Args:
            pointcloud: 点云 (N, 3)
            target_volume: 目标体积
            
        Returns:
            loss: 体积偏差损失
        """
        # 估计当前体积（使用凸包或边界框）
        min_coords = pointcloud.min(dim=0)[0]
        max_coords = pointcloud.max(dim=0)[0]
        current_volume = torch.prod(max_coords - min_coords)
        
        # 体积偏差
        loss = torch.abs(current_volume - target_volume)
        
        return loss
    
    def edge_preservation(
        self,
        pointcloud: torch.Tensor,
        edge_threshold: float = 0.1
    ) -> torch.Tensor:
        """
        边缘保持
        检测并保持尖锐边缘
        
        Args:
            pointcloud: 点云 (N, 3)
            edge_threshold: 边缘检测阈值
            
        Returns:
            loss: 边缘保持损失
        """
        try:
            from pytorch3d.ops import knn_points
            
            # 计算局部密度变化
            knn_result = knn_points(
                pointcloud.unsqueeze(0),
                pointcloud.unsqueeze(0),
                K=10
            )
            
            distances = knn_result.dists.squeeze()
            local_density = 1.0 / (distances.mean(dim=1) + 1e-6)
            
            # 检测密度变化大的区域（可能是边缘）
            density_variance = local_density.var()
            
            # 鼓励适度的密度变化（保持边缘）
            loss = -torch.log(density_variance + 1e-6)
            
            return loss
            
        except ImportError:
            return torch.tensor(0.0, device=pointcloud.device)
    
    def compute_combined_regularization(
        self,
        pointcloud: torch.Tensor,
        weights: Optional[dict] = None,
        **kwargs
    ) -> dict:
        """
        计算组合正则化损失
        
        Args:
            pointcloud: 点云 (N, 3)
            weights: 各项损失权重
            **kwargs: 其他参数
            
        Returns:
            losses: 各项损失字典
        """
        if weights is None:
            weights = {
                'symmetry': 0.01,
                'smoothness': 0.05,
                'compactness': 0.02
            }
        
        losses = {}
        total_loss = torch.tensor(0.0, device=pointcloud.device)
        
        # 对称性
        if 'symmetry' in weights and weights['symmetry'] > 0:
            axis = kwargs.get('symmetry_axis', 0)
            loss = self.symmetry_loss(pointcloud, axis=axis)
            losses['symmetry'] = loss
            total_loss += weights['symmetry'] * loss
        
        # 平滑性
        if 'smoothness' in weights and weights['smoothness'] > 0:
            k = kwargs.get('smoothness_k', 8)
            loss = self.smoothness_loss(pointcloud, k=k)
            losses['smoothness'] = loss
            total_loss += weights['smoothness'] * loss
        
        # 紧凑性
        if 'compactness' in weights and weights['compactness'] > 0:
            loss = self.compactness_loss(pointcloud)
            losses['compactness'] = loss
            total_loss += weights['compactness'] * loss
        
        losses['total'] = total_loss
        
        return losses


if __name__ == "__main__":
    # 测试代码
    print("=== 测试ShapePriorRegularizer ===")
    
    regularizer = ShapePriorRegularizer(device="cpu")
    
    # 创建测试点云
    test_pc = torch.randn(500, 3)
    
    # 测试各项正则化
    print("\n1. 对称性损失:")
    sym_loss = regularizer.symmetry_loss(test_pc, axis=0)
    print(f"   沿x轴对称损失: {sym_loss.item():.4f}")
    
    print("\n2. 平滑性损失:")
    smooth_loss = regularizer.smoothness_loss(test_pc, k=8)
    print(f"   平滑性损失: {smooth_loss.item():.4f}")
    
    print("\n3. 紧凑性损失:")
    compact_loss = regularizer.compactness_loss(test_pc)
    print(f"   紧凑性损失: {compact_loss.item():.4f}")
    
    print("\n4. 平面性损失:")
    planar_loss = regularizer.planarity_loss(test_pc)
    print(f"   平面性损失: {planar_loss.item():.4f}")
    
    print("\n5. 组合正则化:")
    weights = {
        'symmetry': 0.01,
        'smoothness': 0.05,
        'comp