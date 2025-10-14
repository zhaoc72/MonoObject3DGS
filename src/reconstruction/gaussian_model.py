"""
Gaussian Model
3D Gaussian的基础模型定义
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GaussianConfig:
    """Gaussian参数配置"""
    # 球谐函数阶数
    sh_degree: int = 3
    
    # 初始化参数
    init_scale: float = 0.01
    opacity_init: float = 0.1
    
    # 学习率
    position_lr: float = 0.00016
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    
    # 约束
    min_opacity: float = 0.005
    max_scale: float = 10.0


class GaussianModel:
    """
    3D Gaussian基础模型
    定义Gaussian的基本属性和操作
    """
    
    def __init__(self, config: GaussianConfig):
        """
        Args:
            config: Gaussian配置
        """
        self.config = config
        self.max_sh_degree = config.sh_degree
        self.active_sh_degree = 0  # 从0开始，逐步增加
        
        # Gaussian参数（将在初始化时设置）
        self._xyz = None              # 位置 (N, 3)
        self._features_dc = None      # 直流分量 (N, 1, 3)
        self._features_rest = None    # 高阶球谐 (N, K, 3)
        self._scaling = None          # 尺度 (N, 3)
        self._rotation = None         # 旋转(四元数) (N, 4)
        self._opacity = None          # 不透明度 (N, 1)
        
        # 优化相关
        self.xyz_gradient_accum = None
        self.denom = None
        self.max_radii2D = None
        
        self.optimizer = None
        
    def create_from_points(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None
    ):
        """
        从点云创建Gaussians
        
        Args:
            points: 点云坐标 (N, 3)
            colors: 点云颜色 (N, 3), 范围[0, 1]
        """
        num_points = points.shape[0]
        
        # 1. 位置
        self._xyz = nn.Parameter(
            torch.tensor(points, dtype=torch.float32).contiguous()
        )
        
        # 2. 特征（球谐系数）
        if colors is not None:
            # 将RGB转换为球谐系数的DC分量
            colors = torch.tensor(colors, dtype=torch.float32)
            features_dc = self.rgb_to_sh(colors).unsqueeze(1)  # (N, 1, 3)
        else:
            # 默认灰色
            features_dc = torch.zeros((num_points, 1, 3), dtype=torch.float32)
        
        self._features_dc = nn.Parameter(features_dc.contiguous())
        
        # 高阶球谐系数（初始为0）
        num_rest = (self.max_sh_degree + 1) ** 2 - 1
        features_rest = torch.zeros((num_points, num_rest, 3), dtype=torch.float32)
        self._features_rest = nn.Parameter(features_rest.contiguous())
        
        # 3. 尺度（使用knn距离初始化）
        scales = self._init_scales(points)
        self._scaling = nn.Parameter(
            torch.log(torch.tensor(scales, dtype=torch.float32)).contiguous()
        )
        
        # 4. 旋转（初始为单位四元数）
        rots = torch.zeros((num_points, 4), dtype=torch.float32)
        rots[:, 0] = 1.0  # w=1, x=y=z=0
        self._rotation = nn.Parameter(rots.contiguous())
        
        # 5. 不透明度
        opacities = torch.ones((num_points, 1), dtype=torch.float32) * self.config.opacity_init
        self._opacity = nn.Parameter(
            torch.logit(opacities).contiguous()
        )
        
        # 初始化优化相关变量
        self.xyz_gradient_accum = torch.zeros((num_points, 1), dtype=torch.float32)
        self.denom = torch.zeros((num_points, 1), dtype=torch.float32)
        self.max_radii2D = torch.zeros((num_points), dtype=torch.float32)
        
        print(f"✓ 创建了 {num_points} 个Gaussians")
    
    def _init_scales(self, points: np.ndarray, k: int = 3) -> np.ndarray:
        """
        使用k近邻距离初始化尺度
        
        Args:
            points: 点云 (N, 3)
            k: 近邻数量
            
        Returns:
            scales: 初始尺度 (N, 3)
        """
        from scipy.spatial import cKDTree
        
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=k+1)  # k+1因为包含自己
        
        # 使用平均距离作为尺度
        avg_dist = distances[:, 1:].mean(axis=1)  # 排除自己
        
        # 各向同性尺度
        scales = np.tile(avg_dist[:, None] * self.config.init_scale, (1, 3))
        
        return scales
    
    @staticmethod
    def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
        """
        将RGB颜色转换为球谐系数的DC分量
        
        Args:
            rgb: RGB颜色 (N, 3), 范围[0, 1]
            
        Returns:
            sh_dc: 球谐DC分量 (N, 3)
        """
        C0 = 0.28209479177387814  # 1 / sqrt(4*pi)
        return (rgb - 0.5) / C0
    
    @staticmethod
    def sh_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
        """
        将球谐DC分量转换为RGB颜色
        
        Args:
            sh_dc: 球谐DC分量 (N, 3)
            
        Returns:
            rgb: RGB颜色 (N, 3), 范围[0, 1]
        """
        C0 = 0.28209479177387814
        return sh_dc * C0 + 0.5
    
    @property
    def get_xyz(self) -> torch.Tensor:
        """获取位置"""
        return self._xyz
    
    @property
    def get_features(self) -> torch.Tensor:
        """获取完整的球谐特征"""
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self) -> torch.Tensor:
        """获取不透明度"""
        return torch.sigmoid(self._opacity)
    
    @property
    def get_scaling(self) -> torch.Tensor:
        """获取尺度"""
        return torch.exp(self._scaling)
    
    @property
    def get_rotation(self) -> torch.Tensor:
        """获取归一化后的旋转四元数"""
        return nn.functional.normalize(self._rotation)
    
    @property
    def get_colors(self) -> torch.Tensor:
        """获取RGB颜色（仅DC分量）"""
        return self.sh_to_rgb(self._features_dc.squeeze(1))
    
    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        """
        计算3D协方差矩阵
        
        Args:
            scaling_modifier: 尺度修正因子
            
        Returns:
            covariance: 协方差矩阵 (N, 3, 3)
        """
        return self.build_covariance_from_scaling_rotation(
            self.get_scaling * scaling_modifier,
            self.get_rotation
        )
    
    @staticmethod
    def build_covariance_from_scaling_rotation(
        scaling: torch.Tensor,
        rotation: torch.Tensor
    ) -> torch.Tensor:
        """
        从缩放和旋转构建协方差矩阵
        Σ = R S S^T R^T
        
        Args:
            scaling: 缩放 (N, 3)
            rotation: 旋转四元数 (N, 4)
            
        Returns:
            covariance: 协方差矩阵 (N, 3, 3)
        """
        # 构建缩放矩阵 S
        L = torch.zeros((scaling.shape[0], 3, 3), dtype=torch.float, device=scaling.device)
        L[:, 0, 0] = scaling[:, 0]
        L[:, 1, 1] = scaling[:, 1]
        L[:, 2, 2] = scaling[:, 2]
        
        # 四元数转旋转矩阵
        R = GaussianModel.quat_to_rotmat(rotation)
        
        # Σ = R S S^T R^T = (RS)(RS)^T
        M = R @ L
        Sigma = M @ M.transpose(1, 2)
        
        return Sigma
    
    @staticmethod
    def quat_to_rotmat(quaternions: torch.Tensor) -> torch.Tensor:
        """
        四元数转旋转矩阵
        
        Args:
            quaternions: 四元数 (N, 4) [w, x, y, z]
            
        Returns:
            rotation_matrices: 旋转矩阵 (N, 3, 3)
        """
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        
        R = torch.zeros((quaternions.shape[0], 3, 3), device=quaternions.device, dtype=quaternions.dtype)
        
        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - w*z)
        R[:, 0, 2] = 2 * (x*z + w*y)
        R[:, 1, 0] = 2 * (x*y + w*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - w*x)
        R[:, 2, 0] = 2 * (x*z - w*y)
        R[:, 2, 1] = 2 * (y*z + w*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        
        return R
    
    def get_num_points(self) -> int:
        """获取Gaussian数量"""
        return self._xyz.shape[0] if self._xyz is not None else 0
    
    def prune_points(self, mask: torch.Tensor):
        """
        删除指定的点
        
        Args:
            mask: 布尔mask，True表示保留
        """
        valid_points_mask = mask
        
        self._xyz = nn.Parameter(self._xyz[valid_points_mask])
        self._features_dc = nn.Parameter(self._features_dc[valid_points_mask])
        self._features_rest = nn.Parameter(self._features_rest[valid_points_mask])
        self._opacity = nn.Parameter(self._opacity[valid_points_mask])
        self._scaling = nn.Parameter(self._scaling[valid_points_mask])
        self._rotation = nn.Parameter(self._rotation[valid_points_mask])
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    def densify_and_clone(self, mask: torch.Tensor):
        """
        克隆Gaussians
        
        Args:
            mask: 要克隆的Gaussian mask
        """
        new_xyz = self._xyz[mask]
        new_features_dc = self._features_dc[mask]
        new_features_rest = self._features_rest[mask]
        new_opacities = self._opacity[mask]
        new_scaling = self._scaling[mask]
        new_rotation = self._rotation[mask]
        
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz], dim=0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, new_features_dc], dim=0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, new_features_rest], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, new_opacities], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rotation], dim=0))
        
        num_new = mask.sum().item()
        self.xyz_gradient_accum = torch.cat([
            self.xyz_gradient_accum,
            torch.zeros((num_new, 1), device=self._xyz.device)
        ], dim=0)
        self.denom = torch.cat([
            self.denom,
            torch.zeros((num_new, 1), device=self._xyz.device)
        ], dim=0)
        self.max_radii2D = torch.cat([
            self.max_radii2D,
            torch.zeros((num_new,), device=self._xyz.device)
        ], dim=0)
    
    def densify_and_split(self, mask: torch.Tensor, N: int = 2):
        """
        分裂Gaussians
        
        Args:
            mask: 要分裂的Gaussian mask
            N: 分裂数量
        """
        stds = self.get_scaling[mask].repeat(N, 1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        
        new_xyz = self._xyz[mask].repeat(N, 1) + samples
        new_scaling = self._scaling[mask].repeat(N, 1) - torch.log(torch.tensor([1.6], device=self._xyz.device))
        
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz], dim=0))
        self._features_dc = nn.Parameter(torch.cat([
            self._features_dc,
            self._features_dc[mask].repeat(N, 1, 1)
        ], dim=0))
        self._features_rest = nn.Parameter(torch.cat([
            self._features_rest,
            self._features_rest[mask].repeat(N, 1, 1)
        ], dim=0))
        self._opacity = nn.Parameter(torch.cat([
            self._opacity,
            self._opacity[mask].repeat(N, 1)
        ], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling], dim=0))
        self._rotation = nn.Parameter(torch.cat([
            self._rotation,
            self._rotation[mask].repeat(N, 1)
        ], dim=0))
        
        num_new = mask.sum().item() * N
        self.xyz_gradient_accum = torch.cat([
            self.xyz_gradient_accum,
            torch.zeros((num_new, 1), device=self._xyz.device)
        ], dim=0)
        self.denom = torch.cat([
            self.denom,
            torch.zeros((num_new, 1), device=self._xyz.device)
        ], dim=0)
        self.max_radii2D = torch.cat([
            self.max_radii2D,
            torch.zeros((num_new,), device=self._xyz.device)
        ], dim=0)


if __name__ == "__main__":
    # 测试代码
    print("=== 测试GaussianModel ===")
    
    # 创建配置
    config = GaussianConfig()
    
    # 创建模型
    model = GaussianModel(config)
    
    # 从点云创建
    points = np.random.rand(1000, 3) * 2 - 1  # [-1, 1]范围
    colors = np.random.rand(1000, 3)
    
    model.create_from_points(points, colors)
    
    print(f"Gaussian数量: {model.get_num_points()}")
    print(f"位置形状: {model.get_xyz.shape}")
    print(f"特征形状: {model.get_features.shape}")
    print(f"不透明度范围: [{model.get_opacity.min():.3f}, {model.get_opacity.max():.3f}]")
    print(f"尺度范围: [{model.get_scaling.min():.3f}, {model.get_scaling.max():.3f}]")
    
    # 测试协方差矩阵
    cov = model.get_covariance()
    print(f"协方差矩阵形状: {cov.shape}")
    
    print("\n测试完成！")