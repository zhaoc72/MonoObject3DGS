"""
Gaussian Model
3D Gaussian Splatting基础模型
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GaussianConfig:
    """Gaussian配置"""
    sh_degree: int = 3
    init_scale: float = 0.01
    opacity_init: float = 0.1
    scale_activation: str = "exp"
    opacity_activation: str = "sigmoid"
    rotation_activation: str = "normalize"


class GaussianModel(nn.Module):
    """3D Gaussian Splatting模型"""
    
    def __init__(self, config: GaussianConfig):
        super().__init__()
        
        self.config = config
        self.sh_degree = config.sh_degree
        
        # Gaussian参数（初始为空）
        self._xyz = torch.empty(0, 3)
        self._features_dc = torch.empty(0, 1, 3)
        self._features_rest = torch.empty(0, (self.sh_degree + 1) ** 2 - 1, 3)
        self._scaling = torch.empty(0, 3)
        self._rotation = torch.empty(0, 4)
        self._opacity = torch.empty(0, 1)
        
        # 优化相关
        self.xyz_gradient_accum = torch.empty(0, 1)
        self.denom = torch.empty(0, 1)
        self.max_radii2D = torch.empty(0, 1)
        
        self.active_sh_degree = 0
    
    def create_from_points(
        self,
        points: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None
    ):
        """从点云初始化Gaussian"""
        num_points = points.shape[0]
        
        # 位置
        self._xyz = nn.Parameter(points.clone().requires_grad_(True))
        
        # 颜色特征（球谐函数）
        if colors is not None:
            colors = colors.clamp(0, 1)
            features_dc = self.rgb_to_sh(colors).unsqueeze(1)
        else:
            features_dc = torch.ones((num_points, 1, 3)) * 0.5
        
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        
        features_rest = torch.zeros((num_points, (self.sh_degree + 1) ** 2 - 1, 3))
        self._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        
        # 尺度
        if scales is None:
            scales = torch.ones((num_points, 3)) * self.config.init_scale
        else:
            scales = torch.clamp(scales, min=self.config.init_scale)
        
        scales_log = torch.log(scales)
        self._scaling = nn.Parameter(scales_log.requires_grad_(True))
        
        # 旋转（四元数，初始化为单位旋转）
        rotation = torch.zeros((num_points, 4))
        rotation[:, 0] = 1.0
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        
        # 不透明度
        opacity_logit = self.inverse_sigmoid(
            torch.ones((num_points, 1)) * self.config.opacity_init
        )
        self._opacity = nn.Parameter(opacity_logit.requires_grad_(True))
        
        # 优化状态
        self.xyz_gradient_accum = torch.zeros((num_points, 1))
        self.denom = torch.zeros((num_points, 1))
        self.max_radii2D = torch.zeros((num_points, 1))
        
        print(f"  Created {num_points} Gaussians")
    
    @property
    def get_xyz(self) -> torch.Tensor:
        """获取位置"""
        return self._xyz
    
    @property
    def get_features(self) -> torch.Tensor:
        """获取特征"""
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self) -> torch.Tensor:
        """获取不透明度"""
        return self.opacity_activation(self._opacity)
    
    @property
    def get_scaling(self) -> torch.Tensor:
        """获取尺度"""
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self) -> torch.Tensor:
        """获取旋转"""
        return self.rotation_activation(self._rotation)
    
    def opacity_activation(self, x: torch.Tensor) -> torch.Tensor:
        """不透明度激活"""
        return torch.sigmoid(x)
    
    def scaling_activation(self, x: torch.Tensor) -> torch.Tensor:
        """尺度激活"""
        return torch.exp(x)
    
    def rotation_activation(self, x: torch.Tensor) -> torch.Tensor:
        """旋转激活（归一化四元数）"""
        return x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
    
    @staticmethod
    def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
        """Sigmoid逆函数"""
        return torch.log(x / (1 - x + 1e-8) + 1e-8)
    
    @staticmethod
    def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
        """RGB转球谐系数（0阶）"""
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0
    
    @staticmethod
    def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
        """球谐系数转RGB"""
        C0 = 0.28209479177387814
        return sh * C0 + 0.5
    
    def densification_postfix(
        self,
        new_xyz: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_opacities: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor
    ):
        """致密化后处理"""
        d = {
            "_xyz": new_xyz,
            "_features_dc": new_features_dc,
            "_features_rest": new_features_rest,
            "_opacity": new_opacities,
            "_scaling": new_scaling,
            "_rotation": new_rotation
        }
        
        # 优化状态
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in d:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                optimizable_tensors[group["name"]] = (stored_state, d[group["name"]])
        
        # 扩展优化状态
        for name, (old_state, new_param) in optimizable_tensors.items():
            if old_state is not None:
                for key in old_state:
                    if key != "step":
                        old_state[key] = torch.cat(
                            [old_state[key], torch.zeros_like(new_param)],
                            dim=0
                        )
        
        # 更新参数
        for name, param in d.items():
            setattr(self, name, nn.Parameter(param.requires_grad_(True)))
    
    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        N: int = 2
    ):
        """根据梯度分裂Gaussian"""
        n_init_points = self.get_xyz.shape[0]
        
        # 选择要分裂的点
        padded_grad = torch.zeros((n_init_points,), device=grads.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > scene_extent * 0.01
        )
        
        if selected_pts_mask.sum() == 0:
            return
        
        # 分裂
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=stds.device)
        samples = torch.normal(mean=means, std=stds)
        rots = self.get_rotation[selected_pts_mask].repeat(N, 1)
        
        new_xyz = torch.bmm(
            self._build_rotation_matrix(rots),
            samples.unsqueeze(-1)
        ).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        
        new_scaling = self._scaling[selected_pts_mask].repeat(N, 1) - np.log(0.8 * N)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        
        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation
        )
        
        # 删除原始点
        prune_filter = torch.cat([
            selected_pts_mask,
            torch.zeros(N * selected_pts_mask.sum(),
                       dtype=bool, device=selected_pts_mask.device)
        ])
        self.prune_points(prune_filter)
    
    def densify_and_clone(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float
    ):
        """克隆Gaussian"""
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= scene_extent * 0.01
        )
        
        if selected_pts_mask.sum() == 0:
            return
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation
        )
    
    def prune_points(self, mask: torch.Tensor):
        """删除Gaussian"""
        valid_points_mask = ~mask
        
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                for key in stored_state:
                    if key != "step":
                        stored_state[key] = stored_state[key][valid_points_mask]
            
            param_name = group["name"]
            if hasattr(self, param_name):
                param_value = getattr(self, param_name)
                optimizable_tensors[param_name] = param_value[valid_points_mask]
        
        for name, param in optimizable_tensors.items():
            setattr(self, name, nn.Parameter(param.requires_grad_(True)))
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    @staticmethod
    def _build_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
        """从四元数构建旋转矩阵"""
        norm = torch.sqrt(
            q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] +
            q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
        )
        q = q / norm[:, None]
        
        r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        R = torch.zeros((q.size(0), 3, 3), device=q.device)
        
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - r * z)
        R[:, 0, 2] = 2 * (x * z + r * y)
        R[:, 1, 0] = 2 * (x * y + r * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - r * x)
        R[:, 2, 0] = 2 * (x * z - r * y)
        R[:, 2, 1] = 2 * (y * z + r * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        
        return R
    
    def reset_opacity(self):
        """重置不透明度"""
        opacities_new = self.inverse_sigmoid(
            torch.ones_like(self._opacity) * 0.01
        )
        
        self._opacity = nn.Parameter(opacities_new.requires_grad_(True))
    
    @property
    def num_points(self) -> int:
        """Gaussian数量"""
        return self._xyz.shape[0]


# 测试
if __name__ == "__main__":
    config = GaussianConfig(sh_degree=3)
    model = GaussianModel(config)
    
    # 创建测试点云
    points = torch.randn(100, 3)
    colors = torch.rand(100, 3)
    
    model.create_from_points(points, colors)
    
    print(f"Num Gaussians: {model.num_points}")
    print(f"XYZ shape: {model.get_xyz.shape}")
    print(f"Features shape: {model.get_features.shape}")
    print(f"Opacity shape: {model.get_opacity.shape}")