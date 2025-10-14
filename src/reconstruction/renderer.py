"""
Gaussian Renderer
3D Gaussian Splatting渲染器
封装diff-gaussian-rasterization库
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


class GaussianRenderer:
    """
    3D Gaussian Splatting渲染器
    使用可微分光栅化渲染Gaussians
    """
    
    def __init__(
        self,
        image_height: int = 512,
        image_width: int = 512,
        device: str = "cuda"
    ):
        """
        Args:
            image_height: 渲染图像高度
            image_width: 渲染图像宽度
            device: 计算设备
        """
        self.image_height = image_height
        self.image_width = image_width
        self.device = device
        
        # 尝试导入diff-gaussian-rasterization
        try:
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
            self.rasterization_available = True
            self.GaussianRasterizationSettings = GaussianRasterizationSettings
            self.GaussianRasterizer = GaussianRasterizer
            print("✓ GaussianRenderer初始化: 使用diff-gaussian-rasterization")
        except ImportError:
            self.rasterization_available = False
            print("警告: diff-gaussian-rasterization未安装，使用简化渲染器")
    
    def render(
        self,
        gaussians: Dict[str, torch.Tensor],
        camera_params: Dict,
        background_color: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        渲染Gaussians
        
        Args:
            gaussians: Gaussian参数字典，包含:
                - xyz: 位置 (N, 3)
                - features_dc: 球谐DC (N, 1, 3)
                - features_rest: 球谐其余 (N, K, 3)
                - scaling: 尺度 (N, 3) [已exp]
                - rotation: 旋转 (N, 4) [已归一化]
                - opacity: 不透明度 (N, 1) [已sigmoid]
            camera_params: 相机参数
            background_color: 背景颜色
            
        Returns:
            rendered: 渲染结果
                - rgb: RGB图像 (3, H, W)
                - depth: 深度图 (1, H, W)
                - alpha: Alpha通道 (1, H, W)
        """
        if self.rasterization_available:
            return self._render_rasterization(gaussians, camera_params, background_color)
        else:
            return self._render_simple(gaussians, camera_params, background_color)
    
    def _render_rasterization(
        self,
        gaussians: Dict[str, torch.Tensor],
        camera_params: Dict,
        background_color: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """使用diff-gaussian-rasterization渲染"""
        
        # 设置背景
        if background_color is None:
            background_color = torch.ones(3, device=self.device)
        
        # 提取参数
        xyz = gaussians['xyz']
        features_dc = gaussians['features_dc']
        features_rest = gaussians.get('features_rest', torch.zeros_like(features_dc))
        scaling = gaussians['scaling']
        rotation = gaussians['rotation']
        opacity = gaussians['opacity']
        
        # 合并特征
        features = torch.cat([features_dc, features_rest], dim=1)
        
        # 相机参数
        viewmatrix = camera_params['view_matrix']  # (4, 4)
        projmatrix = camera_params['proj_matrix']   # (4, 4)
        cam_pos = camera_params['camera_center']    # (3,)
        
        fx = camera_params['fx']
        fy = camera_params['fy']
        
        # 设置光栅化参数
        raster_settings = self.GaussianRasterizationSettings(
            image_height=self.image_height,
            image_width=self.image_width,
            tanfovx=self.image_width / (2 * fx),
            tanfovy=self.image_height / (2 * fy),
            bg=background_color,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=1,  # 简化，使用1阶球谐
            campos=cam_pos,
            prefiltered=False,
            debug=False
        )
        
        # 创建光栅化器
        rasterizer = self.GaussianRasterizer(raster_settings=raster_settings)
        
        # 渲染
        rendered_image, radii = rasterizer(
            means3D=xyz,
            means2D=torch.zeros_like(xyz[:, :2]),  # 屏幕空间坐标（内部计算）
            shs=features,
            colors_precomp=None,
            opacities=opacity,
            scales=scaling,
            rotations=rotation,
            cov3D_precomp=None
        )
        
        # 简单的深度渲染（需要扩展）
        depth = self._render_depth_simple(xyz, viewmatrix, opacity)
        
        return {
            'rgb': rendered_image,
            'depth': depth,
            'alpha': rendered_image.sum(dim=0, keepdim=True).clamp(0, 1),
            'radii': radii
        }
    
    def _render_simple(
        self,
        gaussians: Dict[str, torch.Tensor],
        camera_params: Dict,
        background_color: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        简化渲染器（当diff-gaussian-rasterization不可用时）
        使用简单的点云投影
        """
        xyz = gaussians['xyz']
        opacity = gaussians['opacity']
        
        # 获取颜色（从球谐DC分量）
        features_dc = gaussians['features_dc']
        C0 = 0.28209479177387814
        colors = features_dc.squeeze(1) * C0 + 0.5
        colors = colors.clamp(0, 1)
        
        # 投影到图像平面
        viewmatrix = camera_params['view_matrix']
        
        # 转换到相机坐标系
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)  # (N, 4)
        xyz_cam = (viewmatrix @ xyz_h.T).T  # (N, 4)
        
        # 透视投影
        fx = camera_params['fx']
        fy = camera_params['fy']
        cx = camera_params.get('cx', self.image_width / 2)
        cy = camera_params.get('cy', self.image_height / 2)
        
        x_proj = xyz_cam[:, 0] / (xyz_cam[:, 2] + 1e-6) * fx + cx
        y_proj = xyz_cam[:, 1] / (xyz_cam[:, 2] + 1e-6) * fy + cy
        z_proj = xyz_cam[:, 2]
        
        # 过滤在图像外的点
        valid = (x_proj >= 0) & (x_proj < self.image_width) & \
                (y_proj >= 0) & (y_proj < self.image_height) & \
                (z_proj > 0)
        
        # 创建图像
        rendered_rgb = torch.ones(3, self.image_height, self.image_width, device=self.device)
        rendered_depth = torch.zeros(1, self.image_height, self.image_width, device=self.device)
        
        if background_color is not None:
            rendered_rgb = background_color.view(3, 1, 1).expand(3, self.image_height, self.image_width)
        
        if valid.any():
            x_proj = x_proj[valid].long()
            y_proj = y_proj[valid].long()
            z_proj = z_proj[valid]
            colors_valid = colors[valid]
            opacity_valid = opacity[valid].squeeze()
            
            # 简单的splat（可以改进）
            for i in range(len(x_proj)):
                x, y = x_proj[i], y_proj[i]
                if 0 <= x < self.image_width and 0 <= y < self.image_height:
                    # Alpha混合
                    alpha = opacity_valid[i]
                    rendered_rgb[:, y, x] = (1 - alpha) * rendered_rgb[:, y, x] + alpha * colors_valid[i]
                    rendered_depth[:, y, x] = max(rendered_depth[:, y, x], z_proj[i])
        
        return {
            'rgb': rendered_rgb,
            'depth': rendered_depth,
            'alpha': (rendered_depth > 0).float()
        }
    
    def _render_depth_simple(
        self,
        xyz: torch.Tensor,
        viewmatrix: torch.Tensor,
        opacity: torch.Tensor
    ) -> torch.Tensor:
        """简单的深度渲染"""
        # 转换到相机坐标
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
        xyz_cam = (viewmatrix @ xyz_h.T).T
        depths = xyz_cam[:, 2]
        
        # 加权平均深度（简化）
        weights = opacity.squeeze()
        weighted_depth = (depths * weights).sum() / (weights.sum() + 1e-6)
        
        depth_map = torch.ones(1, self.image_height, self.image_width, device=self.device) * weighted_depth
        
        return depth_map


def create_camera_params(
    view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor,
    camera_center: torch.Tensor,
    fx: float,
    fy: float,
    cx: Optional[float] = None,
    cy: Optional[float] = None
) -> Dict:
    """
    创建相机参数字典
    
    Args:
        view_matrix: 视图矩阵 (4, 4)
        proj_matrix: 投影矩阵 (4, 4)
        camera_center: 相机中心 (3,)
        fx, fy: 焦距
        cx, cy: 主点
        
    Returns:
        camera_params: 相机参数字典
    """
    return {
        'view_matrix': view_matrix,
        'proj_matrix': proj_matrix,
        'camera_center': camera_center,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }


if __name__ == "__main__":
    # 测试代码
    print("=== 测试GaussianRenderer ===")
    
    # 创建渲染器
    renderer = GaussianRenderer(image_height=256, image_width=256, device="cpu")
    
    # 创建测试Gaussians
    N = 1000
    gaussians = {
        'xyz': torch.randn(N, 3),
        'features_dc': torch.randn(N, 1, 3),
        'features_rest': torch.zeros(N, 0, 3),
        'scaling': torch.ones(N, 3) * 0.01,
        'rotation': torch.cat([torch.ones(N, 1), torch.zeros(N, 3)], dim=1),
        'opacity': torch.ones(N, 1) * 0.5
    }
    
    # 创建相机参数
    view_matrix = torch.eye(4)
    view_matrix[2, 3] = -5  # 相机在z=-5位置
    
    proj_matrix = torch.eye(4)
    
    camera_params = create_camera_params(
        view_matrix=view_matrix,
        proj_matrix=proj_matrix,
        camera_center=torch.tensor([0, 0, -5]),
        fx=525.0,
        fy=525.0
    )
    
    # 渲染
    rendered = renderer.render(gaussians, camera_params)
    
    print(f"渲染结果:")
    print(f"  RGB形状: {rendered['rgb'].shape}")
    print(f"  深度形状: {rendered['depth'].shape}")
    print(f"  Alpha形状: {rendered['alpha'].shape}")
    
    print("\n测试完成！")