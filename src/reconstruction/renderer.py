"""
Gaussian Renderer
简化的Gaussian渲染器
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class GaussianRenderer:
    """简化的Gaussian渲染器"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("✓ GaussianRenderer initialized (simplified)")
    
    def render(
        self,
        gaussians,
        camera_params: Dict,
        background: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        渲染Gaussian
        
        这是简化版本，实际生产环境应使用diff-gaussian-rasterization
        """
        # 获取Gaussian参数
        xyz = gaussians.get_xyz
        features = gaussians.get_features
        opacity = gaussians.get_opacity
        scaling = gaussians.get_scaling
        rotation = gaussians.get_rotation
        
        # 简化渲染：投影到2D并栅格化
        H, W = camera_params.get('height', 512), camera_params.get('width', 512)
        
        # 投影到相机坐标
        projected = self._project_to_camera(xyz, camera_params)
        
        # 创建图像
        if background is None:
            background = torch.zeros((H, W, 3), device=self.device)
        
        rendered_image = background.clone()
        rendered_depth = torch.zeros((H, W), device=self.device)
        
        # 简化：基于深度排序并splat
        depths = projected[:, 2]
        sorted_indices = torch.argsort(depths, descending=True)
        
        # 转换特征为RGB
        colors = self._sh_to_rgb(features[:, 0, :])
        
        for idx in sorted_indices[:1000]:  # 限制数量避免过慢
            x, y, z = projected[idx]
            
            if z <= 0:
                continue
            
            px, py = int(x), int(y)
            
            if 0 <= px < W and 0 <= py < H:
                alpha = opacity[idx].item()
                color = colors[idx]
                
                # Alpha混合
                rendered_image[py, px] = (
                    alpha * color + (1 - alpha) * rendered_image[py, px]
                )
                rendered_depth[py, px] = z
        
        return {
            'image': rendered_image,
            'depth': rendered_depth,
            'projected': projected
        }
    
    def _project_to_camera(
        self,
        points_3d: torch.Tensor,
        camera_params: Dict
    ) -> torch.Tensor:
        """投影3D点到相机"""
        fx = camera_params.get('fx', 525.0)
        fy = camera_params.get('fy', 525.0)
        cx = camera_params.get('cx', 320.0)
        cy = camera_params.get('cy', 240.0)
        
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]
        
        u = fx * x / (z + 1e-6) + cx
        v = fy * y / (z + 1e-6) + cy
        
        projected = torch.stack([u, v, z], dim=1)
        
        return projected
    
    def _sh_to_rgb(self, sh: torch.Tensor) -> torch.Tensor:
        """球谐系数转RGB"""
        C0 = 0.28209479177387814
        rgb = sh * C0 + 0.5
        return torch.clamp(rgb, 0, 1)


# 测试
if __name__ == "__main__":
    from src.reconstruction.gaussian_model import GaussianModel, GaussianConfig
    
    renderer = GaussianRenderer(device="cpu")
    
    config = GaussianConfig()
    model = GaussianModel(config)
    
    points = torch.randn(100, 3) * 2
    points[:, 2] += 5  # 移到相机前方
    colors = torch.rand(100, 3)
    
    model.create_from_points(points, colors)
    
    camera_params = {
        'fx': 525.0, 'fy': 525.0,
        'cx': 256.0, 'cy': 256.0,
        'width': 512, 'height': 512
    }
    
    result = renderer.render(model, camera_params)
    print(f"Rendered image shape: {result['image'].shape}")
    print(f"Rendered depth shape: {result['depth'].shape}")