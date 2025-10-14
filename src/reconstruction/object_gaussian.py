"""
Object Gaussian
物体级3D Gaussian
"""

import torch
import numpy as np
from typing import Dict, Optional
from .gaussian_model import GaussianModel, GaussianConfig


class ObjectGaussian(GaussianModel):
    """物体级Gaussian模型"""
    
    def __init__(
        self,
        object_id: int,
        category: str,
        config: GaussianConfig,
        shape_prior: Optional[torch.Tensor] = None
    ):
        super().__init__(config)
        
        self.object_id = object_id
        self.category = category
        self.shape_prior = shape_prior
        
        # 物体特定信息
        self.observations = []
        self.confidence = 0.0
        self.num_views = 0
    
    def initialize_from_mask_depth(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray,
        image: np.ndarray,
        camera_params: Dict
    ):
        """从mask和深度初始化"""
        # 获取mask内的像素
        ys, xs = np.where(mask)
        
        if len(xs) == 0:
            raise ValueError("Empty mask")
        
        # 采样点（避免过多）
        max_points = 10000
        if len(xs) > max_points:
            indices = np.random.choice(len(xs), max_points, replace=False)
            xs = xs[indices]
            ys = ys[indices]
        
        # 获取深度和颜色
        depths = depth_map[ys, xs]
        colors = image[ys, xs] / 255.0
        
        # 反投影到3D
        fx = camera_params['fx']
        fy = camera_params['fy']
        cx = camera_params['cx']
        cy = camera_params['cy']
        
        points_3d = np.zeros((len(xs), 3), dtype=np.float32)
        points_3d[:, 0] = (xs - cx) * depths / fx
        points_3d[:, 1] = (ys - cy) * depths / fy
        points_3d[:, 2] = depths
        
        # 转换为tensor
        points = torch.from_numpy(points_3d)
        colors = torch.from_numpy(colors.astype(np.float32))
        
        # 估计初始尺度
        scales = self._estimate_scales(points, k=8)
        
        # 创建Gaussian
        self.create_from_points(points, colors, scales)
        
        print(f"  Object {self.object_id} ({self.category}): "
              f"{self.num_points} points initialized")
    
    def _estimate_scales(
        self,
        points: torch.Tensor,
        k: int = 8
    ) -> torch.Tensor:
        """估计初始尺度（基于k近邻）"""
        try:
            # 简化：使用固定尺度
            scales = torch.ones((len(points), 3)) * self.config.init_scale
            return scales
        except:
            return torch.ones((len(points), 3)) * self.config.init_scale
    
    def update_from_observation(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray,
        image: np.ndarray,
        camera_params: Dict
    ):
        """从新观测更新"""
        self.observations.append({
            'mask': mask,
            'depth': depth_map,
            'image': image
        })
        self.num_views += 1
        
        # 可以在这里实现增量更新逻辑
        # 现在先简单记录
    
    def get_bounding_box(self) -> Dict:
        """获取包围盒"""
        xyz = self.get_xyz.detach().cpu().numpy()
        
        min_bound = xyz.min(axis=0)
        max_bound = xyz.max(axis=0)
        center = (min_bound + max_bound) / 2
        size = max_bound - min_bound
        
        return {
            'min': min_bound,
            'max': max_bound,
            'center': center,
            'size': size
        }
    
    def compute_volume(self) -> float:
        """计算体积（近似）"""
        bbox = self.get_bounding_box()
        size = bbox['size']
        return float(np.prod(size))
    
    def export_to_dict(self) -> Dict:
        """导出为字典"""
        return {
            'object_id': self.object_id,
            'category': self.category,
            'num_points': self.num_points,
            'num_views': self.num_views,
            'confidence': self.confidence,
            'bounding_box': self.get_bounding_box(),
            'volume': self.compute_volume()
        }


# 测试
if __name__ == "__main__":
    config = GaussianConfig()
    obj_gaussian = ObjectGaussian(
        object_id=0,
        category="chair",
        config=config
    )
    
    # 创建测试数据
    H, W = 480, 640
    mask = np.zeros((H, W), dtype=bool)
    mask[200:300, 250:350] = True
    
    depth = np.random.rand(H, W) * 3 + 2
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
    camera_params = {
        'fx': 525.0, 'fy': 525.0,
        'cx': 320.0, 'cy': 240.0
    }
    
    obj_gaussian.initialize_from_mask_depth(mask, depth, image, camera_params)
    
    print(f"Bounding box: {obj_gaussian.get_bounding_box()}")
    print(f"Volume: {obj_gaussian.compute_volume():.3f}")