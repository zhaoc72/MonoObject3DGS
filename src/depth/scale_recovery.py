"""
Scale Recovery
从相对深度恢复绝对尺度
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class ScaleRecovery:
    """尺度恢复器"""
    
    def __init__(
        self,
        method: str = "shape_prior",
        reference_objects: Optional[List[str]] = None
    ):
        self.method = method
        
        if reference_objects is None:
            self.reference_objects = ["chair", "table", "person"]
        else:
            self.reference_objects = reference_objects
        
        # 物体典型尺寸（米）
        self.object_sizes = {
            "chair": {"height": 0.9, "width": 0.5, "depth": 0.5},
            "table": {"height": 0.75, "width": 1.5, "depth": 0.8},
            "sofa": {"height": 0.85, "width": 2.0, "depth": 0.9},
            "bed": {"height": 0.5, "width": 2.0, "depth": 1.5},
            "desk": {"height": 0.75, "width": 1.2, "depth": 0.6},
            "person": {"height": 1.7, "width": 0.5, "depth": 0.3},
            "door": {"height": 2.0, "width": 0.9, "depth": 0.05},
            "tv": {"height": 0.7, "width": 1.2, "depth": 0.1}
        }
        
        print(f"✓ ScaleRecovery initialized: {method}")
    
    def recover_scale(
        self,
        depth: np.ndarray,
        objects: List[Dict],
        camera_params: Optional[Dict] = None
    ) -> Tuple[np.ndarray, float]:
        """
        恢复深度尺度
        
        Args:
            depth: 相对深度图 (H, W)
            objects: 检测到的物体列表
            camera_params: 相机参数
            
        Returns:
            metric_depth: 度量深度图
            scale_factor: 尺度因子
        """
        if self.method == "shape_prior":
            return self._recover_from_shape_prior(depth, objects, camera_params)
        elif self.method == "manual":
            return self._recover_manual(depth)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _recover_from_shape_prior(
        self,
        depth: np.ndarray,
        objects: List[Dict],
        camera_params: Optional[Dict] = None
    ) -> Tuple[np.ndarray, float]:
        """使用物体先验恢复尺度"""
        scale_factors = []
        
        for obj in objects:
            category = obj.get('category', 'unknown')
            
            if category not in self.reference_objects:
                continue
            
            if category not in self.object_sizes:
                continue
            
            typical_size = self.object_sizes[category]
            
            scale_factor = self._estimate_scale_from_object(
                obj, depth, typical_size, camera_params
            )
            
            if scale_factor is not None:
                scale_factors.append(scale_factor)
        
        if len(scale_factors) == 0:
            print("Warning: No reference objects found, using default scale")
            scale_factor = 1.0
        else:
            scale_factor = np.median(scale_factors)
        
        metric_depth = depth * scale_factor
        
        print(f"  Scale factor: {scale_factor:.3f}")
        print(f"  Used {len(scale_factors)} reference objects")
        
        return metric_depth, scale_factor
    
    def _estimate_scale_from_object(
        self,
        obj: Dict,
        depth: np.ndarray,
        typical_size: Dict,
        camera_params: Optional[Dict] = None
    ) -> Optional[float]:
        """从单个物体估计尺度"""
        mask = obj['segmentation']
        bbox = obj['bbox']
        x, y, w, h = bbox
        
        # 获取物体深度
        if mask is not None:
            obj_depth = depth[mask]
            if len(obj_depth) == 0:
                return None
            median_depth = np.median(obj_depth)
        else:
            cy, cx = y + h // 2, x + w // 2
            median_depth = depth[cy, cx]
        
        # 估计物体真实高度
        if camera_params is not None:
            fy = camera_params.get('fy', 525.0)
        else:
            fy = depth.shape[0] / (2 * np.tan(np.radians(30)))
        
        estimated_height = (h * median_depth) / fy
        
        # 计算尺度因子
        scale_factor = typical_size["height"] / (estimated_height + 1e-6)
        
        # 合理性检查
        if scale_factor < 0.1 or scale_factor > 10.0:
            return None
        
        return scale_factor
    
    def _recover_manual(
        self,
        depth: np.ndarray,
        manual_scale: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """手动指定尺度"""
        metric_depth = depth * manual_scale
        return metric_depth, manual_scale


# 测试
if __name__ == "__main__":
    recovery = ScaleRecovery()
    
    # 测试数据
    H, W = 480, 640
    depth = np.random.rand(H, W) * 0.5 + 0.5
    
    objects = [{
        'category': 'chair',
        'bbox': [200, 150, 80, 120],
        'segmentation': np.zeros((H, W), dtype=bool)
    }]
    objects[0]['segmentation'][150:270, 200:280] = True
    
    camera_params = {
        'fx': 525.0, 'fy': 525.0,
        'cx': 320.0, 'cy': 240.0
    }
    
    metric_depth, scale = recovery.recover_scale(depth, objects, camera_params)
    
    print(f"Original depth: [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"Metric depth: [{metric_depth.min():.3f}, {metric_depth.max():.3f}] m")