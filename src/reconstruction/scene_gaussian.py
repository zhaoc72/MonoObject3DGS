"""
Scene Gaussians
场景级Gaussian管理
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional
from .object_gaussian import ObjectGaussian
from .gaussian_model import GaussianConfig


class SceneGaussians:
    """场景Gaussian管理器"""
    
    def __init__(self, config: GaussianConfig):
        self.config = config
        self.objects: Dict[int, ObjectGaussian] = {}
        
        print("✓ SceneGaussians initialized")
    
    def add_object(self, object_id: int, obj_gaussian: ObjectGaussian):
        """添加物体"""
        self.objects[object_id] = obj_gaussian
        print(f"  Added object {object_id}: {obj_gaussian.category}, "
              f"{obj_gaussian.num_points} gaussians")
    
    def get_object(self, object_id: int) -> Optional[ObjectGaussian]:
        """获取物体"""
        return self.objects.get(object_id)
    
    def has_object(self, object_id: int) -> bool:
        """检查物体是否存在"""
        return object_id in self.objects
    
    def remove_object(self, object_id: int):
        """移除物体"""
        if object_id in self.objects:
            del self.objects[object_id]
            print(f"  Removed object {object_id}")
    
    @property
    def total_gaussians(self) -> int:
        """总Gaussian数量"""
        return sum(obj.num_points for obj in self.objects.values())
    
    def get_all_points(self) -> torch.Tensor:
        """获取所有点"""
        if len(self.objects) == 0:
            return torch.empty(0, 3)
        
        points = []
        for obj in self.objects.values():
            points.append(obj.get_xyz)
        
        return torch.cat(points, dim=0)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'num_objects': len(self.objects),
            'total_gaussians': self.total_gaussians,
            'objects': {}
        }
        
        for obj_id, obj in self.objects.items():
            stats['objects'][obj_id] = obj.export_to_dict()
        
        return stats
    
    def save(self, save_dir: str):
        """保存场景"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存元数据
        metadata = self.get_statistics()
        with open(save_path / 'scene_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        
        # 保存每个物体
        for obj_id, obj in self.objects.items():
            obj_path = save_path / f"object_{obj_id}"
            obj_path.mkdir(exist_ok=True)
            
            # 保存点云为PLY
            self._save_ply(
                obj_path / f"object_{obj_id}.ply",
                obj.get_xyz.detach().cpu().numpy(),
                obj.get_opacity.detach().cpu().numpy(),
                obj.sh_to_rgb(obj._features_dc.squeeze(1)).detach().cpu().numpy()
            )
            
            # 保存参数
            torch.save({
                'xyz': obj._xyz,
                'features_dc': obj._features_dc,
                'features_rest': obj._features_rest,
                'scaling': obj._scaling,
                'rotation': obj._rotation,
                'opacity': obj._opacity,
                'config': self.config
            }, obj_path / 'gaussian_params.pt')
        
        print(f"✓ Scene saved to {save_path}")
    
    def load(self, load_dir: str):
        """加载场景"""
        load_path = Path(load_dir)
        
        # 加载元数据
        with open(load_path / 'scene_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # 加载每个物体
        for obj_id_str in metadata['objects'].keys():
            obj_id = int(obj_id_str)
            obj_info = metadata['objects'][obj_id_str]
            
            obj_path = load_path / f"object_{obj_id}"
            params = torch.load(obj_path / 'gaussian_params.pt')
            
            # 重建物体
            obj_gaussian = ObjectGaussian(
                object_id=obj_id,
                category=obj_info['category'],
                config=self.config
            )
            
            obj_gaussian._xyz = params['xyz']
            obj_gaussian._features_dc = params['features_dc']
            obj_gaussian._features_rest = params['features_rest']
            obj_gaussian._scaling = params['scaling']
            obj_gaussian._rotation = params['rotation']
            obj_gaussian._opacity = params['opacity']
            
            self.objects[obj_id] = obj_gaussian
        
        print(f"✓ Scene loaded from {load_path}")
        print(f"  {len(self.objects)} objects, {self.total_gaussians} gaussians")
    
    @staticmethod
    def _save_ply(
        path: Path,
        xyz: np.ndarray,
        opacity: np.ndarray,
        rgb: np.ndarray
    ):
        """保存PLY文件"""
        try:
            from plyfile import PlyData, PlyElement
            
            # 准备数据
            opacity = (opacity * 255).astype(np.uint8)
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            
            vertices = np.zeros(
                len(xyz),
                dtype=[
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                    ('alpha', 'u1')
                ]
            )
            
            vertices['x'] = xyz[:, 0]
            vertices['y'] = xyz[:, 1]
            vertices['z'] = xyz[:, 2]
            vertices['red'] = rgb[:, 0]
            vertices['green'] = rgb[:, 1]
            vertices['blue'] = rgb[:, 2]
            vertices['alpha'] = opacity[:, 0]
            
            el = PlyElement.describe(vertices, 'vertex')
            PlyData([el]).write(str(path))
            
        except ImportError:
            print("Warning: plyfile not installed, skipping PLY export")
        except Exception as e:
            print(f"Warning: Failed to save PLY: {e}")


# 测试
if __name__ == "__main__":
    import numpy as np
    
    config = GaussianConfig()
    scene = SceneGaussians(config)
    
    # 添加测试物体
    for i in range(3):
        obj = ObjectGaussian(i, f"object_{i}", config)
        points = torch.randn(100, 3)
        obj.create_from_points(points)
        scene.add_object(i, obj)
    
    stats = scene.get_statistics()
    print(f"\nStatistics:")
    print(f"  Objects: {stats['num_objects']}")
    print(f"  Total Gaussians: {stats['total_gaussians']}")
    
    # 测试保存
    scene.save("test_scene")