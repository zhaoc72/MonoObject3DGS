"""
Scene Gaussians
场景级Gaussian管理器，管理多个物体的Gaussians
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from pathlib import Path
import json

from .object_gaussian import ObjectGaussian
from .gaussian_model import GaussianConfig


class SceneGaussians(nn.Module):
    """
    场景级Gaussian管理器
    管理场景中所有物体的3D Gaussians
    """
    
    def __init__(self, config: Optional[GaussianConfig] = None):
        """
        Args:
            config: Gaussian配置
        """
        super().__init__()
        
        if config is None:
            config = GaussianConfig()
        self.config = config
        
        # 物体字典 {object_id: ObjectGaussian}
        self.objects = nn.ModuleDict()
        
        # 场景统计
        self.total_gaussians = 0
        self.frame_count = 0
        
        print("✓ SceneGaussians初始化完成")
    
    def add_object(
        self,
        object_id: int,
        object_gaussian: ObjectGaussian
    ):
        """
        添加物体
        
        Args:
            object_id: 物体ID
            object_gaussian: 物体Gaussian
        """
        self.objects[str(object_id)] = object_gaussian
        self.total_gaussians += object_gaussian.get_num_points()
        
        print(f"✓ 添加物体 {object_id} ({object_gaussian.category}): "
              f"{object_gaussian.get_num_points()} Gaussians")
    
    def remove_object(self, object_id: int):
        """移除物体"""
        if str(object_id) in self.objects:
            obj = self.objects[str(object_id)]
            self.total_gaussians -= obj.get_num_points()
            del self.objects[str(object_id)]
            print(f"✓ 移除物体 {object_id}")
    
    def get_object(self, object_id: int) -> Optional[ObjectGaussian]:
        """获取物体"""
        return self.objects.get(str(object_id))
    
    def has_object(self, object_id: int) -> bool:
        """检查物体是否存在"""
        return str(object_id) in self.objects
    
    def get_all_gaussians(self) -> Dict[str, torch.Tensor]:
        """
        获取所有物体的合并Gaussian参数
        
        Returns:
            gaussians: 包含所有参数的字典
        """
        if len(self.objects) == 0:
            return {}
        
        all_xyz = []
        all_features_dc = []
        all_features_rest = []
        all_scaling = []
        all_rotation = []
        all_opacity = []
        all_object_ids = []  # 记录每个Gaussian属于哪个物体
        
        for obj_id, obj in self.objects.items():
            num_points = obj.get_num_points()
            
            all_xyz.append(obj._xyz)
            all_features_dc.append(obj._features_dc)
            all_features_rest.append(obj._features_rest)
            all_scaling.append(obj._scaling)
            all_rotation.append(obj._rotation)
            all_opacity.append(obj._opacity)
            all_object_ids.extend([int(obj_id)] * num_points)
        
        return {
            'xyz': torch.cat(all_xyz, dim=0),
            'features_dc': torch.cat(all_features_dc, dim=0),
            'features_rest': torch.cat(all_features_rest, dim=0),
            'scaling': torch.cat(all_scaling, dim=0),
            'rotation': torch.cat(all_rotation, dim=0),
            'opacity': torch.cat(all_opacity, dim=0),
            'object_ids': torch.tensor(all_object_ids)
        }
    
    def get_all_gaussians_processed(self) -> Dict[str, torch.Tensor]:
        """
        获取处理后的Gaussian参数（用于渲染）
        
        Returns:
            processed: 包含处理后参数的字典
        """
        raw = self.get_all_gaussians()
        
        if not raw:
            return {}
        
        return {
            'xyz': raw['xyz'],
            'features_dc': raw['features_dc'],
            'features_rest': raw['features_rest'],
            'scaling': torch.exp(raw['scaling']),  # 转换回实际尺度
            'rotation': nn.functional.normalize(raw['rotation']),  # 归一化四元数
            'opacity': torch.sigmoid(raw['opacity']),  # 转换回[0,1]
            'object_ids': raw['object_ids']
        }
    
    def get_object_by_point(self, point_idx: int) -> Optional[int]:
        """
        根据点索引获取物体ID
        
        Args:
            point_idx: 全局点索引
            
        Returns:
            object_id: 物体ID
        """
        gaussians = self.get_all_gaussians()
        if not gaussians or point_idx >= len(gaussians['object_ids']):
            return None
        
        return gaussians['object_ids'][point_idx].item()
    
    def densify_and_prune_all(
        self,
        max_grad: float = 0.0002,
        min_opacity: float = 0.005,
        extent: float = 5.0,
        max_screen_size: int = 20
    ):
        """对所有物体进行致密化和修剪"""
        for obj_id, obj in self.objects.items():
            initial_count = obj.get_num_points()
            obj.densify_and_prune(max_grad, min_opacity, extent, max_screen_size)
            final_count = obj.get_num_points()
            
            # 更新总数
            self.total_gaussians += (final_count - initial_count)
    
    def reset_opacity_all(self):
        """重置所有物体的不透明度"""
        for obj in self.objects.values():
            obj.reset_opacity()
    
    def get_scene_bounds(self) -> Dict[str, torch.Tensor]:
        """
        获取场景边界
        
        Returns:
            bounds: {min: (3,), max: (3,), center: (3,), extent: float}
        """
        gaussians = self.get_all_gaussians()
        
        if not gaussians:
            return {
                'min': torch.zeros(3),
                'max': torch.zeros(3),
                'center': torch.zeros(3),
                'extent': 0.0
            }
        
        xyz = gaussians['xyz']
        
        min_bound = xyz.min(dim=0)[0]
        max_bound = xyz.max(dim=0)[0]
        center = (min_bound + max_bound) / 2
        extent = (max_bound - min_bound).norm()
        
        return {
            'min': min_bound,
            'max': max_bound,
            'center': center,
            'extent': extent.item()
        }
    
    def get_statistics(self) -> Dict:
        """获取场景统计信息"""
        stats = {
            'num_objects': len(self.objects),
            'total_gaussians': self.total_gaussians,
            'frame_count': self.frame_count,
            'objects': {}
        }
        
        for obj_id, obj in self.objects.items():
            stats['objects'][obj_id] = {
                'category': obj.category,
                'num_gaussians': obj.get_num_points(),
                'confidence': obj.confidence,
                'age': obj.age
            }
        
        return stats
    
    def save(self, output_dir: str):
        """
        保存场景
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存每个物体
        for obj_id, obj in self.objects.items():
            obj_path = output_path / f"object_{obj_id}.ply"
            obj.save_ply(str(obj_path))
        
        # 保存场景元数据
        metadata = {
            'num_objects': len(self.objects),
            'total_gaussians': self.total_gaussians,
            'frame_count': self.frame_count,
            'objects': {}
        }
        
        for obj_id, obj in self.objects.items():
            metadata['objects'][obj_id] = {
                'category': obj.category,
                'num_gaussians': obj.get_num_points(),
                'confidence': obj.confidence,
                'age': obj.age,
                'file': f"object_{obj_id}.ply"
            }
        
        with open(output_path / 'scene_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ 场景保存到: {output_dir}")
        print(f"  - 物体数量: {len(self.objects)}")
        print(f"  - 总Gaussian数: {self.total_gaussians}")
    
    @classmethod
    def load(cls, input_dir: str, config: Optional[GaussianConfig] = None):
        """
        加载场景
        
        Args:
            input_dir: 输入目录
            config: Gaussian配置
            
        Returns:
            scene: SceneGaussians对象
        """
        input_path = Path(input_dir)
        
        # 加载元数据
        with open(input_path / 'scene_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # 创建场景
        scene = cls(config)
        scene.frame_count = metadata.get('frame_count', 0)
        
        # 加载每个物体
        for obj_id, obj_info in metadata['objects'].items():
            # TODO: 实现从PLY加载ObjectGaussian
            # 这里需要实现PLY文件的读取
            print(f"  加载物体 {obj_id}: {obj_info['category']}")
        
        print(f"✓ 场景加载自: {input_dir}")
        
        return scene
    
    def merge_objects(self, obj_id1: int, obj_id2: int, new_id: Optional[int] = None):
        """
        合并两个物体
        
        Args:
            obj_id1: 第一个物体ID
            obj_id2: 第二个物体ID
            new_id: 新物体ID（可选）
        """
        if not self.has_object(obj_id1) or not self.has_object(obj_id2):
            print("错误: 物体不存在")
            return
        
        obj1 = self.get_object(obj_id1)
        obj2 = self.get_object(obj_id2)
        
        # 合并点云
        merged_xyz = torch.cat([obj1._xyz, obj2._xyz], dim=0)
        merged_features_dc = torch.cat([obj1._features_dc, obj2._features_dc], dim=0)
        merged_features_rest = torch.cat([obj1._features_rest, obj2._features_rest], dim=0)
        merged_scaling = torch.cat([obj1._scaling, obj2._scaling], dim=0)
        merged_rotation = torch.cat([obj1._rotation, obj2._rotation], dim=0)
        merged_opacity = torch.cat([obj1._opacity, obj2._opacity], dim=0)
        
        # 创建新物体
        if new_id is None:
            new_id = max(int(k) for k in self.objects.keys()) + 1
        
        merged_obj = ObjectGaussian(
            object_id=new_id,
            category=obj1.category,  # 使用第一个物体的类别
            config=self.config
        )
        
        merged_obj._xyz = nn.Parameter(merged_xyz)
        merged_obj._features_dc = nn.Parameter(merged_features_dc)
        merged_obj._features_rest = nn.Parameter(merged_features_rest)
        merged_obj._scaling = nn.Parameter(merged_scaling)
        merged_obj._rotation = nn.Parameter(merged_rotation)
        merged_obj._opacity = nn.Parameter(merged_opacity)
        
        # 初始化优化变量
        num_points = merged_xyz.shape[0]
        merged_obj.xyz_gradient_accum = torch.zeros((num_points, 1))
        merged_obj.denom = torch.zeros((num_points, 1))
        merged_obj.max_radii2D = torch.zeros((num_points))
        
        # 移除旧物体，添加新物体
        self.remove_object(obj_id1)
        self.remove_object(obj_id2)
        self.add_object(new_id, merged_obj)
        
        print(f"✓ 合并物体 {obj_id1} 和 {obj_id2} -> {new_id}")
    
    def split_object(self, obj_id: int, split_mask: torch.Tensor, new_id: Optional[int] = None):
        """
        分割物体
        
        Args:
            obj_id: 要分割的物体ID
            split_mask: 布尔mask，True的部分分割出去
            new_id: 新物体ID
        """
        if not self.has_object(obj_id):
            print("错误: 物体不存在")
            return
        
        obj = self.get_object(obj_id)
        
        # 分割参数
        split_xyz = obj._xyz[split_mask]
        split_features_dc = obj._features_dc[split_mask]
        split_features_rest = obj._features_rest[split_mask]
        split_scaling = obj._scaling[split_mask]
        split_rotation = obj._rotation[split_mask]
        split_opacity = obj._opacity[split_mask]
        
        # 保留的部分
        obj.prune_points(~split_mask)
        
        # 创建新物体
        if new_id is None:
            new_id = max(int(k) for k in self.objects.keys()) + 1
        
        new_obj = ObjectGaussian(
            object_id=new_id,
            category=obj.category,
            config=self.config
        )
        
        new_obj._xyz = nn.Parameter(split_xyz)
        new_obj._features_dc = nn.Parameter(split_features_dc)
        new_obj._features_rest = nn.Parameter(split_features_rest)
        new_obj._scaling = nn.Parameter(split_scaling)
        new_obj._rotation = nn.Parameter(split_rotation)
        new_obj._opacity = nn.Parameter(split_opacity)
        
        # 初始化优化变量
        num_points = split_xyz.shape[0]
        new_obj.xyz_gradient_accum = torch.zeros((num_points, 1))
        new_obj.denom = torch.zeros((num_points, 1))
        new_obj.max_radii2D = torch.zeros((num_points))
        
        self.add_object(new_id, new_obj)
        
        print(f"✓ 分割物体 {obj_id} -> {new_id}")
    
    def filter_by_confidence(self, min_confidence: float = 0.5):
        """移除低置信度物体"""
        to_remove = []
        for obj_id, obj in self.objects.items():
            if obj.confidence < min_confidence:
                to_remove.append(int(obj_id))
        
        for obj_id in to_remove:
            self.remove_object(obj_id)
        
        if to_remove:
            print(f"✓ 移除了 {len(to_remove)} 个低置信度物体")
    
    def update_frame(self):
        """更新帧计数"""
        self.frame_count += 1


if __name__ == "__main__":
    # 测试代码
    print("=== 测试SceneGaussians ===")
    
    from .gaussian_model import GaussianConfig
    from .object_gaussian import ObjectGaussian
    import numpy as np
    
    # 配置
    config = GaussianConfig()
    
    # 创建场景
    scene = SceneGaussians(config)
    
    # 创建一些测试物体
    print("\n1. 添加物体:")
    for i in range(3):
        obj = ObjectGaussian(
            object_id=i,
            category=["chair", "table", "sofa"][i],
            config=config
        )
        
        # 创建随机点云
        points = np.random.rand(1000, 3) * 2 - 1
        colors = np.random.rand(1000, 3)
        obj.create_from_points(points, colors)
        
        scene.add_object(i, obj)
    
    # 获取统计
    print("\n2. 场景统计:")
    stats = scene.get_statistics()
    print(f"  物体数量: {stats['num_objects']}")
    print(f"  总Gaussian数: {stats['total_gaussians']}")
    
    for obj_id, obj_stats in stats['objects'].items():
        print(f"  物体 {obj_id}: {obj_stats['category']}, {obj_stats['num_gaussians']} Gaussians")
    
    # 获取场景边界
    print("\n3. 场景边界:")
    bounds = scene.get_scene_bounds()
    print(f"  最小值: {bounds['min']}")
    print(f"  最大值: {bounds['max']}")
    print(f"  中心: {bounds['center']}")
    print(f"  范围: {bounds['extent']:.2f}")
    
    # 获取所有Gaussians
    print("\n4. 获取所有Gaussians:")
    all_gaussians = scene.get_all_gaussians()
    print(f"  总点数: {all_gaussians['xyz'].shape[0]}")
    print(f"  物体ID范围: {all_gaussians['object_ids'].min()}-{all_gaussians['object_ids'].max()}")
    
    # 测试合并
    print("\n5. 测试合并物体:")
    scene.merge_objects(0, 1, new_id=10)
    stats = scene.get_statistics()
    print(f"  合并后物体数: {stats['num_objects']}")
    
    # 测试保存
    print("\n6. 测试保存:")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        scene.save(tmpdir)
        print(f"  已保存到: {tmpdir}")
    
    print("\n测试完成！")