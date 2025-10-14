"""
Object Gaussian
单个物体的3D Gaussian表示，集成形状先验
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from .gaussian_model import GaussianModel, GaussianConfig


class ObjectGaussian(GaussianModel):
    """
    物体级3D Gaussian
    继承基础GaussianModel，增加物体特定信息和形状先验集成
    """
    
    def __init__(
        self,
        object_id: int,
        category: str,
        config: GaussianConfig,
        shape_prior: Optional[torch.Tensor] = None
    ):
        """
        Args:
            object_id: 物体ID
            category: 物体类别
            config: Gaussian配置
            shape_prior: 形状先验点云 (N, 3)
        """
        super().__init__(config)
        
        self.object_id = object_id
        self.category = category
        self.shape_prior = shape_prior
        
        # 物体特定的状态
        self.confidence = 1.0  # 置信度
        self.age = 0  # 年龄（帧数）
        self.last_seen = 0  # 最后观测到的帧
        
    def initialize_from_mask_depth(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray,
        image: np.ndarray,
        camera_params: Dict
    ):
        """
        从mask、深度图和图像初始化Gaussian
        
        Args:
            mask: 物体mask (H, W)
            depth_map: 深度图 (H, W)
            image: RGB图像 (H, W, 3)
            camera_params: 相机参数 {fx, fy, cx, cy}
        """
        # 1. 从深度图反投影获得3D点
        points, colors = self._unproject_depth(
            mask,
            depth_map,
            image,
            camera_params
        )
        
        if len(points) == 0:
            raise ValueError(f"物体 {self.object_id} 没有有效的3D点")
        
        # 2. 如果有形状先验，进行融合
        if self.shape_prior is not None:
            points = self._fuse_with_shape_prior(points)
        
        # 3. 创建Gaussians
        self.create_from_points(points, colors)
        
        print(f"✓ 物体 {self.object_id} ({self.category}) 初始化: {len(points)} 个点")
    
    def _unproject_depth(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        image: np.ndarray,
        camera_params: Dict
    ) -> tuple:
        """
        反投影深度到3D点云
        
        Args:
            mask: 物体mask (H, W)
            depth: 深度图 (H, W)
            image: RGB图像 (H, W, 3)
            camera_params: 相机参数
            
        Returns:
            points: 3D点 (N, 3)
            colors: 颜色 (N, 3)
        """
        H, W = mask.shape
        fx = camera_params['fx']
        fy = camera_params['fy']
        cx = camera_params['cx']
        cy = camera_params['cy']
        
        # 获取mask内的像素
        ys, xs = np.where(mask)
        depths = depth[ys, xs]
        
        # 过滤无效深度
        valid = (depths > 0.1) & (depths < 20.0)
        xs, ys, depths = xs[valid], ys[valid], depths[valid]
        
        if len(xs) == 0:
            return np.array([]), np.array([])
        
        # 反投影
        x3d = (xs - cx) * depths / fx
        y3d = (ys - cy) * depths / fy
        z3d = depths
        
        points = np.stack([x3d, y3d, z3d], axis=1)
        
        # 获取颜色
        colors = image[ys, xs].astype(np.float32) / 255.0
        
        # 随机下采样（如果点太多）
        max_points = 10000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        return points, colors
    
    def _fuse_with_shape_prior(
        self,
        points: np.ndarray,
        prior_weight: float = 0.3
    ) -> np.ndarray:
        """
        融合观测点云和形状先验
        
        Args:
            points: 观测点云 (N, 3)
            prior_weight: 先验权重
            
        Returns:
            fused_points: 融合后的点云
        """
        if self.shape_prior is None:
            return points
        
        # 将先验点云对齐到观测点云
        prior_np = self.shape_prior.cpu().numpy()
        
        # 1. 中心对齐
        obs_center = points.mean(axis=0)
        prior_center = prior_np.mean(axis=0)
        
        # 2. 尺度对齐
        obs_scale = np.linalg.norm(points - obs_center, axis=1).max()
        prior_scale = np.linalg.norm(prior_np - prior_center, axis=1).max()
        scale_factor = obs_scale / (prior_scale + 1e-6)
        
        # 对齐先验
        aligned_prior = (prior_np - prior_center) * scale_factor + obs_center
        
        # 3. 融合策略：在观测稀疏的区域添加先验点
        from scipy.spatial import cKDTree
        
        # 构建观测点的KD树
        tree = cKDTree(points)
        
        # 对于每个先验点，检查是否在观测点附近
        distances, _ = tree.query(aligned_prior)
        
        # 选择远离观测点的先验点（填补空洞）
        far_threshold = obs_scale * 0.1
        far_prior_mask = distances > far_threshold
        
        if far_prior_mask.any():
            # 添加这些先验点
            far_prior_points = aligned_prior[far_prior_mask]
            
            # 按权重混合
            num_prior_to_add = int(len(far_prior_points) * prior_weight)
            if num_prior_to_add > 0:
                indices = np.random.choice(
                    len(far_prior_points),
                    min(num_prior_to_add, len(far_prior_points)),
                    replace=False
                )
                fused_points = np.vstack([points, far_prior_points[indices]])
            else:
                fused_points = points
        else:
            fused_points = points
        
        return fused_points
    
    def update_from_observation(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        image: np.ndarray,
        camera_params: Dict
    ):
        """
        使用新的观测更新Gaussian
        
        Args:
            mask: 新的mask
            depth: 新的深度图
            image: 新的图像
            camera_params: 相机参数
        """
        # 获取新的观测点
        new_points, new_colors = self._unproject_depth(
            mask, depth, image, camera_params
        )
        
        if len(new_points) == 0:
            return
        
        # 简单策略：添加新点（可以改进为更新现有点）
        current_points = self._xyz.detach().cpu().numpy()
        current_colors = self.get_colors.detach().cpu().numpy()
        
        # 合并
        merged_points = np.vstack([current_points, new_points])
        merged_colors = np.vstack([current_colors, new_colors])
        
        # 下采样保持合理数量
        max_points = 20000
        if len(merged_points) > max_points:
            indices = np.random.choice(len(merged_points), max_points, replace=False)
            merged_points = merged_points[indices]
            merged_colors = merged_colors[indices]
        
        # 重新初始化
        self.create_from_points(merged_points, merged_colors)
        
        self.age += 1
        self.last_seen = self.age
    
    def densify_and_prune(
        self,
        max_grad: float = 0.0002,
        min_opacity: float = 0.005,
        extent: float = 5.0,
        max_screen_size: int = 20
    ):
        """
        致密化和修剪Gaussians
        
        Args:
            max_grad: 梯度阈值
            min_opacity: 最小不透明度
            extent: 场景范围
            max_screen_size: 最大屏幕尺寸
        """
        # 1. 基于梯度的致密化
        grads = self.xyz_gradient_accum / (self.denom + 1e-8)
        grads[grads.isnan()] = 0.0
        
        # 克隆小尺度、高梯度的Gaussians
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= max_grad,
            True,
            False
        )
        
        scales = self.get_scaling
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scales, dim=1).values <= 0.01 * extent
        )
        
        if selected_pts_mask.any():
            self.densify_and_clone(selected_pts_mask)
        
        # 分裂大尺度、高梯度的Gaussians
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= max_grad,
            True,
            False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scales, dim=1).values > 0.01 * extent
        )
        
        if selected_pts_mask.any():
            self.densify_and_split(selected_pts_mask, N=2)
        
        # 2. 修剪
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        
        if prune_mask.any():
            self.prune_points(~prune_mask)
        
        # 重置梯度累积
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
    
    def reset_opacity(self):
        """重置不透明度到较低值"""
        opacities_new = torch.min(
            self.get_opacity,
            torch.ones_like(self.get_opacity) * 0.01
        )
        self._opacity = nn.Parameter(torch.logit(opacities_new))
    
    def to_dict(self) -> Dict:
        """转换为字典（用于保存）"""
        return {
            'object_id': self.object_id,
            'category': self.category,
            'xyz': self._xyz.detach().cpu(),
            'features_dc': self._features_dc.detach().cpu(),
            'features_rest': self._features_rest.detach().cpu(),
            'scaling': self._scaling.detach().cpu(),
            'rotation': self._rotation.detach().cpu(),
            'opacity': self._opacity.detach().cpu(),
            'confidence': self.confidence,
            'age': self.age,
        }
    
    @classmethod
    def from_dict(cls, data: Dict, config: GaussianConfig):
        """从字典加载"""
        obj = cls(
            object_id=data['object_id'],
            category=data['category'],
            config=config
        )
        
        obj._xyz = nn.Parameter(data['xyz'])
        obj._features_dc = nn.Parameter(data['features_dc'])
        obj._features_rest = nn.Parameter(data['features_rest'])
        obj._scaling = nn.Parameter(data['scaling'])
        obj._rotation = nn.Parameter(data['rotation'])
        obj._opacity = nn.Parameter(data['opacity'])
        obj.confidence = data.get('confidence', 1.0)
        obj.age = data.get('age', 0)
        
        # 初始化优化变量
        num_points = obj._xyz.shape[0]
        obj.xyz_gradient_accum = torch.zeros((num_points, 1))
        obj.denom = torch.zeros((num_points, 1))
        obj.max_radii2D = torch.zeros((num_points))
        
        return obj
    
    def save_ply(self, path: str):
        """保存为PLY格式"""
        from plyfile import PlyData, PlyElement
        
        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.get_opacity.detach().cpu().numpy()
        scale = self.get_scaling.detach().cpu().numpy()
        rotation = self.get_rotation.detach().cpu().numpy()
        
        # 构建PLY格式
        dtype_full = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        print(f"✓ 物体 {self.object_id} 保存到: {path}")


if __name__ == "__main__":
    # 测试代码
    print("=== 测试ObjectGaussian ===")
    
    from .gaussian_model import GaussianConfig
    
    # 配置
    config = GaussianConfig()
    
    # 创建物体Gaussian
    obj_gaussian = ObjectGaussian(
        object_id=0,
        category="chair",
        config=config
    )
    
    # 模拟数据
    H, W = 480, 640
    mask = np.zeros((H, W), dtype=bool)
    mask[100:300, 200:400] = True
    
    depth = np.random.rand(H, W) * 3 + 2  # 2-5米
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
    camera_params = {
        'fx': 525.0,
        'fy': 525.0,
        'cx': 320.0,
        'cy': 240.0
    }
    
    # 初始化
    obj_gaussian.initialize_from_mask_depth(
        mask, depth, image, camera_params
    )
    
    print(f"Gaussian数量: {obj_gaussian.get_num_points()}")
    print(f"物体ID: {obj_gaussian.object_id}")
    print(f"类别: {obj_gaussian.category}")
    
    # 测试致密化
    print("\n测试致密化和修剪:")
    initial_count = obj_gaussian.get_num_points()
    
    # 模拟梯度
    obj_gaussian.xyz_gradient_accum = torch.rand(initial_count, 1) * 0.001
    obj_gaussian.denom = torch.ones(initial_count, 1)
    
    obj_gaussian.densify_and_prune()
    final_count = obj_gaussian.get_num_points()
    
    print(f"初始: {initial_count}, 最终: {final_count}")
    
    # 测试保存
    print("\n测试保存:")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        obj_gaussian.save_ply(f.name)
        print(f"已保存到: {f.name}")
    
    print("\n测试完成！")