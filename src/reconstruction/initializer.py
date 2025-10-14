"""
Gaussian Initializer
Gaussian初始化器，从不同来源初始化3D Gaussians
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy.spatial import cKDTree


class GaussianInitializer:
    """
    Gaussian初始化器
    支持从点云、深度图、SfM等初始化
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 初始化配置
        """
        if config is None:
            config = {}
        
        self.config = config
        self.init_scale = config.get('init_scale', 0.01)
        self.min_points = config.get('min_points', 100)
        
        print("✓ GaussianInitializer初始化完成")
    
    def from_pointcloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        从点云初始化
        
        Args:
            points: 点云坐标 (N, 3)
            colors: 点云颜色 (N, 3), 范围[0, 1]
            normals: 点云法向量 (N, 3)
            
        Returns:
            gaussian_params: 初始化的Gaussian参数
        """
        N = points.shape[0]
        
        if N < self.min_points:
            raise ValueError(f"点云数量太少: {N} < {self.min_points}")
        
        # 1. 位置
        xyz = points.copy()
        
        # 2. 颜色/特征
        if colors is None:
            colors = np.ones((N, 3)) * 0.5  # 默认灰色
        
        # 3. 尺度（使用k近邻距离）
        scales = self._estimate_scales_knn(points, k=3)
        
        # 4. 旋转（使用法向量或默认）
        if normals is not None:
            rotations = self._rotation_from_normals(normals)
        else:
            rotations = np.tile([1, 0, 0, 0], (N, 1))  # 单位四元数
        
        # 5. 不透明度
        opacities = np.ones((N, 1)) * 0.1
        
        return {
            'xyz': xyz,
            'colors': colors,
            'scales': scales,
            'rotations': rotations,
            'opacities': opacities
        }
    
    def from_depth_map(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        camera_params: Dict,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        从深度图初始化
        
        Args:
            depth: 深度图 (H, W)
            image: RGB图像 (H, W, 3)
            camera_params: 相机参数
            mask: 有效区域mask (H, W)
            
        Returns:
            gaussian_params: 初始化的Gaussian参数
        """
        # 反投影深度到3D
        points, colors = self._unproject_depth(
            depth, image, camera_params, mask
        )
        
        # 从点云初始化
        return self.from_pointcloud(points, colors)
    
    def from_multiview(
        self,
        depths: List[np.ndarray],
        images: List[np.ndarray],
        camera_params_list: List[Dict],
        masks: Optional[List[np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        从多视角深度图初始化
        
        Args:
            depths: 深度图列表
            images: 图像列表
            camera_params_list: 相机参数列表
            masks: mask列表
            
        Returns:
            gaussian_params: 初始化的Gaussian参数
        """
        all_points = []
        all_colors = []
        
        for i, (depth, image, cam_params) in enumerate(zip(depths, images, camera_params_list)):
            mask = masks[i] if masks is not None else None
            points, colors = self._unproject_depth(depth, image, cam_params, mask)
            
            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)
        
        if len(all_points) == 0:
            raise ValueError("无法从任何视角提取点")
        
        # 合并所有点
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        
        # 去除重复点（简单的体素下采样）
        merged_points, merged_colors = self._voxel_downsample(
            merged_points, merged_colors, voxel_size=0.01
        )
        
        return self.from_pointcloud(merged_points, merged_colors)
    
    def _unproject_depth(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        camera_params: Dict,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """反投影深度到3D点云"""
        H, W = depth.shape
        fx = camera_params['fx']
        fy = camera_params['fy']
        cx = camera_params.get('cx', W / 2)
        cy = camera_params.get('cy', H / 2)
        
        # 创建像素网格
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # 应用mask
        if mask is not None:
            valid = mask & (depth > 0)
        else:
            valid = depth > 0
        
        u = u[valid]
        v = v[valid]
        depths = depth[valid]
        
        # 反投影
        x = (u - cx) * depths / fx
        y = (v - cy) * depths / fy
        z = depths
        
        points = np.stack([x, y, z], axis=1)
        colors = image[valid].astype(np.float32) / 255.0
        
        return points, colors
    
    def _estimate_scales_knn(
        self,
        points: np.ndarray,
        k: int = 3
    ) -> np.ndarray:
        """使用k近邻估计尺度"""
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=k+1)  # k+1因为包含自己
        
        # 使用平均距离
        avg_dist = distances[:, 1:].mean(axis=1)  # 排除自己
        
        # 各向同性尺度
        scales = np.tile(avg_dist[:, None] * self.init_scale, (1, 3))
        
        return scales
    
    def _rotation_from_normals(
        self,
        normals: np.ndarray
    ) -> np.ndarray:
        """从法向量计算旋转（对齐到z轴）"""
        N = normals.shape[0]
        rotations = np.zeros((N, 4))
        
        # 归一化法向量
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        # z轴
        z_axis = np.array([0, 0, 1])
        
        for i in range(N):
            n = normals[i]
            
            # 计算旋转轴和角度
            axis = np.cross(z_axis, n)
            axis_norm = np.linalg.norm(axis)
            
            if axis_norm < 1e-6:
                # 法向量已经沿z轴
                rotations[i] = [1, 0, 0, 0]
            else:
                axis = axis / axis_norm
                angle = np.arccos(np.clip(np.dot(z_axis, n), -1, 1))
                
                # 轴角到四元数
                half_angle = angle / 2
                rotations[i] = [
                    np.cos(half_angle),
                    axis[0] * np.sin(half_angle),
                    axis[1] * np.sin(half_angle),
                    axis[2] * np.sin(half_angle)
                ]
        
        return rotations
    
    def _voxel_downsample(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        voxel_size: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """体素下采样"""
        # 计算体素索引
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        # 使用字典去重
        voxel_dict = {}
        
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)
        
        # 对每个体素取平均
        downsampled_points = []
        downsampled_colors = []
        
        for indices in voxel_dict.values():
            downsampled_points.append(points[indices].mean(axis=0))
            downsampled_colors.append(colors[indices].mean(axis=0))
        
        return np.array(downsampled_points), np.array(downsampled_colors)
    
    def add_noise_for_robustness(
        self,
        gaussian_params: Dict[str, np.ndarray],
        noise_scale: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """
        添加噪声以提高鲁棒性
        
        Args:
            gaussian_params: Gaussian参数
            noise_scale: 噪声尺度
            
        Returns:
            noisy_params: 添加噪声后的参数
        """
        params = gaussian_params.copy()
        
        # 位置添加高斯噪声
        params['xyz'] = params['xyz'] + np.random.randn(*params['xyz'].shape) * noise_scale
        
        # 尺度添加小扰动
        params['scales'] = params['scales'] * (1 + np.random.randn(*params['scales'].shape) * 0.1)
        params['scales'] = np.clip(params['scales'], 1e-6, None)
        
        return params
    
    def filter_outliers(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        k: int = 20,
        std_ratio: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        过滤离群点
        
        Args:
            points: 点云 (N, 3)
            colors: 颜色 (N, 3)
            k: 近邻数量
            std_ratio: 标准差倍数
            
        Returns:
            filtered_points, filtered_colors
        """
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=k+1)
        
        # 计算平均距离
        avg_distances = distances[:, 1:].mean(axis=1)
        
        # 统计阈值
        mean_dist = avg_distances.mean()
        std_dist = avg_distances.std()
        threshold = mean_dist + std_ratio * std_dist
        
        # 过滤
        inlier_mask = avg_distances < threshold
        
        return points[inlier_mask], colors[inlier_mask]
    
    def random_subsample(
        self,
        gaussian_params: Dict[str, np.ndarray],
        max_points: int
    ) -> Dict[str, np.ndarray]:
        """
        随机下采样到指定数量
        
        Args:
            gaussian_params: Gaussian参数
            max_points: 最大点数
            
        Returns:
            subsampled_params: 下采样后的参数
        """
        N = gaussian_params['xyz'].shape[0]
        
        if N <= max_points:
            return gaussian_params
        
        # 随机选择
        indices = np.random.choice(N, max_points, replace=False)
        
        params = {}
        for key, value in gaussian_params.items():
            params[key] = value[indices]
        
        return params
    
    def farthest_point_sampling(
        self,
        points: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        最远点采样
        
        Args:
            points: 点云 (N, 3)
            n_samples: 采样数量
            
        Returns:
            sampled_indices: 采样的索引
        """
        N = points.shape[0]
        
        if n_samples >= N:
            return np.arange(N)
        
        # 初始化
        sampled_indices = [np.random.randint(N)]
        distances = np.full(N, np.inf)
        
        for _ in range(n_samples - 1):
            # 更新距离
            last_point = points[sampled_indices[-1]]
            dists_to_last = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, dists_to_last)
            
            # 选择最远点
            farthest_idx = np.argmax(distances)
            sampled_indices.append(farthest_idx)
        
        return np.array(sampled_indices)


class SfMInitializer(GaussianInitializer):
    """
    从SfM结果初始化Gaussians
    """
    
    def from_colmap(
        self,
        colmap_path: str
    ) -> Dict[str, np.ndarray]:
        """
        从COLMAP输出初始化
        
        Args:
            colmap_path: COLMAP输出目录
            
        Returns:
            gaussian_params: 初始化的参数
        """
        # TODO: 实现COLMAP读取
        # 需要读取points3D.bin或points3D.txt
        raise NotImplementedError("COLMAP读取尚未实现")


if __name__ == "__main__":
    # 测试代码
    print("=== 测试GaussianInitializer ===")
    
    initializer = GaussianInitializer()
    
    # 测试从点云初始化
    print("\n1. 从点云初始化:")
    points = np.random.rand(1000, 3) * 2 - 1
    colors = np.random.rand(1000, 3)
    normals = np.random.randn(1000, 3)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    params = initializer.from_pointcloud(points, colors, normals)
    
    print(f"  初始化了 {params['xyz'].shape[0]} 个Gaussians")
    print(f"  尺度范围: [{params['scales'].min():.4f}, {params['scales'].max():.4f}]")
    print(f"  颜色范围: [{params['colors'].min():.4f}, {params['colors'].max():.4f}]")
    
    # 测试从深度图初始化
    print("\n2. 从深度图初始化:")
    H, W = 480, 640
    depth = np.random.rand(H, W) * 5 + 2  # 2-7米
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    mask = np.ones((H, W), dtype=bool)
    
    camera_params = {
        'fx': 525.0,
        'fy': 525.0,
        'cx': 320.0,
        'cy': 240.0
    }
    
    params = initializer.from_depth_map(depth, image, camera_params, mask)
    print(f"  从深度图初始化了 {params['xyz'].shape[0]} 个Gaussians")
    
    # 测试多视角初始化
    print("\n3. 从多视角初始化:")
    depths = [depth + np.random.rand(H, W) * 0.1 for _ in range(3)]
    images = [image + np.random.randint(-10, 10, (H, W, 3), dtype=np.int16) for _ in range(3)]
    images = [np.clip(img, 0, 255).astype(np.uint8) for img in images]
    camera_params_list = [camera_params] * 3
    
    params = initializer.from_multiview(depths, images, camera_params_list)
    print(f"  从多视角初始化了 {params['xyz'].shape[0]} 个Gaussians")
    
    # 测试离群点过滤
    print("\n4. 测试离群点过滤:")
    # 添加一些离群点
    outliers = np.random.rand(50, 3) * 10 + 5
    points_with_outliers = np.vstack([points, outliers])
    colors_extended = np.vstack([colors, np.random.rand(50, 3)])
    
    filtered_points, filtered_colors = initializer.filter_outliers(
        points_with_outliers,
        colors_extended
    )
    print(f"  原始点数: {len(points_with_outliers)}")
    print(f"  过滤后: {len(filtered_points)}")
    print(f"  移除了 {len(points_with_outliers) - len(filtered_points)} 个离群点")
    
    # 测试最远点采样
    print("\n5. 测试最远点采样:")
    sampled_indices = initializer.farthest_point_sampling(points, n_samples=100)
    print(f"  从 {len(points)} 个点采样了 {len(sampled_indices)} 个")
    
    # 测试下采样
    print("\n6. 测试下采样:")
    subsampled = initializer.random_subsample(params, max_points=500)
    print(f"  下采样到 {subsampled['xyz'].shape[0]} 个点")
    
    print("\n测试完成！")