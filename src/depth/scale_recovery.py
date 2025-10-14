"""
Scale Recovery
从相对深度恢复绝对尺度
利用物体先验、SfM等方法
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2


class ScaleRecovery:
    """
    尺度恢复器
    将相对深度转换为度量深度
    """
    
    def __init__(
        self,
        method: str = "shape_prior",
        reference_objects: Optional[List[str]] = None
    ):
        """
        Args:
            method: 恢复方法 (shape_prior, sfm, manual)
            reference_objects: 参考物体类别列表
        """
        self.method = method
        
        if reference_objects is None:
            self.reference_objects = ["chair", "table", "person"]
        else:
            self.reference_objects = reference_objects
        
        # 物体典型尺寸数据库（单位：米）
        self.object_sizes = {
            "chair": {"height": 0.9, "width": 0.5, "depth": 0.5},
            "table": {"height": 0.75, "width": 1.5, "depth": 0.8},
            "sofa": {"height": 0.85, "width": 2.0, "depth": 0.9},
            "bed": {"height": 0.5, "width": 2.0, "depth": 1.5},
            "desk": {"height": 0.75, "width": 1.2, "depth": 0.6},
            "person": {"height": 1.7, "width": 0.5, "depth": 0.3},
            "door": {"height": 2.0, "width": 0.9, "depth": 0.05},
            "tv": {"height": 0.7, "width": 1.2, "depth": 0.1},
        }
        
        print(f"✓ ScaleRecovery初始化: method={method}")
    
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
            objects: 检测到的物体列表，每个包含:
                - category: 类别
                - bbox: 边界框 [x, y, w, h]
                - mask: 分割mask
            camera_params: 相机参数 (fx, fy, cx, cy)
            
        Returns:
            metric_depth: 度量深度图 (H, W)
            scale_factor: 尺度因子
        """
        if self.method == "shape_prior":
            return self._recover_from_shape_prior(depth, objects, camera_params)
        elif self.method == "sfm":
            return self._recover_from_sfm(depth, camera_params)
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
        """
        使用物体先验恢复尺度
        
        策略：
        1. 找到参考物体（如椅子、桌子）
        2. 估计物体的真实尺寸
        3. 根据物体在图像中的大小计算尺度因子
        """
        scale_factors = []
        
        for obj in objects:
            category = obj.get('category', 'unknown')
            
            # 只使用参考物体
            if category not in self.reference_objects:
                continue
            
            if category not in self.object_sizes:
                continue
            
            # 获取物体的典型尺寸
            typical_size = self.object_sizes[category]
            
            # 计算物体在深度图中的尺度
            scale_factor = self._estimate_scale_from_object(
                obj,
                depth,
                typical_size,
                camera_params
            )
            
            if scale_factor is not None:
                scale_factors.append(scale_factor)
        
        if len(scale_factors) == 0:
            # 没有参考物体，使用默认尺度
            print("警告: 未找到参考物体，使用默认尺度")
            scale_factor = 1.0
        else:
            # 使用中位数作为最终尺度因子（更鲁棒）
            scale_factor = np.median(scale_factors)
        
        # 应用尺度
        metric_depth = depth * scale_factor
        
        print(f"  尺度因子: {scale_factor:.3f}")
        print(f"  使用了 {len(scale_factors)} 个参考物体")
        
        return metric_depth, scale_factor
    
    def _estimate_scale_from_object(
        self,
        obj: Dict,
        depth: np.ndarray,
        typical_size: Dict,
        camera_params: Optional[Dict] = None
    ) -> Optional[float]:
        """
        从单个物体估计尺度因子
        
        Args:
            obj: 物体信息
            depth: 深度图
            typical_size: 典型尺寸 {"height": h, "width": w, "depth": d}
            camera_params: 相机参数
            
        Returns:
            scale_factor: 尺度因子
        """
        mask = obj['segmentation']
        bbox = obj['bbox']
        x, y, w, h = bbox
        
        # 获取物体区域的深度
        if mask is not None:
            obj_depth = depth[mask]
            if len(obj_depth) == 0:
                return None
            median_depth = np.median(obj_depth)
        else:
            # 使用bbox中心的深度
            cy, cx = y + h // 2, x + w // 2
            median_depth = depth[cy, cx]
        
        # 估计物体的真实高度
        # 使用透视投影关系: 真实高度 / 图像高度 = 深度 / 焦距
        
        if camera_params is not None:
            fy = camera_params.get('fy', 525.0)
        else:
            # 使用典型值（假设FOV约60度）
            fy = depth.shape[0] / (2 * np.tan(np.radians(30)))
        
        # 估计的真实高度 = (像素高度 * 相对深度) / 焦距
        estimated_height = (h * median_depth) / fy
        
        # 计算尺度因子 = 典型高度 / 估计高度
        scale_factor = typical_size["height"] / (estimated_height + 1e-6)
        
        # 合理性检查（尺度因子应该在合理范围内）
        if scale_factor < 0.1 or scale_factor > 10.0:
            return None
        
        return scale_factor
    
    def _recover_from_sfm(
        self,
        depth: np.ndarray,
        camera_params: Optional[Dict] = None
    ) -> Tuple[np.ndarray, float]:
        """
        使用SfM恢复尺度
        
        需要多视角信息，这里提供接口
        实际实现需要COLMAP或其他SfM方法
        """
        # TODO: 集成COLMAP或其他SfM方法
        print("警告: SfM尺度恢复尚未实现，使用默认尺度")
        return depth, 1.0
    
    def _recover_manual(
        self,
        depth: np.ndarray,
        manual_scale: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """
        手动指定尺度
        """
        metric_depth = depth * manual_scale
        return metric_depth, manual_scale
    
    def estimate_scene_scale(
        self,
        depths: List[np.ndarray],
        objects_list: List[List[Dict]],
        camera_params: Optional[Dict] = None
    ) -> float:
        """
        从多帧估计场景的全局尺度
        
        Args:
            depths: 深度图列表
            objects_list: 每帧的物体列表
            camera_params: 相机参数
            
        Returns:
            global_scale: 全局尺度因子
        """
        all_scale_factors = []
        
        for depth, objects in zip(depths, objects_list):
            _, scale_factor = self.recover_scale(depth, objects, camera_params)
            if scale_factor != 1.0:  # 排除默认值
                all_scale_factors.append(scale_factor)
        
        if len(all_scale_factors) == 0:
            return 1.0
        
        # 使用中位数
        global_scale = np.median(all_scale_factors)
        
        print(f"全局尺度估计: {global_scale:.3f} (基于 {len(all_scale_factors)} 帧)")
        
        return global_scale
    
    def align_depth_to_pointcloud(
        self,
        depth: np.ndarray,
        pointcloud: np.ndarray,
        camera_params: Dict
    ) -> Tuple[np.ndarray, float]:
        """
        将深度图对齐到已知点云
        
        Args:
            depth: 相对深度图
            pointcloud: 点云 (N, 3)
            camera_params: 相机参数
            
        Returns:
            aligned_depth: 对齐后的深度
            scale_factor: 尺度因子
        """
        # 从深度图生成点云
        H, W = depth.shape
        fx, fy = camera_params['fx'], camera_params['fy']
        cx, cy = camera_params['cx'], camera_params['cy']
        
        # 创建像素坐标网格
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # 反投影
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        depth_pc = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        # 计算尺度（通过对齐点云）
        # 使用ICP或简单的尺度估计
        
        # 简单方法：比较点云的范围
        depth_extent = np.linalg.norm(depth_pc.max(axis=0) - depth_pc.min(axis=0))
        pc_extent = np.linalg.norm(pointcloud.max(axis=0) - pointcloud.min(axis=0))
        
        scale_factor = pc_extent / (depth_extent + 1e-6)
        
        aligned_depth = depth * scale_factor
        
        return aligned_depth, scale_factor
    
    def validate_scale(
        self,
        metric_depth: np.ndarray,
        objects: List[Dict]
    ) -> Dict[str, float]:
        """
        验证尺度的合理性
        
        Args:
            metric_depth: 度量深度图
            objects: 物体列表
            
        Returns:
            validation_metrics: 验证指标
        """
        metrics = {
            'depth_range': (metric_depth.min(), metric_depth.max()),
            'mean_depth': metric_depth.mean(),
            'object_sizes_reasonable': True,
            'num_validated_objects': 0
        }
        
        # 检查物体尺寸是否合理
        for obj in objects:
            category = obj.get('category', 'unknown')
            if category not in self.object_sizes:
                continue
            
            mask = obj['segmentation']
            if mask is None:
                continue
            
            # 获取物体深度
            obj_depth = metric_depth[mask]
            if len(obj_depth) == 0:
                continue
            
            median_obj_depth = np.median(obj_depth)
            
            # 检查深度是否在合理范围内（0.1米到20米）
            if median_obj_depth < 0.1 or median_obj_depth > 20.0:
                metrics['object_sizes_reasonable'] = False
                break
            
            metrics['num_validated_objects'] += 1
        
        return metrics


class MultiViewScaleRecovery:
    """
    多视角尺度恢复
    利用多个视角的信息提高尺度估计精度
    """
    
    def __init__(self):
        self.scale_history = []
        
    def add_scale_estimate(
        self,
        scale: float,
        confidence: float = 1.0
    ):
        """
        添加一个尺度估计
        
        Args:
            scale: 尺度因子
            confidence: 置信度
        """
        self.scale_history.append({
            'scale': scale,
            'confidence': confidence
        })
    
    def get_robust_scale(self) -> float:
        """
        获取鲁棒的尺度估计
        
        Returns:
            scale: 融合后的尺度
        """
        if len(self.scale_history) == 0:
            return 1.0
        
        # 加权中位数
        scales = np.array([s['scale'] for s in self.scale_history])
        confidences = np.array([s['confidence'] for s in self.scale_history])
        
        # 排序
        sorted_indices = np.argsort(scales)
        sorted_scales = scales[sorted_indices]
        sorted_confidences = confidences[sorted_indices]
        
        # 计算累积权重
        cumsum = np.cumsum(sorted_confidences)
        total_weight = cumsum[-1]
        
        # 找到加权中位数
        median_idx = np.searchsorted(cumsum, total_weight / 2)
        robust_scale = sorted_scales[median_idx]
        
        return robust_scale
    
    def filter_outliers(self, threshold: float = 2.0):
        """
        过滤异常尺度估计
        
        Args:
            threshold: 标准差阈值
        """
        if len(self.scale_history) < 3:
            return
        
        scales = np.array([s['scale'] for s in self.scale_history])
        mean_scale = np.mean(scales)
        std_scale = np.std(scales)
        
        # 保留在阈值范围内的估计
        filtered = []
        for entry in self.scale_history:
            if abs(entry['scale'] - mean_scale) < threshold * std_scale:
                filtered.append(entry)
        
        self.scale_history = filtered


if __name__ == "__main__":
    # 测试代码
    print("=== 测试ScaleRecovery ===")
    
    # 创建测试数据
    H, W = 480, 640
    
    # 生成相对深度图
    depth = np.random.rand(H, W) * 0.5 + 0.5  # 0.5-1.0 (相对深度)
    
    # 创建模拟物体
    objects = [
        {
            'category': 'chair',
            'bbox': [200, 150, 80, 120],
            'segmentation': np.zeros((H, W), dtype=bool)
        },
        {
            'category': 'table',
            'bbox': [300, 200, 150, 100],
            'segmentation': np.zeros((H, W), dtype=bool)
        }
    ]
    
    # 填充mask
    for obj in objects:
        x, y, w, h = obj['bbox']
        obj['segmentation'][y:y+h, x:x+w] = True
    
    # 相机参数
    camera_params = {
        'fx': 525.0,
        'fy': 525.0,
        'cx': 320.0,
        'cy': 240.0
    }
    
    # 测试尺度恢复
    print("\n1. 测试形状先验尺度恢复:")
    scale_recovery = ScaleRecovery(
        method="shape_prior",
        reference_objects=["chair", "table"]
    )
    
    metric_depth, scale_factor = scale_recovery.recover_scale(
        depth,
        objects,
        camera_params
    )
    
    print(f"   原始深度范围: [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"   度量深度范围: [{metric_depth.min():.3f}, {metric_depth.max():.3f}]米")
    print(f"   尺度因子: {scale_factor:.3f}")
    
    # 验证尺度
    print("\n2. 验证尺度合理性:")
    validation = scale_recovery.validate_scale(metric_depth, objects)
    print(f"   深度范围: {validation['depth_range']}")
    print(f"   平均深度: {validation['mean_depth']:.2f}米")
    print(f"   物体尺寸合理: {validation['object_sizes_reasonable']}")
    print(f"   验证的物体数: {validation['num_validated_objects']}")
    
    # 测试多视角尺度恢复
    print("\n3. 测试多视角尺度恢复:")
    multi_view = MultiViewScaleRecovery()
    
    # 添加多个尺度估计
    multi_view.add_scale_estimate(2.5, confidence=0.8)
    multi_view.add_scale_estimate(2.7, confidence=0.9)
    multi_view.add_scale_estimate(2.4, confidence=0.7)
    multi_view.add_scale_estimate(5.0, confidence=0.3)  # 异常值
    
    print(f"   添加了 {len(multi_view.scale_history)} 个尺度估计")
    
    # 过滤异常值
    multi_view.filter_outliers(threshold=2.0)
    print(f"   过滤后剩余 {len(multi_view.scale_history)} 个")
    
    # 获取鲁棒尺度
    robust_scale = multi_view.get_robust_scale()
    print(f"   鲁棒尺度: {robust_scale:.3f}")
    
    # 测试场景尺度估计
    print("\n4. 测试场景尺度估计:")
    depths = [depth + np.random.rand(H, W) * 0.1 for _ in range(3)]
    objects_list = [objects] * 3
    
    global_scale = scale_recovery.estimate_scene_scale(
        depths,
        objects_list,
        camera_params
    )
    print(f"   全局尺度: {global_scale:.3f}")
    
    print("\n测试完成！")