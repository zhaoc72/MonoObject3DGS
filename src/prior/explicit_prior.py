"""
Explicit Shape Prior
显式几何形状先验 - 基于CAD模板
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import trimesh


class ExplicitShapePrior:
    """
    显式形状先验
    使用预定义的几何模板（CAD模型、统计形状模型）
    """
    
    def __init__(self, template_dir: str, device: str = "cuda"):
        """
        Args:
            template_dir: 模板文件目录
            device: 计算设备
        """
        self.template_dir = Path(template_dir)
        self.device = device
        self.templates = {}
        
        # 加载所有可用模板
        self._load_templates()
        
        print(f"✓ ExplicitShapePrior初始化: {len(self.templates)} 个类别")
    
    def _load_templates(self):
        """从目录加载所有模板"""
        if not self.template_dir.exists():
            print(f"警告: 模板目录不存在: {self.template_dir}")
            return
        
        # 支持的格式
        formats = ['*.obj', '*.ply', '*.off']
        
        for category_dir in self.template_dir.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                
                # 查找模板文件
                template_file = None
                for fmt in formats:
                    files = list(category_dir.glob(fmt))
                    if files:
                        template_file = files[0]
                        break
                
                if template_file:
                    try:
                        mesh = trimesh.load(template_file)
                        
                        # 提取顶点作为点云
                        if hasattr(mesh, 'vertices'):
                            vertices = np.array(mesh.vertices, dtype=np.float32)
                            
                            # 中心化和归一化
                            vertices = self._normalize_pointcloud(vertices)
                            
                            self.templates[category] = torch.from_numpy(vertices)
                            print(f"  加载 {category}: {len(vertices)} 个顶点")
                    except Exception as e:
                        print(f"  警告: 无法加载 {category}: {e}")
    
    def _normalize_pointcloud(self, points: np.ndarray) -> np.ndarray:
        """归一化点云到单位球"""
        # 中心化
        center = points.mean(axis=0)
        points = points - center
        
        # 缩放到单位球
        scale = np.linalg.norm(points, axis=1).max()
        if scale > 0:
            points = points / scale
        
        return points
    
    def get_template(self, category: str) -> Optional[torch.Tensor]:
        """
        获取类别模板
        
        Args:
            category: 物体类别
            
        Returns:
            template: 模板点云 (N, 3)
        """
        if category in self.templates:
            return self.templates[category].clone()
        else:
            # 尝试模糊匹配
            for cat in self.templates.keys():
                if category.lower() in cat.lower() or cat.lower() in category.lower():
                    print(f"  使用近似匹配: {category} → {cat}")
                    return self.templates[cat].clone()
            
            print(f"  警告: 未找到类别 '{category}' 的模板")
            return None
    
    def fit_template_to_pointcloud(
        self,
        template: torch.Tensor,
        pointcloud: torch.Tensor,
        method: str = "icp",
        num_iterations: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将模板拟合到观测点云
        
        Args:
            template: 模板点云 (N, 3)
            pointcloud: 观测点云 (M, 3)
            method: 拟合方法 (icp, scale_align)
            num_iterations: 迭代次数
            
        Returns:
            fitted_template: 拟合后的模板 (N, 3)
            rotation: 旋转矩阵 (3, 3)
            translation: 平移向量 (3,)
        """
        if method == "icp":
            return self._fit_icp(template, pointcloud, num_iterations)
        elif method == "scale_align":
            return self._fit_scale_align(template, pointcloud)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _fit_icp(
        self,
        template: torch.Tensor,
        pointcloud: torch.Tensor,
        num_iterations: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用ICP算法拟合"""
        try:
            from pytorch3d.ops import iterative_closest_point
            
            # 确保在正确的设备上
            template = template.to(self.device).unsqueeze(0)
            pointcloud = pointcloud.to(self.device).unsqueeze(0)
            
            # 运行ICP
            icp_result = iterative_closest_point(
                template,
                pointcloud,
                max_iterations=num_iterations
            )
            
            fitted_template = icp_result.Xt.squeeze(0)
            rotation = icp_result.RTs.R.squeeze(0)
            translation = icp_result.RTs.T.squeeze(0)
            
            return fitted_template, rotation, translation
            
        except ImportError:
            print("警告: PyTorch3D未安装，使用简化对齐")
            return self._fit_scale_align(template, pointcloud)
    
    def _fit_scale_align(
        self,
        template: torch.Tensor,
        pointcloud: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """简化的尺度对齐方法"""
        template = template.to(self.device)
        pointcloud = pointcloud.to(self.device)
        
        # 1. 中心对齐
        template_center = template.mean(dim=0)
        pc_center = pointcloud.mean(dim=0)
        
        template_centered = template - template_center
        pc_centered = pointcloud - pc_center
        
        # 2. 尺度对齐
        template_scale = template_centered.norm(dim=1).max()
        pc_scale = pc_centered.norm(dim=1).max()
        scale_factor = pc_scale / (template_scale + 1e-6)
        
        # 3. 应用变换
        fitted_template = template_centered * scale_factor + pc_center
        
        # 返回单位旋转矩阵和平移
        rotation = torch.eye(3, device=self.device)
        translation = pc_center - template_center * scale_factor
        
        return fitted_template, rotation, translation
    
    def compute_prior_loss(
        self,
        predicted_points: torch.Tensor,
        template_points: torch.Tensor,
        loss_type: str = "chamfer"
    ) -> torch.Tensor:
        """
        计算与模板的偏差损失
        
        Args:
            predicted_points: 预测点云 (N, 3)
            template_points: 模板点云 (M, 3)
            loss_type: 损失类型 (chamfer, l2)
            
        Returns:
            loss: 损失值
        """
        if loss_type == "chamfer":
            return self._chamfer_distance(predicted_points, template_points)
        elif loss_type == "l2":
            return self._l2_distance(predicted_points, template_points)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _chamfer_distance(
        self,
        points1: torch.Tensor,
        points2: torch.Tensor
    ) -> torch.Tensor:
        """计算Chamfer距离"""
        try:
            from pytorch3d.loss import chamfer_distance
            
            cd_loss, _ = chamfer_distance(
                points1.unsqueeze(0),
                points2.unsqueeze(0)
            )
            return cd_loss
            
        except ImportError:
            # 简化版Chamfer距离
            return self._chamfer_distance_simple(points1, points2)
    
    def _chamfer_distance_simple(
        self,
        points1: torch.Tensor,
        points2: torch.Tensor
    ) -> torch.Tensor:
        """简化的Chamfer距离计算"""
        # points1 -> points2
        dist1 = torch.cdist(points1, points2)  # (N, M)
        min_dist1 = dist1.min(dim=1)[0].mean()
        
        # points2 -> points1
        min_dist2 = dist1.min(dim=0)[0].mean()
        
        return min_dist1 + min_dist2
    
    def _l2_distance(
        self,
        points1: torch.Tensor,
        points2: torch.Tensor
    ) -> torch.Tensor:
        """L2距离（需要点数量相同）"""
        if points1.shape[0] != points2.shape[0]:
            # 采样到相同数量
            min_points = min(points1.shape[0], points2.shape[0])
            idx1 = torch.randperm(points1.shape[0])[:min_points]
            idx2 = torch.randperm(points2.shape[0])[:min_points]
            points1 = points1[idx1]
            points2 = points2[idx2]
        
        return torch.nn.functional.mse_loss(points1, points2)
    
    def augment_template(
        self,
        template: torch.Tensor,
        augmentation: str = "noise"
    ) -> torch.Tensor:
        """
        增强模板（用于训练时的数据增强）
        
        Args:
            template: 模板点云 (N, 3)
            augmentation: 增强类型
            
        Returns:
            augmented: 增强后的模板
        """
        if augmentation == "noise":
            noise = torch.randn_like(template) * 0.01
            return template + noise
        elif augmentation == "scale":
            scale = torch.rand(1) * 0.2 + 0.9  # 0.9-1.1
            return template * scale
        elif augmentation == "rotation":
            # 随机旋转
            angle = torch.rand(1) * 2 * np.pi
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rot_matrix = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], device=template.device)
            return template @ rot_matrix.T
        else:
            return template


if __name__ == "__main__":
    # 测试代码
    print("=== 测试ExplicitShapePrior ===")
    
    # 创建临时模板目录
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试模板
        chair_dir = Path(tmpdir) / "chair"
        chair_dir.mkdir()
        
        # 生成简单的椅子形状（立方体）
        vertices = np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]
        ])
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(chair_dir / "template.obj")
        
        # 初始化先验
        prior = ExplicitShapePrior(tmpdir, device="cpu")
        
        # 测试获取模板
        print("\n1. 测试获取模板:")
        template = prior.get_template("chair")
        if template is not None:
            print(f"   模板形状: {template.shape}")
            print(f"   模板范围: [{template.min():.2f}, {template.max():.2f}]")
        
        # 测试拟合
        print("\n2. 测试模板拟合:")
        pc = torch.randn(100, 3) * 0.5
        fitted, R, t = prior.fit_template_to_pointcloud(template, pc)
        print(f"   拟合后形状: {fitted.shape}")
        print(f"   旋转矩阵形状: {R.shape}")
        print(f"   平移向量: {t}")
        
        # 测试损失计算
        print("\n3. 测试先验损失:")
        loss = prior.compute_prior_loss(pc, fitted)
        print(f"   Chamfer距离: {loss.item():.4f}")
        
        # 测试增强
        print("\n4. 测试模板增强:")
        augmented = prior.augment_template(template, "noise")
        print(f"   增强后形状: {augmented.shape}")
        
        print("\n测试完成！")