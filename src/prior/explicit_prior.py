"""
Explicit Shape Prior
显式几何形状先验（CAD模板）
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import trimesh


class ExplicitShapePrior:
    """显式形状先验"""
    
    def __init__(self, template_dir: str, device: str = "cuda"):
        self.template_dir = Path(template_dir)
        self.device = device
        self.templates = {}
        
        self._load_templates()
        
        print(f"✓ ExplicitShapePrior initialized: {len(self.templates)} categories")
    
    def _load_templates(self):
        """加载所有模板"""
        if not self.template_dir.exists():
            print(f"Warning: Template directory not found: {self.template_dir}")
            print("Creating dummy templates...")
            self._create_dummy_templates()
            return
        
        formats = ['*.obj', '*.ply', '*.off']
        
        for category_dir in self.template_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
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
                    
                    if hasattr(mesh, 'vertices'):
                        vertices = np.array(mesh.vertices, dtype=np.float32)
                        vertices = self._normalize_pointcloud(vertices)
                        self.templates[category] = torch.from_numpy(vertices)
                        print(f"  Loaded {category}: {len(vertices)} vertices")
                except Exception as e:
                    print(f"  Warning: Could not load {category}: {e}")
    
    def _create_dummy_templates(self):
        """创建简单的占位模板"""
        dummy_categories = ['chair', 'table', 'sofa']
        
        for category in dummy_categories:
            # 创建简单的立方体点云
            n_points = 1000
            vertices = np.random.randn(n_points, 3).astype(np.float32) * 0.5
            vertices = self._normalize_pointcloud(vertices)
            self.templates[category] = torch.from_numpy(vertices)
            print(f"  Created dummy template: {category}")
    
    def _normalize_pointcloud(self, points: np.ndarray) -> np.ndarray:
        """归一化点云"""
        center = points.mean(axis=0)
        points = points - center
        
        scale = np.linalg.norm(points, axis=1).max()
        if scale > 0:
            points = points / scale
        
        return points
    
    def get_template(self, category: str) -> Optional[torch.Tensor]:
        """获取类别模板"""
        if category in self.templates:
            return self.templates[category].clone()
        
        # 模糊匹配
        for cat in self.templates.keys():
            if category.lower() in cat.lower() or cat.lower() in category.lower():
                print(f"  Using approximate match: {category} → {cat}")
                return self.templates[cat].clone()
        
        return None
    
    def fit_template_to_pointcloud(
        self,
        template: torch.Tensor,
        pointcloud: torch.Tensor,
        method: str = "scale_align"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将模板拟合到点云
        
        Returns:
            fitted_template, rotation, translation
        """
        if method == "scale_align":
            return self._fit_scale_align(template, pointcloud)
        else:
            return self._fit_scale_align(template, pointcloud)
    
    def _fit_scale_align(
        self,
        template: torch.Tensor,
        pointcloud: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """简化的尺度对齐"""
        template = template.to(self.device)
        pointcloud = pointcloud.to(self.device)
        
        # 中心对齐
        template_center = template.mean(dim=0)
        pc_center = pointcloud.mean(dim=0)
        
        template_centered = template - template_center
        pc_centered = pointcloud - pc_center
        
        # 尺度对齐
        template_scale = template_centered.norm(dim=1).max()
        pc_scale = pc_centered.norm(dim=1).max()
        scale_factor = pc_scale / (template_scale + 1e-6)
        
        # 应用变换
        fitted_template = template_centered * scale_factor + pc_center
        
        rotation = torch.eye(3, device=self.device)
        translation = pc_center - template_center * scale_factor
        
        return fitted_template, rotation, translation
    
    def compute_prior_loss(
        self,
        predicted_points: torch.Tensor,
        template_points: torch.Tensor,
        loss_type: str = "chamfer"
    ) -> torch.Tensor:
        """计算先验损失"""
        if loss_type == "chamfer":
            return self._chamfer_distance_simple(predicted_points, template_points)
        else:
            return self._l2_distance(predicted_points, template_points)
    
    def _chamfer_distance_simple(
        self,
        points1: torch.Tensor,
        points2: torch.Tensor
    ) -> torch.Tensor:
        """简化的Chamfer距离"""
        # points1 -> points2
        dist1 = torch.cdist(points1, points2)
        min_dist1 = dist1.min(dim=1)[0].mean()
        
        # points2 -> points1
        min_dist2 = dist1.min(dim=0)[0].mean()
        
        return min_dist1 + min_dist2
    
    def _l2_distance(
        self,
        points1: torch.Tensor,
        points2: torch.Tensor
    ) -> torch.Tensor:
        """L2距离"""
        if points1.shape[0] != points2.shape[0]:
            min_points = min(points1.shape[0], points2.shape[0])
            idx1 = torch.randperm(points1.shape[0])[:min_points]
            idx2 = torch.randperm(points2.shape[0])[:min_points]
            points1 = points1[idx1]
            points2 = points2[idx2]
        
        return torch.nn.functional.mse_loss(points1, points2)


# 测试
if __name__ == "__main__":
    prior = ExplicitShapePrior("data/shape_priors/explicit", device="cpu")
    
    # 测试获取模板
    template = prior.get_template("chair")
    if template is not None:
        print(f"Template shape: {template.shape}")
        
        # 测试拟合
        pc = torch.randn(100, 3)
        fitted, R, t = prior.fit_template_to_pointcloud(template, pc)
        print(f"Fitted shape: {fitted.shape}")
        
        # 测试损失
        loss = prior.compute_prior_loss(pc, fitted)
        print(f"Prior loss: {loss.item():.4f}")