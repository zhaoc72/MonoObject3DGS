"""
Adaptive Prior Fusion
自适应形状先验融合 - 核心创新点 ⭐⭐⭐
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .explicit_prior import ExplicitShapePrior
from .implicit_prior import ImplicitShapePrior


@dataclass
class PriorConfig:
    """先验配置"""
    # 基础权重
    explicit_weight: float = 0.5
    implicit_weight: float = 0.3
    
    # 自适应配置
    confidence_based: bool = True
    min_prior_weight: float = 0.1
    max_prior_weight: float = 0.8
    
    # 置信度因子权重
    viewing_coverage_weight: float = 0.4
    segmentation_confidence_weight: float = 0.3
    reconstruction_uncertainty_weight: float = 0.3


class AdaptivePriorFusion:
    """
    自适应先验融合器
    
    核心创新：根据场景条件动态调整先验权重
    - 视角充分 → 弱先验，数据驱动
    - 视角不足 → 强先验，先验引导
    """
    
    def __init__(
        self,
        explicit_prior: ExplicitShapePrior,
        implicit_prior: ImplicitShapePrior,
        config: PriorConfig
    ):
        """
        Args:
            explicit_prior: 显式先验
            implicit_prior: 隐式先验
            config: 配置
        """
        self.explicit_prior = explicit_prior
        self.implicit_prior = implicit_prior
        self.config = config
        
        print("✓ AdaptivePriorFusion初始化完成")
        print(f"  基础权重: explicit={config.explicit_weight}, implicit={config.implicit_weight}")
        print(f"  自适应融合: {'启用' if config.confidence_based else '禁用'}")
    
    def fuse_priors(
        self,
        pointcloud: torch.Tensor,
        category: str,
        viewing_coverage: float,
        segmentation_confidence: float,
        reconstruction_uncertainty: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        融合显式和隐式先验 ⭐ 核心方法
        
        Args:
            pointcloud: 观测点云 (N, 3)
            category: 物体类别
            viewing_coverage: 视角覆盖度 [0, 1]
            segmentation_confidence: 分割置信度 [0, 1]
            reconstruction_uncertainty: 重建不确定性 [0, 1]
            
        Returns:
            fused_pointcloud: 融合后的点云 (M, 3)
            weights_info: 权重信息字典
        """
        # Step 1: 计算自适应权重
        if self.config.confidence_based:
            w_explicit, w_implicit = self.compute_adaptive_weights(
                viewing_coverage,
                segmentation_confidence,
                reconstruction_uncertainty
            )
        else:
            w_explicit = self.config.explicit_weight
            w_implicit = self.config.implicit_weight
        
        w_observation = 1.0 - w_explicit - w_implicit
        
        # Step 2: 获取先验形状
        explicit_shape = None
        implicit_shape = None
        
        # 显式先验
        if w_explicit > 0:
            template = self.explicit_prior.get_template(category)
            if template is not None:
                # 拟合模板到观测
                explicit_shape, _, _ = self.explicit_prior.fit_template_to_pointcloud(
                    template.to(pointcloud.device),
                    pointcloud
                )
        
        # 隐式先验
        if w_implicit > 0:
            implicit_shape = self.implicit_prior.get_category_prior(category)
            if implicit_shape is not None:
                implicit_shape = implicit_shape.to(pointcloud.device)
                # 对齐隐式形状
                implicit_shape = self._align_shape_to_observation(
                    implicit_shape,
                    pointcloud
                )
        
        # Step 3: 融合
        fused = self._weighted_fusion(
            pointcloud,
            explicit_shape,
            implicit_shape,
            w_observation,
            w_explicit,
            w_implicit
        )
        
        # 记录权重信息
        weights_info = {
            'observation': w_observation,
            'explicit': w_explicit,
            'implicit': w_implicit,
            'viewing_coverage': viewing_coverage,
            'segmentation_confidence': segmentation_confidence,
            'reconstruction_uncertainty': reconstruction_uncertainty
        }
        
        return fused, weights_info
    
    def compute_adaptive_weights(
        self,
        viewing_coverage: float,
        segmentation_confidence: float,
        reconstruction_uncertainty: float
    ) -> Tuple[float, float]:
        """
        计算自适应权重 ⭐ 核心算法
        
        策略：
        - 综合置信度高 → 先验权重低（信任观测）
        - 综合置信度低 → 先验权重高（依赖先验）
        
        Args:
            viewing_coverage: 视角覆盖度 [0, 1]
            segmentation_confidence: 分割置信度 [0, 1]
            reconstruction_uncertainty: 重建不确定性 [0, 1]
            
        Returns:
            w_explicit: 显式先验权重
            w_implicit: 隐式先验权重
        """
        # 计算综合置信度
        overall_confidence = (
            self.config.viewing_coverage_weight * viewing_coverage +
            self.config.segmentation_confidence_weight * segmentation_confidence +
            self.config.reconstruction_uncertainty_weight * (1.0 - reconstruction_uncertainty)
        )
        
        # 先验强度 = 1 - 置信度
        # 置信度高时，先验弱；置信度低时，先验强
        prior_strength = 1.0 - overall_confidence
        
        # 计算权重
        w_explicit = self.config.explicit_weight * prior_strength
        w_implicit = self.config.implicit_weight * prior_strength
        
        # 限制在合理范围内
        w_explicit = np.clip(w_explicit, self.config.min_prior_weight, self.config.max_prior_weight)
        w_implicit = np.clip(w_implicit, self.config.min_prior_weight, self.config.max_prior_weight)
        
        # 确保总权重不超过1
        total_prior = w_explicit + w_implicit
        if total_prior > 0.9:  # 留给观测至少10%
            scale = 0.9 / total_prior
            w_explicit *= scale
            w_implicit *= scale
        
        return w_explicit, w_implicit
    
    def _align_shape_to_observation(
        self,
        shape: torch.Tensor,
        observation: torch.Tensor
    ) -> torch.Tensor:
        """
        将先验形状对齐到观测
        
        Args:
            shape: 先验形状 (N, 3)
            observation: 观测点云 (M, 3)
            
        Returns:
            aligned_shape: 对齐后的形状 (N, 3)
        """
        # 中心对齐
        shape_center = shape.mean(dim=0)
        obs_center = observation.mean(dim=0)
        
        shape_centered = shape - shape_center
        obs_centered = observation - obs_center
        
        # 尺度对齐
        shape_scale = shape_centered.norm(dim=1).max()
        obs_scale = obs_centered.norm(dim=1).max()
        
        scale_factor = obs_scale / (shape_scale + 1e-6)
        
        # 应用变换
        aligned = shape_centered * scale_factor + obs_center
        
        return aligned
    
    def _weighted_fusion(
        self,
        observation: torch.Tensor,
        explicit_shape: Optional[torch.Tensor],
        implicit_shape: Optional[torch.Tensor],
        w_obs: float,
        w_exp: float,
        w_imp: float
    ) -> torch.Tensor:
        """
        加权融合三个点云源
        
        Args:
            observation: 观测点云 (N, 3)
            explicit_shape: 显式先验 (M1, 3) or None
            implicit_shape: 隐式先验 (M2, 3) or None
            w_obs, w_exp, w_imp: 权重
            
        Returns:
            fused: 融合后的点云
        """
        all_points = []
        all_weights = []
        
        # 观测点
        if w_obs > 0:
            all_points.append(observation)
            all_weights.append(w_obs)
        
        # 显式先验点
        if w_exp > 0 and explicit_shape is not None:
            # 从显式形状采样点
            num_sample = int(len(observation) * w_exp / (w_obs + 1e-6))
            if num_sample > 0:
                indices = torch.randperm(len(explicit_shape))[:min(num_sample, len(explicit_shape))]
                all_points.append(explicit_shape[indices])
                all_weights.append(w_exp)
        
        # 隐式先验点
        if w_imp > 0 and implicit_shape is not None:
            # 从隐式形状采样点
            num_sample = int(len(observation) * w_imp / (w_obs + 1e-6))
            if num_sample > 0:
                indices = torch.randperm(len(implicit_shape))[:min(num_sample, len(implicit_shape))]
                all_points.append(implicit_shape[indices])
                all_weights.append(w_imp)
        
        if len(all_points) == 0:
            return observation
        
        # 合并所有点
        fused = torch.cat(all_points, dim=0)
        
        return fused
    
    def compute_fused_prior_loss(
        self,
        pointcloud: torch.Tensor,
        category: str,
        weights: Dict[str, float]
    ) -> torch.Tensor:
        """
        计算融合先验损失
        
        Args:
            pointcloud: 当前点云 (N, 3)
            category: 类别
            weights: 权重信息
            
        Returns:
            loss: 融合先验损失
        """
        total_loss = torch.tensor(0.0, device=pointcloud.device)
        
        # 显式先验损失
        if weights['explicit'] > 0:
            template = self.explicit_prior.get_template(category)
            if template is not None:
                template = template.to(pointcloud.device)
                explicit_loss = self.explicit_prior.compute_prior_loss(
                    pointcloud,
                    template
                )
                total_loss += weights['explicit'] * explicit_loss
        
        # 隐式先验损失
        if weights['implicit'] > 0:
            implicit_loss = self.implicit_prior.compute_prior_loss(
                pointcloud,
                category
            )
            total_loss += weights['implicit'] * implicit_loss
        
        return total_loss
    
    def visualize_fusion(
        self,
        observation: torch.Tensor,
        category: str,
        viewing_coverage: float,
        segmentation_confidence: float,
        reconstruction_uncertainty: float
    ) -> Dict:
        """
        可视化融合过程（用于调试和展示）
        
        Returns:
            visualization_data: 包含各个组件的数据
        """
        # 获取权重
        w_exp, w_imp = self.compute_adaptive_weights(
            viewing_coverage,
            segmentation_confidence,
            reconstruction_uncertainty
        )
        w_obs = 1.0 - w_exp - w_imp
        
        # 获取各个组件
        template = self.explicit_prior.get_template(category)
        explicit_shape = None
        if template is not None:
            explicit_shape, _, _ = self.explicit_prior.fit_template_to_pointcloud(
                template.to(observation.device),
                observation
            )
        
        implicit_shape = self.implicit_prior.get_category_prior(category)
        if implicit_shape is not None:
            implicit_shape = self._align_shape_to_observation(
                implicit_shape.to(observation.device),
                observation
            )
        
        return {
            'observation': observation.cpu().numpy(),
            'explicit_prior': explicit_shape.cpu().numpy() if explicit_shape is not None else None,
            'implicit_prior': implicit_shape.cpu().numpy() if implicit_shape is not None else None,
            'weights': {
                'observation': w_obs,
                'explicit': w_exp,
                'implicit': w_imp
            }
        }


if __name__ == "__main__":
    # 测试代码
    print("=== 测试AdaptivePriorFusion ===")
    
    # 初始化组件
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建简单模板
        chair_dir = Path(tmpdir) / "chair"
        chair_dir.mkdir()
        
        import trimesh
        vertices = np.random.rand(100, 3) - 0.5
        mesh = trimesh.Trimesh(vertices=vertices)
        mesh.export(chair_dir / "template.obj")
        
        # 初始化先验
        explicit_prior = ExplicitShapePrior(tmpdir, device="cpu")
        implicit_prior = ImplicitShapePrior(latent_dim=256)
        implicit_prior.add_category_prototype("chair")
        
        # 初始化融合器
        config = PriorConfig(
            explicit_weight=0.5,
            implicit_weight=0.3,
            confidence_based=True
        )
        fusion = AdaptivePriorFusion(explicit_prior, implicit_prior, config)
        
        # 测试不同场景
        test_cases = [
            ("充分观测", 0.9, 0.9, 0.1),
            ("部分遮挡", 0.5, 0.8, 0.3),
            ("严重遮挡", 0.2, 0.6, 0.7)
        ]
        
        print("\n测试自适应权重计算:")
        print("-" * 70)
        print(f"{'场景':<15} {'视角覆盖':<10} {'分割置信':<10} {'重建不确定':<12} {'显式权重':<10} {'隐式权重':<10}")
        print("-" * 70)
        
        for name, vc, sc, ru in test_cases:
            w_exp, w_imp = fusion.compute_adaptive_weights(vc, sc, ru)
            w_obs = 1.0 - w_exp - w_imp
            print(f"{name:<15} {vc:<10.2f} {sc:<10.2f} {ru:<12.2f} {w_exp:<10.3f} {w_imp:<10.3f}")
        
        print("-" * 70)
        
        # 测试融合
        print("\n测试先验融合:")
        test_pc = torch.randn(200, 3)
        
        fused_pc, weights_info = fusion.fuse_priors(
            test_pc,
            "chair",
            viewing_coverage=0.5,
            segmentation_confidence=0.8,
            reconstruction_uncertainty=0.3
        )
        
        print(f"  输入点云: {test_pc.shape}")
        print(f"  融合点云: {fused_pc.shape}")
        print(f"  权重分配:")
        print(f"    观测: {weights_info['observation']:.3f}")
        print(f"    显式: {weights_info['explicit']:.3f}")
        print(f"    隐式: {weights_info['implicit']:.3f}")
        
        # 测试损失计算
        print("\n测试融合损失:")
        loss = fusion.compute_fused_prior_loss(test_pc, "chair", weights_info)
        print(f"  融合先验损失: {loss.item():.4f}")
        
        print("\n测试完成！")