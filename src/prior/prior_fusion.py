"""
Adaptive Prior Fusion
自适应形状先验融合 - 核心创新
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PriorConfig:
    """先验配置"""
    explicit_weight: float = 0.5
    implicit_weight: float = 0.3
    min_prior_weight: float = 0.1
    max_prior_weight: float = 0.9
    confidence_based: bool = True


class AdaptivePriorFusion:
    """自适应形状先验融合器"""
    
    def __init__(
        self,
        explicit_prior,
        implicit_prior,
        config: PriorConfig
    ):
        self.explicit_prior = explicit_prior
        self.implicit_prior = implicit_prior
        self.config = config
        
        print("✓ AdaptivePriorFusion initialized")
        print(f"  Base weights: explicit={config.explicit_weight:.2f}, "
              f"implicit={config.implicit_weight:.2f}")
    
    def compute_adaptive_weights(
        self,
        viewing_coverage: float,
        segmentation_confidence: float,
        reconstruction_uncertainty: float
    ) -> Tuple[float, float, float]:
        """
        计算自适应权重
        
        核心算法：
        - 观测质量高 → 先验权重低（信任观测）
        - 观测质量低 → 先验权重高（依赖先验）
        
        Returns:
            (w_observation, w_explicit, w_implicit)
        """
        if not self.config.confidence_based:
            w_exp = self.config.explicit_weight
            w_imp = self.config.implicit_weight
            w_obs = 1.0 - w_exp - w_imp
            return max(0.0, w_obs), w_exp, w_imp
        
        # 综合观测质量
        observation_quality = (
            0.4 * viewing_coverage +
            0.3 * segmentation_confidence +
            0.3 * (1.0 - reconstruction_uncertainty)
        )
        
        # 先验强度 = 1 - 观测质量
        prior_strength = 1.0 - observation_quality
        
        # 计算权重
        w_explicit = self.config.explicit_weight * prior_strength
        w_implicit = self.config.implicit_weight * prior_strength
        
        # 限制范围
        w_explicit = np.clip(
            w_explicit,
            self.config.min_prior_weight,
            self.config.max_prior_weight
        )
        w_implicit = np.clip(
            w_implicit,
            self.config.min_prior_weight,
            self.config.max_prior_weight
        )
        
        # 确保总权重不超过1
        total_prior = w_explicit + w_implicit
        if total_prior > 0.9:
            scale = 0.9 / total_prior
            w_explicit *= scale
            w_implicit *= scale
        
        w_observation = 1.0 - w_explicit - w_implicit
        
        return w_observation, w_explicit, w_implicit
    
    def fuse_priors(
        self,
        observed_points: torch.Tensor,
        category: str,
        viewing_coverage: float,
        segmentation_confidence: float,
        reconstruction_uncertainty: float
    ) -> Tuple[torch.Tensor, Dict]:
        """
        融合观测和先验
        
        Args:
            observed_points: 观测点云 (N, 3)
            category: 物体类别
            viewing_coverage: 视角覆盖度 [0, 1]
            segmentation_confidence: 分割置信度 [0, 1]
            reconstruction_uncertainty: 重建不确定性 [0, 1]
            
        Returns:
            fused_points: 融合后的点云
            weights_info: 权重信息
        """
        # 计算权重
        w_obs, w_exp, w_imp = self.compute_adaptive_weights(
            viewing_coverage,
            segmentation_confidence,
            reconstruction_uncertainty
        )
        
        weights_info = {
            'observation': w_obs,
            'explicit': w_exp,
            'implicit': w_imp,
            'viewing_coverage': viewing_coverage,
            'seg_confidence': segmentation_confidence,
            'recon_uncertainty': reconstruction_uncertainty
        }
        
        # 获取先验形状
        explicit_shape = None
        implicit_shape = None
        
        # 显式先验
        if w_exp > 0.01:
            template = self.explicit_prior.get_template(category)
            if template is not None:
                explicit_shape = self._align_to_observation(
                    template.to(observed_points.device),
                    observed_points
                )
        
        # 隐式先验
        if w_imp > 0.01:
            implicit_shape = self.implicit_prior.get_category_prior(category)
            if implicit_shape is not None:
                implicit_shape = self._align_to_observation(
                    implicit_shape.to(observed_points.device),
                    observed_points
                )
        
        # 融合
        fused = self._weighted_fusion(
            observed_points,
            explicit_shape,
            implicit_shape,
            w_obs, w_exp, w_imp
        )
        
        return fused, weights_info
    
    def _align_to_observation(
        self,
        prior_points: torch.Tensor,
        observed_points: torch.Tensor
    ) -> torch.Tensor:
        """对齐先验到观测"""
        # 中心对齐
        prior_center = prior_points.mean(dim=0)
        obs_center = observed_points.mean(dim=0)
        
        prior_centered = prior_points - prior_center
        obs_centered = observed_points - obs_center
        
        # 尺度对齐
        prior_scale = prior_centered.norm(dim=1).max()
        obs_scale = obs_centered.norm(dim=1).max()
        
        scale_factor = obs_scale / (prior_scale + 1e-6)
        
        # 应用变换
        aligned = prior_centered * scale_factor + obs_center
        
        return aligned
    
    def _weighted_fusion(
        self,
        observation: torch.Tensor,
        explicit: Optional[torch.Tensor],
        implicit: Optional[torch.Tensor],
        w_obs: float,
        w_exp: float,
        w_imp: float
    ) -> torch.Tensor:
        """加权融合"""
        all_points = [observation]
        
        # 添加显式先验点
        if w_exp > 0.01 and explicit is not None:
            num_sample = int(len(observation) * w_exp / (w_obs + 1e-6))
            if num_sample > 0 and num_sample < len(explicit):
                indices = torch.randperm(len(explicit))[:num_sample]
                all_points.append(explicit[indices])
        
        # 添加隐式先验点
        if w_imp > 0.01 and implicit is not None:
            num_sample = int(len(observation) * w_imp / (w_obs + 1e-6))
            if num_sample > 0 and num_sample < len(implicit):
                indices = torch.randperm(len(implicit))[:num_sample]
                all_points.append(implicit[indices])
        
        fused = torch.cat(all_points, dim=0) if len(all_points) > 1 else observation
        
        return fused
    
    def compute_fused_prior_loss(
        self,
        predicted_points: torch.Tensor,
        category: str,
        weights_info: Dict
    ) -> torch.Tensor:
        """计算融合先验损失"""
        total_loss = torch.tensor(0.0, device=predicted_points.device)
        
        w_exp = weights_info['explicit']
        w_imp = weights_info['implicit']
        
        # 显式先验损失
        if w_exp > 0.01:
            template = self.explicit_prior.get_template(category)
            if template is not None:
                template = template.to(predicted_points.device)
                exp_loss = self.explicit_prior.compute_prior_loss(
                    predicted_points,
                    template
                )
                total_loss += w_exp * exp_loss
        
        # 隐式先验损失
        if w_imp > 0.01:
            imp_loss = self.implicit_prior.compute_prior_loss(
                predicted_points,
                category
            )
            total_loss += w_imp * imp_loss
        
        return total_loss


# 测试
if __name__ == "__main__":
    from src.priors.explicit_prior import ExplicitShapePrior
    from src.priors.implicit_prior import ImplicitShapePrior
    
    explicit = ExplicitShapePrior("data/shape_priors/explicit", device="cpu")
    implicit = ImplicitShapePrior(latent_dim=256)
    implicit.add_category_prototype("chair")
    
    config = PriorConfig(explicit_weight=0.5, implicit_weight=0.3)
    fusion = AdaptivePriorFusion(explicit, implicit, config)
    
    # 测试场景
    scenarios = [
        ("单视角", 0.1, 0.7, 0.8),
        ("部分观测", 0.5, 0.8, 0.4),
        ("充分观测", 0.9, 0.9, 0.2)
    ]
    
    print("\n=== 自适应权重测试 ===")
    for name, vc, sc, ru in scenarios:
        w_obs, w_exp, w_imp = fusion.compute_adaptive_weights(vc, sc, ru)
        print(f"{name}: obs={w_obs:.3f}, exp={w_exp:.3f}, imp={w_imp:.3f}")