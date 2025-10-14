"""
Loss Functions
损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class CompositeLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(
        self,
        lambda_photometric: float = 1.0,
        lambda_depth: float = 0.1,
        lambda_shape_prior: float = 0.5,
        lambda_semantic: float = 0.2,
        lambda_smoothness: float = 0.05,
        lambda_symmetry: float = 0.01,
        use_lpips: bool = False
    ):
        super().__init__()
        
        self.lambda_photometric = lambda_photometric
        self.lambda_depth = lambda_depth
        self.lambda_shape_prior = lambda_shape_prior
        self.lambda_semantic = lambda_semantic
        self.lambda_smoothness = lambda_smoothness
        self.lambda_symmetry = lambda_symmetry
        
        self.use_lpips = use_lpips
        if use_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='vgg')
                print("  LPIPS loss enabled")
            except ImportError:
                print("  Warning: lpips not installed, disabling LPIPS")
                self.use_lpips = False
        
        print(f"✓ CompositeLoss initialized")
        print(f"  Weights: photo={lambda_photometric}, depth={lambda_depth}, "
              f"prior={lambda_shape_prior}, semantic={lambda_semantic}")
    
    def photometric_loss(
        self,
        pred_image: torch.Tensor,
        target_image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """光度损失（L1 + SSIM）"""
        # L1 loss
        l1_loss = F.l1_loss(pred_image, target_image, reduction='none')
        
        if mask is not None:
            l1_loss = (l1_loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            l1_loss = l1_loss.mean()
        
        # SSIM loss
        ssim_loss = 1.0 - self._ssim(pred_image, target_image, mask)
        
        # 组合
        photo_loss = 0.8 * l1_loss + 0.2 * ssim_loss
        
        # LPIPS (可选)
        if self.use_lpips:
            lpips_loss = self.lpips_fn(pred_image, target_image).mean()
            photo_loss = 0.7 * photo_loss + 0.3 * lpips_loss
        
        return photo_loss
    
    def depth_consistency_loss(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """深度一致性损失"""
        # Scale-invariant depth loss
        diff = torch.log(pred_depth + 1e-6) - torch.log(target_depth + 1e-6)
        
        if mask is not None:
            diff = diff * mask
            n = mask.sum() + 1e-6
        else:
            n = diff.numel()
        
        loss = (diff ** 2).sum() / n - (diff.sum() ** 2) / (n ** 2)
        
        return loss
    
    def shape_prior_loss(
        self,
        pred_points: torch.Tensor,
        prior_points: torch.Tensor
    ) -> torch.Tensor:
        """形状先验损失（简化的Chamfer距离）"""
        # 简化版本
        if pred_points.shape[0] != prior_points.shape[0]:
            min_pts = min(pred_points.shape[0], prior_points.shape[0])
            pred_points = pred_points[:min_pts]
            prior_points = prior_points[:min_pts]
        
        return F.mse_loss(pred_points, prior_points)
    
    def semantic_consistency_loss(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """语义一致性损失"""
        # Cosine similarity
        pred_norm = F.normalize(pred_features, dim=1)
        target_norm = F.normalize(target_features, dim=1)
        
        similarity = (pred_norm * target_norm).sum(dim=1)
        loss = 1.0 - similarity
        
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        
        return loss
    
    def smoothness_loss(
        self,
        pointcloud: torch.Tensor,
        k: int = 8
    ) -> torch.Tensor:
        """平滑性损失"""
        # 简化：使用全局方差
        center = pointcloud.mean(dim=0)
        variance = ((pointcloud - center) ** 2).mean()
        return variance * 0.1
    
    def symmetry_loss(
        self,
        pointcloud: torch.Tensor,
        axis: int = 0
    ) -> torch.Tensor:
        """对称性损失"""
        mirrored = pointcloud.clone()
        mirrored[:, axis] = -mirrored[:, axis]
        return F.mse_loss(pointcloud, mirrored)
    
    def forward(
        self,
        pred: Dict,
        target: Dict,
        weights: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            pred: 预测结果字典
            target: 目标字典
            weights: 可选的动态权重
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=pred['image'].device)
        
        # 使用动态权重或默认权重
        if weights is None:
            weights = {}
        
        lambda_photo = weights.get('photometric', self.lambda_photometric)
        lambda_depth = weights.get('depth', self.lambda_depth)
        lambda_prior = weights.get('shape_prior', self.lambda_shape_prior)
        lambda_semantic = weights.get('semantic', self.lambda_semantic)
        lambda_smooth = weights.get('smoothness', self.lambda_smoothness)
        lambda_sym = weights.get('symmetry', self.lambda_symmetry)
        
        # 光度损失
        if 'image' in pred and 'image' in target and lambda_photo > 0:
            mask = target.get('mask', None)
            photo_loss = self.photometric_loss(
                pred['image'], target['image'], mask
            )
            losses['photometric'] = photo_loss
            total_loss += lambda_photo * photo_loss
        
        # 深度损失
        if 'depth' in pred and 'depth' in target and lambda_depth > 0:
            mask = target.get('mask', None)
            depth_loss = self.depth_consistency_loss(
                pred['depth'], target['depth'], mask
            )
            losses['depth'] = depth_loss
            total_loss += lambda_depth * depth_loss
        
        # 形状先验损失
        if 'points' in pred and 'prior_points' in target and lambda_prior > 0:
            prior_loss = self.shape_prior_loss(
                pred['points'], target['prior_points']
            )
            losses['shape_prior'] = prior_loss
            total_loss += lambda_prior * prior_loss
        
        # 语义损失
        if 'features' in pred and 'features' in target and lambda_semantic > 0:
            mask = target.get('mask', None)
            semantic_loss = self.semantic_consistency_loss(
                pred['features'], target['features'], mask
            )
            losses['semantic'] = semantic_loss
            total_loss += lambda_semantic * semantic_loss
        
        # 平滑性
        if 'points' in pred and lambda_smooth > 0:
            smooth_loss = self.smoothness_loss(pred['points'])
            losses['smoothness'] = smooth_loss
            total_loss += lambda_smooth * smooth_loss
        
        # 对称性
        if 'points' in pred and lambda_sym > 0:
            sym_loss = self.symmetry_loss(pred['points'])
            losses['symmetry'] = sym_loss
            total_loss += lambda_sym * sym_loss
        
        losses['total'] = total_loss
        
        return losses
    
    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        window_size: int = 11
    ) -> torch.Tensor:
        """计算SSIM"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 简化版本：使用全局统计
        mu1 = img1.mean()
        mu2 = img2.mean()
        
        sigma1_sq = ((img1 - mu1) ** 2).mean()
        sigma2_sq = ((img2 - mu2) ** 2).mean()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim


# 测试
if __name__ == "__main__":
    loss_fn = CompositeLoss()
    
    # 测试数据
    pred = {
        'image': torch.rand(3, 512, 512),
        'depth': torch.rand(512, 512),
        'points': torch.randn(100, 3)
    }
    
    target = {
        'image': torch.rand(3, 512, 512),
        'depth': torch.rand(512, 512),
        'prior_points': torch.randn(100, 3)
    }
    
    losses = loss_fn(pred, target)
    
    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")