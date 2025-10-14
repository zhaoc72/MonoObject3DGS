"""
Loss Functions for MonoObject3DGS
包含多种损失函数：光度、深度、语义、形状先验等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import lpips


class CompositeLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(
        self,
        lambda_photometric: float = 1.0,
        lambda_depth: float = 0.1,
        lambda_shape_prior: float = 0.3,
        lambda_semantic: float = 0.2,
        lambda_smoothness: float = 0.05,
        lambda_symmetry: float = 0.01,
        use_lpips: bool = True
    ):
        super().__init__()
        
        self.lambda_photometric = lambda_photometric
        self.lambda_depth = lambda_depth
        self.lambda_shape_prior = lambda_shape_prior
        self.lambda_semantic = lambda_semantic
        self.lambda_smoothness = lambda_smoothness
        self.lambda_symmetry = lambda_symmetry
        
        # LPIPS感知损失
        self.use_lpips = use_lpips
        if use_lpips:
            self.lpips_fn = lpips.LPIPS(net='vgg').eval()
            for param in self.lpips_fn.parameters():
                param.requires_grad = False
                
        print("✓ 损失函数初始化完成")
        
    def forward(
        self,
        rendered_image: torch.Tensor,
        gt_image: torch.Tensor,
        rendered_depth: Optional[torch.Tensor] = None,
        gt_depth: Optional[torch.Tensor] = None,
        rendered_semantic: Optional[torch.Tensor] = None,
        gt_semantic: Optional[torch.Tensor] = None,
        gaussians: Optional[torch.Tensor] = None,
        shape_prior_loss: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            rendered_image: 渲染图像 (B, 3, H, W)
            gt_image: 真实图像 (B, 3, H, W)
            rendered_depth: 渲染深度 (B, 1, H, W)
            gt_depth: 真实深度 (B, 1, H, W)
            rendered_semantic: 渲染语义 (B, C, H, W)
            gt_semantic: 真实语义 (B, C, H, W)
            gaussians: Gaussian点云 (N, 3)
            shape_prior_loss: 形状先验损失（预计算）
            mask: 物体mask (B, 1, H, W)
            
        Returns:
            losses: 各项损失的字典
        """
        losses = {}
        total_loss = 0.0
        
        # 1. 光度损失
        photo_loss = self.photometric_loss(
            rendered_image, gt_image, mask
        )
        losses['photometric'] = photo_loss
        total_loss += self.lambda_photometric * photo_loss
        
        # 2. 深度一致性损失
        if rendered_depth is not None and gt_depth is not None:
            depth_loss = self.depth_consistency_loss(
                rendered_depth, gt_depth, mask
            )
            losses['depth'] = depth_loss
            total_loss += self.lambda_depth * depth_loss
            
        # 3. 语义一致性损失
        if rendered_semantic is not None and gt_semantic is not None:
            semantic_loss = self.semantic_consistency_loss(
                rendered_semantic, gt_semantic, mask
            )
            losses['semantic'] = semantic_loss
            total_loss += self.lambda_semantic * semantic_loss
            
        # 4. 形状先验损失
        if shape_prior_loss is not None:
            losses['shape_prior'] = shape_prior_loss
            total_loss += self.lambda_shape_prior * shape_prior_loss
            
        # 5. 平滑性损失
        if gaussians is not None:
            smooth_loss = self.smoothness_loss(gaussians)
            losses['smoothness'] = smooth_loss
            total_loss += self.lambda_smoothness * smooth_loss
            
        # 6. 对称性损失
        if gaussians is not None and self.lambda_symmetry > 0:
            sym_loss = self.symmetry_loss(gaussians)
            losses['symmetry'] = sym_loss
            total_loss += self.lambda_symmetry * sym_loss
            
        losses['total'] = total_loss
        
        return losses
    
    def photometric_loss(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        光度损失 = L1 + SSIM + LPIPS
        """
        # L1损失
        l1_loss = F.l1_loss(rendered, target, reduction='none')
        
        if mask is not None:
            l1_loss = (l1_loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            l1_loss = l1_loss.mean()
            
        # SSIM损失
        ssim_loss = 1.0 - self.ssim(rendered, target, mask)
        
        # 组合L1和SSIM
        combined_loss = 0.8 * l1_loss + 0.2 * ssim_loss
        
        # LPIPS感知损失（如果启用）
        if self.use_lpips:
            # LPIPS需要[-1, 1]范围
            rendered_norm = rendered * 2.0 - 1.0
            target_norm = target * 2.0 - 1.0
            
            with torch.no_grad():
                lpips_loss = self.lpips_fn(rendered_norm, target_norm).mean()
            
            combined_loss = combined_loss + 0.1 * lpips_loss
            
        return combined_loss
    
    def ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        window_size: int = 11
    ) -> torch.Tensor:
        """
        结构相似性指数
        """
        from pytorch_msssim import ssim
        
        # 计算SSIM
        ssim_val = ssim(
            img1, img2,
            data_range=1.0,
            size_average=False,
            win_size=window_size
        )
        
        if mask is not None:
            # 在mask区域内平均
            mask_downsampled = F.avg_pool2d(mask, kernel_size=window_size, stride=1, padding=window_size//2)
            ssim_val = (ssim_val * mask_downsampled).sum() / (mask_downsampled.sum() + 1e-8)
        else:
            ssim_val = ssim_val.mean()
            
        return ssim_val
    
    def depth_consistency_loss(
        self,
        rendered_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        深度一致性损失
        结合L1和梯度损失
        """
        # 只在有效深度区域计算
        valid_mask = (gt_depth > 0) & (gt_depth < 100)
        if mask is not None:
            valid_mask = valid_mask & (mask > 0.5)
            
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=rendered_depth.device)
            
        # L1损失
        l1_loss = F.l1_loss(
            rendered_depth[valid_mask],
            gt_depth[valid_mask]
        )
        
        # 梯度损失（保持边缘）
        grad_loss = self.gradient_loss(rendered_depth, gt_depth, valid_mask)
        
        return l1_loss + 0.5 * grad_loss
    
    def gradient_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """计算梯度损失"""
        # 计算x和y方向的梯度
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # 调整mask大小
        mask_x = mask[:, :, :, 1:]
        mask_y = mask[:, :, 1:, :]
        
        # L1损失
        loss_x = F.l1_loss(
            pred_grad_x[mask_x],
            target_grad_x[mask_x]
        ) if mask_x.sum() > 0 else 0.0
        
        loss_y = F.l1_loss(
            pred_grad_y[mask_y],
            target_grad_y[mask_y]
        ) if mask_y.sum() > 0 else 0.0
        
        return loss_x + loss_y
    
    def semantic_consistency_loss(
        self,
        rendered_semantic: torch.Tensor,
        gt_semantic: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        语义一致性损失
        """
        # 交叉熵损失
        if gt_semantic.dtype == torch.long:
            # 标签格式 (B, H, W)
            loss = F.cross_entropy(
                rendered_semantic,
                gt_semantic,
                reduction='none'
            )
        else:
            # 概率格式 (B, C, H, W)
            loss = F.kl_div(
                F.log_softmax(rendered_semantic, dim=1),
                gt_semantic,
                reduction='none'
            ).sum(dim=1)
            
        if mask is not None:
            loss = (loss * mask.squeeze(1)).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
            
        return loss
    
    def smoothness_loss(
        self,
        gaussians: torch.Tensor,
        k: int = 8
    ) -> torch.Tensor:
        """
        平滑性损失 - 鼓励局部平滑
        """
        from pytorch3d.ops import knn_points
        
        # 计算k近邻
        knn_result = knn_points(
            gaussians.unsqueeze(0),
            gaussians.unsqueeze(0),
            K=k+1
        )
        
        # 排除自己
        neighbor_dists = knn_result.dists[:, :, 1:]
        
        # 平均距离作为平滑性度量
        loss = neighbor_dists.mean()
        
        return loss
    
    def symmetry_loss(
        self,
        gaussians: torch.Tensor,
        axis: int = 0
    ) -> torch.Tensor:
        """
        对称性损失 - 鼓励沿某轴对称
        """
        # 沿指定轴镜像
        mirrored = gaussians.clone()
        mirrored[:, axis] = -mirrored[:, axis]
        
        # 计算Chamfer距离
        from pytorch3d.loss import chamfer_distance
        
        cd_loss, _ = chamfer_distance(
            gaussians.unsqueeze(0),
            mirrored.unsqueeze(0)
        )
        
        return cd_loss


class EdgeAwareLoss(nn.Module):
    """边缘感知损失 - 在物体边界处加强约束"""
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算边缘感知损失
        
        Args:
            rendered: 渲染图像 (B, C, H, W)
            target: 目标图像 (B, C, H, W)
            mask: 物体mask (B, 1, H, W)
        """
        # 检测mask边缘
        edge_mask = self.detect_edges(mask)
        
        # 在边缘处增强L1损失
        loss = F.l1_loss(rendered, target, reduction='none')
        
        # 边缘区域权重更高
        weighted_loss = loss * (1.0 + 2.0 * edge_mask)
        
        return weighted_loss.mean()
    
    @staticmethod
    def detect_edges(mask: torch.Tensor) -> torch.Tensor:
        """检测mask边缘"""
        # Sobel算子
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=mask.device
        ).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=mask.device
        ).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(mask, sobel_x, padding=1)
        grad_y = F.conv2d(mask, sobel_y, padding=1)
        
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # 归一化
        edge_mask = (edge_magnitude > 0.1).float()
        
        return edge_mask


if __name__ == "__main__":
    # 测试损失函数
    print("=== 测试组合损失 ===")
    
    loss_fn = CompositeLoss(use_lpips=False)
    
    # 创建测试数据
    B, C, H, W = 2, 3, 256, 256
    rendered = torch.rand(B, C, H, W)
    target = torch.rand(B, C, H, W)
    mask = (torch.rand(B, 1, H, W) > 0.3).float()
    
    # 计算损失
    losses = loss_fn(
        rendered_image=rendered,
        gt_image=target,
        mask=mask
    )
    
    print("损失:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
        
    print("\n=== 测试边缘感知损失 ===")
    edge_loss_fn = EdgeAwareLoss()
    edge_loss = edge_loss_fn(rendered, target, mask)
    print(f"边缘损失: {edge_loss.item():.4f}")