"""
Unified Trainer for MonoObject3DGS
统一的训练器，支持静态和动态场景
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import json
import numpy as np
from datetime import datetime

from ..segmentation.dinov2_extractor_v2 import DINOv2ExtractorV2
from ..segmentation.sam2_segmenter import SAM2Segmenter
from ..depth.depth_anything_v2_upgraded import DepthAnythingV2Upgraded
from ..priors.prior_fusion import AdaptivePriorFusion
from ..reconstruction.scene_gaussian import SceneGaussians
from ..optimization.losses import CompositeLoss
from .evaluator import Evaluator


class UnifiedTrainer:
    """统一训练器"""
    
    def __init__(
        self,
        config: Dict,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda"
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 初始化组件
        self._init_models()
        self._init_optimizers()
        self._init_losses()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        # 输出目录
        self.output_dir = Path(config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志
        self.train_history = []
        self.val_history = []
        
        print("✓ UnifiedTrainer initialized")
    
    def _init_models(self):
        """初始化模型"""
        print("\n[1/3] Initializing models...")
        
        # DINOv2特征提取器
        self.dinov2 = DINOv2ExtractorV2(
            model_name=self.config['segmentation']['dinov2']['model_name'],
            feature_dim=self.config['segmentation']['dinov2']['feature_dim'],
            device=self.device
        )
        
        # SAM2分割器
        self.sam2 = SAM2Segmenter(
            model_size=self.config['segmentation']['sam2']['model_size'],
            checkpoint=self.config['segmentation']['sam2']['checkpoint'],
            device=self.device,
            mode="image"
        )
        
        # 深度估计器
        self.depth_estimator = DepthAnythingV2Upgraded(
            model_size=self.config['depth']['model_size'],
            metric_depth=self.config['depth']['metric_depth'],
            device=self.device
        )
        
        # 场景Gaussian管理器
        from ..reconstruction.gaussian_model import GaussianConfig
        gaussian_config = GaussianConfig(
            sh_degree=self.config['reconstruction']['gaussian']['sh_degree'],
            init_scale=self.config['reconstruction']['gaussian']['init_scale'],
            opacity_init=self.config['reconstruction']['gaussian']['opacity_init']
        )
        self.scene_gaussians = SceneGaussians(gaussian_config)
        
        # 先验融合器（如果启用）
        if self.config.get('shape_prior', {}).get('enabled', True):
            from ..priors.explicit_prior import ExplicitShapePrior
            from ..priors.implicit_prior import ImplicitShapePrior
            from ..priors.prior_fusion import PriorConfig
            
            explicit_prior = ExplicitShapePrior(
                template_dir=self.config['shape_prior']['explicit']['template_dir'],
                device=self.device
            )
            
            implicit_prior = ImplicitShapePrior(
                latent_dim=self.config['shape_prior']['implicit']['latent_dim']
            ).to(self.device)
            
            prior_config = PriorConfig(
                explicit_weight=self.config['shape_prior']['explicit']['init_weight'],
                implicit_weight=self.config['shape_prior']['implicit']['init_weight']
            )
            
            self.prior_fusion = AdaptivePriorFusion(
                explicit_prior, implicit_prior, prior_config
            )
        else:
            self.prior_fusion = None
    
    def _init_optimizers(self):
        """初始化优化器"""
        print("\n[2/3] Initializing optimizers...")
        
        # 只优化隐式先验（其他模块冻结）
        if self.prior_fusion is not None:
            self.prior_optimizer = torch.optim.Adam(
                self.prior_fusion.implicit_prior.parameters(),
                lr=self.config['optimization']['learning_rate']
            )
        
        # Gaussian参数优化器将动态创建
        self.gaussian_optimizers = {}
    
    def _init_losses(self):
        """初始化损失函数"""
        print("\n[3/3] Initializing losses...")
        
        loss_config = self.config['optimization']['loss']
        self.loss_fn = CompositeLoss(
            lambda_photometric=loss_config['photometric'],
            lambda_depth=loss_config['depth_consistency'],
            lambda_shape_prior=loss_config['shape_prior'],
            lambda_semantic=loss_config['semantic_consistency'],
            lambda_smoothness=loss_config['smoothness'],
            use_lpips=loss_config.get('use_lpips', False)
        ).to(self.device)
        
        # 评估器
        if self.val_loader is not None:
            self.evaluator = Evaluator(self.device)
    
    def train_epoch(self) -> Dict:
        """训练一个epoch"""
        self.scene_gaussians.train()
        
        epoch_losses = {
            'total': 0.0,
            'photometric': 0.0,
            'depth': 0.0,
            'shape_prior': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            images = batch['images'].to(self.device)
            depths = batch.get('depths')
            masks = batch.get('masks')
            
            if depths is not None:
                depths = depths.to(self.device)
            if masks is not None:
                masks = masks.to(self.device)
            
            # 前向传播
            losses = self._forward_step(images, depths, masks, batch)
            
            # 反向传播
            self._backward_step(losses)
            
            # 更新统计
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # 更新进度条
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': losses['total'].item()
                })
            
            self.global_step += 1
        
        # 平均损失
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def _forward_step(
        self,
        images: torch.Tensor,
        depths: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        batch: Dict
    ) -> Dict:
        """前向传播步骤"""
        B = images.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)
        losses = {}
        
        # 对每张图像独立处理
        for b in range(B):
            image = images[b]
            depth = depths[b] if depths is not None else None
            mask = masks[b] if masks is not None else None
            
            # 1. 特征提取
            dense_features = self.dinov2.get_dense_features(
                image.permute(1, 2, 0).cpu().numpy() * 255,
                target_size=(image.shape[1], image.shape[2])
            )
            
            # 2. 分割（如果没有GT mask）
            if mask is None:
                # 使用SAM自动分割
                sam_masks = self.sam2.segment_automatic(
                    image.permute(1, 2, 0).cpu().numpy() * 255
                )
                # 转换为tensor
                # ... (简化，实际需要处理)
            
            # 3. 深度估计（如果没有GT depth）
            if depth is None:
                depth_pred = self.depth_estimator.estimate(
                    image.permute(1, 2, 0).cpu().numpy() * 255
                )
                depth = torch.from_numpy(depth_pred).unsqueeze(0).to(self.device)
            
            # 4. 初始化或更新Gaussians
            # ... (简化，实际需要完整实现)
            
            # 5. 计算损失
            if self.prior_fusion is not None:
                # 形状先验损失
                prior_loss = torch.tensor(0.0, device=self.device)
                for obj_id, obj_gaussian in self.scene_gaussians.objects.items():
                    _, weights_info = self.prior_fusion.fuse_priors(
                        obj_gaussian.get_xyz,
                        obj_gaussian.category,
                        viewing_coverage=0.5,
                        segmentation_confidence=0.8,
                        reconstruction_uncertainty=0.3
                    )
                    
                    obj_prior_loss = self.prior_fusion.compute_fused_prior_loss(
                        obj_gaussian.get_xyz,
                        obj_gaussian.category,
                        weights_info
                    )
                    prior_loss += obj_prior_loss
                
                losses['shape_prior'] = prior_loss
                total_loss += prior_loss
        
        losses['total'] = total_loss
        return losses
    
    def _backward_step(self, losses: Dict):
        """反向传播步骤"""
        # 清零梯度
        if self.prior_fusion is not None:
            self.prior_optimizer.zero_grad()
        
        for optimizer in self.gaussian_optimizers.values():
            optimizer.zero_grad()
        
        # 反向传播
        losses['total'].backward()
        
        # 梯度裁剪
        if self.config['optimization'].get('grad_clip', 0) > 0:
            if self.prior_fusion is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.prior_fusion.implicit_prior.parameters(),
                    self.config['optimization']['grad_clip']
                )
        
        # 更新参数
        if self.prior_fusion is not None:
            self.prior_optimizer.step()
        
        for optimizer in self.gaussian_optimizers.values():
            optimizer.step()
    
    @torch.no_grad()
    def validate(self) -> Dict:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.scene_gaussians.eval()
        
        val_metrics = {
            'loss': 0.0,
            'chamfer_distance': 0.0,
            'f_score': 0.0
        }
        
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch['images'].to(self.device)
            depths = batch.get('depths')
            
            if depths is not None:
                depths = depths.to(self.device)
            
            # 前向传播（不更新参数）
            # ... (简化)
            
            num_batches += 1
        
        # 平均
        for key in val_metrics:
            val_metrics[key] /= max(num_batches, 1)
        
        return val_metrics
    
    def train(self, num_epochs: int):
        """完整训练流程"""
        print("=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_losses = self.train_epoch()
            self.train_history.append(train_losses)
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            
            # 验证
            if self.val_loader is not None and epoch % self.config['logging']['val_interval'] == 0:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)
                
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                
                # 保存最佳模型
                if val_metrics['loss'] < self.best_metric:
                    self.best_metric = val_metrics['loss']
                    self.save_checkpoint('best_model.pth')
                    print("  ✓ New best model saved")
            
            # 定期保存
            if epoch % self.config['logging']['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # 保存历史
            self.save_history()
        
        print("\n" + "=" * 70)
        print("✓ Training completed!")
        print("=" * 70)
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_metric': self.best_metric
        }
        
        # 保存模型状态
        if self.prior_fusion is not None:
            checkpoint['implicit_prior'] = self.prior_fusion.implicit_prior.state_dict()
            checkpoint['prior_optimizer'] = self.prior_optimizer.state_dict()
        
        # 保存场景
        checkpoint['scene_gaussians'] = self.scene_gaussians.get_statistics()
        
        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        print(f"  Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        self.best_metric = checkpoint['best_metric']
        
        if 'implicit_prior' in checkpoint and self.prior_fusion is not None:
            self.prior_fusion.implicit_prior.load_state_dict(checkpoint['implicit_prior'])
        
        if 'prior_optimizer' in checkpoint and self.prior_fusion is not None:
            self.prior_optimizer.load_state_dict(checkpoint['prior_optimizer'])
        
        print(f"✓ Checkpoint loaded from: {checkpoint_path}")
        print(f"  Resuming from epoch {self.current_epoch}")
    
    def save_history(self):
        """保存训练历史"""
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)


class StaticSceneTrainer(UnifiedTrainer):
    """静态场景训练器（单图）"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_video = False


class DynamicSceneTrainer(UnifiedTrainer):
    """动态场景训练器（视频）"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_video = True
    
    def _forward_step(
        self,
        images: torch.Tensor,
        depths: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        batch: Dict
    ) -> Dict:
        """视频序列的前向传播"""
        # images: (B, T, 3, H, W)
        # 处理时序信息
        
        # ... (需要实现视频特有的逻辑)
        
        return super()._forward_step(images, depths, masks, batch)