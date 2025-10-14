"""
Training Script for MonoObject3DGS
单目物体级3D重建训练脚本
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
from datetime import datetime
import wandb
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.segmentation.dinov2_extractor import DINOv2Extractor
from src.segmentation.sam2_segmenter import SAMSegmenter
from src.reconstruction.object_gaussian import ObjectGaussian, SceneGaussians, GaussianConfig
from src.priors.prior_fusion import (
    ExplicitShapePrior, ImplicitShapePrior,
    AdaptivePriorFusion, PriorConfig, ShapePriorRegularizer
)
from src.optimization.losses import CompositeLoss
from src.utils.camera import Camera
from src.utils.visualization import visualize_reconstruction


class Trainer:
    """训练器"""
    
    def __init__(self, config_path: str):
        """初始化训练器"""
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = self.config['experiment']['device']
        self.output_dir = Path(self.config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        seed = self.config['experiment']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 初始化日志
        self.setup_logging()
        
        # 初始化模型
        self.setup_models()
        
        # 初始化损失函数
        self.setup_losses()
        
        # 初始化优化器
        self.setup_optimizers()
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        
        print("=" * 50)
        print("✓ 训练器初始化完成")
        print("=" * 50)
        
    def setup_logging(self):
        """设置日志"""
        exp_name = self.config['experiment']['name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.output_dir / f"{exp_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        if self.config['logging']['use_tensorboard']:
            self.tb_writer = SummaryWriter(self.exp_dir / "tensorboard")
            
        # Weights & Biases
        if self.config['logging']['use_wandb']:
            wandb.init(
                project="MonoObject3DGS",
                name=f"{exp_name}_{timestamp}",
                config=self.config
            )
            
        print(f"✓ 实验目录: {self.exp_dir}")
        
    def setup_models(self):
        """初始化模型"""
        # 1. DINOv2特征提取器
        self.dinov2 = DINOv2Extractor(
            model_name=self.config['segmentation']['dinov2']['checkpoint'],
            feature_dim=self.config['segmentation']['dinov2']['feature_dim'],
            device=self.device
        )
        
        # 2. SAM分割器
        self.sam = SAMSegmenter(
            model_type=self.config['segmentation']['sam']['model_type'],
            checkpoint=self.config['segmentation']['sam']['checkpoint'],
            device=self.device
        )
        
        # 3. 形状先验
        self.explicit_prior = ExplicitShapePrior(
            template_dir=self.config['shape_prior']['explicit']['template_dir']
        )
        
        self.implicit_prior = ImplicitShapePrior(
            latent_dim=self.config['shape_prior']['implicit']['latent_dim'],
            encoder_layers=self.config['shape_prior']['implicit']['encoder_layers'],
            decoder_layers=self.config['shape_prior']['implicit']['decoder_layers']
        ).to(self.device)
        
        # 添加类别原型
        for category in self.config['shape_prior']['explicit']['categories']:
            self.implicit_prior.add_category_prototype(category)
            
        # 4. 先验融合器
        prior_config = PriorConfig(
            explicit_weight=self.config['shape_prior']['explicit']['init_weight'],
            implicit_weight=self.config['shape_prior']['implicit'].get('init_weight', 0.3)
        )
        self.prior_fusion = AdaptivePriorFusion(
            self.explicit_prior,
            self.implicit_prior,
            prior_config
        )
        
        # 5. 场景Gaussians
        self.scene_gaussians = SceneGaussians()
        
        # 6. 正则化器
        self.regularizer = ShapePriorRegularizer()
        
    def setup_losses(self):
        """初始化损失函数"""
        loss_config = self.config['optimization']['loss']
        
        self.loss_fn = CompositeLoss(
            lambda_photometric=loss_config['photometric'],
            lambda_depth=loss_config['depth_consistency'],
            lambda_shape_prior=loss_config['shape_prior'],
            lambda_semantic=loss_config['semantic_consistency'],
            lambda_smoothness=loss_config['smoothness'],
            use_lpips=True
        ).to(self.device)
        
    def setup_optimizers(self):
        """初始化优化器"""
        # Gaussian参数优化器（每个物体独立）
        self.gaussian_optimizers = {}
        
        # 隐式先验优化器
        self.prior_optimizer = torch.optim.Adam(
            self.implicit_prior.parameters(),
            lr=self.config['optimization']['learning_rate']
        )
        
    def add_object_optimizer(self, object_id: int, object_gaussian: ObjectGaussian):
        """为新物体添加优化器"""
        config = self.config['reconstruction']['gaussian']
        
        param_groups = [
            {'params': [object_gaussian._xyz], 'lr': config['position_lr'], 'name': 'xyz'},
            {'params': [object_gaussian._features_dc], 'lr': config['feature_lr'], 'name': 'f_dc'},
            {'params': [object_gaussian._features_rest], 'lr': config['feature_lr'] / 20.0, 'name': 'f_rest'},
            {'params': [object_gaussian._opacity], 'lr': config['opacity_lr'], 'name': 'opacity'},
            {'params': [object_gaussian._scaling], 'lr': config['scaling_lr'], 'name': 'scaling'},
            {'params': [object_gaussian._rotation], 'lr': config['rotation_lr'], 'name': 'rotation'}
        ]
        
        self.gaussian_optimizers[object_id] = torch.optim.Adam(param_groups)
        
    def process_frame(self, image: np.ndarray, depth: np.ndarray, camera_params: dict):
        """
        处理单帧：分割、匹配、重建
        
        Args:
            image: RGB图像
            depth: 深度图
            camera_params: 相机参数
        """
        # 1. 特征提取
        features = self.dinov2.extract_features(image)
        feature_map = self.dinov2.get_dense_features(
            image,
            target_size=(image.shape[0], image.shape[1])
        )
        
        # 2. 分割
        masks = self.sam.segment_automatic(image)
        
        # 3. 特征优化分割
        refined_masks = self.sam.refine_with_features(
            masks,
            feature_map,
            similarity_threshold=0.7
        )
        
        # 4. 语义聚类（识别类别）
        objects_data = []
        for i, mask_dict in enumerate(refined_masks):
            mask = mask_dict['segmentation']
            
            # 简单的类别识别（基于大小和位置）
            # 实际应用中可以使用分类器
            category = self.classify_object(mask, features)
            
            objects_data.append({
                'id': i,
                'category': category,
                'mask': mask,
                'bbox': mask_dict['bbox']
            })
            
        return objects_data
    
    def classify_object(self, mask: np.ndarray, features: dict) -> str:
        """简单的物体分类（实际应该用分类器）"""
        # 基于mask大小和形状的简单分类
        area = mask.sum()
        aspect_ratio = mask.shape[0] / mask.shape[1]
        
        # 简化版本 - 实际应该用CLIP或其他分类器
        if area > 50000:
            return 'table'
        elif area > 20000:
            return 'chair'
        else:
            return 'object'
    
    def initialize_object(
        self,
        object_data: dict,
        depth_map: np.ndarray,
        camera_params: dict
    ) -> ObjectGaussian:
        """初始化物体的Gaussian"""
        category = object_data['category']
        
        # 获取形状先验
        shape_prior = None
        if category in self.config['shape_prior']['explicit']['categories']:
            template = self.explicit_prior.get_template(category)
            if template is not None:
                shape_prior = template.to(self.device)
                
        # 创建Gaussian配置
        gaussian_config = GaussianConfig(
            sh_degree=self.config['reconstruction']['gaussian']['sh_degree'],
            init_scale=self.config['reconstruction']['gaussian']['init_scale'],
            opacity_init=self.config['reconstruction']['gaussian']['opacity_init']
        )
        
        # 初始化物体Gaussian
        object_gaussian = ObjectGaussian(
            object_id=object_data['id'],
            category=category,
            mask=object_data['mask'],
            depth_map=depth_map,
            camera_params=camera_params,
            config=gaussian_config,
            shape_prior=shape_prior
        ).to(self.device)
        
        return object_gaussian
    
    def train_step(self, batch_data: dict):
        """单步训练"""
        images = batch_data['images'].to(self.device)
        depths = batch_data['depths'].to(self.device)
        masks = batch_data['masks'].to(self.device)
        camera_params = batch_data['camera_params']
        
        B = images.shape[0]
        total_losses = {}
        
        # 对每个物体进行优化
        for obj_id, obj_gaussian in self.scene_gaussians.objects.items():
            obj_id_int = int(obj_id)
            
            if obj_id_int not in self.gaussian_optimizers:
                self.add_object_optimizer(obj_id_int, obj_gaussian)
                
            optimizer = self.gaussian_optimizers[obj_id_int]
            optimizer.zero_grad()
            
            # 渲染物体（这里需要实现渲染器）
            rendered = self.render_object(obj_gaussian, camera_params)
            
            # 获取物体的mask
            obj_mask = masks[:, obj_id_int:obj_id_int+1]
            
            # 计算形状先验损失
            viewing_coverage = self.compute_viewing_coverage(camera_params)
            seg_confidence = 0.8  # 从SAM获取
            recon_uncertainty = self.compute_reconstruction_uncertainty()
            
            _, weights_info = self.prior_fusion.fuse_priors(
                obj_gaussian.get_xyz,
                obj_gaussian.category,
                viewing_coverage,
                seg_confidence,
                recon_uncertainty
            )
            
            shape_prior_loss = self.prior_fusion.compute_fused_prior_loss(
                obj_gaussian.get_xyz,
                obj_gaussian.category,
                weights_info
            )
            
            # 计算总损失
            losses = self.loss_fn(
                rendered_image=rendered['rgb'],
                gt_image=images,
                rendered_depth=rendered.get('depth'),
                gt_depth=depths,
                mask=obj_mask,
                gaussians=obj_gaussian.get_xyz,
                shape_prior_loss=shape_prior_loss
            )
            
            # 反向传播
            losses['total'].backward()
            optimizer.step()
            
            # 记录损失
            for k, v in losses.items():
                key = f"obj_{obj_id}/{k}"
                total_losses[key] = v.item()
                
            # 定期致密化和修剪
            if self.should_densify():
                obj_gaussian.densify_and_prune(
                    max_grad=self.config['reconstruction']['per_object']['densify_grad_threshold'],
                    min_opacity=0.005,
                    extent=5.0,
                    max_screen_size=20
                )
                
            # 定期重置不透明度
            if self.should_reset_opacity():
                obj_gaussian.reset_opacity()
                
        # 更新隐式先验
        if self.global_step % 10 == 0:
            self.prior_optimizer.zero_grad()
            prior_loss = self.update_implicit_prior()
            if prior_loss is not None:
                prior_loss.backward()
                self.prior_optimizer.step()
                total_losses['implicit_prior'] = prior_loss.item()
                
        return total_losses
    
    def render_object(self, object_gaussian: ObjectGaussian, camera_params: dict):
        """
        渲染物体（简化版本 - 实际需要使用diff-gaussian-rasterization）
        """
        # 这里应该使用3DGS渲染器
        # 返回渲染的RGB和深度
        # 实际实现需要调用diff_gaussian_rasterization库
        
        # 占位符
        rendered = {
            'rgb': torch.rand(1, 3, 512, 512, device=self.device),
            'depth': torch.rand(1, 1, 512, 512, device=self.device)
        }
        return rendered
    
    def compute_viewing_coverage(self, camera_params: dict) -> float:
        """计算视角覆盖度"""
        # 简化实现
        return 0.5
    
    def compute_reconstruction_uncertainty(self) -> float:
        """计算重建不确定性"""
        # 简化实现
        return 0.3
    
    def should_densify(self) -> bool:
        """是否应该致密化"""
        config = self.config['optimization']
        return (
            config['densify_from_iter'] <= self.global_step <= config['densify_until_iter']
            and self.global_step % config['densify_interval'] == 0
        )
    
    def should_reset_opacity(self) -> bool:
        """是否应该重置不透明度"""
        config = self.config['optimization']
        return self.global_step % config['opacity_reset_interval'] == 0
    
    def update_implicit_prior(self):
        """更新隐式先验"""
        # 收集所有物体的点云
        pointclouds = []
        categories = []
        
        for obj_gaussian in self.scene_gaussians.objects.values():
            pointclouds.append(obj_gaussian.get_xyz)
            categories.append(obj_gaussian.category)
            
        if not pointclouds:
            return None
            
        # 计算重建损失
        total_loss = 0
        for pc, cat in zip(pointclouds, categories):
            loss = self.implicit_prior.compute_prior_loss(pc, cat)
            total_loss += loss
            
        return total_loss / len(pointclouds)
    
    def train(self, dataloader):
        """主训练循环"""
        num_iterations = self.config['optimization']['iterations']
        log_interval = self.config['logging']['log_interval']
        save_interval = self.config['logging']['save_interval']
        
        print(f"\n开始训练，共 {num_iterations} 次迭代")
        print("=" * 50)
        
        pbar = tqdm(total=num_iterations, desc="Training")
        
        while self.global_step < num_iterations:
            for batch_data in dataloader:
                # 训练步骤
                losses = self.train_step(batch_data)
                
                # 日志
                if self.global_step % log_interval == 0:
                    self.log_losses(losses)
                    
                # 保存检查点
                if self.global_step % save_interval == 0:
                    self.save_checkpoint()
                    
                # 可视化
                if self.global_step % self.config['logging']['vis_interval'] == 0:
                    self.visualize_results()
                    
                self.global_step += 1
                pbar.update(1)
                
                if self.global_step >= num_iterations:
                    break
                    
        pbar.close()
        
        # 保存最终模型
        self.save_final_model()
        
        print("\n训练完成！")
        print("=" * 50)
        
    def log_losses(self, losses: dict):
        """记录损失"""
        # TensorBoard
        if hasattr(self, 'tb_writer'):
            for k, v in losses.items():
                self.tb_writer.add_scalar(f"Loss/{k}", v, self.global_step)
                
        # WandB
        if self.config['logging']['use_wandb']:
            wandb.log({f"loss/{k}": v for k, v in losses.items()}, step=self.global_step)
            
        # 控制台
        total_loss = losses.get('total', 0.0)
        tqdm.write(f"Step {self.global_step}: Loss = {total_loss:.4f}")
        
    def save_checkpoint(self):
        """保存检查点"""
        ckpt_dir = self.exp_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'implicit_prior': self.implicit_prior.state_dict(),
            'config': self.config
        }
        
        # 保存每个物体的Gaussian
        for obj_id, obj_gaussian in self.scene_gaussians.objects.items():
            checkpoint[f'object_{obj_id}'] = {
                'gaussian_state': {
                    'xyz': obj_gaussian._xyz,
                    'features_dc': obj_gaussian._features_dc,
                    'features_rest': obj_gaussian._features_rest,
                    'opacity': obj_gaussian._opacity,
                    'scaling': obj_gaussian._scaling,
                    'rotation': obj_gaussian._rotation
                },
                'category': obj_gaussian.category
            }
            
        save_path = ckpt_dir / f"checkpoint_{self.global_step}.pth"
        torch.save(checkpoint, save_path)
        print(f"✓ 保存检查点: {save_path}")
        
    def save_final_model(self):
        """保存最终模型"""
        output_dir = self.exp_dir / "final_model"
        self.scene_gaussians.save_scene(str(output_dir))
        print(f"✓ 保存最终模型: {output_dir}")
        
    def visualize_results(self):
        """可视化结果"""
        # 渲染novel view
        # 保存图像
        pass


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='配置文件路径')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录')
    args = parser.parse_args()
    
    # 初始化训练器
    trainer = Trainer(args.config)
    
    # 加载数据（这里需要实现DataLoader）
    # dataloader = create_dataloader(args.data_dir, trainer.config)
    
    # 由于没有实际数据，这里创建dummy dataloader用于演示
    class DummyDataset:
        def __iter__(self):
            while True:
                yield {
                    'images': torch.rand(1, 3, 512, 512),
                    'depths': torch.rand(1, 1, 512, 512),
                    'masks': torch.rand(1, 5, 512, 512),
                    'camera_params': {
                        'fx': 525.0, 'fy': 525.0,
                        'cx': 256.0, 'cy': 256.0
                    }
                }
    
    dataloader = DummyDataset()
    
    # 开始训练
    trainer.train(dataloader)


if __name__ == "__main__":
    main()