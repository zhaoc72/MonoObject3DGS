"""
Flexible Reconstruction Pipeline
支持不同模式配置和消融实验
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import cv2
import yaml
import argparse
import time
from typing import Dict, Optional
from tqdm import tqdm

# 核心模块加载器
from src.core.module_loader import ModuleLoader

# 其他必要模块
from src.segmentation.semantic_classifier import SemanticClassifier
from src.depth.depth_refiner import DepthRefiner
from src.depth.scale_recovery import ScaleRecovery
from src.priors.explicit_prior import ExplicitShapePrior
from src.priors.implicit_prior import ImplicitShapePrior
from src.priors.prior_fusion import AdaptivePriorFusion, PriorConfig
from src.priors.regularizers import ShapePriorRegularizer
from src.reconstruction.object_gaussian import ObjectGaussian
from src.reconstruction.scene_gaussian import SceneGaussians
from src.reconstruction.gaussian_model import GaussianConfig
from src.optimization.losses import CompositeLoss
from src.utils.visualization import Visualizer


class FlexibleReconstructor:
    """灵活的3D重建器 - 支持多种配置"""
    
    def __init__(self, config_path: str):
        """初始化"""
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config.get('experiment', {}).get('device', 'cuda')
        self.output_dir = Path(self.config.get('experiment', {}).get('output_dir', 'experiments'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模块加载器
        self.module_loader = ModuleLoader(config_path)
        
        print("=" * 70)
        print("MonoObject3DGS - Flexible Reconstruction")
        print("=" * 70)
        
        # 加载模块
        self.modules = self.module_loader.load_all(self.device)
        
        # 初始化其他组件
        self._init_other_components()
        
        # 性能统计
        self.performance_stats = self.module_loader.get_performance_stats()
        self.timing_stats = {}
        
        print("=" * 70)
        print("✓ Flexible reconstructor initialized")
        self._print_config_summary()
        print("=" * 70)
    
    def _init_other_components(self):
        """初始化其他必要组件"""
        # 分类器
        if 'classification' in self.config:
            self.classifier = SemanticClassifier(
                method=self.config['classification'].get('method', 'clip'),
                model_name=self.config['classification'].get('model_name', 'openai/clip-vit-base-patch32'),
                categories=self.config['classification'].get('categories', []),
                device=self.device
            )
        else:
            self.classifier = None
        
        # 深度优化器（如果深度启用）
        if self.performance_stats['depth_enabled']:
            refine_config = self.config.get('depth', {}).get('refine', {})
            self.depth_refiner = DepthRefiner(
                bilateral_filter=refine_config.get('bilateral_filter', True),
                edge_preserving=refine_config.get('edge_preserving', True),
                remove_outliers=refine_config.get('remove_outliers', True)
            )
            
            self.scale_recovery = ScaleRecovery(
                method=self.config.get('depth', {}).get('scale_recovery', {}).get('method', 'shape_prior'),
                reference_objects=self.config.get('depth', {}).get('scale_recovery', {}).get('reference_objects', [])
            )
        else:
            self.depth_refiner = None
            self.scale_recovery = None
        
        # 形状先验
        if 'shape_prior' in self.config:
            self.explicit_prior = ExplicitShapePrior(
                template_dir=self.config['shape_prior']['explicit']['template_dir'],
                device=self.device
            )
            
            self.implicit_prior = ImplicitShapePrior(
                latent_dim=self.config['shape_prior']['implicit']['latent_dim'],
                encoder_hidden=self.config['shape_prior']['implicit']['encoder_layers'],
                decoder_hidden=self.config['shape_prior']['implicit']['decoder_layers'],
                num_output_points=self.config['shape_prior']['implicit']['num_output_points']
            ).to(self.device)
            
            for cat in self.config.get('classification', {}).get('categories', []):
                self.implicit_prior.add_category_prototype(cat)
            
            prior_config = PriorConfig(
                explicit_weight=self.config['shape_prior']['explicit']['init_weight'],
                implicit_weight=self.config['shape_prior']['implicit']['init_weight'],
                confidence_based=self.config['shape_prior']['adaptive']['confidence_based'],
                min_prior_weight=self.config['shape_prior']['adaptive']['min_prior_weight'],
                max_prior_weight=self.config['shape_prior']['adaptive']['max_prior_weight']
            )
            
            self.prior_fusion = AdaptivePriorFusion(
                self.explicit_prior,
                self.implicit_prior,
                prior_config
            )
        else:
            self.prior_fusion = None
        
        # 场景管理器
        if 'reconstruction' in self.config:
            self.scene_gaussians = SceneGaussians(
                config=GaussianConfig(
                    sh_degree=self.config['reconstruction']['gaussian']['sh_degree'],
                    init_scale=self.config['reconstruction']['gaussian']['init_scale'],
                    opacity_init=self.config['reconstruction']['gaussian']['opacity_init']
                )
            )
            
            self.regularizer = ShapePriorRegularizer(device=self.device)
        else:
            self.scene_gaussians = None
            self.regularizer = None
        
        # 可视化
        self.visualizer = Visualizer()
    
    def _print_config_summary(self):
        """打印配置摘要"""
        print(f"\n📋 Configuration Summary:")
        print(f"  Mode: {self.performance_stats['mode']}")
        print(f"  DINOv2: {'✓ Enabled' if self.performance_stats['dinov2_enabled'] else '✗ Disabled'}")
        print(f"  SAM 2: {'✓ Enabled' if self.performance_stats['sam2_enabled'] else '✗ Disabled'}")
        print(f"  Depth: {'✓ Enabled' if self.performance_stats['depth_enabled'] else '✗ Disabled'}")
        print(f"  Expected FPS: {self.performance_stats['expected_fps']}")
        print(f"  GPU Memory: {self.performance_stats['gpu_memory']}")
        print(f"  Accuracy Level: {self.performance_stats['accuracy_level']}")
    
    def reconstruct(self, image_path: str) -> Dict:
        """完整重建流程"""
        print("\n" + "=" * 70)
        print(f"Reconstructing: {image_path}")
        print("=" * 70)
        
        # 重置计时统计
        self.timing_stats = {}
        
        # Step 1: 加载图像
        t0 = time.time()
        print("\n[Step 1/6] Loading image...")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image_rgb.shape[:2]
        print(f"  Image size: {W}x{H}")
        self.timing_stats['load_image'] = time.time() - t0
        
        # Step 2: 语义分割
        t0 = time.time()
        print("\n[Step 2/6] Semantic segmentation...")
        objects = self._segment_and_classify(image_rgb)
        print(f"  Detected {len(objects)} objects")
        self.timing_stats['segmentation'] = time.time() - t0
        
        # Step 3: 深度估计
        t0 = time.time()
        print("\n[Step 3/6] Depth estimation...")
        depth_map, depth_confidence = self._estimate_depth(image_rgb, objects)
        print(f"  Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}] meters")
        self.timing_stats['depth_estimation'] = time.time() - t0
        
        # Step 4: 初始化
        t0 = time.time()
        print("\n[Step 4/6] Initializing 3D Gaussians...")
        camera_params = self._get_camera_params(W, H)
        self._initialize_objects(objects, depth_map, depth_confidence, image_rgb, camera_params)
        self.timing_stats['initialization'] = time.time() - t0
        
        # Step 5: 优化
        t0 = time.time()
        print("\n[Step 5/6] Optimization...")
        self._optimize_scene(image_rgb, depth_map, depth_confidence, camera_params)
        self.timing_stats['optimization'] = time.time() - t0
        
        # Step 6: 保存
        t0 = time.time()
        print("\n[Step 6/6] Saving results...")
        exp_name = Path(image_path).stem
        output_path = self.output_dir / f"{self.performance_stats['mode']}_{exp_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._save_results(output_path, image_rgb, depth_map, depth_confidence, objects)
        self.timing_stats['save_results'] = time.time() - t0
        
        # 生成报告
        report = self._generate_report(objects, depth_map, depth_confidence)
        
        # 打印统计
        self._print_statistics(report)
        
        return {
            'scene': self.scene_gaussians,
            'objects': objects,
            'depth': depth_map,
            'confidence': depth_confidence,
            'output_dir': str(output_path),
            'report': report,
            'timing': self.timing_stats,
            'config': self.performance_stats
        }
    
    def _segment_and_classify(self, image: np.ndarray) -> list:
        """语义感知分割"""
        H, W = image.shape[:2]
        
        # 1. 提取DINOv2特征
        print("  [1/4] Extracting features...")
        if self.performance_stats['dinov2_enabled']:
            dinov2 = self.modules['dinov2']
            
            # 检查是否使用多尺度
            use_multi_scale = self.config.get('dinov2', {}).get('multi_scale', False)
            
            if use_multi_scale:
                scales = self.config.get('dinov2', {}).get('scales', [1.0])
                multi_scale_features = dinov2.get_multi_scale_features(image, scales)
                # 融合多尺度特征
                dense_features = sum(multi_scale_features.values()) / len(multi_scale_features)
            else:
                dense_features = dinov2.get_dense_features(image, target_size=(H, W))
        else:
            print("    ⚠️  DINOv2 disabled - skipping feature extraction")
            dense_features = None
        
        # 2. SAM分割
        print("  [2/4] Running segmentation...")
        sam2 = self.modules['sam2']
        
        if self.performance_stats['sam2_enabled']:
            sam_config = self.config.get('sam2', {})
            masks = sam2.segment_automatic(
                image,
                points_per_side=sam_config.get('points_per_side', 32),
                pred_iou_thresh=sam_config.get('pred_iou_thresh', 0.88),
                stability_score_thresh=sam_config.get('stability_thresh', 0.95),
                min_mask_region_area=sam_config.get('min_mask_area', 500)
            )
            print(f"    Initial masks: {len(masks)}")
            
            # 3. 特征优化（如果DINOv2启用）
            if self.performance_stats['dinov2_enabled'] and \
               sam_config.get('use_feature_refinement', False) and \
               dense_features is not None:
                print("  [3/4] Refining with features...")
                masks = sam2.refine_with_features(
                    masks,
                    dense_features,
                    similarity_threshold=self.config.get('segmentation', {}).get('refine', {}).get('similarity_threshold', 0.7)
                )
                print(f"    Refined masks: {len(masks)}")
            else:
                print("  [3/4] Feature refinement skipped")
        else:
            print("    ⚠️  SAM 2 disabled - using fallback segmentation")
            masks = sam2.segment_automatic(image)
        
        # 4. 分类
        print("  [4/4] Classifying objects...")
        objects = []
        for i, mask_dict in enumerate(masks):
            x, y, w, h = mask_dict['bbox']
            
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = image[y1:y2, x1:x2]
            crop_mask = mask_dict['segmentation'][y1:y2, x1:x2]
            
            if self.classifier is not None:
                predictions = self.classifier.classify(
                    image_crop=crop,
                    mask=crop_mask,
                    top_k=3
                )
                category = predictions[0]['category']
                confidence = predictions[0]['score']
            else:
                category = 'object'
                confidence = 0.5
            
            objects.append({
                'id': i,
                'category': category,
                'confidence': confidence,
                'segmentation': mask_dict['segmentation'],
                'bbox': mask_dict['bbox'],
                'area': mask_dict['area'],
                'predicted_iou': mask_dict.get('predicted_iou', 0.0),
                'stability_score': mask_dict.get('stability_score', 0.0)
            })
        
        return objects
    
    def _estimate_depth(self, image: np.ndarray, objects: list) -> tuple:
        """深度估计"""
        depth_estimator = self.modules['depth']
        
        if self.performance_stats['depth_enabled']:
            # 检查是否使用多尺度
            use_multi_scale = self.config.get('depth', {}).get('multi_scale', False)
            estimate_confidence = self.config.get('depth', {}).get('estimate_confidence', False)
            
            if use_multi_scale:
                print("  Multi-scale depth estimation...")
                depth = depth_estimator.estimate_multi_scale(
                    image,
                    scales=self.config.get('depth', {}).get('scales', [1.0]),
                    fusion_method=self.config.get('depth', {}).get('fusion_method', 'weighted')
                )
                
                if estimate_confidence:
                    _, confidence = depth_estimator.estimate_with_confidence(image)
                else:
                    confidence = np.ones_like(depth) * 0.8
            else:
                if estimate_confidence:
                    print("  Estimating depth with confidence...")
                    depth, confidence = depth_estimator.estimate_with_confidence(image)
                else:
                    print("  Estimating depth...")
                    depth = depth_estimator.estimate(image)
                    confidence = np.ones_like(depth) * 0.8
            
            # 深度优化
            if self.depth_refiner is not None:
                print("  Refining depth...")
                depth = self.depth_refiner.refine(depth, image)
            
            # 尺度恢复
            if self.scale_recovery is not None and len(objects) > 0:
                print("  Recovering metric scale...")
                camera_params = self._get_camera_params(image.shape[1], image.shape[0])
                depth, scale_factor = self.scale_recovery.recover_scale(
                    depth, objects, camera_params
                )
        else:
            print("  ⚠️  Depth disabled - using fallback")
            depth = depth_estimator.estimate(image)
            confidence = np.ones_like(depth) * 0.3  # 低置信度
        
        return depth, confidence
    
    def _initialize_objects(
        self,
        objects: list,
        depth_map: np.ndarray,
        depth_confidence: np.ndarray,
        image: np.ndarray,
        camera_params: Dict
    ):
        """初始化物体"""
        if self.scene_gaussians is None:
            print("  ⚠️  Scene Gaussians not initialized - skipping")
            return
        
        for obj in tqdm(objects, desc="  Initializing"):
            obj_id = obj['id']
            category = obj['category']
            
            # 获取形状先验
            if self.explicit_prior is not None:
                shape_prior = self.explicit_prior.get_template(category)
            else:
                shape_prior = None
            
            obj_gaussian = ObjectGaussian(
                object_id=obj_id,
                category=category,
                config=GaussianConfig(
                    sh_degree=self.config['reconstruction']['gaussian']['sh_degree'],
                    init_scale=self.config['reconstruction']['gaussian']['init_scale'],
                    opacity_init=self.config['reconstruction']['gaussian']['opacity_init']
                ),
                shape_prior=shape_prior
            )
            
            try:
                obj_gaussian.initialize_from_mask_depth(
                    obj['segmentation'],
                    depth_map,
                    image,
                    camera_params
                )
                
                # 存储额外信息
                obj_gaussian.segmentation_confidence = obj['confidence']
                obj_gaussian.stability_score = obj.get('stability_score', 0.0)
                
                self.scene_gaussians.add_object(obj_id, obj_gaussian)
                
            except Exception as e:
                print(f"    Warning: Failed to initialize object {obj_id}: {e}")
                continue
    
    def _optimize_scene(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        depth_confidence: np.ndarray,
        camera_params: Dict
    ):
        """优化场景"""
        if self.scene_gaussians is None or len(self.scene_gaussians.objects) == 0:
            print("  ⚠️  No objects to optimize - skipping")
            return
        
        iterations = self.config['reconstruction']['optimization']['iterations']
        
        # 设置优化器
        optimizers = {}
        for obj_id, obj_gaussian in self.scene_gaussians.objects.items():
            param_groups = [
                {'params': [obj_gaussian._xyz], 
                 'lr': self.config['reconstruction']['gaussian']['position_lr'], 
                 'name': 'xyz'},
                {'params': [obj_gaussian._features_dc], 
                 'lr': self.config['reconstruction']['gaussian']['feature_lr'], 
                 'name': 'f_dc'},
                {'params': [obj_gaussian._features_rest], 
                 'lr': self.config['reconstruction']['gaussian']['feature_lr'] / 20.0, 
                 'name': 'f_rest'},
                {'params': [obj_gaussian._opacity], 
                 'lr': self.config['reconstruction']['gaussian']['opacity_lr'], 
                 'name': 'opacity'},
                {'params': [obj_gaussian._scaling], 
                 'lr': self.config['reconstruction']['gaussian']['scaling_lr'], 
                 'name': 'scaling'},
                {'params': [obj_gaussian._rotation], 
                 'lr': self.config['reconstruction']['gaussian']['rotation_lr'], 
                 'name': 'rotation'}
            ]
            optimizers[obj_id] = torch.optim.Adam(param_groups)
            obj_gaussian.optimizer = optimizers[obj_id]
        
        # 损失函数
        loss_config = self.config['reconstruction']['loss']
        loss_fn = CompositeLoss(
            lambda_photometric=loss_config['photometric'],
            lambda_depth=loss_config['depth_consistency'],
            lambda_shape_prior=loss_config['shape_prior'],
            lambda_semantic=loss_config['semantic_consistency'],
            lambda_smoothness=loss_config['smoothness'],
            lambda_symmetry=loss_config['symmetry'],
            use_lpips=False
        ).to(self.device)
        
        pbar = tqdm(range(iterations), desc="  Optimizing")
        
        viewing_coverage = self.config['shape_prior']['adaptive']['viewing_coverage']
        seg_confidence = self.config['shape_prior']['adaptive']['segmentation_confidence']
        recon_uncertainty = self.config['shape_prior']['adaptive']['reconstruction_uncertainty']
        
        for iter_num in pbar:
            total_loss = 0.0
            
            for obj_id, obj_gaussian in self.scene_gaussians.objects.items():
                optimizer = optimizers[obj_id]
                optimizer.zero_grad()
                
                # 形状先验损失
                if self.prior_fusion is not None:
                    obj_seg_conf = getattr(obj_gaussian, 'segmentation_confidence', seg_confidence)
                    
                    _, weights_info = self.prior_fusion.fuse_priors(
                        obj_gaussian.get_xyz,
                        obj_gaussian.category,
                        viewing_coverage,
                        obj_seg_conf,
                        recon_uncertainty
                    )
                    
                    shape_prior_loss = self.prior_fusion.compute_fused_prior_loss(
                        obj_gaussian.get_xyz,
                        obj_gaussian.category,
                        weights_info
                    )
                else:
                    shape_prior_loss = torch.tensor(0.0, device=self.device)
                
                # 正则化损失
                if self.regularizer is not None:
                    reg_losses = self.regularizer.compute_combined_regularization(
                        obj_gaussian.get_xyz,
                        weights={
                            'smoothness': loss_config['smoothness'],
                            'compactness': 0.02,
                            'symmetry': loss_config['symmetry']
                        }
                    )
                    reg_loss = reg_losses['total']
                else:
                    reg_loss = torch.tensor(0.0, device=self.device)
                
                loss = shape_prior_loss + reg_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # 致密化
                if self._should_densify(iter_num):
                    try:
                        obj_gaussian.densify_and_prune(
                            max_grad=self.config['reconstruction']['optimization']['densify_grad_threshold'],
                            min_opacity=0.005,
                            extent=5.0,
                            max_screen_size=20
                        )
                    except:
                        pass
                
                # 重置不透明度
                if self._should_reset_opacity(iter_num):
                    obj_gaussian.reset_opacity()
            
            # 更新进度条
            if iter_num % 10 == 0:
                avg_loss = total_loss / max(1, len(self.scene_gaussians.objects))
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    def _should_densify(self, iter_num: int) -> bool:
        """是否致密化"""
        config = self.config['reconstruction']['optimization']
        return (
            config['densify_from_iter'] <= iter_num <= config['densify_until_iter']
            and iter_num % config['densify_interval'] == 0
        )
    
    def _should_reset_opacity(self, iter_num: int) -> bool:
        """是否重置不透明度"""
        interval = self.config['reconstruction']['optimization']['opacity_reset_interval']
        return iter_num % interval == 0 and iter_num > 0
    
    def _get_camera_params(self, width: int, height: int) -> Dict:
        """获取相机参数"""
        fov = self.config['camera']['fov']
        fx = fy = width / (2 * np.tan(np.radians(fov / 2)))
        
        return {
            'fx': fx,
            'fy': fy,
            'cx': width / 2,
            'cy': height / 2,
            'width': width,
            'height': height
        }
    
    def _save_results(
        self,
        output_path: Path,
        image: np.ndarray,
        depth_map: np.ndarray,
        depth_confidence: np.ndarray,
        objects: list
    ):
        """保存结果"""
        if self.scene_gaussians is not None:
            self.scene_gaussians.save(str(output_path / 'scene'))
        
        # 深度
        depth_vis = self.visualizer.visualize_depth(depth_map)
        cv2.imwrite(str(output_path / 'depth.png'), depth_vis)
        
        conf_vis = (depth_confidence * 255).astype(np.uint8)
        conf_colored = cv2.applyColorMap(conf_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_path / 'depth_confidence.png'), conf_colored)
        
        # 分割
        seg_vis = self.visualizer.visualize_masks(image, objects)
        cv2.imwrite(str(output_path / 'segmentation.png'), 
                   cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR))
        
        # 统计
        if self.scene_gaussians is not None:
            stats = self.scene_gaussians.get_statistics()
        else:
            stats = {'num_objects': len(objects), 'total_gaussians': 0}
        
        # 添加配置和计时信息
        stats['config'] = self.performance_stats
        stats['timing'] = self.timing_stats
        stats['total_time'] = sum(self.timing_stats.values())
        
        import json
        with open(output_path / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, 
                     default=lambda x: float(x) if isinstance(x, (np.float32, np.float64, np.int64)) else x)
        
        print(f"  ✓ Results saved to: {output_path}")
    
    def _generate_report(
        self,
        objects: list,
        depth_map: np.ndarray,
        depth_confidence: np.ndarray
    ) -> Dict:
        """生成报告"""
        report = {
            'mode': self.performance_stats['mode'],
            'config': self.performance_stats,
            'timing': self.timing_stats,
            'total_time': sum(self.timing_stats.values()),
            'summary': {
                'total_objects': len(objects),
                'total_gaussians': self.scene_gaussians.total_gaussians if self.scene_gaussians else 0
            },
            'quality_metrics': {
                'avg_segmentation_confidence': float(np.mean([obj['confidence'] for obj in objects])) if objects else 0.0,
                'avg_depth_confidence': float(depth_confidence.mean()),
                'depth_range': [float(depth_map.min()), float(depth_map.max())]
            }
        }
        
        return report
    
    def _print_statistics(self, report: Dict):
        """打印统计信息"""
        print("\n" + "=" * 70)
        print("📊 Reconstruction Report")
        print("=" * 70)
        
        print(f"\n⚙️  Mode: {report['mode']}")
        print(f"\n⏱️  Timing:")
        for key, value in report['timing'].items():
            print(f"  {key}: {value:.3f}s")
        print(f"  Total: {report['total_time']:.3f}s")
        
        print(f"\n📈 Summary:")
        print(f"  Objects: {report['summary']['total_objects']}")
        print(f"  Gaussians: {report['summary']['total_gaussians']}")
        
        print(f"\n🎯 Quality:")
        print(f"  Seg Confidence: {report['quality_metrics']['avg_segmentation_confidence']:.3f}")
        print(f"  Depth Confidence: {report['quality_metrics']['avg_depth_confidence']:.3f}")
        print(f"  Depth Range: {report['quality_metrics']['depth_range']}")
        
        print("=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Flexible 3D Reconstruction')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['high_accuracy', 'real_time', 'balanced', 
                               'ablation_no_dinov2', 'ablation_no_depth', 'ablation_minimal'],
                       help='Reconstruction mode')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # 配置文件路径
    config_path = f'configs/modes/{args.mode}.yaml'
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # 初始化重建器
    reconstructor = FlexibleReconstructor(config_path)
    
    if args.output:
        reconstructor.output_dir = Path(args.output)
    
    # 执行重建
    result = reconstructor.reconstruct(args.image)
    
    print(f"\n✓ Done! Check results in: {result['output_dir']}")


if __name__ == "__main__":
    main()