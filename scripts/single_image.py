"""
Single Image V2 - 使用最新SOTA模型
DINOv2 Large + SAM 2 + Depth Anything V2
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import cv2
import yaml
from tqdm import tqdm
from typing import Dict, Optional
import argparse

# V2模块
from src.segmentation.dinov2_extractor_v2 import DINOv2ExtractorV2
from src.segmentation.sam2_segmenter import SAM2Segmenter
from src.segmentation.semantic_classifier import SemanticClassifier
from src.depth.depth_anything_v2 import DepthAnythingV2
from src.depth.depth_refiner import DepthRefiner
from src.depth.scale_recovery import ScaleRecovery

# 保持原有模块
from src.priors.explicit_prior import ExplicitShapePrior
from src.priors.implicit_prior import ImplicitShapePrior
from src.priors.prior_fusion import AdaptivePriorFusion, PriorConfig
from src.priors.regularizers import ShapePriorRegularizer
from src.reconstruction.object_gaussian import ObjectGaussian
from src.reconstruction.scene_gaussian import SceneGaussians
from src.reconstruction.gaussian_model import GaussianConfig
from src.optimization.losses import CompositeLoss
from src.utils.visualization import Visualizer


class SingleImageReconstructorV2:
    """单张图片3D重建器 - V2升级版"""
    
    def __init__(self, config_path: str):
        """初始化"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['experiment']['device']
        self.output_dir = Path(self.config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("MonoObject3DGS V2 - Single Image Reconstruction")
        print("🚀 Powered by DINOv2 Large + SAM 2 + Depth Anything V2")
        print("=" * 70)
        
        self._init_modules()
        
        print("=" * 70)
        print("✓ All V2 modules initialized successfully")
        print("=" * 70)
    
    def _init_modules(self):
        """初始化所有模块"""
        print("\n[1/8] Loading DINOv2 V2 (Large model)...")
        self.dinov2 = DINOv2ExtractorV2(
            model_name=self.config['segmentation']['dinov2']['model_name'],
            feature_dim=self.config['segmentation']['dinov2']['feature_dim'],
            device=self.device,
            use_registers=self.config['segmentation']['dinov2'].get('use_registers', True),
            enable_xformers=self.config['segmentation']['dinov2'].get('enable_xformers', True)
        )
        
        print("\n[2/8] Loading SAM 2 (Image mode)...")
        self.segmenter = SAM2Segmenter(
            model_size=self.config['segmentation']['sam2']['model_size'],
            checkpoint=self.config['segmentation']['sam2']['checkpoint'],
            device=self.device,
            mode="image"
        )
        
        print("\n[3/8] Loading semantic classifier...")
        self.classifier = SemanticClassifier(
            method=self.config['classification']['method'],
            model_name=self.config['classification'].get('model_name', 'openai/clip-vit-large-patch14'),
            categories=self.config['classification']['categories'],
            device=self.device
        )
        
        print("\n[4/8] Loading Depth Anything V2...")
        self.depth_estimator = DepthAnythingV2(
            model_size=self.config['depth']['model_size'],
            metric_depth=self.config['depth'].get('metric_depth', True),
            device=self.device,
            max_depth=self.config['depth'].get('max_depth', 20.0)
        )
        
        print("\n[5/8] Initializing depth refiner...")
        self.depth_refiner = DepthRefiner(
            bilateral_filter=self.config['depth']['refine']['bilateral_filter'],
            edge_preserving=self.config['depth']['refine']['edge_preserving'],
            remove_outliers=self.config['depth']['refine']['remove_outliers']
        )
        
        print("\n[6/8] Loading scale recovery...")
        self.scale_recovery = ScaleRecovery(
            method=self.config['depth']['scale_recovery']['method'],
            reference_objects=self.config['depth']['scale_recovery']['reference_objects']
        )
        
        print("\n[7/8] Loading shape priors...")
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
        
        for category in self.config['classification']['categories']:
            self.implicit_prior.add_category_prototype(category)
        
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
        
        print("\n[8/8] Initializing scene manager...")
        self.scene_gaussians = SceneGaussians(
            config=GaussianConfig(
                sh_degree=self.config['reconstruction']['gaussian']['sh_degree'],
                init_scale=self.config['reconstruction']['gaussian']['init_scale'],
                opacity_init=self.config['reconstruction']['gaussian']['opacity_init']
            )
        )
        
        self.regularizer = ShapePriorRegularizer(device=self.device)
        self.visualizer = Visualizer()
    
    def reconstruct(self, image_path: str) -> Dict:
        """完整重建流程 - V2"""
        print("\n" + "=" * 70)
        print(f"Reconstructing: {image_path}")
        print("=" * 70)
        
        # Step 1: 加载图像
        print("\n[Step 1/7] Loading image...")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image_rgb.shape[:2]
        print(f"  Image size: {W}x{H}")
        
        # Step 2: 增强语义分割（V2特性）
        print("\n[Step 2/7] Enhanced semantic segmentation (V2)...")
        objects = self._segment_and_classify_v2(image_rgb)
        print(f"  Detected {len(objects)} objects:")
        for obj in objects:
            print(f"    - {obj['category']} (conf: {obj['confidence']:.3f})")
        
        # Step 3: 增强深度估计（V2特性）
        print("\n[Step 3/7] Enhanced depth estimation (V2)...")
        depth_map, depth_confidence = self._estimate_depth_v2(image_rgb, objects)
        print(f"  Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}] meters")
        print(f"  Confidence: [{depth_confidence.min():.2f}, {depth_confidence.max():.2f}]")
        
        # Step 4: 物体级初始化
        print("\n[Step 4/7] Object-level 3D Gaussian initialization...")
        camera_params = self._get_camera_params(W, H)
        self._initialize_objects(objects, depth_map, depth_confidence, image_rgb, camera_params)
        
        # Step 5: 形状先验约束优化（V2增强）
        print("\n[Step 5/7] Shape prior constrained optimization (V2)...")
        self._optimize_with_priors_v2(image_rgb, depth_map, depth_confidence, camera_params)
        
        # Step 6: 保存结果
        print("\n[Step 6/7] Saving results...")
        exp_name = Path(image_path).stem
        output_path = self.output_dir / f"single_image_v2_{exp_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._save_results(output_path, image_rgb, depth_map, depth_confidence, objects)
        
        # Step 7: 生成报告
        print("\n[Step 7/7] Generating report...")
        report = self._generate_report(objects, depth_map, depth_confidence)
        
        print("\n" + "=" * 70)
        print("✓ V2 Reconstruction completed!")
        print(f"  Output directory: {output_path}")
        print(f"  Total objects: {len(self.scene_gaussians.objects)}")
        print(f"  Total Gaussians: {self.scene_gaussians.total_gaussians}")
        print(f"  Average depth confidence: {depth_confidence.mean():.3f}")
        print("=" * 70)
        
        return {
            'scene': self.scene_gaussians,
            'objects': objects,
            'depth': depth_map,
            'confidence': depth_confidence,
            'output_dir': str(output_path),
            'report': report
        }
    
    def _segment_and_classify_v2(self, image: np.ndarray) -> list:
        """增强版语义感知分割 - V2"""
        H, W = image.shape[:2]
        
        # 1. 提取DINOv2 V2特征
        print("  [1/5] Extracting DINOv2 V2 features (large model)...")
        features = self.dinov2.extract_features(image, return_registers=True)
        
        # 多尺度特征（V2新特性）
        if self.config['segmentation']['dinov2'].get('multi_scale', False):
            print("  [2/5] Extracting multi-scale features...")
            multi_scale_features = self.dinov2.get_multi_scale_features(
                image, 
                scales=self.config['segmentation']['dinov2'].get('scales', [1.0, 0.75, 0.5])
            )
            # 融合多尺度特征
            dense_features = sum(multi_scale_features.values()) / len(multi_scale_features)
        else:
            dense_features = self.dinov2.get_dense_features(image, target_size=(H, W))
        
        # 2. SAM 2自动分割
        print("  [3/5] Running SAM 2 segmentation...")
        masks = self.segmenter.segment_automatic(
            image,
            points_per_side=self.config['segmentation']['sam2'].get('points_per_side', 32),
            pred_iou_thresh=self.config['segmentation']['sam2'].get('pred_iou_thresh', 0.88),
            stability_score_thresh=self.config['segmentation']['sam2'].get('stability_thresh', 0.95),
            min_mask_region_area=self.config['segmentation']['sam2'].get('min_mask_area', 500)
        )
        print(f"    Initial masks: {len(masks)}")
        
        # 3. 特征优化分割（V2增强）
        if self.config['segmentation']['refine']['use_features']:
            print("  [4/5] Refining with DINOv2 features...")
            masks = self.segmenter.refine_with_features(
                masks,
                dense_features,
                similarity_threshold=self.config['segmentation']['refine']['similarity_threshold']
            )
            print(f"    Refined masks: {len(masks)}")
            
            # 多尺度优化（V2新特性）
            if self.config['segmentation']['refine'].get('multi_scale_refine', False):
                print("    Multi-scale refinement...")
                for scale, scale_features in multi_scale_features.items():
                    if scale != 1.0:
                        masks = self.segmenter.refine_with_features(
                            masks, scale_features,
                            similarity_threshold=self.config['segmentation']['refine']['similarity_threshold'] * 0.95
                        )
        
        # 4. 合并相似mask
        print("  [5/5] Merging similar masks...")
        masks = self.segmenter.merge_similar_masks(
            masks,
            iou_threshold=self.config['segmentation']['refine']['merge_iou_threshold']
        )
        print(f"    Final masks: {len(masks)}")
        
        # 5. 分类物体
        print("  Classifying objects with CLIP...")
        objects = []
        for i, mask_dict in enumerate(masks):
            x, y, w, h = mask_dict['bbox']
            
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = image[y1:y2, x1:x2]
            crop_mask = mask_dict['segmentation'][y1:y2, x1:x2]
            
            predictions = self.classifier.classify(
                image_crop=crop,
                mask=crop_mask,
                top_k=3
            )
            
            objects.append({
                'id': i,
                'category': predictions[0]['category'],
                'confidence': predictions[0]['score'],
                'predictions': predictions,  # V2: 保存所有预测
                'segmentation': mask_dict['segmentation'],
                'bbox': mask_dict['bbox'],
                'area': mask_dict['area'],
                'predicted_iou': mask_dict.get('predicted_iou', 0.0),
                'stability_score': mask_dict.get('stability_score', 0.0)
            })
        
        return objects
    
    def _estimate_depth_v2(
        self, 
        image: np.ndarray, 
        objects: list
    ) -> tuple:
        """增强版深度估计 - V2"""
        # 1. 多尺度深度估计（V2特性）
        if self.config['depth'].get('multi_scale', {}).get('enabled', False):
            print("  [1/4] Multi-scale depth estimation...")
            depth = self.depth_estimator.estimate_multi_scale(
                image,
                scales=self.config['depth']['multi_scale'].get('scales', [1.0, 0.75, 0.5]),
                fusion_method=self.config['depth']['multi_scale'].get('fusion_method', 'weighted')
            )
            
            # 获取置信度
            _, confidence = self.depth_estimator.estimate_with_confidence(image)
        else:
            print("  [1/4] Estimating depth with confidence...")
            depth, confidence = self.depth_estimator.estimate_with_confidence(image)
        
        # 2. 深度优化
        print("  [2/4] Refining depth...")
        
        # 基于置信度的加权优化（V2特性）
        if self.config['depth']['refine'].get('confidence_weighted', False):
            # 低置信度区域增强去噪
            low_conf_mask = confidence < 0.5
            if low_conf_mask.sum() > 0:
                depth_denoised = cv2.bilateralFilter(
                    depth.astype(np.float32), 9, 75, 75
                )
                depth[low_conf_mask] = depth_denoised[low_conf_mask]
        
        depth = self.depth_refiner.refine(depth, image)
        
        # 3. 尺度恢复（使用置信度）
        print("  [3/4] Recovering metric scale...")
        camera_params = self._get_camera_params(image.shape[1], image.shape[0])
        
        if self.config['depth']['scale_recovery'].get('use_confidence', False):
            # V2: 只使用高置信度物体进行尺度恢复
            high_conf_objects = [
                obj for obj in objects 
                if obj['confidence'] > 0.7 and obj['category'] in 
                self.config['depth']['scale_recovery']['reference_objects']
            ]
            depth, scale_factor = self.scale_recovery.recover_scale(
                depth, high_conf_objects, camera_params
            )
        else:
            depth, scale_factor = self.scale_recovery.recover_scale(
                depth, objects, camera_params
            )
        
        # 4. 深度边缘检测和修复（V2特性）
        print("  [4/4] Detecting and inpainting depth edges...")
        depth_edges = self.depth_estimator.compute_depth_edges(depth, threshold=0.1)
        
        # 修复深度不连续区域
        if depth_edges.sum() > 0:
            depth = self.depth_estimator.inpaint_depth(depth, depth_edges, method='telea')
        
        return depth, confidence
    
    def _initialize_objects(
        self,
        objects: list,
        depth_map: np.ndarray,
        depth_confidence: np.ndarray,
        image: np.ndarray,
        camera_params: Dict
    ):
        """初始化所有物体的3D Gaussians - V2"""
        for obj in tqdm(objects, desc="  Initializing objects"):
            obj_id = obj['id']
            category = obj['category']
            
            # 获取形状先验
            shape_prior = self.explicit_prior.get_template(category)
            
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
                # V2: 使用深度置信度加权初始化
                mask = obj['segmentation']
                
                # 只使用高置信度深度点
                if self.config['shape_prior']['adaptive'].get('use_depth_confidence', False):
                    high_conf_mask = depth_confidence > 0.6
                    final_mask = mask & high_conf_mask
                    
                    # 如果过滤后点太少，回退到原始mask
                    if final_mask.sum() < 100:
                        final_mask = mask
                else:
                    final_mask = mask
                
                obj_gaussian.initialize_from_mask_depth(
                    final_mask,
                    depth_map,
                    image,
                    camera_params
                )
                
                # V2: 存储额外信息
                obj_gaussian.segmentation_confidence = obj['confidence']
                obj_gaussian.stability_score = obj.get('stability_score', 0.0)
                
                self.scene_gaussians.add_object(obj_id, obj_gaussian)
                
            except Exception as e:
                print(f"    Warning: Failed to initialize object {obj_id}: {e}")
                continue
    
    def _optimize_with_priors_v2(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        depth_confidence: np.ndarray,
        camera_params: Dict
    ):
        """使用形状先验约束优化 - V2增强"""
        iterations = self.config['reconstruction']['optimization']['iterations']
        
        # 为所有物体设置优化器
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
        
        # V2: 深度置信度加权
        use_depth_conf_weight = loss_config.get('depth_confidence_weight', False)
        
        pbar = tqdm(range(iterations), desc="  Optimizing")
        
        viewing_coverage = self.config['shape_prior']['adaptive']['viewing_coverage']
        seg_confidence = self.config['shape_prior']['adaptive']['segmentation_confidence']
        recon_uncertainty = self.config['shape_prior']['adaptive']['reconstruction_uncertainty']
        
        for iter_num in pbar:
            total_loss = 0.0
            loss_details = {}
            
            for obj_id, obj_gaussian in self.scene_gaussians.objects.items():
                optimizer = optimizers[obj_id]
                optimizer.zero_grad()
                
                # V2: 使用物体的分割置信度调整先验权重
                obj_seg_conf = getattr(obj_gaussian, 'segmentation_confidence', seg_confidence)
                
                _, weights_info = self.prior_fusion.fuse_priors(
                    obj_gaussian.get_xyz,
                    obj_gaussian.category,
                    viewing_coverage,
                    obj_seg_conf,  # V2: 使用实际置信度
                    recon_uncertainty
                )
                
                # 形状先验损失
                shape_prior_loss = self.prior_fusion.compute_fused_prior_loss(
                    obj_gaussian.get_xyz,
                    obj_gaussian.category,
                    weights_info
                )
                
                # V2: 深度置信度加权的正则化
                if use_depth_conf_weight:
                    # 获取物体区域的平均深度置信度
                    mask = obj_gaussian.segmentation if hasattr(obj_gaussian, 'segmentation') else None
                    if mask is not None:
                        avg_depth_conf = depth_confidence[mask].mean()
                        # 低置信度区域增强正则化
                        reg_weight_factor = 2.0 - avg_depth_conf  # [1.0, 2.0]
                    else:
                        reg_weight_factor = 1.0
                else:
                    reg_weight_factor = 1.0
                
                # 正则化损失
                reg_losses = self.regularizer.compute_combined_regularization(
                    obj_gaussian.get_xyz,
                    weights={
                        'smoothness': loss_config['smoothness'] * reg_weight_factor,
                        'compactness': 0.02 * reg_weight_factor,
                        'symmetry': loss_config['symmetry']
                    }
                )
                
                loss = shape_prior_loss + reg_losses['total']
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # 记录详细损失
                if iter_num % 100 == 0:
                    loss_details[f'obj_{obj_id}'] = {
                        'prior': shape_prior_loss.item(),
                        'reg': reg_losses['total'].item(),
                        'weights': weights_info
                    }
                
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
            
            # V2: 详细日志
            if iter_num % 100 == 0 and iter_num > 0:
                print(f"\n  Iter {iter_num}: Avg loss = {avg_loss:.4f}")
                for obj_id, details in list(loss_details.items())[:3]:  # 只打印前3个
                    print(f"    {obj_id}: prior={details['prior']:.4f}, "
                          f"reg={details['reg']:.4f}, "
                          f"w_obs={details['weights']['observation']:.3f}")
    
    def _should_densify(self, iter_num: int) -> bool:
        """是否应该致密化"""
        config = self.config['reconstruction']['optimization']
        return (
            config['densify_from_iter'] <= iter_num <= config['densify_until_iter']
            and iter_num % config['densify_interval'] == 0
        )
    
    def _should_reset_opacity(self, iter_num: int) -> bool:
        """是否应该重置不透明度"""
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
        """保存重建结果 - V2"""
        # 保存场景
        self.scene_gaussians.save(str(output_path / 'scene'))
        
        # V2: 保存深度和置信度
        depth_vis = self.visualizer.visualize_depth(depth_map)
        cv2.imwrite(str(output_path / 'depth.png'), depth_vis)
        
        conf_vis = (depth_confidence * 255).astype(np.uint8)
        conf_colored = cv2.applyColorMap(conf_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_path / 'depth_confidence.png'), conf_colored)
        
        # 保存分割
        seg_vis = self.visualizer.visualize_masks(image, objects)
        cv2.imwrite(str(output_path / 'segmentation.png'), 
                   cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR))
        
        # V2: 保存深度边缘
        edges = self.depth_estimator.compute_depth_edges(depth_map)
        cv2.imwrite(str(output_path / 'depth_edges.png'), 
                   (edges * 255).astype(np.uint8))
        
        # 保存统计
        stats = self.scene_gaussians.get_statistics()
        
        # V2: 添加额外统计
        stats['v2_info'] = {
            'avg_segmentation_confidence': float(np.mean([obj['confidence'] for obj in objects])),
            'avg_depth_confidence': float(depth_confidence.mean()),
            'depth_confidence_std': float(depth_confidence.std()),
            'num_high_conf_objects': sum(1 for obj in objects if obj['confidence'] > 0.8),
            'model_versions': {
                'dinov2': 'large',
                'sam': '2.0',
                'depth': 'anything_v2'
            }
        }
        
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
        """生成重建报告 - V2新特性"""
        report = {
            'summary': {
                'total_objects': len(objects),
                'total_gaussians': self.scene_gaussians.total_gaussians,
                'avg_gaussians_per_object': self.scene_gaussians.total_gaussians / max(1, len(objects))
            },
            'quality_metrics': {
                'avg_segmentation_confidence': float(np.mean([obj['confidence'] for obj in objects])),
                'avg_depth_confidence': float(depth_confidence.mean()),
                'depth_confidence_std': float(depth_confidence.std()),
                'high_quality_objects': sum(1 for obj in objects if obj['confidence'] > 0.8)
            },
            'objects': []
        }
        
        for obj in objects:
            obj_report = {
                'id': obj['id'],
                'category': obj['category'],
                'confidence': float(obj['confidence']),
                'area': int(obj['area']),
                'stability_score': float(obj.get('stability_score', 0.0))
            }
            
            if obj['id'] in self.scene_gaussians.objects:
                obj_gaussian = self.scene_gaussians.objects[obj['id']]
                obj_report['num_gaussians'] = obj_gaussian.num_points
                obj_report['bounding_box'] = obj_gaussian.get_bounding_box()
            
            report['objects'].append(obj_report)
        
        return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Single Image 3D Reconstruction V2')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--config', type=str, default='configs/single_image_v2.yaml', 
                       help='Config file')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # 初始化V2重建器
    reconstructor = SingleImageReconstructorV2(args.config)
    
    if args.output:
        reconstructor.output_dir = Path(args.output)
    
    # 执行重建
    result = reconstructor.reconstruct(args.image)
    
    # 打印报告
    print("\n" + "=" * 70)
    print("📊 Reconstruction Report")
    print("=" * 70)
    report = result['report']
    print(f"\n Summary:")
    print(f"  Objects: {report['summary']['total_objects']}")
    print(f"  Gaussians: {report['summary']['total_gaussians']}")
    print(f"  Avg Gaussians/Object: {report['summary']['avg_gaussians_per_object']:.1f}")
    print(f"\n Quality Metrics:")
    print(f"  Segmentation Confidence: {report['quality_metrics']['avg_segmentation_confidence']:.3f}")
    print(f"  Depth Confidence: {report['quality_metrics']['avg_depth_confidence']:.3f}")
    print(f"  High Quality Objects: {report['quality_metrics']['high_quality_objects']}")
    print("=" * 70)
    
    print(f"\n✓ Done! Check results in: {result['output_dir']}")


if __name__ == "__main__":
    main()