"""
Single Image 3D Reconstruction
单目单张图片的物体级3D重建
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

from src.segmentation.dinov2_extractor import DINOv2Extractor
from src.segmentation.sam_segmenter import SAMSegmenter
from src.segmentation.semantic_classifier import SemanticClassifier
from src.depth.depth_estimator import DepthEstimator
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


class SingleImageReconstructor:
    """单张图片3D重建器"""
    
    def __init__(self, config_path: str):
        """初始化"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['experiment']['device']
        self.output_dir = Path(self.config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("MonoObject3DGS - Single Image Reconstruction")
        print("=" * 70)
        
        self._init_modules()
        
        print("=" * 70)
        print("✓ All modules initialized successfully")
        print("=" * 70)
    
    def _init_modules(self):
        """初始化所有模块"""
        print("\n[1/8] Loading DINOv2 feature extractor...")
        self.dinov2 = DINOv2Extractor(
            model_name=self.config['segmentation']['dinov2']['model_name'],
            feature_dim=self.config['segmentation']['dinov2']['feature_dim'],
            device=self.device
        )
        
        print("\n[2/8] Loading SAM segmentation model...")
        self.segmenter = SAMSegmenter(
            model_type=self.config['segmentation']['sam']['model_type'],
            checkpoint=self.config['segmentation']['sam']['checkpoint'],
            device=self.device,
            points_per_side=self.config['segmentation']['sam']['points_per_side'],
            pred_iou_thresh=self.config['segmentation']['sam']['pred_iou_thresh']
        )
        
        print("\n[3/8] Loading semantic classifier...")
        self.classifier = SemanticClassifier(
            method=self.config['classification']['method'],
            categories=self.config['classification']['categories'],
            device=self.device
        )
        
        print("\n[4/8] Loading depth estimator...")
        self.depth_estimator = DepthEstimator(
            method=self.config['depth']['method'],
            model_size=self.config['depth']['model_size'],
            device=self.device
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
        """完整重建流程"""
        print("\n" + "=" * 70)
        print(f"Reconstructing: {image_path}")
        print("=" * 70)
        
        # Step 1: 加载图像
        print("\n[Step 1/6] Loading image...")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image_rgb.shape[:2]
        print(f"  Image size: {W}x{H}")
        
        # Step 2: 语义分割
        print("\n[Step 2/6] Semantic-aware segmentation...")
        objects = self._segment_and_classify(image_rgb)
        print(f"  Detected {len(objects)} objects:")
        for obj in objects:
            print(f"    - {obj['category']} (confidence: {obj['confidence']:.2f})")
        
        # Step 3: 深度估计
        print("\n[Step 3/6] Monocular depth estimation...")
        depth_map = self._estimate_depth(image_rgb, objects)
        print(f"  Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}] meters")
        
        # Step 4: 物体级初始化
        print("\n[Step 4/6] Object-level 3D Gaussian initialization...")
        camera_params = self._get_camera_params(W, H)
        self._initialize_objects(objects, depth_map, image_rgb, camera_params)
        
        # Step 5: 形状先验约束优化
        print("\n[Step 5/6] Shape prior constrained optimization...")
        self._optimize_with_priors(image_rgb, depth_map, camera_params)
        
        # Step 6: 保存结果
        print("\n[Step 6/6] Saving results...")
        exp_name = Path(image_path).stem
        output_path = self.output_dir / f"single_image_{exp_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._save_results(output_path, image_rgb, depth_map, objects)
        
        print("\n" + "=" * 70)
        print("✓ Reconstruction completed!")
        print(f"  Output directory: {output_path}")
        print(f"  Total objects: {len(self.scene_gaussians.objects)}")
        print(f"  Total Gaussians: {self.scene_gaussians.total_gaussians}")
        print("=" * 70)
        
        return {
            'scene': self.scene_gaussians,
            'objects': objects,
            'depth': depth_map,
            'output_dir': str(output_path)
        }
    
    def _segment_and_classify(self, image: np.ndarray) -> list:
        """分割并分类"""
        H, W = image.shape[:2]
        
        print("  Extracting DINOv2 features...")
        features = self.dinov2.extract_features(image)
        dense_features = self.dinov2.get_dense_features(image, target_size=(H, W))
        
        print("  Running SAM segmentation...")
        masks = self.segmenter.segment_automatic(
            image,
            min_area=self.config['segmentation']['sam']['min_mask_area']
        )
        print(f"    Initial masks: {len(masks)}")
        
        if self.config['segmentation']['refine']['use_features']:
            print("  Refining with features...")
            masks = self.segmenter.refine_with_features(
                masks,
                dense_features,
                similarity_threshold=self.config['segmentation']['refine']['similarity_threshold']
            )
            print(f"    Refined masks: {len(masks)}")
        
        print("  Merging similar masks...")
        masks = self.segmenter.merge_similar_masks(
            masks,
            iou_threshold=self.config['segmentation']['refine']['merge_iou_threshold']
        )
        print(f"    Final masks: {len(masks)}")
        
        print("  Classifying objects...")
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
                top_k=1
            )
            
            objects.append({
                'id': i,
                'category': predictions[0]['category'],
                'confidence': predictions[0]['score'],
                'segmentation': mask_dict['segmentation'],
                'bbox': mask_dict['bbox'],
                'area': mask_dict['area']
            })
        
        return objects
    
    def _estimate_depth(self, image: np.ndarray, objects: list) -> np.ndarray:
        """估计并优化深度"""
        print("  Estimating depth...")
        depth = self.depth_estimator.estimate(image)
        
        print("  Refining depth...")
        depth = self.depth_refiner.refine(depth, image)
        
        print("  Recovering metric scale...")
        camera_params = self._get_camera_params(image.shape[1], image.shape[0])
        depth, scale_factor = self.scale_recovery.recover_scale(
            depth, objects, camera_params
        )
        
        return depth
    
    def _initialize_objects(
        self,
        objects: list,
        depth_map: np.ndarray,
        image: np.ndarray,
        camera_params: Dict
    ):
        """初始化所有物体的3D Gaussians"""
        for obj in tqdm(objects, desc="  Initializing objects"):
            obj_id = obj['id']
            category = obj['category']
            
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
                obj_gaussian.initialize_from_mask_depth(
                    mask=obj['segmentation'],
                    depth_map=depth_map,
                    image=image,
                    camera_params=camera_params
                )
                
                self.scene_gaussians.add_object(obj_id, obj_gaussian)
                
            except Exception as e:
                print(f"    Warning: Failed to initialize object {obj_id}: {e}")
                continue
    
    def _optimize_with_priors(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        camera_params: Dict
    ):
        """使用形状先验约束优化"""
        iterations = self.config['reconstruction']['optimization']['iterations']
        
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
        
        loss_fn = CompositeLoss(
            lambda_photometric=self.config['reconstruction']['loss']['photometric'],
            lambda_depth=self.config['reconstruction']['loss']['depth_consistency'],
            lambda_shape_prior=self.config['reconstruction']['loss']['shape_prior'],
            lambda_semantic=self.config['reconstruction']['loss']['semantic_consistency'],
            lambda_smoothness=self.config['reconstruction']['loss']['smoothness'],
            lambda_symmetry=self.config['reconstruction']['loss']['symmetry'],
            use_lpips=False
        ).to(self.device)
        
        pbar = tqdm(range(iterations), desc="  Optimizing")
        
        viewing_coverage = self.config['shape_prior']['adaptive']['viewing_coverage']
        seg_confidence = 0.8
        recon_uncertainty = 0.7
        
        for iter_num in pbar:
            total_loss = 0.0
            
            for obj_id, obj_gaussian in self.scene_gaussians.objects.items():
                optimizer = optimizers[obj_id]
                optimizer.zero_grad()
                
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
                
                reg_losses = self.regularizer.compute_combined_regularization(
                    obj_gaussian.get_xyz,
                    weights={
                        'smoothness': self.config['reconstruction']['loss']['smoothness'],
                        'compactness': 0.02,
                        'symmetry': self.config['reconstruction']['loss']['symmetry']
                    }
                )
                
                loss = shape_prior_loss + reg_losses['total']
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
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
                
                if self._should_reset_opacity(iter_num):
                    obj_gaussian.reset_opacity()
            
            if iter_num % 100 == 0:
                pbar.set_postfix({'loss': total_loss / max(1, len(self.scene_gaussians.objects))})
    
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
        objects: list
    ):
        """保存重建结果"""
        self.scene_gaussians.save(str(output_path / 'scene'))
        
        depth_vis = self.visualizer.visualize_depth(depth_map)
        cv2.imwrite(str(output_path / 'depth.png'), depth_vis)
        
        seg_vis = self.visualizer.visualize_masks(image, objects)
        cv2.imwrite(str(output_path / 'segmentation.png'), cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR))
        
        stats = self.scene_gaussians.get_statistics()
        import json
        with open(output_path / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64, np.int64)) else x)
        
        print(f"  Saved to: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Single Image 3D Reconstruction')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--config', type=str, default='configs/single_image.yaml', help='Config file')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    reconstructor = SingleImageReconstructor(args.config)
    
    if args.output:
        reconstructor.output_dir = Path(args.output)
    
    result = reconstructor.reconstruct(args.image)
    
    print("\n✓ Done! Check results in:", result['output_dir'])


if __name__ == "__main__":
    main()