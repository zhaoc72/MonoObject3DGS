"""
Video 3D Reconstruction
单目视频的物体级3D重建（实时+质量平衡）
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
from collections import deque
import argparse

from src.segmentation.fast_segmenter import FastSegmenter
from src.segmentation.dinov2_extractor import DINOv2Extractor
from src.segmentation.semantic_classifier import SemanticClassifier
from src.segmentation.object_tracker import ObjectTracker
from src.depth.depth_anything_v2 import DepthAnythingV2
from src.depth.depth_refiner import DepthRefiner, DepthConsistencyRefiner
from src.priors.explicit_prior import ExplicitShapePrior
from src.priors.implicit_prior import ImplicitShapePrior
from src.priors.prior_fusion import AdaptivePriorFusion, PriorConfig
from src.reconstruction.object_gaussian import ObjectGaussian
from src.reconstruction.scene_gaussian import SceneGaussians
from src.reconstruction.gaussian_model import GaussianConfig
from src.optimization.losses import CompositeLoss


class VideoReconstructor:
    """视频3D重建器"""
    
    def __init__(self, config_path: str):
        """初始化"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['experiment']['device']
        self.output_dir = Path(self.config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("MonoObject3DGS - Video Reconstruction")
        print("=" * 70)
        
        self._init_modules()
        
        # 视频专用状态
        self.keyframes = []
        self.frame_buffer = deque(maxlen=30)
        self.frame_count = 0
        
        print("=" * 70)
        print("✓ All modules initialized for video processing")
        print("=" * 70)
    
    def _init_modules(self):
        """初始化模块（轻量级版本）"""
        print("\n[1/9] Loading DINOv2 (base model for speed)...")
        self.dinov2 = DINOv2Extractor(
            model_name="facebook/dinov2-base",  # base而不是large
            feature_dim=768,
            device=self.device
        )
        
        print("\n[2/9] Loading FastSAM (lightweight segmenter)...")
        self.segmenter = FastSegmenter(
            method="fastsam",
            checkpoint=self.config['segmentation']['sam']['checkpoint'],
            device=self.device
        )
        
        print("\n[3/9] Initializing object tracker...")
        self.tracker = ObjectTracker(
            max_age=self.config['segmentation']['tracking']['max_age'],
            min_hits=self.config['segmentation']['tracking']['min_hits'],
            iou_threshold=self.config['segmentation']['tracking']['iou_threshold'],
            feature_threshold=self.config['segmentation']['tracking']['feature_threshold']
        )
        
        print("\n[4/9] Loading semantic classifier...")
        self.classifier = SemanticClassifier(
            method=self.config['classification']['method'],
            categories=self.config['classification']['categories'],
            device=self.device
        )
        
        print("\n[5/9] Loading depth estimator...")
        self.depth_estimator = DepthAnythingV2(
            model_size=self.config['depth']['model_size'],
            device=self.device
        )
        
        print("\n[6/9] Initializing temporal depth refiner...")
        self.temporal_refiner = DepthConsistencyRefiner(
            window_size=self.config['depth']['temporal']['window_size']
        )
        
        print("\n[7/9] Loading shape priors...")
        self.explicit_prior = ExplicitShapePrior(
            template_dir=self.config['shape_prior']['explicit']['template_dir'],
            device=self.device
        )
        
        self.implicit_prior = ImplicitShapePrior(
            latent_dim=self.config['shape_prior']['implicit']['latent_dim']
        ).to(self.device)
        
        for category in self.config['classification']['categories']:
            self.implicit_prior.add_category_prototype(category)
        
        prior_config = PriorConfig(
            explicit_weight=self.config['shape_prior']['explicit']['init_weight'],
            implicit_weight=self.config['shape_prior']['implicit']['init_weight'],
            confidence_based=True
        )
        
        self.prior_fusion = AdaptivePriorFusion(
            self.explicit_prior,
            self.implicit_prior,
            prior_config
        )
        
        print("\n[8/9] Initializing scene manager...")
        self.scene_gaussians = SceneGaussians(
            config=GaussianConfig(
                sh_degree=self.config['reconstruction']['gaussian']['sh_degree'],
                init_scale=self.config['reconstruction']['gaussian']['init_scale'],
                opacity_init=self.config['reconstruction']['gaussian']['opacity_init']
            )
        )
        
        print("\n[9/9] Setting up loss function...")
        self.loss_fn = CompositeLoss(
            lambda_photometric=self.config['reconstruction']['loss']['photometric'],
            lambda_depth=self.config['reconstruction']['loss']['depth_consistency'],
            lambda_shape_prior=self.config['reconstruction']['loss']['shape_prior'],
            lambda_semantic=self.config['reconstruction']['loss']['semantic_consistency'],
            lambda_smoothness=self.config['reconstruction']['loss']['smoothness'],
            use_lpips=False  # 视频实时处理不用LPIPS
        ).to(self.device)
    
    def reconstruct(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> Dict:
        """
        从视频重建3D场景
        
        Args:
            video_path: 视频路径
            max_frames: 最大处理帧数
        """
        print("\n" + "=" * 70)
        print(f"Reconstructing from video: {video_path}")
        print("=" * 70)
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Video info: {W}x{H}, {fps:.1f} FPS, {total_frames} frames")
        
        # 相机参数
        camera_params = self._get_camera_params(W, H)
        
        # 输出目录
        exp_name = Path(video_path).stem
        output_path = self.output_dir / f"video_{exp_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 主处理循环
        print("\nProcessing video frames...")
        pbar = tqdm(total=total_frames, desc="Processing")
        
        prev_frame = None
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_id >= max_frames):
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 实时处理单帧
            frame_result = self._process_frame_realtime(
                frame_rgb,
                frame_id,
                camera_params,
                prev_frame
            )
            
            # 保存到缓冲区
            self.frame_buffer.append(frame_result)
            
            # 判断关键帧
            if self._is_keyframe(frame_id, frame_result):
                self.keyframes.append({
                    'frame_id': frame_id,
                    'image': frame_rgb,
                    'depth': frame_result['depth'],
                    'objects': frame_result['objects']
                })
                
                # 批量优化（使用关键帧）
                if len(self.keyframes) >= self.config['reconstruction']['keyframe']['min_keyframes']:
                    self._optimize_with_keyframes(camera_params)
            
            prev_frame = frame_rgb
            frame_id += 1
            
            # 更新进度
            if frame_id % 10 == 0:
                stats = self.scene_gaussians.get_statistics()
                pbar.set_postfix({
                    'objects': stats['num_objects'],
                    'gaussians': stats['total_gaussians'],
                    'keyframes': len(self.keyframes)
                })
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        # 最终全局优化
        print("\nPerforming final global optimization...")
        self._final_optimization(camera_params)
        
        # 保存结果
        print("\nSaving reconstruction results...")
        self._save_results(output_path, fps)
        
        print("\n" + "=" * 70)
        print("✓ Video reconstruction completed!")
        print(f"  Output directory: {output_path}")
        print(f"  Processed frames: {frame_id}")
        print(f"  Keyframes: {len(self.keyframes)}")
        print(f"  Objects: {len(self.scene_gaussians.objects)}")
        print(f"  Total Gaussians: {self.scene_gaussians.total_gaussians}")
        print("=" * 70)
        
        return {
            'scene': self.scene_gaussians,
            'keyframes': self.keyframes,
            'output_dir': str(output_path),
            'num_frames': frame_id
        }
    
    def _process_frame_realtime(
        self,
        image: np.ndarray,
        frame_id: int,
        camera_params: Dict,
        prev_frame: Optional[np.ndarray]
    ) -> Dict:
        """实时处理单帧（不能阻塞）"""
        H, W = image.shape[:2]
        
        # 1. 深度估计
        depth = self.depth_estimator.estimate(image)
        
        # 1.1 时序优化
        self.temporal_refiner.add_frame(depth)
        if prev_frame is not None and self.config['depth']['temporal']['enabled']:
            depth = self.temporal_refiner.refine_temporal(depth)
        
        # 2. 关键帧分割 vs 非关键帧传播
        is_keyframe = (frame_id % self.config['segmentation']['async']['keyframe_interval'] == 0)
        
        if is_keyframe or frame_id == 0:
            # 关键帧：完整分割
            masks = self._segment_keyframe(image)
            
            # 提取特征用于追踪
            mask_features = self._extract_mask_features(image, masks)
            
            # 追踪
            tracked_objects = self.tracker.update(masks, mask_features)
            
            # 分类（只对新物体）
            for obj in tracked_objects:
                if 'category' not in obj or obj.get('category') == 'unknown':
                    obj['category'] = self._classify_object(image, obj)
        else:
            # 非关键帧：使用光流传播（简化）
            if len(self.frame_buffer) > 0:
                # 复用上一帧结果
                tracked_objects = self.frame_buffer[-1]['objects']
            else:
                tracked_objects = []
        
        # 3. 增量更新物体Gaussians
        for obj in tracked_objects:
            obj_id = obj['id']
            category = obj.get('category', 'object')
            
            if not self.scene_gaussians.has_object(obj_id):
                # 新物体：初始化
                self._initialize_object(
                    obj_id,
                    category,
                    obj['segmentation'],
                    depth,
                    image,
                    camera_params
                )
            else:
                # 已存在：增量更新（轻量级）
                if is_keyframe:
                    obj_gaussian = self.scene_gaussians.get_object(obj_id)
                    obj_gaussian.update_from_observation(
                        obj['segmentation'],
                        depth,
                        image,
                        camera_params
                    )
        
        return {
            'frame_id': frame_id,
            'objects': tracked_objects,
            'depth': depth,
            'is_keyframe': is_keyframe
        }
    
    def _segment_keyframe(self, image: np.ndarray) -> list:
        """关键帧完整分割"""
        # FastSAM分割
        masks = self.segmenter.segment(
            image,
            min_area=self.config['segmentation']['sam']['min_mask_area']
        )
        return masks
    
    def _extract_mask_features(self, image: np.ndarray, masks: list) -> Optional[torch.Tensor]:
        """提取mask特征（用于追踪）"""
        if len(masks) == 0:
            return None
        
        H, W = image.shape[:2]
        dense_features = self.dinov2.get_dense_features(image, target_size=(H, W))
        
        mask_features = []
        for mask_dict in masks:
            mask = mask_dict['segmentation']
            mask_tensor = torch.from_numpy(mask).to(self.device)
            
            # 提取mask区域的平均特征
            masked_features = dense_features * mask_tensor.unsqueeze(0).unsqueeze(0)
            
            if mask_tensor.sum() > 0:
                avg_feature = masked_features.sum(dim=(2, 3)) / mask_tensor.sum()
                mask_features.append(avg_feature.squeeze(0))
            else:
                mask_features.append(torch.zeros(768, device=self.device))
        
        return torch.stack(mask_features)
    
    def _classify_object(self, image: np.ndarray, obj: Dict) -> str:
        """分类单个物体"""
        x, y, w, h = obj['bbox']
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
        
        if x2 <= x1 or y2 <= y1:
            return 'unknown'
        
        crop = image[y1:y2, x1:x2]
        crop_mask = obj['segmentation'][y1:y2, x1:x2]
        
        predictions = self.classifier.classify(
            image_crop=crop,
            mask=crop_mask,
            top_k=1
        )
        
        return predictions[0]['category']
    
    def _initialize_object(
        self,
        obj_id: int,
        category: str,
        mask: np.ndarray,
        depth: np.ndarray,
        image: np.ndarray,
        camera_params: Dict
    ):
        """初始化新物体"""
        # 获取形状先验
        shape_prior = self.explicit_prior.get_template(category)
        
        # 创建物体Gaussian
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
                mask, depth, image, camera_params
            )
            self.scene_gaussians.add_object(obj_id, obj_gaussian)
        except Exception as e:
            print(f"Warning: Failed to initialize object {obj_id}: {e}")
    
    def _is_keyframe(self, frame_id: int, frame_result: Dict) -> bool:
        """判断是否为关键帧"""
        # 策略1: 固定间隔
        interval = self.config['reconstruction']['keyframe'].get('interval', 10)
        if frame_id % interval == 0:
            return True
        
        # 策略2: 新物体出现
        current_obj_ids = {obj['id'] for obj in frame_result['objects']}
        scene_obj_ids = {int(k) for k in self.scene_gaussians.objects.keys()}
        
        if len(current_obj_ids - scene_obj_ids) > 0:
            return True
        
        return False
    
    def _optimize_with_keyframes(self, camera_params: Dict):
        """使用关键帧批量优化"""
        # 选择最近的关键帧
        max_kf = self.config['reconstruction']['keyframe']['max_keyframes']
        recent_keyframes = self.keyframes[-max_kf:]
        
        if len(recent_keyframes) < self.config['reconstruction']['keyframe']['min_keyframes']:
            return
        
        # 对每个物体进行优化
        for obj_id, obj_gaussian in self.scene_gaussians.objects.items():
            # 计算视角覆盖度（动态）
            viewing_coverage = self._compute_viewing_coverage(obj_id, recent_keyframes)
            
            # 自适应先验权重
            _, weights_info = self.prior_fusion.fuse_priors(
                obj_gaussian.get_xyz,
                obj_gaussian.category,
                viewing_coverage,
                seg_confidence=0.8,
                recon_uncertainty=0.3
            )
            
            # 轻量级优化（少量迭代）
            self._optimize_object(obj_gaussian, weights_info, iterations=100)
    
    def _compute_viewing_coverage(self, obj_id: int, keyframes: list) -> float:
        """计算物体的视角覆盖度"""
        appearances = sum(
            1 for kf in keyframes
            if any(obj['id'] == obj_id for obj in kf['objects'])
        )
        
        # 假设50帧达到满覆盖
        full_coverage_frames = self.config['shape_prior']['adaptive'].get('coverage_full_frames', 50)
        coverage = min(1.0, appearances / full_coverage_frames)
        
        return coverage
    
    def _optimize_object(
        self,
        obj_gaussian: ObjectGaussian,
        weights_info: Dict,
        iterations: int = 100
    ):
        """优化单个物体"""
        optimizer = torch.optim.Adam([
            {'params': [obj_gaussian._xyz], 'lr': 0.00016},
            {'params': [obj_gaussian._scaling], 'lr': 0.005},
            {'params': [obj_gaussian._rotation], 'lr': 0.001},
            {'params': [obj_gaussian._opacity], 'lr': 0.05}
        ])
        
        for _ in range(iterations):
            optimizer.zero_grad()
            
            # 形状先验损失
            prior_loss = self.prior_fusion.compute_fused_prior_loss(
                obj_gaussian.get_xyz,
                obj_gaussian.category,
                weights_info
            )
            
            prior_loss.backward()
            optimizer.step()
    
    def _final_optimization(self, camera_params: Dict):
        """最终全局优化"""
        print("  Using all keyframes for global optimization...")
        
        iterations = 500
        
        # 为所有物体设置优化器
        optimizers = {}
        for obj_id, obj_gaussian in self.scene_gaussians.objects.items():
            param_groups = [
                {'params': [obj_gaussian._xyz], 'lr': 0.00016},
                {'params': [obj_gaussian._features_dc], 'lr': 0.0025},
                {'params': [obj_gaussian._opacity], 'lr': 0.05},
                {'params': [obj_gaussian._scaling], 'lr': 0.005},
                {'params': [obj_gaussian._rotation], 'lr': 0.001}
            ]
            optimizers[obj_id] = torch.optim.Adam(param_groups)
        
        pbar = tqdm(range(iterations), desc="  Final optimization")
        
        for iter_num in pbar:
            total_loss = 0.0
            
            for obj_id, obj_gaussian in self.scene_gaussians.objects.items():
                optimizer = optimizers[obj_id]
                optimizer.zero_grad()
                
                # 使用所有关键帧计算覆盖度
                viewing_coverage = self._compute_viewing_coverage(obj_id, self.keyframes)
                
                _, weights_info = self.prior_fusion.fuse_priors(
                    obj_gaussian.get_xyz,
                    obj_gaussian.category,
                    viewing_coverage,
                    seg_confidence=0.8,
                    recon_uncertainty=0.2  # 多帧后不确定性降低
                )
                
                prior_loss = self.prior_fusion.compute_fused_prior_loss(
                    obj_gaussian.get_xyz,
                    obj_gaussian.category,
                    weights_info
                )
                
                prior_loss.backward()
                optimizer.step()
                
                total_loss += prior_loss.item()
            
            if iter_num % 50 == 0:
                pbar.set_postfix({'loss': total_loss / len(self.scene_gaussians.objects)})
    
    def _get_camera_params(self, width: int, height: int) -> Dict:
        """获取相机参数"""
        fov = self.config['camera']['fov']
        fx = fy = width / (2 * np.tan(np.radians(fov / 2)))
        
        return {
            'fx': fx, 'fy': fy,
            'cx': width / 2, 'cy': height / 2,
            'width': width, 'height': height
        }
    
    def _save_results(self, output_path: Path, fps: float):
        """保存结果"""
        # 1. 保存场景
        self.scene_gaussians.save(str(output_path / 'scene'))
        
        # 2. 保存关键帧信息
        import json
        keyframe_info = [
            {
                'frame_id': kf['frame_id'],
                'num_objects': len(kf['objects']),
                'objects': [
                    {
                        'id': obj['id'],
                        'category': obj.get('category', 'unknown'),
                        'bbox': obj['bbox']
                    }
                    for obj in kf['objects']
                ]
            }
            for kf in self.keyframes
        ]
        
        with open(output_path / 'keyframes.json', 'w') as f:
            json.dump(keyframe_info, f, indent=2)
        
        # 3. 保存统计
        stats = self.scene_gaussians.get_statistics()
        with open(output_path / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 4. 渲染关键帧可视化
        vis_dir = output_path / 'keyframe_visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        for i, kf in enumerate(self.keyframes[:10]):  # 保存前10个关键帧
            vis = self._visualize_frame(kf['image'], kf['objects'])
            cv2.imwrite(
                str(vis_dir / f'keyframe_{kf["frame_id"]:05d}.jpg'),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            )
        
        print(f"  Saved to: {output_path}")
    
    def _visualize_frame(self, image: np.ndarray, objects: list) -> np.ndarray:
        """可视化单帧"""
        vis = image.copy()
        overlay = np.zeros_like(image)
        
        np.random.seed(42)
        colors = np.random.randint(50, 255, (100, 3))  # 预生成足够的颜色
        
        for obj in objects:
            obj_id = obj['id']
            mask = obj['segmentation']
            color = colors[obj_id % len(colors)]
            
            # 填充
            overlay[mask] = color * 0.6
            
            # 轮廓
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, color.tolist(), 2)
            
            # 标签
            x, y, w, h = obj['bbox']
            label = f"ID:{obj_id}"
            if 'category' in obj:
                label += f" {obj['category']}"
            
            cv2.putText(
                vis, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        return vis


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Video 3D Reconstruction')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--config', type=str, default='configs/video.yaml', help='Config file')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum frames to process')
    
    args = parser.parse_args()
    
    # 初始化重建器
    reconstructor = VideoReconstructor(args.config)
    
    # 如果指定输出目录，覆盖配置
    if args.output:
        reconstructor.output_dir = Path(args.output)
    
    # 执行重建
    result = reconstructor.reconstruct(args.video, max_frames=args.max_frames)
    
    print("\n✓ Done! Check results in:", result['output_dir'])


if __name__ == "__main__":
    main()