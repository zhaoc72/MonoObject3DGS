"""
Segmentation Pipeline
完整的分割流程管道：特征提取 -> 分割 -> 追踪 -> 分类
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import cv2
from tqdm import tqdm

from .dinov2_extractor import DINOv2Extractor
from .sam_segmenter import SAMSegmenter
from .object_tracker import ObjectTracker
from .semantic_classifier import SemanticClassifier


class SegmentationPipeline:
    """
    完整的分割流程管道
    整合DINOv2、SAM、Tracker和Classifier
    """
    
    def __init__(
        self,
        device: str = "cuda",
        use_tracking: bool = True,
        use_classification: bool = True,
        config: Optional[Dict] = None
    ):
        """
        Args:
            device: 计算设备
            use_tracking: 是否启用跨帧追踪
            use_classification: 是否启用语义分类
            config: 配置字典
        """
        self.device = device
        self.use_tracking = use_tracking
        self.use_classification = use_classification
        
        # 默认配置
        if config is None:
            config = self._default_config()
        self.config = config
        
        # 初始化各个组件
        print("初始化分割管道...")
        self._init_components()
        
        print("✓ SegmentationPipeline初始化完成")
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'dinov2': {
                'model_name': 'facebook/dinov2-base',
                'feature_dim': 768
            },
            'sam': {
                'model_type': 'vit_h',
                'checkpoint': 'data/checkpoints/sam_vit_h_4b8939.pth',
                'points_per_side': 32,
                'pred_iou_thresh': 0.88
            },
            'tracker': {
                'max_age': 30,
                'min_hits': 3,
                'iou_threshold': 0.3,
                'feature_threshold': 0.7
            },
            'classifier': {
                'method': 'clip',
                'categories': None
            },
            'refine': {
                'use_features': True,
                'similarity_threshold': 0.7,
                'merge_similar': True,
                'iou_threshold': 0.5
            }
        }
    
    def _init_components(self):
        """初始化所有组件"""
        # DINOv2特征提取器
        self.feature_extractor = DINOv2Extractor(
            model_name=self.config['dinov2']['model_name'],
            feature_dim=self.config['dinov2']['feature_dim'],
            device=self.device
        )
        
        # SAM分割器
        self.segmenter = SAMSegmenter(
            model_type=self.config['sam']['model_type'],
            checkpoint=self.config['sam']['checkpoint'],
            device=self.device,
            points_per_side=self.config['sam']['points_per_side'],
            pred_iou_thresh=self.config['sam']['pred_iou_thresh']
        )
        
        # 物体追踪器
        if self.use_tracking:
            self.tracker = ObjectTracker(
                max_age=self.config['tracker']['max_age'],
                min_hits=self.config['tracker']['min_hits'],
                iou_threshold=self.config['tracker']['iou_threshold'],
                feature_threshold=self.config['tracker']['feature_threshold']
            )
        
        # 语义分类器
        if self.use_classification:
            self.classifier = SemanticClassifier(
                method=self.config['classifier']['method'],
                categories=self.config['classifier']['categories'],
                device=self.device
            )
    
    def process_frame(
        self,
        image: np.ndarray,
        return_features: bool = False,
        return_visualization: bool = False
    ) -> Dict:
        """
        处理单帧图像
        
        Args:
            image: RGB图像 (H, W, 3)
            return_features: 是否返回特征
            return_visualization: 是否返回可视化结果
            
        Returns:
            result字典包含:
                - objects: 物体列表
                - features: (可选) 特征图
                - visualization: (可选) 可视化图像
        """
        H, W = image.shape[:2]
        
        # 1. 特征提取
        features = self.feature_extractor.extract_features(image)
        dense_features = self.feature_extractor.get_dense_features(
            image,
            target_size=(H, W)
        )
        
        # 2. 分割
        masks = self.segmenter.segment_automatic(image)
        
        # 3. 特征优化分割
        if self.config['refine']['use_features']:
            masks = self.segmenter.refine_with_features(
                masks,
                dense_features,
                similarity_threshold=self.config['refine']['similarity_threshold']
            )
        
        # 4. 合并相似mask
        if self.config['refine']['merge_similar']:
            masks = self.segmenter.merge_similar_masks(
                masks,
                iou_threshold=self.config['refine']['iou_threshold']
            )
        
        # 5. 提取每个mask的特征（用于追踪和分类）
        mask_features = []
        for mask_dict in masks:
            mask = mask_dict['segmentation']
            # 提取mask区域的平均特征
            mask_tensor = torch.from_numpy(mask).to(self.device)
            masked_features = dense_features * mask_tensor.unsqueeze(0).unsqueeze(0)
            
            if mask_tensor.sum() > 0:
                avg_feature = masked_features.sum(dim=(2, 3)) / mask_tensor.sum()
                mask_features.append(avg_feature.squeeze(0))
            else:
                mask_features.append(torch.zeros(self.config['dinov2']['feature_dim']).to(self.device))
        
        if mask_features:
            mask_features = torch.stack(mask_features)
        else:
            mask_features = None
        
        # 6. 追踪
        if self.use_tracking:
            tracked_masks = self.tracker.update(masks, mask_features)
        else:
            # 不追踪时，简单分配ID
            tracked_masks = []
            for i, mask in enumerate(masks):
                mask['id'] = i
                tracked_masks.append(mask)
        
        # 7. 分类
        objects = []
        for i, mask_dict in enumerate(tracked_masks):
            obj = {
                'id': mask_dict['id'],
                'segmentation': mask_dict['segmentation'],
                'bbox': mask_dict['bbox'],
                'area': mask_dict['area']
            }
            
            if self.use_classification:
                # 裁剪物体区域
                x, y, w, h = mask_dict['bbox']
                x, y = max(0, x), max(0, y)
                x2, y2 = min(W, x + w), min(H, y + h)
                
                crop = image[y:y2, x:x2]
                crop_mask = mask_dict['segmentation'][y:y2, x:x2]
                
                # 分类
                if crop.size > 0:
                    predictions = self.classifier.classify(
                        image_crop=crop,
                        mask=crop_mask,
                        top_k=3
                    )
                    obj['category'] = predictions[0]['category']
                    obj['category_score'] = predictions[0]['score']
                    obj['category_predictions'] = predictions
                else:
                    obj['category'] = 'unknown'
                    obj['category_score'] = 0.0
            
            objects.append(obj)
        
        # 构建返回结果
        result = {'objects': objects}
        
        if return_features:
            result['features'] = {
                'dense': dense_features,
                'cls': features.get('cls_token'),
                'patch': features.get('patch_tokens')
            }
        
        if return_visualization:
            result['visualization'] = self._visualize_result(image, objects)
        
        return result
    
    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        save_visualization: bool = True,
        max_frames: Optional[int] = None
    ) -> List[Dict]:
        """
        处理视频序列
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            save_visualization: 是否保存可视化结果
            max_frames: 最大处理帧数
            
        Returns:
            results: 每帧的处理结果列表
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"处理视频: {video_path}")
        print(f"  分辨率: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  总帧数: {total_frames}")
        
        # 准备输出
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if save_visualization:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_video = cv2.VideoWriter(
                    str(output_path / 'segmentation.mp4'),
                    fourcc,
                    fps,
                    (width, height)
                )
        
        results = []
        frame_id = 0
        
        with tqdm(total=total_frames, desc="处理帧") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_id >= max_frames):
                    break
                
                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 处理帧
                result = self.process_frame(
                    frame_rgb,
                    return_visualization=save_visualization
                )
                result['frame_id'] = frame_id
                results.append(result)
                
                # 保存可视化
                if save_visualization and output_dir is not None:
                    vis = result['visualization']
                    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                    out_video.write(vis_bgr)
                
                frame_id += 1
                pbar.update(1)
        
        cap.release()
        if save_visualization and output_dir is not None:
            out_video.release()
        
        print(f"\n✓ 处理完成，共 {len(results)} 帧")
        
        # 保存结果
        if output_dir is not None:
            self._save_results(results, output_dir)
        
        return results
    
    def process_image_sequence(
        self,
        image_dir: str,
        output_dir: Optional[str] = None,
        image_extension: str = "jpg",
        save_visualization: bool = True
    ) -> List[Dict]:
        """
        处理图像序列
        
        Args:
            image_dir: 图像目录
            output_dir: 输出目录
            image_extension: 图像扩展名
            save_visualization: 是否保存可视化
            
        Returns:
            results: 每帧的处理结果列表
        """
        image_path = Path(image_dir)
        image_files = sorted(image_path.glob(f"*.{image_extension}"))
        
        if len(image_files) == 0:
            raise ValueError(f"在 {image_dir} 中未找到 .{image_extension} 文件")
        
        print(f"处理图像序列: {image_dir}")
        print(f"  图像数量: {len(image_files)}")
        
        # 准备输出
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if save_visualization:
                (output_path / 'visualizations').mkdir(exist_ok=True)
        
        results = []
        
        for frame_id, img_file in enumerate(tqdm(image_files, desc="处理图像")):
            # 加载图像
            image = cv2.imread(str(img_file))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 处理
            result = self.process_frame(
                image_rgb,
                return_visualization=save_visualization
            )
            result['frame_id'] = frame_id
            result['image_path'] = str(img_file)
            results.append(result)
            
            # 保存可视化
            if save_visualization and output_dir is not None:
                vis = result['visualization']
                vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(output_path / 'visualizations' / f'{frame_id:05d}.jpg'),
                    vis_bgr
                )
        
        print(f"\n✓ 处理完成，共 {len(results)} 帧")
        
        # 保存结果
        if output_dir is not None:
            self._save_results(results, output_dir)
        
        return results
    
    def _visualize_result(
        self,
        image: np.ndarray,
        objects: List[Dict]
    ) -> np.ndarray:
        """可视化分割结果"""
        vis = image.copy()
        H, W = image.shape[:2]
        
        # 创建彩色mask覆盖层
        overlay = np.zeros_like(image)
        
        # 为每个物体分配颜色
        np.random.seed(42)
        colors = np.random.randint(0, 255, (len(objects), 3))
        
        for i, obj in enumerate(objects):
            mask = obj['segmentation']
            color = colors[i % len(colors)]
            
            # 填充mask
            overlay[mask] = color * 0.6 + overlay[mask] * 0.4
            
            # 绘制边界
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, color.tolist(), 2)
            
            # 绘制bbox和标签
            x, y, w, h = obj['bbox']
            cv2.rectangle(vis, (x, y), (x + w, y + h), color.tolist(), 2)
            
            # 标签
            label = f"ID:{obj['id']}"
            if 'category' in obj:
                label += f" {obj['category']}"
                if 'category_score' in obj:
                    label += f" ({obj['category_score']:.2f})"
            
            # 绘制标签背景
            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )
            cv2.rectangle(
                vis,
                (x, y - text_h - 5),
                (x + text_w, y),
                color.tolist(),
                -1
            )
            cv2.putText(
                vis,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # 混合原图和覆盖层
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        return vis
    
    def _save_results(self, results: List[Dict], output_dir: str):
        """保存处理结果"""
        import json
        
        output_path = Path(output_dir)
        
        # 保存为JSON（不包含大型数组）
        simplified_results = []
        for result in results:
            simplified = {
                'frame_id': result['frame_id'],
                'objects': []
            }
            
            for obj in result['objects']:
                obj_info = {
                    'id': int(obj['id']),
                    'bbox': [int(x) for x in obj['bbox']],
                    'area': int(obj['area'])
                }
                
                if 'category' in obj:
                    obj_info['category'] = obj['category']
                    obj_info['category_score'] = float(obj['category_score'])
                
                simplified['objects'].append(obj_info)
            
            simplified_results.append(simplified)
        
        with open(output_path / 'results.json', 'w') as f:
            json.dump(simplified_results, f, indent=2)
        
        print(f"✓ 结果保存到: {output_path / 'results.json'}")
        
        # 保存统计信息
        self._save_statistics(results, output_path)
    
    def _save_statistics(self, results: List[Dict], output_path: Path):
        """保存统计信息"""
        stats = {
            'total_frames': len(results),
            'total_objects_detected': sum(len(r['objects']) for r in results),
            'avg_objects_per_frame': np.mean([len(r['objects']) for r in results]),
            'unique_object_ids': set()
        }
        
        if self.use_classification:
            category_counts = {}
            for result in results:
                for obj in result['objects']:
                    if 'category' in obj:
                        cat = obj['category']
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                    
                    stats['unique_object_ids'].add(obj['id'])
            
            stats['category_distribution'] = category_counts
        
        stats['unique_object_ids'] = len(stats['unique_object_ids'])
        
        # 保存统计
        import json
        with open(output_path / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ 统计信息保存到: {output_path / 'statistics.json'}")
        
        # 打印统计摘要
        print("\n=== 处理统计 ===")
        print(f"总帧数: {stats['total_frames']}")
        print(f"检测到的物体总数: {stats['total_objects_detected']}")
        print(f"平均每帧物体数: {stats['avg_objects_per_frame']:.2f}")
        print(f"唯一物体ID数: {stats['unique_object_ids']}")
        
        if self.use_classification:
            print("\n类别分布:")
            for cat, count in sorted(
                stats['category_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"  {cat}: {count}")
    
    def reset(self):
        """重置管道状态"""
        if self.use_tracking:
            self.tracker.reset()


class BatchProcessor:
    """
    批量处理器
    用于高效处理大量图像/视频
    """
    
    def __init__(
        self,
        pipeline: SegmentationPipeline,
        batch_size: int = 4,
        num_workers: int = 0
    ):
        """
        Args:
            pipeline: 分割管道
            batch_size: 批大小
            num_workers: 工作线程数
        """
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def process_images_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Dict]:
        """
        批量处理图像
        
        Args:
            images: 图像列表
            
        Returns:
            results: 处理结果列表
        """
        results = []
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            # 处理批次中的每个图像
            for img in batch:
                result = self.pipeline.process_frame(img)
                results.append(result)
        
        return results


class InteractivePipeline(SegmentationPipeline):
    """
    交互式分割管道
    支持用户交互式指定物体
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_image = None
        self.current_features = None
    
    def load_image(self, image: np.ndarray):
        """加载图像并预计算特征"""
        self.current_image = image
        
        # 预计算特征
        features = self.feature_extractor.extract_features(image)
        self.current_features = self.feature_extractor.get_dense_features(
            image,
            target_size=(image.shape[0], image.shape[1])
        )
        
        print(f"✓ 图像加载完成: {image.shape}")
    
    def segment_with_points(
        self,
        point_coords: List[Tuple[int, int]],
        point_labels: List[int]
    ) -> Dict:
        """
        使用点提示分割
        
        Args:
            point_coords: 点坐标列表 [(x, y), ...]
            point_labels: 点标签列表 [1=前景, 0=背景, ...]
            
        Returns:
            result: 分割结果
        """
        if self.current_image is None:
            raise ValueError("请先使用 load_image() 加载图像")
        
        # 转换为numpy数组
        coords = np.array(point_coords, dtype=np.float32)
        labels = np.array(point_labels, dtype=np.int32)
        
        # SAM分割
        masks, scores, _ = self.segmenter.segment_with_points(
            self.current_image,
            coords,
            labels
        )
        
        # 选择最佳mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        # 创建结果
        mask_dict = {
            'segmentation': best_mask,
            'bbox': self._mask_to_bbox(best_mask),
            'area': best_mask.sum(),
            'score': scores[best_idx]
        }
        
        # 分类
        if self.use_classification:
            x, y, w, h = mask_dict['bbox']
            crop = self.current_image[y:y+h, x:x+w]
            crop_mask = best_mask[y:y+h, x:x+w]
            
            predictions = self.classifier.classify(
                image_crop=crop,
                mask=crop_mask,
                top_k=3
            )
            
            mask_dict['category'] = predictions[0]['category']
            mask_dict['category_score'] = predictions[0]['score']
            mask_dict['category_predictions'] = predictions
        
        return mask_dict
    
    def segment_with_box(
        self,
        bbox: Tuple[int, int, int, int]
    ) -> Dict:
        """
        使用边界框提示分割
        
        Args:
            bbox: 边界框 (x1, y1, x2, y2)
            
        Returns:
            result: 分割结果
        """
        if self.current_image is None:
            raise ValueError("请先使用 load_image() 加载图像")
        
        box = np.array(bbox, dtype=np.float32)
        
        # SAM分割
        masks, scores, _ = self.segmenter.segment_with_box(
            self.current_image,
            box
        )
        
        best_mask = masks[0]
        
        # 创建结果
        mask_dict = {
            'segmentation': best_mask,
            'bbox': self._mask_to_bbox(best_mask),
            'area': best_mask.sum(),
            'score': scores[0]
        }
        
        # 分类
        if self.use_classification:
            x, y, w, h = mask_dict['bbox']
            crop = self.current_image[y:y+h, x:x+w]
            crop_mask = best_mask[y:y+h, x:x+w]
            
            predictions = self.classifier.classify(
                image_crop=crop,
                mask=crop_mask,
                top_k=3
            )
            
            mask_dict['category'] = predictions[0]['category']
            mask_dict['category_score'] = predictions[0]['score']
            mask_dict['category_predictions'] = predictions
        
        return mask_dict
    
    def get_similarity_map(
        self,
        query_point: Tuple[int, int]
    ) -> np.ndarray:
        """
        获取与查询点的相似度图
        用于可视化引导
        
        Args:
            query_point: 查询点 (x, y)
            
        Returns:
            similarity_map: 相似度图 (H, W)
        """
        if self.current_image is None:
            raise ValueError("请先使用 load_image() 加载图像")
        
        return self.feature_extractor.compute_similarity_map(
            self.current_image,
            query_point,
            target_size=(self.current_image.shape[0], self.current_image.shape[1])
        )
    
    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> List[int]:
        """将mask转换为bbox"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return [0, 0, 0, 0]
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]


if __name__ == "__main__":
    # 测试代码
    print("=== 测试SegmentationPipeline ===")
    
    # 初始化管道
    pipeline = SegmentationPipeline(
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_tracking=True,
        use_classification=True
    )
    
    # 测试单帧处理
    print("\n1. 单帧处理测试:")
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    result = pipeline.process_frame(
        test_image,
        return_features=False,
        return_visualization=True
    )
    
    print(f"检测到 {len(result['objects'])} 个物体")
    for obj in result['objects']:
        print(f"  ID {obj['id']}: {obj['category']} ({obj['category_score']:.3f})")
    
    # 测试图像序列处理
    print("\n2. 图像序列处理测试:")
    
    # 创建临时图像序列
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    temp_input = os.path.join(temp_dir, 'input')
    temp_output = os.path.join(temp_dir, 'output')
    os.makedirs(temp_input, exist_ok=True)
    
    # 生成测试图像
    print(f"创建测试图像到: {temp_input}")
    for i in range(5):
        test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(temp_input, f'{i:05d}.jpg'), test_img)
    
    # 处理序列
    results = pipeline.process_image_sequence(
        temp_input,
        temp_output,
        save_visualization=True
    )
    
    print(f"处理了 {len(results)} 帧")
    
    # 测试交互式管道
    print("\n3. 交互式管道测试:")
    interactive = InteractivePipeline(
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_classification=True
    )
    
    # 加载图像
    interactive.load_image(test_image)
    
    # 点提示分割
    result = interactive.segment_with_points(
        point_coords=[(256, 256), (200, 200)],
        point_labels=[1, 1]
    )
    print(f"点提示分割: {result['category']} (面积: {result['area']})")
    
    # 框提示分割
    result = interactive.segment_with_box(
        bbox=(100, 100, 400, 400)
    )
    print(f"框提示分割: {result['category']} (面积: {result['area']})")
    
    # 相似度图
    sim_map = interactive.get_similarity_map((256, 256))
    print(f"相似度图: {sim_map.shape}, 范围: [{sim_map.min():.3f}, {sim_map.max():.3f}]")
    
    # 清理
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n所有测试完成！")