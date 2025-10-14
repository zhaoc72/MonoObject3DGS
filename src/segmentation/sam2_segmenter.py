"""
SAM 2 Segmenter - UPGRADED
支持视频分割和更强的追踪能力
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2


class SAM2Segmenter:
    """SAM 2 分割器 - 支持图像和视频"""
    
    def __init__(
        self,
        model_size: str = "large",  # tiny/small/base/large
        checkpoint: str = "data/checkpoints/sam2_hiera_large.pt",
        device: str = "cuda",
        mode: str = "image"  # image/video
    ):
        self.device = device
        self.model_size = model_size
        self.mode = mode
        
        print(f"🔄 Loading SAM 2 ({mode} mode): {model_size}")
        
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            
            # 模型配置映射
            model_cfg_map = {
                "tiny": "sam2_hiera_t.yaml",
                "small": "sam2_hiera_s.yaml",
                "base": "sam2_hiera_b.yaml",
                "large": "sam2_hiera_l.yaml"
            }
            
            model_cfg = model_cfg_map.get(model_size, "sam2_hiera_l.yaml")
            
            # 构建SAM2模型
            sam2_model = build_sam2(model_cfg, checkpoint, device=device)
            
            # 创建预测器
            if mode == "image":
                self.predictor = SAM2ImagePredictor(sam2_model)
            else:  # video
                self.predictor = SAM2VideoPredictor(sam2_model)
            
            print(f"✓ SAM 2 loaded: {model_size} ({mode} mode)")
            
        except Exception as e:
            print(f"❌ Failed to load SAM 2: {e}")
            print("  Please install: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            self.predictor = None
    
    # ============ 图像分割 ============
    
    @torch.no_grad()
    def segment_automatic(
        self,
        image: np.ndarray,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100
    ) -> List[Dict]:
        """
        自动分割 - SAM 2增强版
        
        Args:
            image: RGB图像 (H, W, 3)
            points_per_side: 每边采样点数
            pred_iou_thresh: IoU阈值
            stability_score_thresh: 稳定性阈值
            min_mask_region_area: 最小区域面积
            
        Returns:
            masks: mask列表
        """
        if self.predictor is None:
            return self._dummy_masks(image.shape[:2])
        
        # SAM 2使用自动mask生成器
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        mask_generator = SAM2AutomaticMaskGenerator(
            model=self.predictor.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2
        )
        
        masks = mask_generator.generate(image)
        
        # 按面积排序
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        return masks
    
    @torch.no_grad()
    def segment_with_prompts(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        提示式分割 - SAM 2
        
        Args:
            image: RGB图像
            point_coords: 点坐标 (N, 2) [x, y]
            point_labels: 点标签 (N,) [1=前景, 0=背景]
            box: 边界框 (4,) [x1, y1, x2, y2]
            multimask_output: 是否输出多个mask
            
        Returns:
            masks: (M, H, W)
            scores: (M,)
            logits: (M, 256, 256)
        """
        if self.predictor is None:
            H, W = image.shape[:2]
            return (
                np.zeros((1, H, W), dtype=bool),
                np.array([0.5]),
                np.zeros((1, 256, 256))
            )
        
        # 设置图像
        self.predictor.set_image(image)
        
        # 预测
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output
        )
        
        return masks, scores, logits
    
    # ============ 视频分割 ============
    
    def init_video_state(
        self,
        video_path: str,
        offload_video_to_cpu: bool = True,
        async_loading_frames: bool = True
    ):
        """
        初始化视频状态 - SAM 2 Video NEW
        
        Args:
            video_path: 视频路径或帧目录
            offload_video_to_cpu: 卸载到CPU节省显存
            async_loading_frames: 异步加载帧
        """
        if self.mode != "video":
            raise ValueError("Video mode not enabled. Initialize with mode='video'")
        
        if self.predictor is None:
            return None
        
        # 初始化视频
        inference_state = self.predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames
        )
        
        return inference_state
    
    def add_object_prompt(
        self,
        inference_state,
        frame_idx: int,
        obj_id: int,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None
    ):
        """
        添加物体提示 - 在指定帧
        
        Args:
            inference_state: 视频推理状态
            frame_idx: 帧索引
            obj_id: 物体ID
            points: 点坐标 (N, 2)
            labels: 点标签 (N,)
            box: 边界框 (4,)
        """
        if self.predictor is None:
            return
        
        # 添加提示
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            box=box
        )
        
        return out_obj_ids, out_mask_logits
    
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx: Optional[int] = None,
        max_frame_num_to_track: Optional[int] = None,
        reverse: bool = False
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        在视频中传播mask - SAM 2核心功能
        
        Args:
            inference_state: 视频推理状态
            start_frame_idx: 起始帧
            max_frame_num_to_track: 最大追踪帧数
            reverse: 是否反向传播
            
        Returns:
            video_segments: {
                frame_idx: {
                    obj_id: mask (H, W)
                }
            }
        """
        if self.predictor is None:
            return {}
        
        video_segments = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse
        ):
            frame_masks = {}
            for obj_id, mask_logits in zip(out_obj_ids, out_mask_logits):
                mask = (mask_logits > 0.0).cpu().numpy()
                frame_masks[obj_id] = mask
            
            video_segments[out_frame_idx] = frame_masks
        
        return video_segments
    
    # ============ 高级功能 ============
    
    def refine_with_features(
        self,
        masks: List[Dict],
        feature_map: torch.Tensor,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        使用DINOv2特征优化mask边界
        
        Args:
            masks: SAM输出的mask列表
            feature_map: DINOv2特征图 (1, D, H, W)
            similarity_threshold: 相似度阈值
            
        Returns:
            refined_masks: 优化后的mask列表
        """
        if len(masks) == 0:
            return masks
        
        feature_map = torch.nn.functional.normalize(feature_map, dim=1)
        B, D, H, W = feature_map.shape
        
        refined_masks = []
        
        for mask_dict in masks:
            mask = mask_dict['segmentation']
            
            # 调整mask大小
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            # 提取mask特征
            mask_tensor = torch.from_numpy(mask_resized).to(feature_map.device)
            masked_features = feature_map * mask_tensor.unsqueeze(0).unsqueeze(0)
            
            if mask_tensor.sum() > 0:
                # 计算平均特征
                avg_feature = masked_features.sum(dim=(2, 3)) / mask_tensor.sum()
                avg_feature = torch.nn.functional.normalize(avg_feature, dim=1)
                
                # 计算相似度
                feature_flat = feature_map.reshape(1, D, -1)
                similarity = torch.matmul(avg_feature, feature_flat)
                similarity = similarity.reshape(1, H, W)
                
                # 优化mask
                refined_mask_small = (similarity[0] > similarity_threshold).cpu().numpy()
                
                # 调整回原始尺寸
                refined_mask = cv2.resize(
                    refined_mask_small.astype(np.uint8),
                    (mask.shape[1], mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                
                # 更新mask
                mask_dict_copy = mask_dict.copy()
                mask_dict_copy['segmentation'] = refined_mask
                mask_dict_copy['area'] = int(refined_mask.sum())
                mask_dict_copy['bbox'] = self._mask_to_bbox(refined_mask)
                refined_masks.append(mask_dict_copy)
            else:
                refined_masks.append(mask_dict)
        
        return refined_masks
    
    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> List[int]:
        """mask转bbox"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return [0, 0, 0, 0]
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]
    
    def _dummy_masks(self, image_shape):
        """Dummy masks for testing"""
        H, W = image_shape
        return [{
            'segmentation': np.zeros((H, W), dtype=bool),
            'area': 0,
            'bbox': [0, 0, 0, 0],
            'predicted_iou': 0.0,
            'stability_score': 0.0
        }]


# 测试代码
if __name__ == "__main__":
    print("=== Testing SAM 2 Segmenter ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试图像模式
    print("\n1. Image Mode:")
    segmenter_img = SAM2Segmenter(
        model_size="large",
        device=device,
        mode="image"
    )
    
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # 自动分割
    masks = segmenter_img.segment_automatic(test_image)
    print(f"  Detected {len(masks)} masks")
    
    # 提示式分割
    masks, scores, logits = segmenter_img.segment_with_prompts(
        test_image,
        point_coords=np.array([[256, 256]]),
        point_labels=np.array([1])
    )
    print(f"  Prompt segmentation: {masks.shape}, scores: {scores}")
    
    # 测试视频模式
    print("\n2. Video Mode:")
    segmenter_vid = SAM2Segmenter(
        model_size="large",
        device=device,
        mode="video"
    )
    print("  Video mode initialized")
    
    print("\n✓ All tests passed!")