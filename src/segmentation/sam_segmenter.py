"""
SAM Segmenter
Segment Anything Model分割器
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2


class SAMSegmenter:
    """SAM分割器"""
    
    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint: str = "data/checkpoints/sam_vit_h_4b8939.pth",
        device: str = "cuda",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88
    ):
        self.device = device
        self.model_type = model_type
        
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            
            # 加载SAM
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=device)
            
            # 自动mask生成器
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100
            )
            
            # 提示式预测器
            self.predictor = SamPredictor(sam)
            
            print(f"✓ SAM loaded: {model_type}")
            
        except Exception as e:
            print(f"Warning: Could not load SAM: {e}")
            print("Using dummy segmenter")
            self.mask_generator = None
            self.predictor = None
    
    def segment_automatic(
        self,
        image: np.ndarray,
        min_area: int = 100,
        max_area: Optional[int] = None
    ) -> List[Dict]:
        """
        自动分割
        
        Args:
            image: RGB图像 (H, W, 3)
            min_area: 最小面积
            max_area: 最大面积
            
        Returns:
            masks: mask列表
        """
        if self.mask_generator is None:
            # Dummy masks
            H, W = image.shape[:2]
            return [{
                'segmentation': np.zeros((H, W), dtype=bool),
                'area': 0,
                'bbox': [0, 0, 0, 0],
                'predicted_iou': 0.0,
                'stability_score': 0.0
            }]
        
        masks = self.mask_generator.generate(image)
        
        # 过滤
        if max_area is None:
            max_area = image.shape[0] * image.shape[1] * 0.9
        
        filtered = [
            m for m in masks
            if min_area <= m['area'] <= max_area
        ]
        
        return sorted(filtered, key=lambda x: x['area'], reverse=True)
    
    def refine_with_features(
        self,
        masks: List[Dict],
        feature_map: torch.Tensor,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        使用特征优化mask
        
        Args:
            masks: SAM输出的mask列表
            feature_map: 特征图 (1, D, H, W)
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
                
                mask_dict_copy = mask_dict.copy()
                mask_dict_copy['segmentation'] = refined_mask
                mask_dict_copy['area'] = int(refined_mask.sum())
                refined_masks.append(mask_dict_copy)
            else:
                refined_masks.append(mask_dict)
        
        return refined_masks
    
    def merge_similar_masks(
        self,
        masks: List[Dict],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """
        合并相似mask
        
        Args:
            masks: mask列表
            iou_threshold: IoU阈值
            
        Returns:
            merged_masks: 合并后的mask列表
        """
        if len(masks) <= 1:
            return masks
        
        merged = []
        used = set()
        
        for i, mask1 in enumerate(masks):
            if i in used:
                continue
            
            current_mask = mask1['segmentation'].copy()
            used.add(i)
            
            for j, mask2 in enumerate(masks[i+1:], start=i+1):
                if j in used:
                    continue
                
                # 计算IoU
                intersection = np.logical_and(current_mask, mask2['segmentation']).sum()
                union = np.logical_or(current_mask, mask2['segmentation']).sum()
                iou = intersection / union if union > 0 else 0
                
                if iou > iou_threshold:
                    current_mask = np.logical_or(current_mask, mask2['segmentation'])
                    used.add(j)
            
            # 创建合并后的mask字典
            merged_dict = {
                'segmentation': current_mask,
                'area': int(current_mask.sum()),
                'bbox': self._mask_to_bbox(current_mask)
            }
            merged.append(merged_dict)
        
        return merged
    
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


# 测试
if __name__ == "__main__":
    segmenter = SAMSegmenter(device="cpu")
    
    # 测试图像
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 自动分割
    masks = segmenter.segment_automatic(image)
    print(f"Detected {len(masks)} masks")