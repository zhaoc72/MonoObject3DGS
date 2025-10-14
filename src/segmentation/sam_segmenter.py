"""
SAM Segmenter
Segment Anything Model分割器
"""

import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from typing import List, Dict, Optional, Tuple
import cv2


class SAMSegmenter:
    """SAM分割器，支持自动和提示式分割"""
    
    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint: str = "sam_vit_h_4b8939.pth",
        device: str = "cuda",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
    ):
        self.device = device
        
        # 加载SAM模型
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
            min_mask_region_area=100,
        )
        
        # 提示式预测器
        self.predictor = SamPredictor(sam)
        
        print(f"✓ SAM模型加载成功: {model_type}")
        
    def segment_automatic(
        self,
        image: np.ndarray,
        min_area: int = 100,
        max_area: Optional[int] = None
    ) -> List[Dict]:
        """
        自动分割图像
        
        Args:
            image: RGB图像 (H, W, 3)
            min_area: 最小mask面积
            max_area: 最大mask面积
            
        Returns:
            masks: mask列表，每个包含:
                - segmentation: (H, W) bool mask
                - bbox: [x, y, w, h]
                - area: 面积
                - predicted_iou: 预测的IoU
                - stability_score: 稳定性分数
        """
        masks = self.mask_generator.generate(image)
        
        # 过滤
        if max_area is None:
            max_area = image.shape[0] * image.shape[1] * 0.9
            
        filtered_masks = []
        for mask in masks:
            area = mask['area']
            if min_area <= area <= max_area:
                filtered_masks.append(mask)
                
        # 按面积排序
        filtered_masks = sorted(filtered_masks, key=lambda x: x['area'], reverse=True)
        
        return filtered_masks
    
    def segment_with_points(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用点提示进行分割
        
        Args:
            image: RGB图像
            point_coords: 点坐标 (N, 2) [x, y]
            point_labels: 点标签 (N,) 1=前景, 0=背景
            
        Returns:
            masks: (N_masks, H, W)
            scores: (N_masks,)
            logits: (N_masks, H, W)
        """
        self.predictor.set_image(image)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        
        return masks, scores, logits
    
    def segment_with_box(
        self,
        image: np.ndarray,
        box: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用边界框提示进行分割
        
        Args:
            image: RGB图像
            box: 边界框 [x1, y1, x2, y2]
            
        Returns:
            masks, scores, logits
        """
        self.predictor.set_image(image)
        
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=False,
        )
        
        return masks, scores, logits
    
    def refine_with_features(
        self,
        masks: List[Dict],
        feature_map: torch.Tensor,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        使用特征相似度优化mask
        
        Args:
            masks: SAM输出的mask列表
            feature_map: DINOv2特征图 (1, D, H, W)
            similarity_threshold: 相似度阈值
            
        Returns:
            refined_masks: 优化后的mask列表
        """
        import torch.nn.functional as F
        
        feature_map = F.normalize(feature_map, dim=1)
        B, D, H, W = feature_map.shape
        
        refined_masks = []
        
        for mask_dict in masks:
            mask = mask_dict['segmentation']
            
            # 调整mask大小以匹配特征图
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            # 提取mask内的平均特征
            mask_tensor = torch.from_numpy(mask_resized).to(feature_map.device)
            masked_features = feature_map * mask_tensor.unsqueeze(0).unsqueeze(0)
            
            if mask_tensor.sum() > 0:
                avg_feature = masked_features.sum(dim=(2, 3)) / mask_tensor.sum()
                avg_feature = F.normalize(avg_feature, dim=1)
                
                # 计算特征相似度
                feature_flat = feature_map.reshape(1, D, -1)
                similarity = torch.matmul(avg_feature, feature_flat)
                similarity = similarity.reshape(1, H, W)
                
                # 根据相似度优化mask
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
                mask_dict_copy['area'] = refined_mask.sum()
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
        合并相似的mask
        
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
                'area': current_mask.sum(),
                'bbox': self._mask_to_bbox(current_mask),
            }
            merged.append(merged_dict)
            
        return merged
    
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
    print("=== 测试SAMSegmenter ===")
    
    # 注意：需要先下载SAM模型
    # wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    
    try:
        segmenter = SAMSegmenter(checkpoint="sam_vit_h_4b8939.pth")
        
        # 加载测试图像
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # 自动分割
        print("\n1. 测试自动分割:")
        masks = segmenter.segment_automatic(test_image)
        print(f"   检测到 {len(masks)} 个物体")
        
        if masks:
            print(f"   第一个mask面积: {masks[0]['area']}")
            print(f"   第一个mask bbox: {masks[0]['bbox']}")
        
        # 点提示分割
        print("\n2. 测试点提示分割:")
        point_coords = np.array([[256, 256]])
        point_labels = np.array([1])
        
        masks_p, scores, _ = segmenter.segment_with_points(
            test_image,
            point_coords,
            point_labels
        )
        print(f"   生成了 {len(masks_p)} 个mask候选")
        print(f"   分数: {scores}")
        
        # 框提示分割
        print("\n3. 测试框提示分割:")
        box = np.array([100, 100, 400, 400])
        masks_b, scores_b, _ = segmenter.segment_with_box(test_image, box)
        print(f"   生成的mask形状: {masks_b.shape}")
        
        print("\n测试完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        print("提示: 请确保已下载SAM模型权重")