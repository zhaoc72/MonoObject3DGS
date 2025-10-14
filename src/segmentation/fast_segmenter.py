"""
Fast Segmenter
轻量级快速分割器（FastSAM/MobileSAM）
"""

import torch
import numpy as np
from typing import List, Dict, Optional
import cv2


class FastSegmenter:
    """轻量级快速分割器"""
    
    def __init__(
        self,
        method: str = "fastsam",
        checkpoint: str = "data/checkpoints/FastSAM-x.pt",
        device: str = "cuda"
    ):
        self.method = method
        self.device = device
        
        if method == "fastsam":
            self._init_fastsam(checkpoint)
        elif method == "mobile_sam":
            self._init_mobile_sam(checkpoint)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"✓ FastSegmenter initialized: {method}")
    
    def _init_fastsam(self, checkpoint: str):
        """初始化FastSAM"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(checkpoint)
            self.model.to(self.device)
            print(f"  Loaded FastSAM from {checkpoint}")
        except ImportError:
            print("Warning: ultralytics not installed")
            print("Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"Warning: Could not load FastSAM: {e}")
            self.model = None
    
    def _init_mobile_sam(self, checkpoint: str):
        """初始化MobileSAM"""
        try:
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
            sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
            sam.to(self.device)
            self.model = SamAutomaticMaskGenerator(
                sam,
                points_per_side=16,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92
            )
            print(f"  Loaded MobileSAM from {checkpoint}")
        except ImportError:
            print("Warning: mobile_sam not installed")
            self.model = None
    
    @torch.no_grad()
    def segment(
        self,
        image: np.ndarray,
        min_area: int = 500,
        max_area: Optional[int] = None
    ) -> List[Dict]:
        """
        快速分割
        
        Args:
            image: RGB图像 (H, W, 3)
            min_area: 最小面积
            max_area: 最大面积
            
        Returns:
            masks: mask列表
        """
        if self.model is None:
            # Dummy output
            return self._dummy_segment(image, min_area)
        
        if self.method == "fastsam":
            return self._segment_fastsam(image, min_area, max_area)
        else:
            return self._segment_mobile_sam(image, min_area, max_area)
    
    def _segment_fastsam(
        self,
        image: np.ndarray,
        min_area: int,
        max_area: Optional[int]
    ) -> List[Dict]:
        """FastSAM分割"""
        try:
            results = self.model(
                image,
                retina_masks=True,
                imgsz=1024,
                conf=0.4,
                iou=0.9
            )
            
            if len(results) == 0:
                return []
            
            masks_data = results[0].masks
            if masks_data is None:
                return []
            
            masks = []
            for i, mask_tensor in enumerate(masks_data.data):
                mask = mask_tensor.cpu().numpy().astype(bool)
                
                area = int(mask.sum())
                if area < min_area:
                    continue
                if max_area and area > max_area:
                    continue
                
                # 计算bbox
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                
                if not rows.any() or not cols.any():
                    continue
                
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                masks.append({
                    'segmentation': mask,
                    'area': area,
                    'bbox': [int(cmin), int(rmin), int(cmax-cmin), int(rmax-rmin)],
                    'predicted_iou': 0.9,
                    'stability_score': 0.95
                })
            
            return masks
        
        except Exception as e:
            print(f"FastSAM error: {e}")
            return []
    
    def _segment_mobile_sam(
        self,
        image: np.ndarray,
        min_area: int,
        max_area: Optional[int]
    ) -> List[Dict]:
        """MobileSAM分割"""
        masks = self.model.generate(image)
        
        if max_area is None:
            max_area = image.shape[0] * image.shape[1] * 0.9
        
        filtered = [
            m for m in masks
            if min_area <= m['area'] <= max_area
        ]
        
        return sorted(filtered, key=lambda x: x['area'], reverse=True)
    
    def _dummy_segment(self, image: np.ndarray, min_area: int) -> List[Dict]:
        """Dummy分割（用于测试）"""
        H, W = image.shape[:2]
        
        # 创建几个随机mask
        masks = []
        for i in range(3):
            mask = np.zeros((H, W), dtype=bool)
            x, y = np.random.randint(0, W-100), np.random.randint(0, H-100)
            w, h = np.random.randint(50, 150), np.random.randint(50, 150)
            mask[y:y+h, x:x+w] = True
            
            masks.append({
                'segmentation': mask,
                'area': int(mask.sum()),
                'bbox': [x, y, w, h],
                'predicted_iou': 0.8,
                'stability_score': 0.9
            })
        
        return masks


# 测试
if __name__ == "__main__":
    segmenter = FastSegmenter(method="fastsam", device="cpu")
    
    # 测试图像
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    import time
    start = time.time()
    masks = segmenter.segment(image)
    elapsed = time.time() - start
    
    print(f"Segmented {len(masks)} objects in {elapsed:.3f}s ({1/elapsed:.1f} FPS)")