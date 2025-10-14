"""
Segmentation Utilities
分割相关工具函数
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask的IoU"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def mask_to_bbox(mask: np.ndarray) -> List[int]:
    """mask转bbox"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]


def visualize_masks(
    image: np.ndarray,
    masks: List[Dict],
    alpha: float = 0.5
) -> np.ndarray:
    """可视化mask"""
    vis = image.copy()
    overlay = np.zeros_like(image)
    
    np.random.seed(42)
    colors = np.random.randint(50, 255, (len(masks), 3))
    
    for i, mask_dict in enumerate(masks):
        mask = mask_dict['segmentation']
        color = colors[i % len(colors)]
        
        overlay[mask] = color
        
        # 轮廓
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color.tolist(), 2)
    
    vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
    return vis