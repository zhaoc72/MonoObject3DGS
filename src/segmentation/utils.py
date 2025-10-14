"""
Segmentation Utils
分割模块的实用工具函数
"""

import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional
from scipy.ndimage import binary_erosion, binary_dilation
from skimage import measure
import matplotlib.pyplot as plt


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算两个mask的IoU
    
    Args:
        mask1: 第一个mask (H, W)
        mask2: 第二个mask (H, W)
        
    Returns:
        iou: IoU值
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def mask_to_boundary(
    mask: np.ndarray,
    dilation_ratio: float = 0.02
) -> np.ndarray:
    """
    将mask转换为边界
    
    Args:
        mask: 输入mask (H, W)
        dilation_ratio: 膨胀比例
        
    Returns:
        boundary: 边界mask
    """
    h, w = mask.shape
    dilation_size = int(dilation_ratio * np.sqrt(h * w))
    
    # 膨胀和腐蚀
    kernel = np.ones((3, 3), np.uint8)
    dilated = binary_dilation(mask, iterations=dilation_size)
    eroded = binary_erosion(mask, iterations=dilation_size)
    
    # 边界 = 膨胀 - 腐蚀
    boundary = dilated.astype(np.uint8) - eroded.astype(np.uint8)
    
    return boundary.astype(bool)


def mask_to_polygon(mask: np.ndarray, tolerance: float = 1.0) -> List[np.ndarray]:
    """
    将mask转换为多边形
    
    Args:
        mask: 输入mask (H, W)
        tolerance: 简化容差
        
    Returns:
        polygons: 多边形列表，每个为 (N, 2) 数组
    """
    # 查找轮廓
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        # 简化多边形
        epsilon = tolerance * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 转换为 (N, 2) 格式
        polygon = approx.reshape(-1, 2)
        if len(polygon) >= 3:  # 至少3个点
            polygons.append(polygon)
    
    return polygons


def polygon_to_mask(
    polygon: np.ndarray,
    height: int,
    width: int
) -> np.ndarray:
    """
    将多边形转换为mask
    
    Args:
        polygon: 多边形顶点 (N, 2)
        height: 图像高度
        width: 图像宽度
        
    Returns:
        mask: 生成的mask (H, W)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 绘制填充多边形
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    
    return mask.astype(bool)


def merge_masks(
    masks: List[np.ndarray],
    method: str = "union"
) -> np.ndarray:
    """
    合并多个mask
    
    Args:
        masks: mask列表
        method: 合并方法 ("union", "intersection")
        
    Returns:
        merged: 合并后的mask
    """
    if len(masks) == 0:
        raise ValueError("masks列表为空")
    
    if method == "union":
        merged = masks[0].copy()
        for mask in masks[1:]:
            merged = np.logical_or(merged, mask)
    elif method == "intersection":
        merged = masks[0].copy()
        for mask in masks[1:]:
            merged = np.logical_and(merged, mask)
    else:
        raise ValueError(f"未知方法: {method}")
    
    return merged


def split_mask_into_components(mask: np.ndarray) -> List[np.ndarray]:
    """
    将mask分割为独立的连通组件
    
    Args:
        mask: 输入mask (H, W)
        
    Returns:
        components: 组件mask列表
    """
    # 连通组件分析
    labeled = measure.label(mask, connectivity=2)
    
    components = []
    for region in measure.regionprops(labeled):
        component = (labeled == region.label)
        components.append(component)
    
    return components


def filter_small_masks(
    masks: List[Dict],
    min_area: int = 100
) -> List[Dict]:
    """
    过滤掉小mask
    
    Args:
        masks: mask字典列表
        min_area: 最小面积
        
    Returns:
        filtered: 过滤后的mask列表
    """
    return [m for m in masks if m['area'] >= min_area]


def remove_mask_holes(
    mask: np.ndarray,
    min_hole_size: int = 100
) -> np.ndarray:
    """
    移除mask中的小孔洞
    
    Args:
        mask: 输入mask (H, W)
        min_hole_size: 最小孔洞大小
        
    Returns:
        filled: 填充后的mask
    """
    # 反转mask以找到孔洞
    inverted = ~mask
    
    # 标记连通组件
    labeled = measure.label(inverted, connectivity=2)
    
    # 移除小组件（孔洞）
    filled = mask.copy()
    for region in measure.regionprops(labeled):
        if region.area < min_hole_size:
            filled[labeled == region.label] = True
    
    return filled


def smooth_mask_boundary(
    mask: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    平滑mask边界
    
    Args:
        mask: 输入mask (H, W)
        kernel_size: 核大小
        
    Returns:
        smoothed: 平滑后的mask
    """
    # 使用形态学操作平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 闭操作（先膨胀后腐蚀）
    smoothed = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel
    )
    
    # 开操作（先腐蚀后膨胀）
    smoothed = cv2.morphologyEx(
        smoothed,
        cv2.MORPH_OPEN,
        kernel
    )
    
    return smoothed.astype(bool)


def crop_image_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 10,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    使用mask裁剪图像
    
    Args:
        image: 输入图像 (H, W, 3)
        mask: mask (H, W)
        padding: 边界填充
        background_color: 背景颜色
        
    Returns:
        cropped_image: 裁剪的图像
        cropped_mask: 裁剪的mask
        bbox: 原图中的bbox (x, y, w, h)
    """
    # 获取bbox
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return image, mask, (0, 0, image.shape[1], image.shape[0])
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # 添加padding
    H, W = image.shape[:2]
    rmin = max(0, rmin - padding)
    rmax = min(H, rmax + padding + 1)
    cmin = max(0, cmin - padding)
    cmax = min(W, cmax + padding + 1)
    
    # 裁剪
    cropped_image = image[rmin:rmax, cmin:cmax].copy()
    cropped_mask = mask[rmin:rmax, cmin:cmax].copy()
    
    # 应用背景色
    cropped_image[~cropped_mask] = background_color
    
    bbox = (cmin, rmin, cmax - cmin, rmax - rmin)
    
    return cropped_image, cropped_mask, bbox


def visualize_masks(
    image: np.ndarray,
    masks: List[Dict],
    show_labels: bool = True,
    alpha: float = 0.5,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    可视化多个mask
    
    Args:
        image: 输入图像 (H, W, 3)
        masks: mask字典列表
        show_labels: 是否显示标签
        alpha: 透明度
        save_path: 保存路径
        
    Returns:
        vis: 可视化图像
    """
    vis = image.copy()
    
    # 为每个mask分配颜色
    np.random.seed(42)
    colors = np.random.randint(50, 255, (len(masks), 3))
    
    # 创建覆盖层
    overlay = np.zeros_like(image)
    
    for i, mask_dict in enumerate(masks):
        mask = mask_dict['segmentation']
        color = colors[i % len(colors)]
        
        # 填充mask
        overlay[mask] = color
        
        # 绘制轮廓
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color.tolist(), 2)
        
        # 显示标签
        if show_labels:
            x, y, w, h = mask_dict['bbox']
            
            label = f"ID:{mask_dict.get('id', i)}"
            if 'category' in mask_dict:
                label += f" {mask_dict['category']}"
            
            cv2.putText(
                vis,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color.tolist(),
                2
            )
    
    # 混合
    vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
    
    if save_path is not None:
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    return vis


def compute_mask_statistics(mask: np.ndarray) -> Dict:
    """
    计算mask的统计信息
    
    Args:
        mask: 输入mask (H, W)
        
    Returns:
        stats: 统计信息字典
    """
    # 连通性分析
    labeled = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labeled)
    
    if len(regions) == 0:
        return {
            'area': 0,
            'perimeter': 0,
            'circularity': 0,
            'bbox': (0, 0, 0, 0),
            'centroid': (0, 0),
            'num_components': 0
        }
    
    # 使用最大组件
    main_region = max(regions, key=lambda r: r.area)
    
    # 计算圆形度
    area = main_region.area
    perimeter = main_region.perimeter
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    stats = {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'bbox': main_region.bbox,  # (min_row, min_col, max_row, max_col)
        'centroid': main_region.centroid,  # (row, col)
        'num_components': len(regions),
        'solidity': main_region.solidity,
        'eccentricity': main_region.eccentricity
    }
    
    return stats


def batch_process_masks(
    masks: List[np.ndarray],
    operation: str,
    **kwargs
) -> List[np.ndarray]:
    """
    批量处理mask
    
    Args:
        masks: mask列表
        operation: 操作类型 ("smooth", "fill_holes", "remove_small")
        **kwargs: 操作参数
        
    Returns:
        processed: 处理后的mask列表
    """
    processed = []
    
    for mask in masks:
        if operation == "smooth":
            result = smooth_mask_boundary(mask, **kwargs)
        elif operation == "fill_holes":
            result = remove_mask_holes(mask, **kwargs)
        elif operation == "remove_small":
            # 移除小组件
            components = split_mask_into_components(mask)
            min_area = kwargs.get('min_area', 100)
            large_components = [c for c in components if c.sum() >= min_area]
            result = merge_masks(large_components, "union") if large_components else mask
        else:
            result = mask
        
        processed.append(result)
    
    return processed


def compare_masks(
    mask1: np.ndarray,
    mask2: np.ndarray,
    visualize: bool = False
) -> Dict:
    """
    比较两个mask
    
    Args:
        mask1: 第一个mask
        mask2: 第二个mask
        visualize: 是否可视化
        
    Returns:
        comparison: 比较结果
    """
    iou = compute_mask_iou(mask1, mask2)
    
    # 计算Dice系数
    intersection = np.logical_and(mask1, mask2).sum()
    dice = 2 * intersection / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) > 0 else 0
    
    # 计算面积差异
    area_diff = abs(mask1.sum() - mask2.sum())
    area_ratio = min(mask1.sum(), mask2.sum()) / max(mask1.sum(), mask2.sum()) if max(mask1.sum(), mask2.sum()) > 0 else 0
    
    comparison = {
        'iou': iou,
        'dice': dice,
        'area_diff': area_diff,
        'area_ratio': area_ratio
    }
    
    if visualize:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(mask1, cmap='gray')
        axes[0].set_title('Mask 1')
        axes[0].axis('off')
        
        axes[1].imshow(mask2, cmap='gray')
        axes[1].set_title('Mask 2')
        axes[1].axis('off')
        
        axes[2].imshow(np.logical_and(mask1, mask2), cmap='gray')
        axes[2].set_title(f'Intersection (IoU={iou:.3f})')
        axes[2].axis('off')
        
        axes[3].imshow(np.logical_or(mask1, mask2), cmap='gray')
        axes[3].set_title('Union')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return comparison


if __name__ == "__main__":
    # 测试工具函数
    print("=== 测试Segmentation Utils ===")
    
    # 创建测试mask
    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[20:80, 20:80] = True
    
    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[40:90, 40:90] = True
    
    # 测试IoU
    iou = compute_mask_iou(mask1, mask2)
    print(f"\n1. IoU测试: {iou:.3f}")
    
    # 测试边界提取
    boundary = mask_to_boundary(mask1)
    print(f"2. 边界提取: {boundary.sum()} 个边界像素")
    
    # 测试多边形转换
    polygons = mask_to_polygon(mask1)
    print(f"3. 多边形转换: {len(polygons)} 个多边形")
    if polygons:
        print(f"   第一个多边形: {len(polygons[0])} 个顶点")
    
    # 测试mask合并
    merged = merge_masks([mask1, mask2], method="union")
    print(f"4. Mask合并: {merged.sum()} 个像素")
    
    # 测试连通组件分割
    components = split_mask_into_components(merged)
    print(f"5. 连通组件: {len(components)} 个组件")
    
    # 测试平滑
    smoothed = smooth_mask_boundary(mask1, kernel_size=5)
    print(f"6. 平滑: 原始 {mask1.sum()} -> 平滑后 {smoothed.sum()} 像素")
    
    # 测试统计
    stats = compute_mask_statistics(mask1)
    print(f"\n7. Mask统计:")
    print(f"   面积: {stats['area']}")
    print(f"   周长: {stats['perimeter']:.2f}")
    print(f"   圆形度: {stats['circularity']:.3f}")
    print(f"   质心: ({stats['centroid'][0]:.1f}, {stats['centroid'][1]:.1f})")
    
    # 测试mask比较
    comparison = compare_masks(mask1, mask2, visualize=False)
    print(f"\n8. Mask比较:")
    print(f"   IoU: {comparison['iou']:.3f}")
    print(f"   Dice: {comparison['dice']:.3f}")
    print(f"   面积比: {comparison['area_ratio']:.3f}")
    
    # 测试批量处理
    masks = [mask1, mask2, merged]
    processed = batch_process_masks(masks, "smooth", kernel_size=3)
    print(f"\n9. 批量处理: 处理了 {len(processed)} 个mask")
    
    print("\n所有测试完成！")