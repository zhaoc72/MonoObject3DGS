"""
Visualization Utilities
可视化工具
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Dict, List


class Visualizer:
    """可视化器"""
    
    @staticmethod
    def visualize_depth(
        depth: np.ndarray,
        colormap: int = cv2.COLORMAP_TURBO
    ) -> np.ndarray:
        """可视化深度图"""
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_vis = (depth_norm * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, colormap)
        return depth_colored
    
    @staticmethod
    def visualize_masks(
        image: np.ndarray,
        masks: List[Dict],
        alpha: float = 0.5,
        show_labels: bool = True
    ) -> np.ndarray:
        """可视化分割mask"""
        vis = image.copy()
        overlay = np.zeros_like(image)
        
        np.random.seed(42)
        colors = np.random.randint(50, 255, (len(masks), 3))
        
        for i, mask_dict in enumerate(masks):
            mask = mask_dict['segmentation']
            color = colors[i % len(colors)]
            
            # 填充
            overlay[mask] = color
            
            # 轮廓
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, color.tolist(), 2)
            
            # 标签
            if show_labels:
                x, y, w, h = mask_dict['bbox']
                label = f"Object {i}"
                if 'category' in mask_dict:
                    label = mask_dict['category']
                
                cv2.putText(
                    vis, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
        
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
        return vis
    
    @staticmethod
    def create_grid(
        images: List[np.ndarray],
        rows: int,
        cols: int,
        titles: Optional[List[str]] = None
    ) -> np.ndarray:
        """创建图像网格"""
        if len(images) == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        H, W = images[0].shape[:2]
        grid = np.zeros((H * rows, W * cols, 3), dtype=np.uint8)
        
        for idx, img in enumerate(images[:rows * cols]):
            row = idx // cols
            col = idx % cols
            
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            grid[row*H:(row+1)*H, col*W:(col+1)*W] = img
            
            if titles and idx < len(titles):
                cv2.putText(
                    grid, titles[idx],
                    (col*W + 10, row*H + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
        
        return grid
    
    @staticmethod
    def plot_3d_points(
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        title: str = "3D Point Cloud"
    ):
        """绘制3D点云"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if colors is None:
            ax.scatter(
                    points[:, 0], points[:, 1], points[:, 2],
                    c='b', marker='.', s=1, alpha=0.6
                )
        else:
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=colors, marker='.', s=1, alpha=0.6
            )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # 设置相等的坐标轴比例
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.show()


# 测试
if __name__ == "__main__":
    vis = Visualizer()
    
    # 测试深度可视化
    depth = np.random.rand(480, 640) * 5 + 2
    depth_vis = vis.visualize_depth(depth)
    print(f"Depth visualization shape: {depth_vis.shape}")
    
    # 测试mask可视化
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    masks = [
        {
            'segmentation': np.zeros((480, 640), dtype=bool),
            'bbox': [100, 100, 150, 150],
            'category': 'chair'
        }
    ]
    masks[0]['segmentation'][100:250, 100:250] = True
    
    mask_vis = vis.visualize_masks(image, masks)
    print(f"Mask visualization shape: {mask_vis.shape}")