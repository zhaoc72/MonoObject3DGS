"""
Camera Utilities
相机工具
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class Camera:
    """相机类"""
    
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        R: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        
        # 外参（世界坐标到相机坐标）
        self.R = R if R is not None else np.eye(3)
        self.t = t if t is not None else np.zeros(3)
        
        # 内参矩阵
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def project_points(
        self,
        points_3d: np.ndarray
    ) -> np.ndarray:
        """投影3D点到图像"""
        # 世界坐标到相机坐标
        points_cam = (self.R @ points_3d.T + self.t.reshape(3, 1)).T
        
        # 投影到图像
        points_2d = (self.K @ points_cam.T).T
        points_2d = points_2d[:, :2] / (points_2d[:, 2:3] + 1e-6)
        
        return points_2d
    
    def unproject_depth(
        self,
        depth_map: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """从深度图反投影到3D"""
        H, W = depth_map.shape
        
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        if mask is not None:
            ys = ys[mask]
            xs = xs[mask]
            depths = depth_map[mask]
        else:
            ys = ys.flatten()
            xs = xs.flatten()
            depths = depth_map.flatten()
        
        # 反投影
        points_cam = np.zeros((len(xs), 3))
        points_cam[:, 0] = (xs - self.cx) * depths / self.fx
        points_cam[:, 1] = (ys - self.cy) * depths / self.fy
        points_cam[:, 2] = depths
        
        # 相机坐标到世界坐标
        points_world = (self.R.T @ (points_cam - self.t).T).T
        
        return points_world
    
    def get_fov(self) -> Tuple[float, float]:
        """获取视场角"""
        fov_x = 2 * np.arctan(self.width / (2 * self.fx))
        fov_y = 2 * np.arctan(self.height / (2 * self.fy))
        return np.degrees(fov_x), np.degrees(fov_y)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'fx': self.fx, 'fy': self.fy,
            'cx': self.cx, 'cy': self.cy,
            'width': self.width, 'height': self.height,
            'R': self.R.tolist(),
            't': self.t.tolist()
        }


class CameraTrajectory:
    """相机轨迹生成器"""
    
    @staticmethod
    def circular_trajectory(
        num_views: int,
        radius: float = 5.0,
        elevation: float = 30.0,
        center: np.ndarray = np.zeros(3)
    ) -> List[Dict]:
        """生成圆形轨迹"""
        cameras = []
        
        elevation_rad = np.radians(elevation)
        
        for i in range(num_views):
            azimuth = 2 * np.pi * i / num_views
            
            # 相机位置
            x = radius * np.cos(elevation_rad) * np.cos(azimuth)
            y = radius * np.sin(elevation_rad)
            z = radius * np.cos(elevation_rad) * np.sin(azimuth)
            
            cam_pos = center + np.array([x, y, z])
            
            # 朝向中心
            forward = center - cam_pos
            forward = forward / (np.linalg.norm(forward) + 1e-6)
            
            # 上向量
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
            right = right / (np.linalg.norm(right) + 1e-6)
            up = np.cross(right, forward)
            
            # 旋转矩阵（相机到世界）
            R_c2w = np.stack([right, up, -forward], axis=1)
            R_w2c = R_c2w.T
            
            t_w2c = -R_w2c @ cam_pos
            
            cameras.append({
                'R': R_w2c,
                't': t_w2c,
                'position': cam_pos
            })
        
        return cameras


# 测试
if __name__ == "__main__":
    camera = Camera(
        fx=525.0, fy=525.0,
        cx=320.0, cy=240.0,
        width=640, height=480
    )
    
    # 测试投影
    points_3d = np.random.randn(10, 3) * 2
    points_3d[:, 2] += 5  # 移到相机前方
    
    points_2d = camera.project_points(points_3d)
    print(f"Projected points shape: {points_2d.shape}")
    
    # 测试轨迹
    trajectory = CameraTrajectory.circular_trajectory(num_views=36)
    print(f"Generated {len(trajectory)} camera poses")