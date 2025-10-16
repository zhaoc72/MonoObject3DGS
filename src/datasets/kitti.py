"""
KITTI Dataset Loader
自动驾驶场景数据集 - 户外3D物体检测和重建

支持:
- KITTI Object Detection (3D bounding boxes)
- KITTI Raw Data (sequences)
- KITTI Depth (LiDAR深度)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import cv2
from .base_dataset import StaticSceneDataset, DynamicSceneDataset, load_image


class KITTIDataset(StaticSceneDataset):
    """
    KITTI 3D Object Detection数据集
    
    数据结构:
    kitti/
    ├── training/
    │   ├── image_2/         # 左相机RGB
    │   ├── image_3/         # 右相机RGB
    │   ├── calib/           # 相机标定
    │   ├── label_2/         # 3D标注
    │   └── velodyne/        # LiDAR点云
    └── testing/
    """
    
    # KITTI类别
    CATEGORIES = [
        'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
        'Cyclist', 'Tram', 'Misc', 'DontCare'
    ]
    
    def __init__(
        self,
        camera: str = 'image_2',  # image_2 (left) or image_3 (right)
        use_lidar_depth: bool = True,
        **kwargs
    ):
        self.camera = camera
        self.use_lidar_depth = use_lidar_depth
        super().__init__(**kwargs)
    
    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        samples = []
        
        # 数据目录
        if self.split in ['train', 'val']:
            data_dir = self.data_root / 'training'
        else:
            data_dir = self.data_root / 'testing'
        
        image_dir = data_dir / self.camera
        if not image_dir.exists():
            print(f"Warning: Image directory not found: {image_dir}")
            return samples
        
        # 获取所有图像
        image_files = sorted(image_dir.glob("*.png"))
        
        # 加载train/val split
        if self.split in ['train', 'val']:
            split_file = self.data_root / f"{self.split}.txt"
            if split_file.exists():
                with open(split_file, 'r') as f:
                    split_indices = set(int(line.strip()) for line in f.readlines())
                
                image_files = [f for f in image_files if int(f.stem) in split_indices]
        
        # 创建样本
        for img_file in image_files:
            idx = int(img_file.stem)
            
            sample = {
                'idx': idx,
                'image_path': str(img_file),
                'calib_path': str(data_dir / 'calib' / f"{idx:06d}.txt"),
                'label_path': str(data_dir / 'label_2' / f"{idx:06d}.txt"),
                'velodyne_path': str(data_dir / 'velodyne' / f"{idx:06d}.bin")
            }
            
            samples.append(sample)
        
        return samples
    
    def _load_sample(self, idx: int) -> Dict:
        """加载单个样本"""
        sample_info = self.samples[idx]
        
        # 加载图像
        image = load_image(sample_info['image_path'], self.image_size)
        
        sample = {
            'image': image,
            'scene_id': f"kitti_{sample_info['idx']:06d}",
            'frame_id': sample_info['idx']
        }
        
        # 加载标定
        calib = self._load_calibration(sample_info['calib_path'])
        sample['camera'] = calib
        
        # 加载标签（3D boxes）
        if Path(sample_info['label_path']).exists():
            objects = self._load_labels(sample_info['label_path'])
            
            if self.load_masks and len(objects) > 0:
                # KITTI只有2D bbox，生成简单的mask
                masks = []
                categories = []
                
                for obj in objects:
                    mask = self._bbox_to_mask(obj['bbox_2d'], image.shape[:2])
                    masks.append(mask)
                    categories.append(obj['type'])
                
                sample['masks'] = masks
                sample['categories'] = categories
                sample['objects_3d'] = objects  # 保存3D信息
        
        # 加载深度（从LiDAR投影）
        if self.load_depth and self.use_lidar_depth:
            if Path(sample_info['velodyne_path']).exists():
                depth = self._project_lidar_to_image(
                    sample_info['velodyne_path'],
                    calib,
                    image.shape[:2]
                )
                sample['depth'] = depth
        
        return sample
    
    def _load_calibration(self, calib_path: str) -> Dict:
        """加载相机标定"""
        calib = {}
        
        if not Path(calib_path).exists():
            # 返回默认值
            H, W = self.image_size
            return {
                'P2': np.eye(3, 4) * W / 2,
                'R0_rect': np.eye(3),
                'Tr_velo_to_cam': np.eye(3, 4)
            }
        
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if ':' not in line:
                    continue
                
                key, value = line.split(':', 1)
                key = key.strip()
                
                if key in ['P0', 'P1', 'P2', 'P3']:
                    calib[key] = np.array([float(x) for x in value.split()]).reshape(3, 4)
                elif key == 'R0_rect':
                    calib[key] = np.array([float(x) for x in value.split()]).reshape(3, 3)
                elif key in ['Tr_velo_to_cam', 'Tr_imu_to_velo']:
                    calib[key] = np.array([float(x) for x in value.split()]).reshape(3, 4)
        
        # 提取内参
        P2 = calib.get('P2', np.eye(3, 4))
        calib['fx'] = P2[0, 0]
        calib['fy'] = P2[1, 1]
        calib['cx'] = P2[0, 2]
        calib['cy'] = P2[1, 2]
        
        return calib
    
    def _load_labels(self, label_path: str) -> List[Dict]:
        """加载3D标签"""
        objects = []
        
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                
                obj = {
                    'type': parts[0],
                    'truncated': float(parts[1]),
                    'occluded': int(parts[2]),
                    'alpha': float(parts[3]),
                    'bbox_2d': [float(x) for x in parts[4:8]],  # [x1, y1, x2, y2]
                    'dimensions': [float(x) for x in parts[8:11]],  # [h, w, l]
                    'location': [float(x) for x in parts[11:14]],  # [x, y, z]
                    'rotation_y': float(parts[14])
                }
                
                if len(parts) > 15:
                    obj['score'] = float(parts[15])
                
                # 过滤DontCare
                if obj['type'] != 'DontCare':
                    objects.append(obj)
        
        return objects
    
    def _bbox_to_mask(
        self,
        bbox_2d: List[float],
        image_shape: tuple
    ) -> np.ndarray:
        """2D bbox转mask"""
        H, W = image_shape
        mask = np.zeros((H, W), dtype=bool)
        
        x1, y1, x2, y2 = bbox_2d
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        
        mask[y1:y2, x1:x2] = True
        
        return mask
    
    def _project_lidar_to_image(
        self,
        velodyne_path: str,
        calib: Dict,
        image_shape: tuple
    ) -> np.ndarray:
        """将LiDAR点云投影到图像生成深度图"""
        H, W = image_shape
        depth = np.zeros((H, W), dtype=np.float32)
        
        # 加载LiDAR点云
        points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
        points = points[:, :3]  # (x, y, z)
        
        # 转换到相机坐标系
        # velo -> cam: Tr_velo_to_cam
        # cam -> rect: R0_rect
        # rect -> image: P2
        
        Tr = calib.get('Tr_velo_to_cam', np.eye(3, 4))
        R0 = calib.get('R0_rect', np.eye(3))
        P2 = calib.get('P2', np.eye(3, 4))
        
        # 齐次坐标
        points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        
        # velo -> cam
        points_cam = (Tr @ points_homo.T).T
        
        # cam -> rect
        points_rect = (R0 @ points_cam.T).T
        
        # 过滤相机后面的点
        mask = points_rect[:, 2] > 0
        points_rect = points_rect[mask]
        
        # rect -> image
        points_homo_rect = np.concatenate([points_rect, np.ones((points_rect.shape[0], 1))], axis=1)
        points_img = (P2 @ points_homo_rect.T).T
        
        # 归一化
        points_img[:, 0] /= points_img[:, 2]
        points_img[:, 1] /= points_img[:, 2]
        
        # 深度值
        depths = points_img[:, 2]
        
        # 投影到图像
        us = points_img[:, 0]
        vs = points_img[:, 1]
        
        # 过滤范围外的点
        valid = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
        us = us[valid].astype(np.int32)
        vs = vs[valid].astype(np.int32)
        depths = depths[valid]
        
        # 填充深度图（取最近的点）
        for u, v, d in zip(us, vs, depths):
            if depth[v, u] == 0 or d < depth[v, u]:
                depth[v, u] = d
        
        # 调整大小
        if depth.shape[:2] != self.image_size:
            depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]))
        
        return depth


class KITTIVideoDataset(DynamicSceneDataset):
    """KITTI Raw Data视频序列"""
    
    def __init__(self, **kwargs):
        # KITTI Raw数据结构不同，需要单独处理
        super().__init__(**kwargs)
    
    def _load_samples(self) -> List[Dict]:
        """加载序列"""
        # 简化：从training数据创建伪序列
        # 实际使用时应该加载KITTI Raw Data
        samples = []
        
        # TODO: 实现KITTI Raw Data的序列加载
        # KITTI Raw结构: date/drive/image_02/data/*.png
        
        return samples
    
    def _load_sample_by_info(self, sample_info: Dict) -> Dict:
        """加载帧"""
        # TODO: 实现
        return {}


def prepare_kitti(
    kitti_root: str,
    output_root: str
):
    """预处理KITTI数据"""
    from tqdm import tqdm
    
    kitti_path = Path(kitti_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Preparing KITTI Dataset")
    print("=" * 70)
    
    # 统计training数据
    train_dir = kitti_path / 'training'
    
    stats = {
        'total_images': 0,
        'with_label': 0,
        'with_lidar': 0,
        'categories': {}
    }
    
    if train_dir.exists():
        image_files = list((train_dir / 'image_2').glob("*.png"))
        stats['total_images'] = len(image_files)
        
        for img_file in tqdm(image_files, desc="Processing"):
            idx = int(img_file.stem)
            
            # 检查label
            label_file = train_dir / 'label_2' / f"{idx:06d}.txt"
            if label_file.exists():
                stats['with_label'] += 1
                
                # 统计类别
                with open(label_file, 'r') as f:
                    for line in f.readlines():
                        obj_type = line.strip().split()[0]
                        if obj_type != 'DontCare':
                            stats['categories'][obj_type] = stats['categories'].get(obj_type, 0) + 1
            
            # 检查LiDAR
            velo_file = train_dir / 'velodyne' / f"{idx:06d}.bin"
            if velo_file.exists():
                stats['with_lidar'] += 1
    
    # 打印统计
    print(f"\n=== Statistics ===")
    print(f"Total images: {stats['total_images']}")
    print(f"With labels: {stats['with_label']} ({stats['with_label']/max(1,stats['total_images'])*100:.1f}%)")
    print(f"With LiDAR: {stats['with_lidar']} ({stats['with_lidar']/max(1,stats['total_images'])*100:.1f}%)")
    
    print(f"\n=== Category Distribution ===")
    for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
        print(f"  {cat:20s}: {count:5d}")
    
    # 保存统计
    import json
    with open(output_path / 'kitti_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 创建train/val split (80/20)
    if stats['total_images'] > 0:
        indices = list(range(stats['total_images']))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_point = int(len(indices) * 0.8)
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]
        
        with open(output_path / 'train.txt', 'w') as f:
            f.write('\n'.join(map(str, train_indices)))
        
        with open(output_path / 'val.txt', 'w') as f:
            f.write('\n'.join(map(str, val_indices)))
        
        print(f"\n=== Splits Created ===")
        print(f"Train: {len(train_indices)}")
        print(f"Val: {len(val_indices)}")
    
    print("\n" + "=" * 70)
    print("✓ KITTI preparation completed!")
    print("=" * 70)


if __name__ == "__main__":
    # 测试
    dataset = KITTIDataset(
        data_root="/path/to/kitti",
        split="train",
        image_size=(384, 1280),  # KITTI原始尺寸比例
        use_lidar_depth=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample: {sample.keys()}")
        if 'objects_3d' in sample:
            print(f"3D objects: {len(sample['objects_3d'])}")