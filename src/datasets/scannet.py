"""
ScanNet Dataset Loader
支持ScanNet的RGB-D数据和实例分割
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
import json
from .base_dataset import StaticSceneDataset, DynamicSceneDataset, load_image, load_depth, load_mask


class ScanNetDataset(StaticSceneDataset):
    """
    ScanNet静态场景数据集
    
    数据结构:
    scannet/
    ├── scans/
    │   ├── scene0000_00/
    │   │   ├── color/           # RGB图像
    │   │   ├── depth/           # 深度图
    │   │   ├── instance/        # 实例分割
    │   │   ├── label/           # 语义标签
    │   │   └── pose/            # 相机位姿
    │   └── ...
    └── scannetv2-labels.combined.tsv  # 类别映射
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_label_mapping()
    
    def _load_label_mapping(self):
        """加载类别映射"""
        label_file = self.data_root / "scannetv2-labels.combined.tsv"
        if label_file.exists():
            self.label_map = {}
            with open(label_file, 'r') as f:
                lines = f.readlines()[1:]  # 跳过表头
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        label_id = int(parts[0])
                        category = parts[1]
                        self.label_map[label_id] = category
        else:
            self.label_map = {}
    
    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        samples = []
        
        split_file = self.data_root / f"scannetv2_{self.split}.txt"
        if not split_file.exists():
            print(f"Warning: Split file not found: {split_file}")
            # 自动发现场景
            scene_dirs = sorted((self.data_root / "scans").glob("scene*"))
        else:
            with open(split_file, 'r') as f:
                scene_names = [line.strip() for line in f.readlines()]
            scene_dirs = [self.data_root / "scans" / name for name in scene_names]
        
        for scene_dir in scene_dirs:
            if not scene_dir.exists():
                continue
            
            color_dir = scene_dir / "color"
            if not color_dir.exists():
                continue
            
            # 获取所有RGB图像
            image_files = sorted(color_dir.glob("*.jpg"))
            
            for img_file in image_files:
                frame_id = int(img_file.stem)
                
                sample = {
                    'scene_id': scene_dir.name,
                    'frame_id': frame_id,
                    'image_path': str(img_file),
                    'depth_path': str(scene_dir / "depth" / f"{frame_id}.png"),
                    'instance_path': str(scene_dir / "instance" / f"{frame_id}.png"),
                    'label_path': str(scene_dir / "label" / f"{frame_id}.png"),
                    'pose_path': str(scene_dir / "pose" / f"{frame_id}.txt")
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
            'scene_id': sample_info['scene_id'],
            'frame_id': sample_info['frame_id']
        }
        
        # 加载深度
        if self.load_depth and Path(sample_info['depth_path']).exists():
            depth = load_depth(sample_info['depth_path'], self.image_size)
            sample['depth'] = depth
        
        # 加载实例masks
        if self.load_masks and Path(sample_info['instance_path']).exists():
            instance_map = cv2.imread(
                sample_info['instance_path'],
                cv2.IMREAD_UNCHANGED
            )
            
            if instance_map is not None:
                # 调整大小
                if instance_map.shape[:2] != self.image_size:
                    instance_map = cv2.resize(
                        instance_map,
                        (self.image_size[1], self.image_size[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                
                # 提取每个实例的mask
                masks = []
                categories = []
                instance_ids = np.unique(instance_map)
                instance_ids = instance_ids[instance_ids > 0]  # 排除背景
                
                for inst_id in instance_ids[:self.max_objects]:
                    mask = (instance_map == inst_id)
                    
                    # 加载对应的语义标签
                    if Path(sample_info['label_path']).exists():
                        label_map = cv2.imread(
                            sample_info['label_path'],
                            cv2.IMREAD_UNCHANGED
                        )
                        if label_map is not None:
                            label_id = np.median(label_map[mask]).astype(int)
                            category = self.label_map.get(label_id, 'unknown')
                        else:
                            category = 'unknown'
                    else:
                        category = 'unknown'
                    
                    masks.append(mask)
                    categories.append(category)
                
                sample['masks'] = masks
                sample['categories'] = categories
        
        # 加载相机参数
        if Path(sample_info['pose_path']).exists():
            pose = np.loadtxt(sample_info['pose_path'])
            
            # ScanNet使用固定内参
            sample['camera'] = {
                'fx': 577.870605,
                'fy': 577.870605,
                'cx': 319.5,
                'cy': 239.5,
                'pose': pose  # 4x4变换矩阵
            }
        
        return sample


class ScanNetVideoDataset(DynamicSceneDataset):
    """ScanNet视频序列数据集"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_label_mapping()
    
    def _load_label_mapping(self):
        """加载类别映射"""
        label_file = self.data_root / "scannetv2-labels.combined.tsv"
        if label_file.exists():
            self.label_map = {}
            with open(label_file, 'r') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        label_id = int(parts[0])
                        category = parts[1]
                        self.label_map[label_id] = category
        else:
            self.label_map = {}
    
    def _load_samples(self) -> List[Dict]:
        """加载样本列表（序列的起始帧）"""
        samples = []
        
        split_file = self.data_root / f"scannetv2_{self.split}.txt"
        if split_file.exists():
            with open(split_file, 'r') as f:
                scene_names = [line.strip() for line in f.readlines()]
            scene_dirs = [self.data_root / "scans" / name for name in scene_names]
        else:
            scene_dirs = sorted((self.data_root / "scans").glob("scene*"))
        
        for scene_dir in scene_dirs:
            color_dir = scene_dir / "color"
            if not color_dir.exists():
                continue
            
            image_files = sorted(color_dir.glob("*.jpg"))
            
            # 每隔sequence_length创建一个样本
            for i in range(0, len(image_files) - self.sequence_length * self.frame_step, 
                          self.sequence_length * self.frame_step):
                frame_id = int(image_files[i].stem)
                
                sample = {
                    'scene_id': scene_dir.name,
                    'frame_id': frame_id,
                    'scene_dir': scene_dir
                }
                
                samples.append(sample)
        
        return samples
    
    def _load_sample_by_info(self, sample_info: Dict) -> Dict:
        """根据样本信息加载数据"""
        scene_dir = sample_info['scene_dir']
        frame_id = sample_info['frame_id']
        
        # 构建路径
        image_path = scene_dir / "color" / f"{frame_id}.jpg"
        depth_path = scene_dir / "depth" / f"{frame_id}.png"
        instance_path = scene_dir / "instance" / f"{frame_id}.png"
        label_path = scene_dir / "label" / f"{frame_id}.png"
        pose_path = scene_dir / "pose" / f"{frame_id}.txt"
        
        # 加载数据（复用静态数据集的逻辑）
        sample = {
            'scene_id': sample_info['scene_id'],
            'frame_id': frame_id,
            'image_path': str(image_path),
            'depth_path': str(depth_path),
            'instance_path': str(instance_path),
            'label_path': str(label_path),
            'pose_path': str(pose_path)
        }
        
        # 使用静态数据集的加载方法
        static_dataset = ScanNetDataset(
            data_root=self.data_root,
            split=self.split,
            image_size=self.image_size,
            max_objects=self.max_objects,
            load_depth=self.load_depth,
            load_masks=self.load_masks,
            augmentation=False
        )
        static_dataset.label_map = self.label_map
        static_dataset.samples = [sample]
        
        return static_dataset._load_sample(0)


# 数据预处理脚本
def prepare_scannet(
    scannet_root: str,
    output_root: str,
    splits: List[str] = ['train', 'val', 'test']
):
    """
    预处理ScanNet数据
    - 提取关键帧
    - 生成实例分割
    - 统计数据集信息
    """
    import shutil
    from tqdm import tqdm
    
    scannet_path = Path(scannet_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Preparing ScanNet Dataset")
    print("=" * 70)
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        split_file = scannet_path / f"scannetv2_{split}.txt"
        if not split_file.exists():
            print(f"  Split file not found: {split_file}")
            continue
        
        with open(split_file, 'r') as f:
            scene_names = [line.strip() for line in f.readlines()]
        
        stats = {
            'scenes': len(scene_names),
            'frames': 0,
            'objects': 0
        }
        
        for scene_name in tqdm(scene_names, desc=f"  {split}"):
            scene_dir = scannet_path / "scans" / scene_name
            
            if not scene_dir.exists():
                continue
            
            # 统计帧数
            color_files = list((scene_dir / "color").glob("*.jpg"))
            stats['frames'] += len(color_files)
            
            # 统计实例数
            instance_dir = scene_dir / "instance"
            if instance_dir.exists():
                for inst_file in instance_dir.glob("*.png"):
                    inst_map = cv2.imread(str(inst_file), cv2.IMREAD_UNCHANGED)
                    if inst_map is not None:
                        num_instances = len(np.unique(inst_map)) - 1
                        stats['objects'] += num_instances
        
        print(f"\n  {split} Statistics:")
        print(f"    Scenes: {stats['scenes']}")
        print(f"    Frames: {stats['frames']}")
        print(f"    Objects: {stats['objects']}")
        print(f"    Avg objects/frame: {stats['objects'] / max(1, stats['frames']):.2f}")
        
        # 保存统计信息
        with open(output_path / f"scannet_{split}_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✓ ScanNet preparation completed!")
    print("=" * 70)


if __name__ == "__main__":
    import cv2
    
    # 测试数据加载
    dataset = ScanNetDataset(
        data_root="/path/to/scannet",
        split="train",
        image_size=(512, 512),
        load_depth=True,
        load_masks=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试单个样本
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    if 'depth' in sample:
        print(f"Depth shape: {sample['depth'].shape}")
    if 'masks' in sample:
        print(f"Masks shape: {sample['masks'].shape}")