"""
Base Dataset for MonoObject3DGS
统一的数据集基类，支持单图和视频数据
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import cv2
from abc import ABC, abstractmethod


class BaseDataset(Dataset, ABC):
    """数据集基类"""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        max_objects: int = 20,
        load_depth: bool = True,
        load_masks: bool = True,
        augmentation: bool = True
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.max_objects = max_objects
        self.load_depth = load_depth
        self.load_masks = load_masks
        self.augmentation = augmentation
        
        # 加载数据列表
        self.samples = self._load_samples()
        
        print(f"✓ {self.__class__.__name__} loaded: {len(self.samples)} samples")
    
    @abstractmethod
    def _load_samples(self) -> List[Dict]:
        """加载样本列表 - 由子类实现"""
        pass
    
    @abstractmethod
    def _load_sample(self, idx: int) -> Dict:
        """加载单个样本 - 由子类实现"""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取样本"""
        sample = self._load_sample(idx)
        
        # 数据增强
        if self.augmentation and self.split == "train":
            sample = self._augment(sample)
        
        # 转换为tensor
        sample = self._to_tensor(sample)
        
        return sample
    
    def _augment(self, sample: Dict) -> Dict:
        """数据增强"""
        image = sample['image']
        
        # 随机翻转
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            if 'depth' in sample:
                sample['depth'] = cv2.flip(sample['depth'], 1)
            if 'masks' in sample:
                sample['masks'] = [cv2.flip(m, 1) for m in sample['masks']]
        
        # 随机颜色抖动
        if np.random.rand() > 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] += np.random.uniform(-10, 10)
            hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)
            hsv[:, :, 2] *= np.random.uniform(0.8, 1.2)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        sample['image'] = image
        return sample
    
    def _to_tensor(self, sample: Dict) -> Dict:
        """转换为tensor"""
        # 图像: (H, W, 3) -> (3, H, W)
        image = torch.from_numpy(sample['image']).permute(2, 0, 1).float() / 255.0
        sample['image'] = image
        
        # 深度: (H, W) -> (1, H, W)
        if 'depth' in sample:
            depth = torch.from_numpy(sample['depth']).unsqueeze(0).float()
            sample['depth'] = depth
        
        # Masks: List[(H, W)] -> (N, H, W)
        if 'masks' in sample:
            masks = [torch.from_numpy(m) for m in sample['masks']]
            if len(masks) > 0:
                sample['masks'] = torch.stack(masks).float()
            else:
                sample['masks'] = torch.zeros(0, *self.image_size).float()
        
        # 相机参数
        if 'camera' in sample:
            for key in ['fx', 'fy', 'cx', 'cy']:
                if key in sample['camera']:
                    sample['camera'][key] = torch.tensor(sample['camera'][key]).float()
        
        return sample
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """批处理整理函数"""
        collated = {
            'images': torch.stack([s['image'] for s in batch]),
            'scene_ids': [s['scene_id'] for s in batch],
            'frame_ids': [s['frame_id'] for s in batch]
        }
        
        # 深度（可选）
        if 'depth' in batch[0]:
            collated['depths'] = torch.stack([s['depth'] for s in batch])
        
        # Masks（填充到最大数量）
        if 'masks' in batch[0]:
            max_objects = max(s['masks'].shape[0] for s in batch)
            max_objects = min(max_objects, self.max_objects)
            
            padded_masks = []
            for s in batch:
                masks = s['masks'][:max_objects]
                if masks.shape[0] < max_objects:
                    padding = torch.zeros(
                        max_objects - masks.shape[0],
                        *masks.shape[1:]
                    )
                    masks = torch.cat([masks, padding], dim=0)
                padded_masks.append(masks)
            
            collated['masks'] = torch.stack(padded_masks)
        
        # 相机参数
        if 'camera' in batch[0]:
            collated['cameras'] = [s['camera'] for s in batch]
        
        return collated


class StaticSceneDataset(BaseDataset):
    """静态场景数据集（单图）"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_video = False


class DynamicSceneDataset(BaseDataset):
    """动态场景数据集（视频序列）"""
    
    def __init__(
        self,
        sequence_length: int = 10,
        frame_step: int = 1,
        **kwargs
    ):
        self.sequence_length = sequence_length
        self.frame_step = frame_step
        super().__init__(**kwargs)
        self.is_video = True
    
    def __getitem__(self, idx: int) -> Dict:
        """获取视频序列"""
        # 获取起始帧
        start_sample = self.samples[idx]
        scene_id = start_sample['scene_id']
        start_frame = start_sample['frame_id']
        
        # 获取序列
        sequence = []
        for i in range(self.sequence_length):
            frame_id = start_frame + i * self.frame_step
            
            # 查找对应帧
            frame_sample = None
            for s in self.samples:
                if s['scene_id'] == scene_id and s['frame_id'] == frame_id:
                    frame_sample = s
                    break
            
            if frame_sample is None:
                # 如果找不到，复制最后一帧
                frame_sample = sequence[-1] if sequence else start_sample
            
            # 加载帧数据
            frame_data = self._load_sample_by_info(frame_sample)
            
            # 增强和转换
            if self.augmentation and self.split == "train":
                frame_data = self._augment(frame_data)
            frame_data = self._to_tensor(frame_data)
            
            sequence.append(frame_data)
        
        # 整理为批次格式
        collated = {
            'images': torch.stack([f['image'] for f in sequence]),
            'scene_id': scene_id,
            'frame_ids': [f['frame_id'] for f in sequence]
        }
        
        if 'depth' in sequence[0]:
            collated['depths'] = torch.stack([f['depth'] for f in sequence])
        
        if 'masks' in sequence[0]:
            # 对齐所有帧的物体数量
            max_objects = max(f['masks'].shape[0] for f in sequence)
            max_objects = min(max_objects, self.max_objects)
            
            padded_masks = []
            for f in sequence:
                masks = f['masks'][:max_objects]
                if masks.shape[0] < max_objects:
                    padding = torch.zeros(
                        max_objects - masks.shape[0],
                        *masks.shape[1:]
                    )
                    masks = torch.cat([masks, padding], dim=0)
                padded_masks.append(masks)
            
            collated['masks'] = torch.stack(padded_masks)
        
        if 'camera' in sequence[0]:
            collated['cameras'] = [f['camera'] for f in sequence]
        
        return collated
    
    @abstractmethod
    def _load_sample_by_info(self, sample_info: Dict) -> Dict:
        """根据样本信息加载数据"""
        pass


# 工具函数
def load_image(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """加载并调整图像大小"""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Cannot load image: {path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if size is not None:
        image = cv2.resize(image, (size[1], size[0]))
    
    return image


def load_depth(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """加载深度图"""
    # 支持多种深度格式
    if path.endswith('.npy'):
        depth = np.load(path)
    elif path.endswith('.png'):
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
    elif path.endswith('.exr'):
        import OpenEXR
        import Imath
        exr_file = OpenEXR.InputFile(path)
        dw = exr_file.header()['dataWindow']
        size_exr = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        depth = np.frombuffer(
            exr_file.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT)),
            dtype=np.float32
        ).reshape(size_exr)
    else:
        raise ValueError(f"Unsupported depth format: {path}")
    
    if size is not None and depth.shape[:2] != size:
        depth = cv2.resize(depth, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    
    return depth


def load_mask(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """加载mask"""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot load mask: {path}")
    
    mask = (mask > 127).astype(bool)
    
    if size is not None:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (size[1], size[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    
    return mask