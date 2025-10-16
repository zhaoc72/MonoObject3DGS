"""
Pix3D Dataset Loader
单物体3D重建数据集 - 包含精确的3D CAD模型和2D标注

数据结构:
pix3d/
├── img/                 # RGB图像
├── mask/                # 实例分割mask
├── model/               # 3D CAD模型
└── pix3d.json          # 标注文件
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import cv2
from .base_dataset import StaticSceneDataset, load_image, load_mask


class Pix3DDataset(StaticSceneDataset):
    """
    Pix3D数据集
    
    特点:
    - 单物体场景
    - 精确的3D CAD模型
    - 准确的相机参数和姿态
    - 9个类别: bed, bookcase, chair, desk, misc, sofa, table, tool, wardrobe
    """
    
    def __init__(self, **kwargs):
        # Pix3D类别
        self.categories = [
            'bed', 'bookcase', 'chair', 'desk', 'misc',
            'sofa', 'table', 'tool', 'wardrobe'
        ]
        
        super().__init__(**kwargs)
    
    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        # 加载标注文件
        annotation_file = self.data_root / "pix3d.json"
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        print(f"  Total annotations: {len(annotations)}")
        
        # 过滤并创建样本
        samples = []
        
        for idx, ann in enumerate(annotations):
            # 根据split过滤
            if 'split' in ann:
                if ann['split'] != self.split:
                    continue
            else:
                # 如果没有split信息，使用简单划分
                if self.split == 'train' and idx % 10 < 8:
                    pass
                elif self.split == 'val' and idx % 10 == 8:
                    pass
                elif self.split == 'test' and idx % 10 == 9:
                    pass
                else:
                    continue
            
            # 检查文件是否存在
            img_path = self.data_root / ann['img']
            mask_path = self.data_root / ann['mask']
            model_path = self.data_root / ann['model']
            
            if not img_path.exists():
                continue
            
            sample = {
                'idx': idx,
                'image_path': str(img_path),
                'mask_path': str(mask_path) if mask_path.exists() else None,
                'model_path': str(model_path) if model_path.exists() else None,
                'category': ann['category'],
                'bbox': ann['bbox'],  # [x, y, width, height]
                'focal_length': ann.get('focal_length', 35.0),
                'inplane_rotation': ann.get('inplane_rotation', 0.0),
                'truncated': ann.get('truncated', False),
                'occluded': ann.get('occluded', False),
                'slightly_occluded': ann.get('slightly_occluded', False)
            }
            
            # 3D模型信息
            if 'model' in ann:
                sample['model_info'] = {
                    'scale': ann.get('3d_model_scale', 1.0),
                    'rotation': ann.get('rot_mat', None),
                    'translation': ann.get('trans_mat', None)
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
            'scene_id': f"pix3d_{sample_info['idx']}",
            'frame_id': 0,  # 单帧
            'category': sample_info['category']
        }
        
        # 加载mask
        if self.load_masks and sample_info['mask_path'] is not None:
            mask = load_mask(sample_info['mask_path'], self.image_size)
            
            # Pix3D是单物体，所以只有一个mask
            sample['masks'] = [mask]
            sample['categories'] = [sample_info['category']]
        
        # 估计深度（Pix3D没有真实深度，需要从3D模型和相机参数计算）
        if self.load_depth:
            # 简化：使用固定深度或从bbox估计
            depth = self._estimate_depth_from_bbox(
                sample_info['bbox'],
                sample_info['focal_length'],
                image.shape[:2]
            )
            sample['depth'] = depth
        
        # 相机参数
        H, W = self.image_size
        focal_length = sample_info['focal_length']
        
        # 从焦距计算内参（假设35mm传感器）
        sensor_width = 36.0  # mm
        fx = fy = (focal_length / sensor_width) * W
        
        sample['camera'] = {
            'fx': fx,
            'fy': fy,
            'cx': W / 2.0,
            'cy': H / 2.0,
            'focal_length': focal_length,
            'inplane_rotation': sample_info['inplane_rotation']
        }
        
        # 3D模型信息（用于形状先验）
        if 'model_info' in sample_info:
            sample['model_info'] = sample_info['model_info']
            sample['model_path'] = sample_info['model_path']
        
        # 质量标记
        sample['quality'] = {
            'truncated': sample_info['truncated'],
            'occluded': sample_info['occluded'],
            'slightly_occluded': sample_info['slightly_occluded']
        }
        
        return sample
    
    def _estimate_depth_from_bbox(
        self,
        bbox: List[float],
        focal_length: float,
        image_size: tuple
    ) -> np.ndarray:
        """从bbox估计深度"""
        H, W = image_size
        x, y, w, h = bbox
        
        # 创建深度图
        depth = np.ones((H, W), dtype=np.float32) * 10.0  # 背景深度
        
        # 物体区域深度（基于bbox大小的简单估计）
        # 假设物体的真实大小约1-2米，根据投影大小反推深度
        object_size = 1.5  # 假设平均物体尺寸
        sensor_width = 36.0
        fx = (focal_length / sensor_width) * W
        
        # 深度 = (真实尺寸 * 焦距) / 像素尺寸
        estimated_depth = (object_size * fx) / max(w, h)
        estimated_depth = np.clip(estimated_depth, 1.0, 10.0)
        
        # 填充物体区域
        y1, y2 = int(y), int(y + h)
        x1, x2 = int(x), int(x + w)
        
        y1, y2 = max(0, y1), min(H, y2)
        x1, x2 = max(0, x1), min(W, x2)
        
        depth[y1:y2, x1:x2] = estimated_depth
        
        return depth
    
    def load_3d_model(self, model_path: str) -> Optional[np.ndarray]:
        """加载3D CAD模型"""
        try:
            import trimesh
            mesh = trimesh.load(model_path)
            
            # 提取顶点
            vertices = np.array(mesh.vertices, dtype=np.float32)
            return vertices
        
        except Exception as e:
            print(f"Warning: Failed to load 3D model {model_path}: {e}")
            return None


def prepare_pix3d(
    pix3d_root: str,
    output_root: str
):
    """
    预处理Pix3D数据
    - 验证数据完整性
    - 生成统计信息
    - 创建split文件
    """
    import shutil
    from tqdm import tqdm
    
    pix3d_path = Path(pix3d_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Preparing Pix3D Dataset")
    print("=" * 70)
    
    # 加载标注
    annotation_file = pix3d_path / "pix3d.json"
    if not annotation_file.exists():
        print(f"Error: Annotation file not found: {annotation_file}")
        return
    
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"\nTotal annotations: {len(annotations)}")
    
    # 统计信息
    stats = {
        'total': len(annotations),
        'categories': {},
        'with_mask': 0,
        'with_model': 0,
        'truncated': 0,
        'occluded': 0,
        'splits': {'train': 0, 'val': 0, 'test': 0}
    }
    
    valid_samples = []
    
    for ann in tqdm(annotations, desc="Processing"):
        # 检查文件存在性
        img_path = pix3d_path / ann['img']
        mask_path = pix3d_path / ann['mask']
        model_path = pix3d_path / ann['model']
        
        if not img_path.exists():
            continue
        
        valid_samples.append(ann)
        
        # 统计类别
        category = ann['category']
        stats['categories'][category] = stats['categories'].get(category, 0) + 1
        
        # 统计mask和模型
        if mask_path.exists():
            stats['with_mask'] += 1
        if model_path.exists():
            stats['with_model'] += 1
        
        # 统计遮挡
        if ann.get('truncated', False):
            stats['truncated'] += 1
        if ann.get('occluded', False):
            stats['occluded'] += 1
        
        # 统计split
        split = ann.get('split', 'train')
        stats['splits'][split] = stats['splits'].get(split, 0) + 1
    
    stats['valid'] = len(valid_samples)
    
    # 打印统计
    print(f"\n=== Statistics ===")
    print(f"Valid samples: {stats['valid']} / {stats['total']}")
    print(f"With mask: {stats['with_mask']} ({stats['with_mask']/stats['valid']*100:.1f}%)")
    print(f"With model: {stats['with_model']} ({stats['with_model']/stats['valid']*100:.1f}%)")
    print(f"Truncated: {stats['truncated']}")
    print(f"Occluded: {stats['occluded']}")
    
    print(f"\n=== Category Distribution ===")
    for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
        print(f"  {cat:15s}: {count:4d} ({count/stats['valid']*100:.1f}%)")
    
    print(f"\n=== Split Distribution ===")
    for split, count in stats['splits'].items():
        print(f"  {split:10s}: {count:4d} ({count/stats['valid']*100:.1f}%)")
    
    # 保存统计
    with open(output_path / 'pix3d_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 如果没有split信息，创建split
    if all(stats['splits'][s] == 0 for s in ['train', 'val', 'test']):
        print("\n=== Creating splits ===")
        # 按类别分层划分
        splits = {'train': [], 'val': [], 'test': []}
        
        for category in stats['categories'].keys():
            cat_samples = [s for s in valid_samples if s['category'] == category]
            n = len(cat_samples)
            
            # 80% train, 10% val, 10% test
            n_train = int(n * 0.8)
            n_val = int(n * 0.1)
            
            for i, sample in enumerate(cat_samples):
                if i < n_train:
                    sample['split'] = 'train'
                    splits['train'].append(sample)
                elif i < n_train + n_val:
                    sample['split'] = 'val'
                    splits['val'].append(sample)
                else:
                    sample['split'] = 'test'
                    splits['test'].append(sample)
        
        # 保存更新的标注
        with open(output_path / 'pix3d_with_splits.json', 'w') as f:
            json.dump(valid_samples, f, indent=2)
        
        print(f"  Train: {len(splits['train'])}")
        print(f"  Val: {len(splits['val'])}")
        print(f"  Test: {len(splits['test'])}")
    
    print("\n" + "=" * 70)
    print("✓ Pix3D preparation completed!")
    print(f"  Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    # 测试数据加载
    dataset = Pix3DDataset(
        data_root="/path/to/pix3d",
        split="train",
        image_size=(512, 512),
        load_depth=True,
        load_masks=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Category: {sample['category']}")
        if 'depth' in sample:
            print(f"Depth shape: {sample['depth'].shape}")
        if 'masks' in sample:
            print(f"Num masks: {len(sample['masks'])}")