"""
CO3D (Common Objects in 3D) v2 Dataset Loader
多视角物体数据集 - 50个类别

数据结构:
co3d/
├── <category>/
│   ├── <sequence_id>/
│   │   ├── images/          # RGB图像
│   │   ├── masks/           # 前景mask
│   │   ├── depths/          # 深度图（部分有）
│   │   └── frame_annotations.jgz  # 标注
│   └── ...
└── set_lists/              # train/val/test划分
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import gzip
import cv2
from .base_dataset import StaticSceneDataset, DynamicSceneDataset, load_image, load_mask


class CO3DDataset(StaticSceneDataset):
    """
    CO3Dv2静态数据集（单帧）
    
    特点:
    - 50个物体类别
    - 多视角数据
    - 精确的相机参数
    - 部分场景有深度
    """
    
    # CO3D v2 类别
    CATEGORIES = [
        'apple', 'backpack', 'banana', 'baseballbat', 'baseballglove',
        'bench', 'bicycle', 'bottle', 'bowl', 'broccoli',
        'cake', 'car', 'carrot', 'cellphone', 'chair',
        'cup', 'donut', 'hairdryer', 'handbag', 'hotdog',
        'hydrant', 'keyboard', 'kite', 'laptop', 'microwave',
        'motorcycle', 'mouse', 'orange', 'parkingmeter', 'pizza',
        'plant', 'remote', 'sandwich', 'skateboard', 'stopsign',
        'suitcase', 'teddybear', 'toaster', 'toilet', 'toybus',
        'toyplane', 'toytrain', 'toytruck', 'tv', 'umbrella',
        'vase', 'wineglass', 'book', 'couch', 'ball'
    ]
    
    def __init__(
        self,
        category: Optional[str] = None,  # 指定类别，None则使用所有
        min_quality: float = 0.5,  # 最小图像质量
        **kwargs
    ):
        self.category_filter = category
        self.min_quality = min_quality
        super().__init__(**kwargs)
    
    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        samples = []
        
        # 加载split列表
        set_list_file = self.data_root / "set_lists" / f"{self.split}.json"
        
        if set_list_file.exists():
            with open(set_list_file, 'r') as f:
                set_lists = json.load(f)
        else:
            # 如果没有官方split，自动发现
            set_lists = self._discover_sequences()
        
        # 遍历类别
        categories = [self.category_filter] if self.category_filter else self.CATEGORIES
        
        for category in categories:
            category_dir = self.data_root / category
            if not category_dir.exists():
                continue
            
            # 获取该类别的序列
            if category in set_lists:
                sequences = set_lists[category]
            else:
                sequences = [d.name for d in category_dir.iterdir() if d.is_dir()]
            
            # 遍历序列
            for seq_name in sequences:
                seq_dir = category_dir / seq_name
                
                # 加载序列标注
                seq_samples = self._load_sequence(seq_dir, category, seq_name)
                samples.extend(seq_samples)
        
        return samples
    
    def _load_sequence(
        self,
        seq_dir: Path,
        category: str,
        seq_name: str
    ) -> List[Dict]:
        """加载单个序列"""
        samples = []
        
        # 加载标注文件
        annotation_file = seq_dir / "frame_annotations.jgz"
        if not annotation_file.exists():
            return samples
        
        try:
            with gzip.open(annotation_file, 'rt') as f:
                annotations = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {annotation_file}: {e}")
            return samples
        
        # 遍历帧
        for frame_ann in annotations:
            frame_id = frame_ann['frame_number']
            image_path = seq_dir / "images" / f"frame{frame_id:06d}.jpg"
            
            if not image_path.exists():
                continue
            
            # 质量过滤
            if frame_ann.get('image_quality_score', 1.0) < self.min_quality:
                continue
            
            sample = {
                'category': category,
                'sequence_name': seq_name,
                'frame_id': frame_id,
                'image_path': str(image_path),
                'mask_path': str(seq_dir / "masks" / f"frame{frame_id:06d}.png"),
                'depth_path': str(seq_dir / "depths" / f"frame{frame_id:06d}.png"),
                'camera': self._parse_camera(frame_ann),
                'quality_score': frame_ann.get('image_quality_score', 1.0)
            }
            
            samples.append(sample)
        
        return samples
    
    def _parse_camera(self, frame_ann: Dict) -> Dict:
        """解析相机参数"""
        camera = frame_ann.get('viewpoint', {})
        
        # CO3D使用标准化的相机参数
        return {
            'R': np.array(camera.get('R', [[1,0,0],[0,1,0],[0,0,1]])),
            'T': np.array(camera.get('T', [0,0,0])),
            'focal_length': np.array(camera.get('focal_length', [1.0, 1.0])),
            'principal_point': np.array(camera.get('principal_point', [0.5, 0.5])),
            'intrinsics': camera.get('intrinsics_format', 'ndc')  # 'ndc' or 'screen'
        }
    
    def _discover_sequences(self) -> Dict:
        """自动发现序列"""
        set_lists = {}
        
        for category in self.CATEGORIES:
            category_dir = self.data_root / category
            if category_dir.exists():
                sequences = [d.name for d in category_dir.iterdir() if d.is_dir()]
                set_lists[category] = sequences
        
        return set_lists
    
    def _load_sample(self, idx: int) -> Dict:
        """加载单个样本"""
        sample_info = self.samples[idx]
        
        # 加载图像
        image = load_image(sample_info['image_path'], self.image_size)
        
        sample = {
            'image': image,
            'scene_id': f"{sample_info['category']}_{sample_info['sequence_name']}",
            'frame_id': sample_info['frame_id'],
            'category': sample_info['category']
        }
        
        # 加载mask
        if self.load_masks and Path(sample_info['mask_path']).exists():
            mask = load_mask(sample_info['mask_path'], self.image_size)
            sample['masks'] = [mask]
            sample['categories'] = [sample_info['category']]
        
        # 加载深度
        if self.load_depth and Path(sample_info['depth_path']).exists():
            depth = cv2.imread(
                sample_info['depth_path'],
                cv2.IMREAD_ANYDEPTH
            ).astype(np.float32) / 1000.0
            
            if depth.shape[:2] != self.image_size:
                depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]))
            
            sample['depth'] = depth
        
        # 相机参数（转换为像素坐标）
        cam = sample_info['camera']
        H, W = self.image_size
        
        if cam['intrinsics'] == 'ndc':
            # NDC坐标转像素坐标
            fx = cam['focal_length'][0] * W / 2
            fy = cam['focal_length'][1] * H / 2
            cx = (cam['principal_point'][0] + 1) * W / 2
            cy = (cam['principal_point'][1] + 1) * H / 2
        else:
            # 已经是像素坐标
            fx = cam['focal_length'][0]
            fy = cam['focal_length'][1]
            cx = cam['principal_point'][0]
            cy = cam['principal_point'][1]
        
        sample['camera'] = {
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'R': cam['R'], 'T': cam['T']
        }
        
        return sample


class CO3DVideoDataset(DynamicSceneDataset):
    """CO3D视频序列数据集"""
    
    CATEGORIES = CO3DDataset.CATEGORIES
    
    def __init__(
        self,
        category: Optional[str] = None,
        min_quality: float = 0.5,
        **kwargs
    ):
        self.category_filter = category
        self.min_quality = min_quality
        super().__init__(**kwargs)
    
    def _load_samples(self) -> List[Dict]:
        """加载序列列表"""
        samples = []
        
        # 遍历类别
        categories = [self.category_filter] if self.category_filter else self.CATEGORIES
        
        for category in categories:
            category_dir = self.data_root / category
            if not category_dir.exists():
                continue
            
            # 遍历序列
            for seq_dir in category_dir.iterdir():
                if not seq_dir.is_dir():
                    continue
                
                # 检查标注文件
                annotation_file = seq_dir / "frame_annotations.jgz"
                if not annotation_file.exists():
                    continue
                
                # 加载标注统计帧数
                try:
                    with gzip.open(annotation_file, 'rt') as f:
                        annotations = json.load(f)
                    
                    num_frames = len(annotations)
                    
                    # 确保有足够的帧
                    if num_frames < self.sequence_length:
                        continue
                    
                    sample = {
                        'category': category,
                        'sequence_name': seq_dir.name,
                        'sequence_dir': seq_dir,
                        'num_frames': num_frames
                    }
                    
                    samples.append(sample)
                
                except Exception as e:
                    continue
        
        return samples
    
    def _load_sample_by_info(self, sample_info: Dict) -> Dict:
        """加载序列中的单帧"""
        # 复用静态数据集的逻辑
        static_dataset = CO3DDataset(
            data_root=self.data_root,
            split=self.split,
            image_size=self.image_size,
            category=sample_info['category'],
            load_depth=self.load_depth,
            load_masks=self.load_masks,
            augmentation=False
        )
        
        # 加载该序列的所有帧
        seq_samples = static_dataset._load_sequence(
            sample_info['sequence_dir'],
            sample_info['category'],
            sample_info['sequence_name']
        )
        
        # 找到对应的帧
        frame_id = sample_info.get('frame_id', 0)
        for s in seq_samples:
            if s['frame_id'] == frame_id:
                static_dataset.samples = [s]
                return static_dataset._load_sample(0)
        
        # 如果找不到，返回第一帧
        if seq_samples:
            static_dataset.samples = [seq_samples[0]]
            return static_dataset._load_sample(0)
        
        # 返回空样本
        return {
            'image': np.zeros((*self.image_size, 3), dtype=np.uint8),
            'scene_id': sample_info['sequence_name'],
            'frame_id': 0
        }


def prepare_co3d(
    co3d_root: str,
    output_root: str,
    categories: Optional[List[str]] = None
):
    """预处理CO3D数据"""
    from tqdm import tqdm
    
    co3d_path = Path(co3d_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Preparing CO3Dv2 Dataset")
    print("=" * 70)
    
    if categories is None:
        categories = CO3DDataset.CATEGORIES
    
    stats = {
        'categories': {},
        'total_sequences': 0,
        'total_frames': 0
    }
    
    for category in tqdm(categories, desc="Categories"):
        cat_dir = co3d_path / category
        if not cat_dir.exists():
            continue
        
        cat_stats = {
            'sequences': 0,
            'frames': 0,
            'with_depth': 0
        }
        
        for seq_dir in cat_dir.iterdir():
            if not seq_dir.is_dir():
                continue
            
            annotation_file = seq_dir / "frame_annotations.jgz"
            if not annotation_file.exists():
                continue
            
            try:
                with gzip.open(annotation_file, 'rt') as f:
                    annotations = json.load(f)
                
                num_frames = len(annotations)
                cat_stats['sequences'] += 1
                cat_stats['frames'] += num_frames
                
                # 检查深度
                depth_dir = seq_dir / "depths"
                if depth_dir.exists() and len(list(depth_dir.glob("*.png"))) > 0:
                    cat_stats['with_depth'] += 1
            
            except:
                continue
        
        stats['categories'][category] = cat_stats
        stats['total_sequences'] += cat_stats['sequences']
        stats['total_frames'] += cat_stats['frames']
    
    # 打印统计
    print(f"\n=== Overall Statistics ===")
    print(f"Total sequences: {stats['total_sequences']}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Avg frames/sequence: {stats['total_frames']/max(1, stats['total_sequences']):.1f}")
    
    print(f"\n=== Per-Category Statistics ===")
    for cat, cat_stats in sorted(stats['categories'].items(), key=lambda x: -x[1]['frames']):
        print(f"  {cat:20s}: {cat_stats['sequences']:4d} seqs, "
              f"{cat_stats['frames']:6d} frames, "
              f"{cat_stats['with_depth']:4d} with depth")
    
    # 保存统计
    with open(output_path / 'co3d_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✓ CO3D preparation completed!")
    print("=" * 70)


if __name__ == "__main__":
    # 测试
    dataset = CO3DDataset(
        data_root="/path/to/co3d",
        split="train",
        category="chair",  # 测试单个类别
        image_size=(512, 512)
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample: {sample.keys()}")
        print(f"Category: {sample['category']}")
        print(f"Image shape: {sample['image'].shape}")