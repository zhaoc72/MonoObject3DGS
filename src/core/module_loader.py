"""
Module Loader - 灵活加载模型
支持不同配置和消融实验
"""

import torch
import numpy as np
from typing import Dict, Optional, Any
from pathlib import Path
import yaml


class ModuleConfig:
    """模块配置类"""
    
    def __init__(self, config_dict: Dict):
        self.config = config_dict
    
    def is_enabled(self, module_name: str) -> bool:
        """检查模块是否启用"""
        return self.config.get(module_name, {}).get('enabled', True)
    
    def get(self, module_name: str, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(module_name, {}).get(key, default)


class DINOv2Loader:
    """DINOv2加载器"""
    
    @staticmethod
    def load(config: ModuleConfig, device: str = "cuda"):
        """根据配置加载DINOv2"""
        if not config.is_enabled('dinov2'):
            print("⚠️  DINOv2 disabled - using dummy extractor")
            return DummyFeatureExtractor(device)
        
        model_size = config.get('dinov2', 'model_size', 'base')
        
        # 模型映射
        model_map = {
            'large': 'facebook/dinov2-large',
            'base': 'facebook/dinov2-base',
            'small': 'facebook/dinov2-small'
        }
        
        feature_dim_map = {
            'large': 1024,
            'base': 768,
            'small': 384
        }
        
        model_name = model_map.get(model_size, model_map['base'])
        feature_dim = feature_dim_map.get(model_size, 768)
        
        print(f"🔄 Loading DINOv2: {model_size} ({feature_dim}D)")
        
        from src.segmentation.dinov2_extractor_v2 import DINOv2ExtractorV2
        
        return DINOv2ExtractorV2(
            model_name=model_name,
            feature_dim=feature_dim,
            device=device,
            use_registers=config.get('dinov2', 'use_registers', True),
            enable_xformers=config.get('dinov2', 'enable_xformers', True)
        )


class SAM2Loader:
    """SAM 2加载器"""
    
    @staticmethod
    def load(config: ModuleConfig, device: str = "cuda", mode: str = "image"):
        """根据配置加载SAM 2"""
        if not config.is_enabled('sam2'):
            print("⚠️  SAM 2 disabled - using dummy segmenter")
            return DummySegmenter(device)
        
        model_size = config.get('sam2', 'model_size', 'base')
        
        # 检查点映射
        checkpoint_map = {
            'large': 'sam2_hiera_large.pt',
            'base+': 'sam2_hiera_base_plus.pt',
            'base': 'sam2_hiera_base.pt',
            'small': 'sam2_hiera_small.pt',
            'tiny': 'sam2_hiera_tiny.pt'
        }
        
        checkpoint_file = checkpoint_map.get(model_size, checkpoint_map['base'])
        checkpoint_path = f"data/checkpoints/{checkpoint_file}"
        
        print(f"🔄 Loading SAM 2: {model_size}")
        
        from src.segmentation.sam2_segmenter import SAM2Segmenter
        
        return SAM2Segmenter(
            model_size=model_size,
            checkpoint=checkpoint_path,
            device=device,
            mode=mode
        )


class DepthLoader:
    """Depth Anything V2加载器"""
    
    @staticmethod
    def load(config: ModuleConfig, device: str = "cuda"):
        """根据配置加载Depth Anything V2"""
        if not config.is_enabled('depth'):
            print("⚠️  Depth estimation disabled - using fallback")
            return DummyDepthEstimator(config, device)
        
        model_size = config.get('depth', 'model_size', 'vitb')
        
        print(f"🔄 Loading Depth Anything V2: {model_size}")
        
        from src.depth.depth_anything_v2_upgraded import DepthAnythingV2Upgraded
        
        return DepthAnythingV2Upgraded(
            model_size=model_size,
            metric_depth=config.get('depth', 'metric_depth', True),
            device=device,
            max_depth=config.get('depth', 'max_depth', 20.0)
        )


class DummyFeatureExtractor:
    """Dummy特征提取器（消融实验用）"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.feature_dim = 768
        print("  Using dummy feature extractor")
    
    def extract_features(self, image: np.ndarray, **kwargs) -> Dict:
        """返回随机特征"""
        H, W = image.shape[:2]
        patch_h = patch_w = H // 14
        return {
            'cls_token': torch.randn(1, self.feature_dim).to(self.device),
            'patch_tokens': torch.randn(1, patch_h * patch_w, self.feature_dim).to(self.device),
            'patch_h': patch_h,
            'patch_w': patch_w
        }
    
    def get_dense_features(self, image: np.ndarray, target_size=None) -> torch.Tensor:
        """返回随机密集特征"""
        if target_size is None:
            target_size = image.shape[:2]
        H, W = target_size
        return torch.randn(1, self.feature_dim, H, W).to(self.device)
    
    def get_multi_scale_features(self, image: np.ndarray, scales=None):
        """多尺度dummy特征"""
        if scales is None:
            scales = [1.0]
        
        H, W = image.shape[:2]
        result = {}
        for scale in scales:
            result[scale] = torch.randn(1, self.feature_dim, H, W).to(self.device)
        return result


class DummySegmenter:
    """Dummy分割器（消融实验用）"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("  Using dummy segmenter")
    
    def segment_automatic(self, image: np.ndarray, **kwargs) -> list:
        """返回简单的网格分割"""
        H, W = image.shape[:2]
        
        # 简单的4x4网格分割
        masks = []
        grid_h, grid_w = H // 4, W // 4
        
        for i in range(4):
            for j in range(4):
                mask = np.zeros((H, W), dtype=bool)
                y1, y2 = i * grid_h, (i + 1) * grid_h
                x1, x2 = j * grid_w, (j + 1) * grid_w
                mask[y1:y2, x1:x2] = True
                
                masks.append({
                    'segmentation': mask,
                    'area': int(mask.sum()),
                    'bbox': [x1, y1, grid_w, grid_h],
                    'predicted_iou': 0.7,
                    'stability_score': 0.8
                })
        
        return masks
    
    def refine_with_features(self, masks: list, features, **kwargs) -> list:
        """不做任何优化"""
        return masks


class DummyDepthEstimator:
    """Dummy深度估计器（消融实验用）"""
    
    def __init__(self, config: ModuleConfig, device: str = "cuda"):
        self.device = device
        self.config = config
        
        # 获取fallback配置
        fallback_method = config.get('depth', 'fallback', {}).get('method', 'uniform')
        self.default_depth = config.get('depth', 'fallback', {}).get('default_depth', 5.0)
        self.depth_range = config.get('depth', 'fallback', {}).get('depth_range', [2.0, 8.0])
        self.method = fallback_method
        
        print(f"  Using fallback depth: {fallback_method} (depth={self.default_depth}m)")
    
    def estimate(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """返回固定或随机深度"""
        H, W = image.shape[:2]
        
        if self.method == 'uniform':
            # 统一深度
            depth = np.ones((H, W)) * self.default_depth
        elif self.method == 'random':
            # 随机深度
            depth = np.random.uniform(
                self.depth_range[0], 
                self.depth_range[1], 
                (H, W)
            )
        elif self.method == 'plane':
            # 平面深度（简单的透视）
            y_coords = np.arange(H).reshape(-1, 1).repeat(W, axis=1)
            depth = self.default_depth * (1 + (y_coords / H) * 0.5)
        else:
            depth = np.ones((H, W)) * self.default_depth
        
        return depth.astype(np.float32)
    
    def estimate_with_confidence(self, image: np.ndarray, **kwargs):
        """返回固定深度+低置信度"""
        depth = self.estimate(image)
        confidence = np.ones_like(depth) * 0.3  # 低置信度
        return depth, confidence
    
    def estimate_multi_scale(self, image: np.ndarray, **kwargs):
        """多尺度也返回同样的深度"""
        return self.estimate(image)


class ModuleLoader:
    """统一的模块加载器"""
    
    def __init__(self, config_path: str):
        """
        初始化模块加载器
        
        Args:
            config_path: 配置文件路径（支持模式配置）
        """
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        self.config = ModuleConfig(raw_config)
        self.mode_name = self.config.get('mode', 'name', 'unknown')
        self.mode_description = self.config.get('mode', 'description', '')
        
        print("=" * 70)
        print(f"📦 Module Loader Initialized")
        print(f"   Mode: {self.mode_name}")
        print(f"   Description: {self.mode_description}")
        print("=" * 70)
    
    def load_all(self, device: str = "cuda") -> Dict:
        """加载所有模块"""
        modules = {}
        
        print("\n🔧 Loading modules...")
        
        # 加载DINOv2
        print("\n[1/3] DINOv2:")
        modules['dinov2'] = DINOv2Loader.load(self.config, device)
        
        # 加载SAM 2
        print("\n[2/3] SAM 2:")
        modules['sam2'] = SAM2Loader.load(self.config, device, mode='image')
        
        # 加载Depth
        print("\n[3/3] Depth Estimator:")
        modules['depth'] = DepthLoader.load(self.config, device)
        
        print("\n" + "=" * 70)
        print("✓ All modules loaded")
        print("=" * 70)
        
        return modules
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        return {
            'mode': self.mode_name,
            'expected_fps': self.config.get('performance', 'expected_fps', 'N/A'),
            'gpu_memory': self.config.get('performance', 'gpu_memory', 'N/A'),
            'accuracy_level': self.config.get('performance', 'accuracy_level', 'N/A'),
            'dinov2_enabled': self.config.is_enabled('dinov2'),
            'sam2_enabled': self.config.is_enabled('sam2'),
            'depth_enabled': self.config.is_enabled('depth')
        }


# 测试代码
if __name__ == "__main__":
    print("=== Testing Module Loader ===\n")
    
    # 测试不同模式
    modes = [
        'configs/modes/high_accuracy.yaml',
        'configs/modes/real_time.yaml',
        'configs/modes/balanced.yaml',
        'configs/modes/ablation_no_dinov2.yaml',
        'configs/modes/ablation_no_depth.yaml'
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for mode_path in modes:
        if Path(mode_path).exists():
            print(f"\n{'='*70}")
            print(f"Testing: {mode_path}")
            print('='*70)
            
            loader = ModuleLoader(mode_path)
            modules = loader.load_all(device)
            
            stats = loader.get_performance_stats()
            print(f"\n📊 Performance Stats:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            
            print(f"\n✓ {mode_path} tested successfully")
        else:
            print(f"⚠️  Config not found: {mode_path}")