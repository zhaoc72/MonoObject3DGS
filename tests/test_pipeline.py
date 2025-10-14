"""
Pipeline Test
测试完整流程
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import cv2


def test_dependencies():
    """测试依赖"""
    print("\n=== Testing Dependencies ===")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except:
        print("✗ PyTorch")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except:
        print("✗ OpenCV")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except:
        print("✗ Transformers")
        return False
    
    return True


def test_modules():
    """测试模块导入"""
    print("\n=== Testing Module Imports ===")
    
    try:
        from src.segmentation.dinov2_extractor import DINOv2Extractor
        print("✓ DINOv2Extractor")
    except Exception as e:
        print(f"✗ DINOv2Extractor: {e}")
        return False
    
    try:
        from src.depth.depth_estimator import DepthEstimator
        print("✓ DepthEstimator")
    except Exception as e:
        print(f"✗ DepthEstimator: {e}")
        return False
    
    try:
        from src.priors.explicit_prior import ExplicitShapePrior
        print("✓ ExplicitShapePrior")
    except Exception as e:
        print(f"✗ ExplicitShapePrior: {e}")
        return False
    
    try:
        from src.reconstruction.gaussian_model import GaussianModel
        print("✓ GaussianModel")
    except Exception as e:
        print(f"✗ GaussianModel: {e}")
        return False
    
    return True


def test_single_image_pipeline():
    """测试单图流程"""
    print("\n=== Testing Single Image Pipeline ===")
    
    try:
        from scripts.single_image import SingleImageReconstructor
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        test_path = "data/examples/test_single.jpg"
        Path("data/examples").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(test_path, test_image)
        
        print("✓ Single image pipeline imports successful")
        return True
        
    except Exception as e:
        print(f"✗ Single image pipeline: {e}")
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("MonoObject3DGS - Pipeline Test")
    print("=" * 70)
    
    success = True
    
    if not test_dependencies():
        print("\n✗ Dependency test failed")
        success = False
    
    if not test_modules():
        print("\n✗ Module import test failed")
        success = False
    
    if not test_single_image_pipeline():
        print("\n✗ Pipeline test failed")
        success = False
    
    print("\n" + "=" * 70)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 70)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())