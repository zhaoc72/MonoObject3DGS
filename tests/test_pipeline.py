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

def test_single_image():
    """测试单图重建"""
    print("\n=== Testing Single Image Reconstruction ===")
    
    from scripts.single_image import SingleImageReconstructor
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    test_path = "data/examples/test_single.jpg"
    cv2.imwrite(test_path, test_image)
    
    # 初始化（使用简化配置）
    reconstructor = SingleImageReconstructor("configs/single_image.yaml")
    
    # 测试各模块
    print("\n1. Testing segmentation...")
    objects = reconstructor._segment_and_classify(test_image)
    print(f"   Detected {len(objects)} objects")
    
    print("\n2. Testing depth estimation...")
    depth = reconstructor._estimate_depth(test_image)
    print(f"   Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
    
    print("\n✓ Single image pipeline test passed!")

def test_video():
    """测试视频重建"""
    print("\n=== Testing Video Reconstruction ===")
    
    from scripts.video import VideoReconstructor
    
    # 创建测试视频
    test_video = "data/examples/test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video, fourcc, 30.0, (640, 480))
    
    for i in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    # 初始化
    reconstructor = VideoReconstructor("configs/video.yaml")
    
    # 测试处理
    print("\n1. Testing video processing...")
    result = reconstructor.reconstruct(test_video, max_frames=10)
    print(f"   Processed {result['num_frames']} frames")
    print(f"   Keyframes: {len(result['keyframes'])}")
    
    print("\n✓ Video pipeline test passed!")

if __name__ == "__main__":
    # 检查依赖
    print("Checking dependencies...")
    try:
        import torch
        import transformers
        import cv2
        print("✓ Core dependencies OK")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        sys.exit(1)
    
    # 运行测试
    try:
        test_single_image()
    except Exception as e:
        print(f"\n✗ Single image test failed: {e}")
    
    try:
        test_video()
    except Exception as e:
        print(f"\n✗ Video test failed: {e}")
          
    print("\n" + "=" * 70)
    print("✓ All tests completed!")
    print("=" * 70)