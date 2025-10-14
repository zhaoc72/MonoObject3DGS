"""
Data Preprocessing
数据预处理工具
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import argparse
from tqdm import tqdm
import json


def preprocess_images(input_dir: str, output_dir: str, target_size: tuple = (640, 480)):
    """
    预处理图像序列
    - 调整大小
    - 去模糊检测
    - 提取关键帧
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))
    
    print(f"Preprocessing {len(image_files)} images...")
    
    processed_info = []
    
    for i, img_file in enumerate(tqdm(image_files)):
        # 读取图像
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        # 调整大小
        if image.shape[:2] != target_size:
            image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # 模糊检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 保存
        output_file = output_path / f"{i:05d}.jpg"
        cv2.imwrite(str(output_file), image)
        
        processed_info.append({
            'frame_id': i,
            'filename': output_file.name,
            'blur_score': float(blur_score),
            'is_blurry': blur_score < 100
        })
    
    # 保存元数据
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(processed_info, f, indent=2)
    
    print(f"✓ Processed {len(processed_info)} images")
    print(f"  Blurry images: {sum(1 for x in processed_info if x['is_blurry'])}")


def extract_keyframes(video_path: str, output_dir: str, interval: int = 10):
    """
    从视频提取关键帧
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Extracting keyframes from {video_path}")
    print(f"  Total frames: {total_frames}, FPS: {fps:.1f}")
    print(f"  Keyframe interval: {interval}")
    
    frame_id = 0
    keyframe_id = 0
    
    pbar = tqdm(total=total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id % interval == 0:
            output_file = output_path / f"keyframe_{keyframe_id:05d}.jpg"
            cv2.imwrite(str(output_file), frame)
            keyframe_id += 1
        
        frame_id += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
    print(f"✓ Extracted {keyframe_id} keyframes")


def main():
    parser = argparse.ArgumentParser(description='Data Preprocessing')
    parser.add_argument('--mode', type=str, choices=['images', 'video'], required=True)
    parser.add_argument('--input', type=str, required=True, help='Input directory or video file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--interval', type=int, default=10, help='Keyframe interval for video')
    parser.add_argument('--size', type=str, default='640x480', help='Target size (WxH)')
    
    args = parser.parse_args()
    
    # 解析尺寸
    w, h = map(int, args.size.split('x'))
    target_size = (h, w)
    
    if args.mode == 'images':
        preprocess_images(args.input, args.output, target_size)
    elif args.mode == 'video':
        extract_keyframes(args.input, args.output, args.interval)


if __name__ == "__main__":
    main()