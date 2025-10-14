"""
Visualization Tools
可视化重建结果
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import json


def visualize_scene(scene_dir: str):
    """可视化重建场景"""
    scene_path = Path(scene_dir)
    
    # 加载元数据
    with open(scene_path / 'scene_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print("=" * 70)
    print(f"Scene: {scene_dir}")
    print("=" * 70)
    print(f"Objects: {metadata['num_objects']}")
    print(f"Total Gaussians: {metadata['total_gaussians']}")
    print("\nObject details:")
    
    for obj_id, obj_info in metadata['objects'].items():
        print(f"  Object {obj_id}:")
        print(f"    Category: {obj_info['category']}")
        print(f"    Gaussians: {obj_info['num_gaussians']}")
        print(f"    Confidence: {obj_info.get('confidence', 'N/A')}")


def visualize_pointcloud(scene_dir: str, object_id: int = None):
    """可视化3D点云"""
    scene_path = Path(scene_dir)
    
    # 这里需要加载实际的点云数据
    # 简化版本：从PLY文件加载
    
    try:
        from plyfile import PlyData
        
        if object_id is not None:
            ply_file = scene_path / f"object_{object_id}.ply"
        else:
            # 可视化第一个物体
            ply_files = list(scene_path.glob("object_*.ply"))
            if not ply_files:
                print("No PLY files found")
                return
            ply_file = ply_files[0]
        
        print(f"Loading {ply_file}...")
        plydata = PlyData.read(str(ply_file))
        
        # 提取顶点
        vertices = plydata['vertex']
        x = vertices['x']
        y = vertices['y']
        z = vertices['z']
        
        # 可视化
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制点云
        ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Point Cloud - {ply_file.name}')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("plyfile not installed. Install with: pip install plyfile")
    except Exception as e:
        print(f"Error: {e}")


def create_comparison_video(
    original_video: str,
    keyframe_dir: str,
    output_video: str
):
    """创建对比视频（原始 vs 分割）"""
    cap = cv2.VideoCapture(original_video)
    if not cap.isOpened():
        print(f"Cannot open video: {original_video}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 输出视频（并排显示）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width * 2, height))
    
    keyframe_path = Path(keyframe_dir)
    keyframe_files = sorted(keyframe_path.glob("*.jpg"))
    
    frame_id = 0
    
    print(f"Creating comparison video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 查找对应的关键帧可视化
        if frame_id < len(keyframe_files):
            seg_frame = cv2.imread(str(keyframe_files[frame_id]))
            if seg_frame is not None:
                seg_frame = cv2.resize(seg_frame, (width, height))
            else:
                seg_frame = frame.copy()
        else:
            seg_frame = frame.copy()
        
        # 并排显示
        combined = np.hstack([frame, seg_frame])
        
        # 添加标签
        cv2.putText(combined, 'Original', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, 'Segmentation', (width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(combined)
        frame_id += 1
    
    cap.release()
    out.release()
    
    print(f"✓ Comparison video saved to: {output_video}")


def plot_statistics(scene_dir: str):
    """绘制统计图表"""
    scene_path = Path(scene_dir)
    
    # 加载统计数据
    with open(scene_path / 'statistics.json', 'r') as f:
        stats = json.load(f)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 物体类别分布
    if 'objects' in stats:
        categories = {}
        for obj_info in stats['objects'].values():
            cat = obj_info['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        axes[0, 0].bar(categories.keys(), categories.values())
        axes[0, 0].set_title('Object Category Distribution')
        axes[0, 0].set_xlabel('Category')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Gaussian数量分布
    if 'objects' in stats:
        gaussian_counts = [obj['num_gaussians'] for obj in stats['objects'].values()]
        axes[0, 1].hist(gaussian_counts, bins=20, edgecolor='black')
        axes[0, 1].set_title('Gaussians per Object Distribution')
        axes[0, 1].set_xlabel('Number of Gaussians')
        axes[0, 1].set_ylabel('Frequency')
    
    # 3. 物体大小（Gaussian数量）
    if 'objects' in stats:
        obj_ids = list(stats['objects'].keys())
        gaussian_counts = [stats['objects'][oid]['num_gaussians'] for oid in obj_ids]
        
        axes[1, 0].barh(obj_ids[:10], gaussian_counts[:10])  # 显示前10个
        axes[1, 0].set_title('Top 10 Objects by Gaussians')
        axes[1, 0].set_xlabel('Number of Gaussians')
        axes[1, 0].set_ylabel('Object ID')
    
    # 4. 总体信息
    info_text = f"""
    Total Objects: {stats['num_objects']}
    Total Gaussians: {stats['total_gaussians']}
    Avg Gaussians/Object: {stats['total_gaussians']/stats['num_objects']:.1f}
    """
    
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(scene_path / 'statistics.png', dpi=150)
    plt.show()
    
    print(f"✓ Statistics plot saved to: {scene_path / 'statistics.png'}")


def main():
    parser = argparse.ArgumentParser(description='Visualization Tools')
    parser.add_argument('--mode', type=str, 
                       choices=['scene', 'pointcloud', 'comparison', 'stats'],
                       required=True)
    parser.add_argument('--scene-dir', type=str, help='Scene directory')
    parser.add_argument('--object-id', type=int, help='Object ID for pointcloud')
    parser.add_argument('--original-video', type=str, help='Original video for comparison')
    parser.add_argument('--keyframe-dir', type=str, help='Keyframe directory')
    parser.add_argument('--output', type=str, help='Output file')
    
    args = parser.parse_args()
    
    if args.mode == 'scene':
        if not args.scene_dir:
            print("Error: --scene-dir required")
            return
        visualize_scene(args.scene_dir)
    
    elif args.mode == 'pointcloud':
        if not args.scene_dir:
            print("Error: --scene-dir required")
            return
        visualize_pointcloud(args.scene_dir, args.object_id)
    
    elif args.mode == 'comparison':
        if not all([args.original_video, args.keyframe_dir, args.output]):
            print("Error: --original-video, --keyframe-dir, --output required")
            return
        create_comparison_video(args.original_video, args.keyframe_dir, args.output)
    
    elif args.mode == 'stats':
        if not args.scene_dir:
            print("Error: --scene-dir required")
            return
        plot_statistics(args.scene_dir)


if __name__ == "__main__":
    main()