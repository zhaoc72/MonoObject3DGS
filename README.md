"# MonoObject3DGS" 
# MonoObject3DGS

**单目物体级3D重建系统** - 基于DINOv2+SAM+3D Gaussian Splatting的语义感知重建

## 🌟 核心特性

- **语义感知分割**：DINOv2特征 + SAM高质量分割
- **物体级重建**：独立重建每个物体，而非整个场景
- **自适应形状先验**：显式CAD模板 + 隐式学习先验的自适应融合
- **单图/视频支持**：同时支持单张图片和视频序列输入
- **实时处理**：视频模式下使用FastSAM实现实时分割

## 📋 系统架构


# Flexible Configuration Guide

## 🚀 Quick Start

### 1. 高准确度模式（推荐用于研究）
```bash
python scripts/reconstruct_flexible.py \
    --image data/test.jpg \
    --mode high_accuracy


python scripts/reconstruct_flexible.py \
    --image data/test.jpg \
    --mode real_time

python scripts/reconstruct_flexible.py \
    --image data/test.jpg \
    --mode balanced
    
# 无DINOv2
python scripts/reconstruct_flexible.py \
    --image data/test.jpg \
    --mode ablation_no_dinov2

# 无深度估计
python scripts/reconstruct_flexible.py \
    --image data/test.jpg \
    --mode ablation_no_depth

# 最小配置
python scripts/reconstruct_flexible.py \
    --image data/test.jpg \
    --mode ablation_minimal