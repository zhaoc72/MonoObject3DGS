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



# MonoObject3DGS (Restructured)


一个可扩展的单目（单图/视频）3D Gaussian Splatting (3DGS) 重建框架，集成 DINOv2+SAM/FastSAM
先验融合与可配置数据管线。此版本为工程化重构：数据/管线/模型分层、配置可组合、脚本简洁、测试可跑。


## 支持数据集（本版本适配骨架）
- 静态（单图/若干图，不含时序）：ScanNet、Pix3D、CO3Dv2、ImageNet3D、KITTI
- 动态（视频/多视角有时序）：ScanNet、CO3Dv2、KITTI、vKITTI2


> 说明：本仓库提供**统一接口与最小实现**。实际训练/评测需自行准备数据与权重。没有 GPU/torch 时
也能在 CPU 上以占位逻辑跑通流程与测试。


## 快速开始
```bash
# 单图重建（balanced 模式）
python scripts/reconstruct_flexible.py --image assets/example.jpg --mode balanced


# 视频重建
python scripts/reconstruct_flexible.py --video assets/example.mp4 --mode balanced


# 训练（先验适配占位循环）
python scripts/train.py --dataset co3dv2