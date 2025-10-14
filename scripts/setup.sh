#!/bin/bash

echo "========================================="
echo "MonoObject3DGS Setup"
echo "========================================="

# 1. 创建目录结构
echo "[1/5] Creating directory structure..."
mkdir -p data/{checkpoints,shape_priors/{explicit,implicit},examples}
mkdir -p experiments
mkdir -p logs

# 2. 安装Python依赖
echo "[2/5] Installing Python dependencies..."
pip install -r requirements.txt

# 3. 下载预训练模型
echo "[3/5] Downloading pretrained models..."
bash scripts/download_models.sh

# 4. 编译CUDA扩展（可选）
echo "[4/5] Compiling CUDA extensions (optional)..."
# pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization

# 5. 测试安装
echo "[5/5] Testing installation..."
python -c "
import torch
import numpy as np
import cv2
print('✓ Basic dependencies OK')

try:
    from transformers import AutoModel
    print('✓ Transformers OK')
except:
    print('✗ Transformers failed')

try:
    from segment_anything import sam_model_registry
    print('✓ SAM OK')
except:
    print('✗ SAM not installed (run: pip install git+https://github.com/facebookresearch/segment-anything.git)')
"

echo ""
echo "========================================="
echo "✓ Setup completed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Download models: bash scripts/download_models.sh"
echo "  2. Run single image: python scripts/single_image.py --image data/examples/test.jpg"
echo "  3. Run video: python scripts/video.py --video data/examples/test.mp4"