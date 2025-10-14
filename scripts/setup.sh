#!/bin/bash

echo "========================================="
echo "MonoObject3DGS Setup"
echo "========================================="

echo "[1/5] Creating directory structure..."
mkdir -p data/{checkpoints,shape_priors/{explicit,implicit},examples}
mkdir -p experiments
mkdir -p logs

echo "[2/5] Installing Python dependencies..."
pip install -r requirements.txt

echo "[3/5] Downloading pretrained models..."
echo "Please run: bash scripts/download_models.sh"

echo "[4/5] Creating example shape priors..."
python3 << 'EOF'
import numpy as np
import os
from pathlib import Path

# 创建简单的形状模板
categories = {
    'chair': {'h': 0.9, 'w': 0.5, 'd': 0.5},
    'table': {'h': 0.75, 'w': 1.5, 'd': 0.8},
    'sofa': {'h': 0.85, 'w': 2.0, 'd': 0.9}
}

base_dir = Path('data/shape_priors/explicit')
base_dir.mkdir(parents=True, exist_ok=True)

for cat, dims in categories.items():
    cat_dir = base_dir / cat
    cat_dir.mkdir(exist_ok=True)
    
    # 创建简单的点云
    n_points = 1000
    points = np.random.randn(n_points, 3) * 0.5
    points[:, 0] *= dims['w']
    points[:, 1] *= dims['h']
    points[:, 2] *= dims['d']
    
    # 保存为简单格式
    np.save(cat_dir / 'template.npy', points)
    print(f'Created template for {cat}')
EOF

echo "[5/5] Testing installation..."
python3 -c "
import torch
import numpy as np
import cv2
print('✓ Basic dependencies OK')

try:
    from transformers import AutoModel
    print('✓ Transformers OK')
except:
    print('✗ Transformers failed')

print('')
print('Setup completed! Next steps:')
print('  1. Download models: bash scripts/download_models.sh')
print('  2. Run example: python scripts/single_image.py --image data/examples/test.jpg')
"

echo ""
echo "========================================="
echo "✓ Setup completed!"
echo "========================================="