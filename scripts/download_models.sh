#!/bin/bash

echo "Downloading pretrained models..."

cd data/checkpoints

# 1. SAM模型
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    echo "[1/3] Downloading SAM ViT-H (2.4GB)..."
    wget -q --show-progress \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

# 2. FastSAM模型（可选）
if [ ! -f "FastSAM-x.pt" ]; then
    echo "[2/3] Downloading FastSAM-x (138MB)..."
    wget -q --show-progress \
        https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v0.1.0/FastSAM-x.pt
fi

# 3. 示例形状先验（创建简单模板）
echo "[3/3] Creating example shape priors..."
cd ../shape_priors/explicit

python3 << 'EOF'
import numpy as np
import trimesh
import os

# 创建简单的CAD模板
categories = {
    'chair': {'height': 0.9, 'width': 0.5, 'depth': 0.5},
    'table': {'height': 0.75, 'width': 1.5, 'depth': 0.8},
    'sofa': {'height': 0.85, 'width': 2.0, 'depth': 0.9}
}

for cat, dims in categories.items():
    os.makedirs(cat, exist_ok=True)
    
    # 创建简单的盒子作为模板
    vertices = np.array([
        [-dims['width']/2, 0, -dims['depth']/2],
        [dims['width']/2, 0, -dims['depth']/2],
        [dims['width']/2, 0, dims['depth']/2],
        [-dims['width']/2, 0, dims['depth']/2],
        [-dims['width']/2, dims['height'], -dims['depth']/2],
        [dims['width']/2, dims['height'], -dims['depth']/2],
        [dims['width']/2, dims['height'], dims['depth']/2],
        [-dims['width']/2, dims['height'], dims['depth']/2]
    ])
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # 底面
        [4, 5, 6], [4, 6, 7],  # 顶面
        [0, 1, 5], [0, 5, 4],  # 前面
        [2, 3, 7], [2, 7, 6],  # 后面
        [0, 3, 7], [0, 7, 4],  # 左面
        [1, 2, 6], [1, 6, 5]   # 右面
    ])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(f'{cat}/template.obj')
    print(f'✓ Created template for {cat}')
EOF

cd ../../..

echo ""
echo "✓ All models downloaded!"