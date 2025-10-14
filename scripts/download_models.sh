#!/bin/bash

echo "Downloading pretrained models..."

mkdir -p data/checkpoints
cd data/checkpoints

# SAM模型
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    echo "[1/2] Downloading SAM ViT-H (2.4GB)..."
    wget -q --show-progress \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

echo ""
echo "✓ All models downloaded!"
echo ""
echo "Note: DINOv2 and Depth Anything V2 will be downloaded automatically"
echo "from Hugging Face on first use."