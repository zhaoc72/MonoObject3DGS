#!/bin/bash

echo "========================================"
echo "Downloading V2 Models"
echo "========================================"
echo ""

mkdir -p data/checkpoints
cd data/checkpoints

# SAM 2模型
echo "[1/3] Downloading SAM 2 models..."
if [ ! -f "sam2_hiera_large.pt" ]; then
    echo "  Downloading SAM 2 Large (2.9GB)..."
    wget -q --show-progress \
        https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
fi

echo ""
echo "[2/3] Depth Anything V2 models..."
echo "  Note: Will be downloaded automatically from Hugging Face"
echo "  Models: vits, vitb, vitl (metric versions)"

echo ""
echo "[3/3] DINOv2 models..."
echo "  Note: Will be downloaded automatically from Hugging Face"
echo "  Model: facebook/dinov2-large (1.3GB)"

echo ""
echo "✓ Model download configured!"
echo ""
echo "Models will auto-download on first use:"
echo "  - DINOv2 Large: ~1.3GB"
echo "  - SAM 2 Large: ~2.9GB"
echo "  - Depth Anything V2: ~1.5GB"
echo ""