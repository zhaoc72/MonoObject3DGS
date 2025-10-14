#!/bin/bash

# MonoObject3DGS Quick Start Script
# 快速设置和测试整个系统

set -e  # 遇到错误立即退出

echo "================================"
echo "MonoObject3DGS Quick Start"
echo "================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查Python版本
echo -e "${YELLOW}[1/8] 检查Python环境...${NC}"
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

if ! python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${RED}错误: 需要Python 3.8或更高版本${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python版本检查通过${NC}"
echo ""

# 检查CUDA
echo -e "${YELLOW}[2/8] 检查CUDA环境...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo -e "${GREEN}✓ CUDA可用${NC}"
else
    echo -e "${YELLOW}警告: 未检测到CUDA，将使用CPU模式${NC}"
fi
echo ""

# 创建目录结构
echo -e "${YELLOW}[3/8] 创建目录结构...${NC}"
mkdir -p data/{downloads,processed,shape_priors/{explicit,implicit},checkpoints}
mkdir -p experiments
mkdir -p logs
echo -e "${GREEN}✓ 目录创建完成${NC}"
echo ""

# 下载预训练模型
echo -e "${YELLOW}[4/8] 下载预训练模型...${NC}"

# SAM模型
if [ ! -f "data/checkpoints/sam_vit_h_4b8939.pth" ]; then
    echo "下载SAM模型 (约2.4GB)..."
    wget -q --show-progress \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
        -O data/checkpoints/sam_vit_h_4b8939.pth || {
        echo -e "${YELLOW}SAM模型下载失败，请手动下载${NC}"
    }
else
    echo "SAM模型已存在"
fi

echo -e "${GREEN}✓ 模型下载完成${NC}"
echo ""

# 安装依赖
echo -e "${YELLOW}[5/8] 安装Python依赖...${NC}"
pip install -q -r requirements.txt
echo -e "${GREEN}✓ 依赖安装完成${NC}"
echo ""

# 安装3DGS渲染器
echo -e "${YELLOW}[6/8] 安装3D Gaussian Splatting渲染器...${NC}"
if ! python -c "import diff_gaussian_rasterization" 2>/dev/null; then
    echo "安装diff-gaussian-rasterization..."
    pip install -q git+https://github.com/graphdeco-inria/diff-gaussian-rasterization || {
        echo -e "${YELLOW}警告: 3DGS渲染器安装失败，某些功能可能不可用${NC}"
    }
else
    echo "3DGS渲染器已安装"
fi
echo -e "${GREEN}✓ 渲染器安装完成${NC}"
echo ""

# 下载示例数据
echo -e "${YELLOW}[7/8] 准备示例数据...${NC}"
if [ ! -d "data/example_scene" ]; then
    echo "创建示例场景..."
    python << EOF
import numpy as np
from PIL import Image
import os

# 创建示例图像
os.makedirs("data/example_scene/images", exist_ok=True)
os.makedirs("data/example_scene/depth", exist_ok=True)

for i in range(10):
    # 生成示例RGB图像
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    Image.fromarray(img).save(f"data/example_scene/images/{i:05d}.jpg")
    
    # 生成示例深度图
    depth = np.random.rand(512, 512) * 5.0 + 2.0
    depth_img = (depth * 1000).astype(np.uint16)
    Image.fromarray(depth_img).save(f"data/example_scene/depth/{i:05d}.png")

print("✓ 示例数据创建完成")
EOF
else
    echo "示例数据已存在"
fi
echo -e "${GREEN}✓ 数据准备完成${NC}"
echo ""

# 运行测试
echo -e "${YELLOW}[8/8] 运行系统测试...${NC}"
python << EOF
import sys
sys.path.append('.')

print("测试模块导入...")
try:
    from src.segmentation.dinov2_extractor import DINOv2Extractor
    print("  ✓ DINOv2模块")
    
    from src.segmentation.sam_segmenter import SAMSegmenter
    print("  ✓ SAM模块")
    
    from src.reconstruction.object_gaussian import ObjectGaussian
    print("  ✓ Gaussian重建模块")
    
    from src.priors.prior_fusion import AdaptivePriorFusion
    print("  ✓ 形状先验模块")
    
    from src.optimization.losses import CompositeLoss
    print("  ✓ 损失函数模块")
    
    print("\n所有模块导入成功！")
    
except Exception as e:
    print(f"\n错误: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 系统测试通过${NC}"
else
    echo -e "${RED}✗ 系统测试失败${NC}"
    exit 1
fi
echo ""

# 完成
echo "================================"
echo -e "${GREEN}设置完成！${NC}"
echo "================================"
echo ""
echo "下一步操作:"
echo ""
echo "1. 查看演示notebook:"
echo "   jupyter notebook notebooks/demo.ipynb"
echo ""
echo "2. 在示例数据上训练:"
echo "   python scripts/train.py --config configs/default.yaml --data_dir data/example_scene"
echo ""
echo "3. 在自己的数据上训练:"
echo "   # 将图像放到 data/your_scene/images/"
echo "   python scripts/preprocess_data.py --data_dir data/your_scene"
echo "   python scripts/train.py --config configs/default.yaml --data_dir data/your_scene"
echo ""
echo "4. 可视化结果:"
echo "   python scripts/visualize.py --checkpoint experiments/your_exp/final_model"
echo ""
echo "更多信息请查看 README.md"
echo ""