#!/bin/bash

# 快速测试不同模式
echo "========================================"
echo "Quick Mode Testing"
echo "========================================"
echo ""

TEST_IMAGE="data/examples/test.jpg"
OUTPUT_DIR="experiments/quick_test"

# 检查测试图像
if [ ! -f "$TEST_IMAGE" ]; then
    echo "❌ Test image not found: $TEST_IMAGE"
    echo "Please create a test image first"
    exit 1
fi

# 测试所有模式
modes=("high_accuracy" "balanced" "real_time" "ablation_no_dinov2" "ablation_no_depth" "ablation_minimal")

for mode in "${modes[@]}"; do
    echo ""
    echo "========================================" 
    echo "Testing mode: $mode"
    echo "========================================"
    
    python scripts/reconstruct_flexible.py \
        --image "$TEST_IMAGE" \
        --mode "$mode" \
        --output "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✓ $mode completed"
    else
        echo "✗ $mode failed"
    fi
done

echo ""
echo "========================================"
echo "All tests completed!"
echo "Results in: $OUTPUT_DIR"
echo "========================================"