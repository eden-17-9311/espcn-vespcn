#!/bin/bash
# 快速使用脚本：提取训练集和测试集

# 配置
INPUT_DIR="./videos"
OUTPUT_DIR="./data/vimeo90k"
TEST_PATTERN="test"  # 文件名包含 'test' 的作为测试集

echo "=========================================="
echo "Step 1: 提取并分离训练/测试集"
echo "=========================================="
echo "训练集: 所有不包含 '$TEST_PATTERN' 的视频"
echo "测试集: 包含 '$TEST_PATTERN' 的视频 (如: test1.mp4, test2.mp4)"
echo ""

# 提取帧
python extract_frames.py \
  --mode extract \
  --input_dir $INPUT_DIR \
  --output_dir $OUTPUT_DIR \
  --max_frames 1000000 \
  --test_pattern $TEST_PATTERN

echo ""
echo "=========================================="
echo "Step 2: 生成 LR 版本"
echo "=========================================="

# 生成 LR
python extract_frames.py \
  --mode create_lr \
  --input_dir $OUTPUT_DIR \
  --output_dir $OUTPUT_DIR \
  --downscale_factor 4

echo ""
echo "✅ 完成！数据结构："
echo "  训练集: $OUTPUT_DIR/sequences/"
echo "  测试集 GT: $OUTPUT_DIR/test/sequences/"
echo "  测试集 LR: $OUTPUT_DIR/test/sequences_lrx4/"
