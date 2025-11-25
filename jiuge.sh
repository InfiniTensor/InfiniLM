#!/bin/bash

# Jiuge模型运行脚本
# 使用NVIDIA显卡运行9G4B模型

set -e  # 遇到错误立即退出

echo "=========================================="
echo "🚀 启动 Jiuge 模型 (9G4B) - NVIDIA版本"
echo "=========================================="

# 设置参数
MODEL_DIR="/home/featurize/work/InfiniFamily/9G4B"
DEVICE="--nvidia"
N_DEVICE=1
SCRIPT_PATH="python scripts/jiuge.py"

# 检查模型目录是否存在
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 错误: 模型目录不存在: $MODEL_DIR"
    echo "请检查路径是否正确"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "scripts/jiuge.py" ]; then
    echo "❌ 错误: 未找到jiuge.py脚本: scripts/jiuge.py"
    echo "请确保在当前目录下运行此脚本"
    exit 1
fi

echo "📁 模型路径: $MODEL_DIR"
echo "🎯 设备类型: NVIDIA GPU"
echo "💻 设备数量: $N_DEVICE"
echo ""

# 运行模型
echo "🔄 启动模型..."
$SCRIPT_PATH $DEVICE $MODEL_DIR $N_DEVICE

echo ""
echo "=========================================="
echo "✅ 模型运行完成"
echo "=========================================="