#!/bin/bash
set -e

PROJECT_ROOT="/home/wuwei/InfiniLM"
BUILD_DIR="$PROJECT_ROOT/build"
TARGET_DIR="$PROJECT_ROOT/python/infinilm/lib"

# 解析命令行参数
KV_CACHING="OFF"
CLASSIC_LLAMA="OFF"

while [[ $# -gt 0 ]]; do
    case $1 in
        --kv-caching)
            KV_CACHING="ON"
            shift
            ;;
        --classic-llama)
            CLASSIC_LLAMA="ON"
            shift
            ;;
        --clean)
            rm -rf "$BUILD_DIR"
            shift
            ;;
        *)
            echo "Usage: $0 [--kv-caching] [--classic-llama] [--clean]"
            exit 1
            ;;
    esac
done

echo "=== Building InfiniLM C++ Extension ==="
echo "KV Caching: $KV_CACHING"
echo "Classic Llama: $CLASSIC_LLAMA"
echo ""

# 创建目录
mkdir -p "$BUILD_DIR"
mkdir -p "$TARGET_DIR"

cd "$BUILD_DIR"

# 配置 CMake
echo "1. Configuring CMake..."
cmake .. \
    -DUSE_KV_CACHING=$KV_CACHING \
    -DUSE_CLASSIC_LLAMA=$CLASSIC_LLAMA

# 构建
echo "2. Building _infinilm target..."
cmake --build . --target _infinilm -j$(nproc)

# 查找并复制 .so 文件
echo "3. Installing module to Python package..."
SO_FILE=$(find . -name "_infinilm*.so" -type f | head -1)

if [ -z "$SO_FILE" ]; then
    echo "ERROR: Could not find _infinilm.so"
    echo "Searching in: $BUILD_DIR"
    find "$BUILD_DIR" -name "*.so" -type f
    exit 1
fi

echo "Found: $SO_FILE"
cp "$SO_FILE" "$TARGET_DIR/"
echo "Copied to: $TARGET_DIR/$(basename $SO_FILE)"

# 创建 __init__.py 如果不存在
if [ ! -f "$TARGET_DIR/__init__.py" ]; then
    echo "Creating $TARGET_DIR/__init__.py"
    cat > "$TARGET_DIR/__init__.py" << 'EOF'
# Import the C++ extension module
import os
import sys

# Get the directory of this file
_dir = os.path.dirname(os.path.abspath(__file__))

# Find the _infinilm shared library
for _file in os.listdir(_dir):
    if _file.startswith('_infinilm') and _file.endswith('.so'):
        _module_path = os.path.join(_dir, _file)
        break
else:
    raise ImportError("Could not find _infinilm shared library")

# Import the module
import importlib.util
_spec = importlib.util.spec_from_file_location("_infinilm", _module_path)
_infinilm = importlib.util.module_from_spec(_spec)
sys.modules["_infinilm"] = _infinilm
_spec.loader.exec_module(_infinilm)
EOF
fi

cd "$PROJECT_ROOT"

# 验证安装
echo ""
echo "4. Verifying installation..."
python3 -c "
import sys
sys.path.insert(0, 'python')
from infinilm.lib import _infinilm
print('Module loaded successfully!')
print(f'Module location: {_infinilm.__file__}')
"

echo ""
echo "=== Build completed successfully! ==="
echo ""
echo "To use in Python:"
echo "  from infinilm.lib import _infinilm"
