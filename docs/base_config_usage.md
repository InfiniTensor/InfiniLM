# BaseTestConfig 使用文档

## 概述

`BaseTestConfig` 是 InfiniLM 项目的统一配置基类，为各个测试脚本提供通用的命令行参数解析和配置管理功能。

## 特性

- **统一参数管理**: 提供所有测试脚本共用的基础参数
- **设备类型映射**: 支持多种硬件设备的自动类型转换
- **灵活扩展**: 使用 `parse_known_args()` 容错处理，允许脚本添加特定参数
- **类型安全**: 自动将设备字符串转换为对应的 `DeviceType` 枚举

## 支持的设备类型

| 设备名称 | DeviceType 枚举值 |
|---------|-----------------|
| `cpu` | `DEVICE_TYPE_CPU` |
| `nvidia` | `DEVICE_TYPE_NVIDIA` |
| `qy` | `DEVICE_TYPE_QY` |
| `cambricon` | `DEVICE_TYPE_CAMBRICON` |
| `ascend` | `DEVICE_TYPE_ASCEND` |
| `metax` | `DEVICE_TYPE_METAX` |
| `moore` | `DEVICE_TYPE_MOORE` |
| `iluvatar` | `DEVICE_TYPE_ILUVATAR` |
| `kunlun` | `DEVICE_TYPE_KUNLUN` |
| `hygon` | `DEVICE_TYPE_HYGON` |

## 通用参数说明

| 参数 | 类型 | 是否必需 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `--model_path` | str | ✓ 是 | - | 模型文件路径 |
| `--device` | str | 否 | `cpu` | 目标设备类型（见上表） |
| `--ndev` | int | 否 | `1` | 使用的设备数量 |
| `--verbose` | flag | 否 | `False` | 启用详细输出模式 |

## 基本使用

### 1. 直接使用（测试）

```bash
python scripts/base_config.py --model_path /path/to/model
```

### 2. 在脚本中继承使用

```python
from base_config import BaseTestConfig

class MyTestConfig(BaseTestConfig):
    def __init__(self):
        super().__init__()
        # 添加脚本特定参数
        self.parser.add_argument("--my_param", type=int, default=10)
        self.my_param = self.args.my_param

# 使用
cfg = MyTestConfig()
print(f"模型路径: {cfg.model_path}")
print(f"设备类型: {cfg.device_type}")
print(f"自定义参数: {cfg.my_param}")
```

## 命令行示例

### 基础用法
```bash
python your_script.py --model_path ./models/llama2
```

### 使用 NVIDIA GPU
```bash
python your_script.py --model_path ./models/llama2 --device nvidia --ndev 2
```

### 使用 QY 设备并启用详细输出
```bash
python your_script.py --model_path ./models/llama2 --device qy --verbose
```

### 结合自定义参数
```bash
python your_script.py --model_path ./models/llama2 --device nvidia --ndev 4 --batch_size 32
```

## 类属性说明

初始化后可访问的属性：

- `model_path` (str): 模型路径
- `ndev` (int): 设备数量
- `verbose` (bool): 详细输出开关
- `device_name` (str): 设备名称（原始输入）
- `device_type` (DeviceType): 设备类型枚举值
- `args` (Namespace): 解析后的参数命名空间
- `extra` (list): 未解析的额外参数

## 设计模式说明

### 参数解析策略

使用 `parse_known_args()` 而非 `parse_args()`，这使得：

1. **允许未知参数**: 脚本可以添加自己的参数而不会报错
2. **参数链传递**: 子类可以在解析后处理额外参数
3. **容错性强**: 配置变更时不会中断现有脚本

### 设备类型映射

- 大小写不敏感（自动转换为小写）
- 未知设备默认回退到 CPU
- 映射逻辑集中管理，便于维护

## 错误处理

### 缺少必需参数
```bash
# 错误示例：缺少 --model_path
python your_script.py
# 输出：error: the following arguments are required: --model_path
```

### 未知设备类型
```bash
python your_script.py --model_path ./models --device unknown
# 行为：默认使用 CPU，不会报错
```

## 扩展指南

### 添加新的设备类型

修改 `_get_device_type` 方法中的 `DEVICE_TYPE_MAP`：

```python
DEVICE_TYPE_MAP = {
    # ... 现有映射 ...
    "new_device": DeviceType.DEVICE_TYPE_NEW,
}
```

### 添加新的通用参数

修改 `_add_common_args` 方法：

```python
def _add_common_args(self):
    # ... 现有参数 ...
    self.parser.add_argument("--new_param", type=str, default="default")
```

## 注意事项

1. **参数顺序**: 命令行参数顺序不影响解析结果
2. **类型转换**: `--ndev` 等整数参数会自动验证类型
3. **参数覆盖**: 后出现的参数会覆盖前面的同名参数
4. **帮助信息**: 使用 `--help` 查看所有可用参数

## 相关文件

- `scripts/jiuge_config.py`: 九歌评测配置（继承自 BaseTestConfig）
- `scripts/jiuge_ppl_config.py`: 九歌 PPL 配置（继承自 BaseTestConfig）
- `scripts/jiuge.py`: 九歌评测主脚本

## 版本信息

- 创建日期: 2026-03-16
- 适用版本: InfiniLM (feat-new-work 分支及以后)
