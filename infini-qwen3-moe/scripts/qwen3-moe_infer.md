# Qwen3-MoE 推理指南

本文档介绍如何使用 InfiniCore 框架运行 Qwen3-MoE (Mixture of Experts) 模型进行推理。

## 概述

Qwen3-MoE 是基于 Qwen3 架构的混合专家模型，具有以下特点：

- **稀疏激活**: 每个 token 只激活少量专家 (top-k)
- **高效推理**: 通过专家选择减少计算量
- **负载均衡**: 专家使用统计和负载监控
- **分布式支持**: 专家权重在多设备间分区
- **灵活配置**: 支持混合 MLP 和 MoE 层

## 模型架构

### MoE 层配置

```python
# 典型的 Qwen3-MoE 配置参数
{
    "num_experts": 64,              # 每个 MoE 层的专家数量
    "num_experts_per_tok": 2,       # 每个 token 选择的专家数量 (top-k)
    "moe_intermediate_size": 1536,  # MoE 专家的中间维度
    "decoder_sparse_step": 2,       # MoE 层的间隔 (每2层一个MoE层)
    "mlp_only_layers": [0, 1],      # 指定使用普通 MLP 的层
    "norm_topk_prob": True,         # 是否归一化 top-k 概率
    "router_aux_loss_coef": 0.01    # 路由辅助损失系数
}
```

### 层类型判断

模型中的层分为两种类型：

1. **MoE 层**: 当满足以下条件时使用 MoE
   - `(layer_idx + 1) % decoder_sparse_step == 0`
   - `layer_idx` 不在 `mlp_only_layers` 中
   - `num_experts > 0`

2. **普通 MLP 层**: 其他情况使用标准 MLP

## 环境配置

### 依赖要求

```bash
# Python 依赖
pip install torch transformers safetensors

# InfiniCore 框架
export INFINI_ROOT=~/.infini
# 确保 libinfinicore_infer.so 在 LD_LIBRARY_PATH 中
```

### 编译项目

```bash
# 构建 InfiniCore 推理库
xmake build

# 确认编译成功
ls build/linux/x86_64/debug/libinfinicore_infer.so
```

## 模型准备

### 模型格式

支持以下模型格式：
- **Safetensors**: 推荐格式，加载速度快
- **PyTorch**: 标准 `.bin` 格式

### 权重命名规范

Qwen3-MoE 模型需要遵循以下权重命名规范：

```python
# 基础权重
model.embed_tokens.weight
model.norm.weight
lm_head.weight

# 注意力权重 (每层)
model.layers.{i}.input_layernorm.weight
model.layers.{i}.self_attn.q_proj.weight
model.layers.{i}.self_attn.k_proj.weight
model.layers.{i}.self_attn.v_proj.weight
model.layers.{i}.self_attn.o_proj.weight
model.layers.{i}.self_attn.q_norm.weight  # Qwen3 特有
model.layers.{i}.self_attn.k_norm.weight  # Qwen3 特有

# MLP 权重 (普通层)
model.layers.{i}.post_attention_layernorm.weight
model.layers.{i}.mlp.gate_proj.weight
model.layers.{i}.mlp.up_proj.weight
model.layers.{i}.mlp.down_proj.weight

# MoE 权重 (MoE层)
model.layers.{i}.mlp.gate.weight  # 路由器权重
model.layers.{i}.mlp.experts.{j}.gate_proj.weight  # 专家j的gate投影
model.layers.{i}.mlp.experts.{j}.up_proj.weight    # 专家j的up投影
model.layers.{i}.mlp.experts.{j}.down_proj.weight  # 专家j的down投影
```

## 使用方法

### 基本推理

```bash
# CPU 推理
python scripts/qwen3_moe.py /path/to/qwen3-moe-model --cpu

# GPU 推理 (单卡)
python scripts/qwen3_moe.py /path/to/qwen3-moe-model --nvidia

# 多卡推理
python scripts/qwen3_moe.py /path/to/qwen3-moe-model --nvidia 4

# 开启调试模式
python scripts/qwen3_moe.py /path/to/qwen3-moe-model --nvidia --debug
```

### 参数说明

- `model_path`: 模型文件目录路径
- `--cpu/--nvidia/--cambricon/--ascend`: 设备类型选择
- `n_device`: 使用的设备数量 (默认: 1)
- `--debug`: 开启详细调试信息

### Python API 使用

```python
from scripts.qwen3_moe import Qwen3MoeForCausalLM
from libinfinicore_infer import DeviceType

# 创建模型实例
model = Qwen3MoeForCausalLM(
    model_dir_path="/path/to/qwen3-moe-model",
    device=DeviceType.DEVICE_TYPE_NVIDIA,
    ndev=2,  # 使用2张GPU
    max_tokens=4096
)

# 生成文本
output, avg_time = model.generate(
    input_content="介绍一下人工智能的发展历史",
    max_steps=100,
    temperature=0.7,
    topk=50,
    topp=0.8
)

print(f"生成内容: {output}")
print(f"平均耗时: {avg_time:.2f}ms/token")

# 查看专家使用统计
model.print_router_stats()

# 清理资源
model.destroy_model_instance()
```

### 批量推理

```python
from scripts.qwen3_moe import Qwen3MoeForCausalLM
from scripts.infer_task import Qwen3MoeInferTask, Qwen3MoeKVCache

model = Qwen3MoeForCausalLM("model_path")

# 创建多个推理任务
tasks = []
for i, text in enumerate(["问题1", "问题2", "问题3"]):
    tokens = model.tokenizer.encode(text)
    task = Qwen3MoeInferTask(
        tokens=tokens,
        temperature=0.7,
        topk=50,
        topp=0.8,
        task_id=i
    )
    task.bind_kvcache(Qwen3MoeKVCache(model))
    tasks.append(task)

# 批量推理
outputs = model.batch_infer_one_round(tasks)
print(f"输出 tokens: {outputs}")
```

## 性能优化

### 设备分布策略

对于多设备推理，权重按以下策略分区：

1. **注意力权重**: 按头维度分区
   - Q 投影: `[d, d/ndev]`
   - K/V 投影: `[d, (nkvh*dh)/ndev]`
   - O 投影: `[d/ndev, d]`

2. **MLP 权重**: 按中间维度分区
   - Gate/Up 投影: `[d, di/ndev]`
   - Down 投影: `[di/ndev, d]`

3. **MoE 专家权重**: 专家在设备间分区
   - 每个设备处理 `⌈num_experts/ndev⌉` 个专家
   - 专家权重按中间维度分区: `[d, moe_di/ndev]`

### 内存优化

```python
# 配置最大上下文长度
model = Qwen3MoeForCausalLM(
    model_path,
    max_tokens=2048  # 减少内存使用
)

# 使用较小的批次大小
batch_size = 4  # 根据GPU内存调整
```

### 专家负载均衡

模型自动跟踪专家使用统计：

```python
# 获取特定层的专家统计
layer_stats = model.get_router_stats(layer_idx=5)
print(f"层5专家使用: {layer_stats}")

# 打印所有MoE层的统计
model.print_router_stats()
```

理想情况下，专家使用应该相对均衡 (balance ≈ 1.0)。

## 调试和监控

### 调试模式

```bash
# 开启调试模式
python scripts/qwen3_moe.py model_path --debug
```

调试模式提供：
- 详细的推理步骤日志
- 张量范围验证
- 专家选择追踪
- 性能分析信息

### 常见问题排查

#### 1. 权重加载失败

```
错误: Unsupported weight naming scheme for Qwen3-MoE
```

**解决方案**:
- 检查模型配置文件 `config.json`
- 确认权重文件包含 MoE 相关权重
- 验证权重命名是否符合规范

#### 2. 专家权重缺失

```
错误: KeyError: 'model.layers.0.mlp.experts.0.gate_proj.weight'
```

**解决方案**:
- 确认模型确实是 MoE 架构
- 检查 `num_experts` 配置
- 验证专家权重存在

#### 3. 设备内存不足

```
错误: CUDA out of memory
```

**解决方案**:
- 减少 `max_tokens` 参数
- 使用更多设备分区
- 降低批次大小

#### 4. 专家负载不均衡

```
Expert load balance variance: 1250.5 (high variance)
```

**解决方案**:
- 检查路由器初始化
- 调整 `router_aux_loss_coef`
- 考虑重新训练或微调

## 性能基准

### 典型性能指标

| 配置 | 设备 | 吞吐量 (tokens/s) | 延迟 (ms/token) |
|------|------|------------------|----------------|
| 7B MoE | 1x A100 | 150-200 | 5-7 |
| 7B MoE | 2x A100 | 280-350 | 3-4 |
| 14B MoE | 4x A100 | 200-250 | 4-5 |

实际性能取决于：
- 模型大小和专家数量
- 输入序列长度
- 硬件配置
- 专家激活模式

### 与标准模型对比

相比于相同参数量的标准 Transformer：
- **推理速度**: 提升 1.5-2.5x (取决于专家激活比例)
- **内存使用**: 增加 1.2-1.8x (存储所有专家权重)
- **计算效率**: 提升显著 (仅激活部分专家)

## 最佳实践

### 1. 模型部署

- 使用 Safetensors 格式提升加载速度
- 预先验证模型权重完整性
- 配置合适的设备分区策略

### 2. 推理优化

- 根据应用场景调整 `num_experts_per_tok`
- 监控专家负载均衡情况
- 使用批量推理提升吞吐量

### 3. 资源管理

- 及时释放 KV 缓存
- 合理配置内存池大小
- 监控设备间通信开销

### 4. 质量控制

- 定期检查专家使用统计
- 验证输出质量一致性
- 比较与标准模型的性能差异

## 故障排除

如果遇到问题，请按以下步骤排查：

1. **检查环境配置**
   ```bash
   python -c "from libinfinicore_infer import *; print('API available')"
   ```

2. **验证模型格式**
   ```python
   import json
   with open("model_path/config.json") as f:
       config = json.load(f)
       print(f"Model type: {config.get('model_type')}")
       print(f"Experts: {config.get('num_experts', 0)}")
   ```

3. **测试基础功能**
   ```bash
   python scripts/qwen3_moe.py model_path --debug
   ```

4. **查看详细日志**
   - 启用调试模式
   - 检查专家统计信息
   - 分析性能瓶颈

如需更多帮助，请参考 InfiniCore 文档或提交 issue。