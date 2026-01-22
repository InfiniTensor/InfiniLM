# LLaDA Python Implementation

本文档展示了如何在Python中使用LLaDA（Masked Diffusion Language Model）进行文本生成。

## 概述

LLaDA是一种基于掩码扩散的语言模型，它通过逐步从全掩码序列还原为有意义文本来生成文本。这个实现包含了与原始LLaDA论文相同的核心算法：

- 基于掩码的扩散生成过程
- Gumbel噪声采样
- 低置信度重掩码策略
- 分块半自回归生成
- 无监督分类器指导(CFG)

## 主要功能

### 1. 基本generate函数

`generate()` 函数提供了LLaDA的核心生成逻辑，但使用占位符模型前向传播。

```python
model.generate(
    prompts="Your prompt here",
    max_steps=128,        # 采样步数
    gen_length=128,       # 生成长度
    block_length=128,     # 块长度
    temperature_=0.0,     # 采样温度
    cfg_scale=0.0,        # CFG尺度
    remasking='low_confidence',  # 重掩码策略
    verbose=True          # 详细输出
)
```

### 2. C++模型集成函数

`generate_with_cpp_model()` 函数展示了如何与C++模型接口集成：

```python
model.generate_with_cpp_model(
    prompts="Your prompt here",
    max_steps=128,
    gen_length=128,
    block_length=32,      # 使用分块生成
    temperature_=0.5,
    cfg_scale=1.0,        # 启用CFG
    verbose=True
)
```

## 核心算法

### 1. 掩码扩散过程

LLaDA从全掩码序列开始，逐步还原token：

```python
# 初始化为全掩码
x = torch.full((batch_size, prompt_length + gen_length), mask_id, dtype=torch.long)
x[:, :prompt_length] = input_ids  # 保留prompt部分
```

### 2. Gumbel噪声采样

使用Gumbel-Max采样进行分类分布采样：

```python
def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
```

### 3. 低置信度重掩码

在每一步中，选择置信度最低的位置进行重新掩码：

```python
if remasking == 'low_confidence':
    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

# 选择置信度最低的位置进行转移
_, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
```

### 4. 分块生成

支持半自回归分块生成，提高生成长文本的效率：

```python
# 将生成长度分成多个块
num_blocks = gen_length // block_length
for num_block in range(num_blocks):
    # 处理当前块
    block_mask_index = (x[:, block_start:block_end] == mask_id)
```

## 参数说明

- **prompts**: 输入提示（字符串或字符串列表）
- **max_steps**: 采样步数（建议≤gen_length）
- **gen_length**: 生成长度
- **block_length**: 块长度（≤gen_length，如果小于gen_length则使用半自回归）
- **temperature_**: 采样温度（0=确定性，>0=随机性）
- **cfg_scale**: 无监督分类器指导尺度（0=禁用，>0=启用）
- **remasking**: 重掩码策略（'low_confidence'或'random'）
- **mask_id**: 掩码token ID（LLaDA中为126336）
- **logits_eos_inf**: 是否将EOS token logits设为-inf
- **confidence_eos_eot_inf**: 是否将EOS和EOT token置信度设为-inf

## C++集成指南

当前实现已经集成了C++模型接口，通过以下组件：

### 1. InferTask和BatchedTask
- `InferTask`: 封装单个推理请求的参数和状态
- `LLaDABatchedTask`: 批量处理多个InferTask，转换为C++接口需要的格式

### 2. C++接口方法

**batch_infer_one_round(tasks)**:
- 输入: InferTask对象列表
- 输出: 生成的token ID列表
- 用于采样推理

**forward_logits_batch(input_ids_tensor, attention_mask_tensor)**:
- 输入: PyTorch张量格式的token IDs
- 输出: logits张量 [batch_size, seq_len, vocab_size]
- 用于获取完整logits进行LLaDA采样

### 3. 自动错误处理
- C++模型调用失败时自动降级到占位符logits
- 详细的调试输出帮助排查问题

### 4. 已实现的C++接口
```python
# 在scripts/libinfinicore_infer/llada.py中已定义：
- inferBatchLLaDA: 批量采样推理
- forwardBatchLLaDA: 批量logits计算
```

## 使用示例

```python
# 导入LLaDA模型
from scripts.llada import LLaDAForCauslLM

# 加载模型
model = LLaDAForCauslLM(
    model_dir_path="/path/to/llada/model",
    device=DeviceType.DEVICE_TYPE_CPU,
    ndev=1
)

# 基本生成
result = model.generate(
    prompts="The future of AI is",
    max_steps=64,
    gen_length=64,
    temperature_=0.0,
    verbose=True
)

# 使用C++模型集成
result = model.generate_with_cpp_model(
    prompts="Explain quantum computing:",
    max_steps=128,
    gen_length=128,
    block_length=32,
    temperature_=0.7,
    cfg_scale=1.0,
    remasking='low_confidence',
    verbose=True
)
```

## 注意事项

1. **模型接口**: 需要根据您的具体C++模型接口调整`cpp_model_forward`函数
2. **内存管理**: 确保PyTorch张量和C++内存之间的正确同步
3. **设备兼容性**: 确保PyTorch张量与C++模型在相同的设备上（CPU/GPU）
4. **性能优化**: 对于生产环境，考虑批量处理和内存优化
5. **token ID**: 确保使用正确的掩码token ID（126336 for LLaDA）

## 测试

运行测试函数：

```bash
python scripts/llada.py
```

这将执行多个测试用例，验证基本生成、C++集成和高级参数功能。

## 扩展

可以基于此实现添加：
- 更多的重掩码策略
- 不同的采样方法
- 批量处理优化
- 更多的高级控制参数