根据您的指令和之前的分析，我制定了以下计划来彻底修复精度问题并进行验证。

核心思路是：不仅在 `inject` 阶段，在 `forward` 阶段的 Gather Buffer 也必须初始化为 `-inf`，以防止 Padding 区域的 0 值干扰 Softmax 计算。

### 1. 代码完善 (C++)
在 `src/models/Qwen3MoE/Qwen3MoE.cpp` 中继续修改 `forwardQwen3MoEAttention` 函数：
- 找到 `k_padded_gather` 和 `v_padded_gather` 的初始化代码。
- 将原本的 `cudaMemsetAsync` (清零) 替换为我们新实现的 `launch_fill_val_bf16` (填充 `-inf`)。
- **原因**：这是解决 `past=0` 场景下精度下降（0.99 -> 0.98/0.96）的关键，确保 Padding 不会参与 Attention 权重计算。

### 2. 编译与运行
使用您提供的完整环境命令进行编译和测试：
```bash
cd '/data/users/lviy/InfiniLM' ; source /data/apps/env.sh && source /data/apps/miniforge3/etc/profile.d/conda.sh && conda activate py313 && export INFINI_ROOT=$HOME/.infini && export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$LD_LIBRARY_PATH && export PATH="/data/apps/xmake/bin:/usr/local/cuda/bin:$PATH" && xmake && srun --gres=gpu:nvidia:1 --cpus-per-task=8 --mem=16G python test/models/qwen3_moe/attention_test.py --model_path "/data/shared/models/Qwen3-30B-A3B-Instruct-2507-Layer-0" --nvidia
```

### 3. 结果验证
观察输出日志：
- **Debug Log**: 确认 `[Inject]` 的分配逻辑是否如预期（Batch 0 Alloc, Batch 3 Reuse）。
- **Cosine Similarity**: 检查是否恢复到 > 0.99（预期 0.0000 问题应随之解决，因为构建环境已修复）。

如果测试通过，我将删除 Debug Print 并交付最终代码。