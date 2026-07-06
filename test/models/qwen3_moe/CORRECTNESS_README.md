# Qwen3-MoE 端到端正确性验证（prefill 首步对齐 HF）

验证 InfiniLM 适配的 Qwen3-30B-A3B-Thinking-2507 与 HuggingFace 参照在 **prefill 首步**
的一致性。核心指标是**首个生成 token（= prefill logits 的 argmax）与 HF 一致**，并顺带校验
**prompt 分词（chat template + `<think>` 前缀）与 HF 一致**。

## 为什么是"首 token"而不是"整段 logits bit-exact"

- 本模型 **run-to-run 非确定**（TP all-reduce 规约顺序 + BF16 `index_add_` 原子加），
  同一二进制两次贪心生成会从中途分叉 → 整条 token 序列 / 逐元素 logit 对齐不可靠。
- **首 token 的 argmax 对微小数值噪声鲁棒**，是贪心解码真正依赖的量，也正是赛题要求的
  "输出 token 一致性"最本质的一步。脚本额外报告"前缀一致长度"作为更强的参考信号。

## 为什么两阶段

30B 模型放不下两份（InfiniLM TP=2 + HF 参照）。所以：
1. 先单独用 HF 生成参照并存盘（`dump_hf_reference.py`）；
2. 再单独用 InfiniLM 加载并对比（`check_prefill_logits.py`）。

HF 仅作**测试参照**，不进入 InfiniLM 推理路径（符合赛题规则）。

## 用法

### 1) 生成 HF 参照（HF 单独占用显存/内存）

```bash
cd /data/InfiniLM
pip install transformers torch    # 若未装

# GPU（device_map=auto 可跨多卡）：
python3 test/models/qwen3_moe/dump_hf_reference.py \
    --model /data/huggingface_home/Qwen3-30B-A3B-Thinking-2507 \
    --device cuda --out /tmp/qwen3_ref.json
# 或 CPU（~60GB 内存，慢但不占 GPU）：--device cpu
```

### 2) InfiniLM 对比（确保 HF 进程已退出、显存已释放）

```bash
python3 test/models/qwen3_moe/check_prefill_logits.py \
    --model /data/huggingface_home/Qwen3-30B-A3B-Thinking-2507 \
    --device metax --tp 2 --ref /tmp/qwen3_ref.json
```

退出码 0 = PASS，非 0 = FAIL（可用于 CI）。

## 判读

- **全部 PASS**：first_tok 与 prompt_tok 都对齐 → 权重加载 / chat template / 路由正确。
- **first_tok 或 prompt_tok 不一致 → 真 bug**（权重/模板/路由适配错误）。
- 仅个别 **prefix 早停但 first_tok 全 ==** → 通常是非确定性或近似平票，非适配错误。

## 参数

- `--max-new-tokens`（dump 侧，默认 16）：生成多少 token 供比对前缀。
- `--prefix-threshold`（check 侧，默认 1）：PASS 要求每条前缀一致的最少 token 数；
  设 1 即"首 token 对齐"。
- prompts 固定在 `dump_hf_reference.py` 的 `DEFAULT_PROMPTS`，并写入参照文件，
  checker 复用同一批 messages，保证两侧 prompt 完全一致。
