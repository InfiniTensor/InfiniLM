# 项目 #2 基线差距清单

最新进展见 v16（2026-09-01）：投机采样方向开工并已在 5090 验收——
prompt-lookup（零训练、模型无关的 n-gram draft）+ spec×chunked 融合 +
融合单前向（主采样与 draft 验证一次完成，纯 decode 批次形状固定
b×(k+1)，为 verify 图化备好可录制形状）+ 自适应收益门控（低命中负载
开投机不致亏）。纯 Python 实现，stub 测试 11 用例全绿；5090 实测
w4/w5/w7 提速 2.1~2.4×、w6 吞吐 +57%、w1 打平，高命中负载输出与非
投机逐 token 相同（开放生成的分歧定位为 near-tie argmax 翻转，非逻辑
bug，详见 v16 验收结果节）。v15 的 kernel 级归因结论不变：小模型
decode 的常数项（小 kernel 延迟 + 图外 CPU）决定了投机必须走融合
单前向才有净收益。

---

## v16（2026-09-01）：prompt-lookup 投机采样 + spec×chunked 融合

动机：v15 归因认定 kernel 微优化天花板清晰，数量级杠杆在 FP8 与投机
采样。仓库已有投机链路（speculative_runner.py）但三个硬限制：draft
只支持 minicpm_eagle（Qwen3-0.6B/1.7B 没有现成 EAGLE 头）、与 v13
chunked prefill 在引擎 gate 互斥、且投机模式全程 eager（target 主前向
走 forward_raw 拿 hidden states，被 rank_worker 的
!sample_all_positions 门禁挡在图外）。v16 按分析的第一、三步落地
（跳过需要训练的 EAGLE 头）：① prompt-lookup 打通链路，③ spec×
chunked 融合 + verify 形状固定化。

### 改动清单（全部纯 Python）

- `config/engine_config.py`：新增 `speculative_method`（None/"eagle"/
  "prompt_lookup"）；归一化——给了 `draft_model_path` 未指定方法时默认
  "eagle"（旧调用方行为不变）；校验组合合法性。
- `llm/model_runner/speculative_runner.py`：
  - prompt_lookup draft：`_draft_prompt_lookup_tokens` 在请求自身
    prompt+已生成序列中找当前后缀的最近一次出现，取其后 k 个 token
    （n-gram 范围由 `INFINILM_PROMPT_LOOKUP_MIN/MAX_NGRAM` 控制，默认
    2/4；只影响命中质量，正确性由 verify 保证）。
  - **融合单前向**（`_forward_fused_prompt_lookup`）：纯 decode 批次时
    每请求一行固定 k+1 个 token——[已提交尾 token x, d1..dk]（不足 k
    个用尾 token 补齐，验证逻辑对任意 draft 内容自校正），一次前向同时
    拿到主采样 o[0]（=target_token）与逐位置验证输出 o[1..k]，接受最长
    匹配前缀+1 修正 token，未接受槽位 rollback。与两前向流程逐 token
    等价，但省掉独立 verify 前向的 launch/CPU 常数项——这是 0.6B 级别
    小模型上投机有没有净收益的分水岭（v15：eager 前向常数项与图化
    decode 同量级）。
  - 逐请求相位判定：`num_scheduled_tokens` 非空时按请求判断，中段
    chunk 请求不产 token、不进 draft/verify（否则 append_verify_slots
    会破坏块表不变量）；混排批次回退两前向流程。
  - batch 阈值：`len(requests) > INFINILM_SPEC_MAX_BATCH_SIZE`（默认
    32）回退常规前向——大 batch decode 转向计算受限，投机放大每步
    计算量反而降吞吐（w3 bs=32 类负载的保护）。
  - **自适应收益门控**：滑动窗口（`INFINILM_SPEC_GATE_WINDOW`，默认
    32 请求步）内平均每步产出低于 `INFINILM_SPEC_MIN_AVG_TOKENS`
    （默认 2.0，实测盈亏平衡点：eager 投机步成本 ≈ 2× 图化 decode
    步）时回退常规前向 `INFINILM_SPEC_GATE_COOLDOWN` 步（默认 64），
    期满自动重试。开放生成等低命中负载开投机不致亏（见验收结果）。
  - verify 槽位分配失败（KV 块不足）退化为本请求不投机，不再抛异常。
  - 接受率埋点：`get_acceptance_stats()`（drafted/accepted/accept_rate/
    avg_tokens_per_step/verify_alloc_failures/gate_triggered），经
    `ModelRunner.get_speculative_stats()` 暴露。
- `llm/model_runner/model_runner.py`：投机开关从 `draft_model_path is
  not None` 改为 `speculative_method is not None`。
- `llm/llm.py`：LLM/AsyncLLMEngine 新增 `speculative_method` 参数；
  **拆除 spec×chunked 互斥 gate**（`_chunked_prefill_supported` 不再
  排除投机路径；mamba/多模态处理器的排除保留）。
- `dev_perf/bench.py`：`--speculative-method {none,eagle,prompt_lookup}`
  / `--draft-model` / `--num-draft-tokens`；每个 workload 记录
  spec_accept_rate 与 spec_avg_tokens_per_step 增量并随 JSON 落盘。
- `dev_perf/workload.py`：新增 w7_repetitive_copy（pattern 续写负载，
  prompt-lookup 接受率上限的演示；不依赖指令遵循能力）。

### 本机验证（无 GPU，stub 模式）

`test/test_prompt_lookup_spec.py`（复用 test_chunked_prefill.py 的 stub
harness，target 模型用确定性伪模型替代）7 用例 + 既有 4 用例全绿：

- lookup 辅助函数语义（最近出现/min-max 边界/k 截断/未命中）；
- 高命中场景输出与非投机真值逐 token 相同，accept_rate>0.9、
  avg_tokens_per_step>1.4，且 decode 批次的 forward_raw 调用形状全是
  b×(k+1)（证明走了融合路径）；
- draft 整体拒绝/部分接受时输出仍精确、verify 槽位回滚、块无泄漏；
- spec×chunked 混排：中段 chunk 产出 []、完成 chunk 即投机、混排批次
  走两前向、纯 decode 批次走融合（按调用形状断言）；
- 旧式整段 prefill 批次投机、非 greedy 回退；
- 自适应门控：零命中负载攒满 32 步窗口后触发回退，冷静期内不再产生
  融合前向调用，输出仍逐 token 精确。

### 5090 验收步骤

```bash
# 基线（无投机）
python dev_perf/bench.py --engine infinilm --model Qwen/Qwen3-0.6B \
    --enable-graph --attn-backend hybrid \
    --only w1_short_decode,w4_long_decode,w7_repetitive_copy --dump-outputs
# 投机（prompt_lookup, k=4）
python dev_perf/bench.py --engine infinilm --model Qwen/Qwen3-0.6B \
    --enable-graph --attn-backend hybrid \
    --speculative-method prompt_lookup --num-draft-tokens 4 \
    --only w1_short_decode,w4_long_decode,w7_repetitive_copy --dump-outputs
# 正确性：贪心输出必须逐 token 相同
python dev_perf/compare_outputs.py results/infinilm_*baseline*.json results/infinilm_*spec*.json
```

看点：w7 的 spec_accept_rate 与 avg_tokens_per_step（预期接近 k）、
w1/w4 的 ms/tok 差（开放生成接受率低，可能接近打平——这本身就是
prompt-lookup 的已知边界，EAGLE 头是后续解）、w3 不回归（batch 阈值
保护）。spec×chunked 组合验收：`INFINILM_ENABLE_CHUNKED_PREFILL=1` +
`--speculative-method prompt_lookup` 跑 w5/w6 对拍。

### 5090 验收结果（2026-09-01，Qwen3-0.6B，hybrid + graph，k=4）

**性能**（该 VM 当天有分时 CPU 争抢，绝对 ms/tok 在观测窗口内漂过
2~3×；表中为紧邻交错对拍的数字，比值才是可靠信号。接受率是确定性
计数，多轮逐位相同）：

| workload | 基线 | prompt_lookup | 收益 | accept | avg tok/step |
|---|---|---|---|---|---|
| w1_short_decode | 2.844 / 2.683 ms/tok | 2.791 / 2.692 | ≈打平（门控生效） | 0.781 | 1.68 |
| w4_long_decode | 3.174 / 3.144 | 1.360 / 1.288 | **2.3~2.4×** | 0.982 | 4.49 |
| w7_repetitive_copy | 3.169 / 3.176 | 1.427 / 1.401 | **2.2~2.3×** | 0.958 | 4.03 |
| w5_concurrent_prefill（chunked） | 2.364 | 1.114 | **2.1×** | 0.926 | 4.57 |
| w6_decode_stall（chunked） | 1564.6 tok/s | 2451.3 tok/s | **+57%** | — | — |

门控的价值有直接对照：无门控的首轮（机器较空闲窗口）w1 从 1.744
回归到 2.551 ms/tok（-46%，accept 仅 0.55）；加门控后 w1 与基线
打平（128 个 token 里约 32+32 步投机探测，其余回退图化 decode），
高命中负载不受影响。

**正确性**：

- 基线确定性成立：同配置连跑两次 w1/w4 输出逐 token 相同。
- 全 exact：w7（512 tok）、w5（8/8 请求）——高命中负载里正确 token
  的 logit 遥遥领先，数值噪声翻不动 argmax。
- 非 exact：w1（token 9 分歧）、w4（token 32）、w6（7/9 exact，
  最早 token 1）。定位为 **near-tie argmax 翻转**，非逻辑 bug：
  - HF 探针实测分歧位置的 top-2 logit 间隙：w1 = 0.125、w4 = 0.000
    （完全平局）；分歧两侧文本都通顺且语义等价（"对猫的误解" vs
    "的好奇心"、"如何进行交流" vs "如何交流"）。
  - 机制：spec 的 verify/融合前向与基线 decode 的 batch 形状不同
    （b×(k+1) eager vs b×1 图化），归约顺序差异产生 O(0.01~0.1)
    的数值噪声，足以翻转平局。w6 的 token-1 分歧是同一机制上移
    一层：spec 改变各请求进度 → 混排批次组成不同 → 主前向数值微差。
  - 结论：投机采样"数学无损"指分布等价，不是逐位等价（vLLM 同样
    不保证逐位一致）。验收门槛应理解为：高命中负载必须 exact
    （w5/w7 已满足），开放生成负载看分歧点 logit 间隙是否近平局。
- 反面边界不变：大 batch（w3 bs=32 类）由 batch 阈值直接回退，不回归。

**w1 类开放生成要真正获益，需要 EAGLE 头**（接受率 0.55 → 2.5~3.5
才有净收益），这是后续项；prompt-lookup 的定位是把链路和验收跑通 +
覆盖重复性负载（代码、总结、agent 循环输出）。

### 遗留：verify/融合前向的 CUDA graph 录制（③b 的 C++ 半）

融合前向目前 eager（sample_all_positions=True 被
`csrc/engine/rank_worker.cpp:428` 的门禁挡在图外）。录图改动点
（已定位，待有 GPU 构建环境时实施）：

1. `rank_worker.cpp:428` 放宽 `!sample_all_positions` 门禁，让
   all-position 输入可进编译器；
2. `paged_compiler.cpp` 新增按 (batch, k+1) 键的 verify 图表（tuple
   key 仿 static_batching_compiler.hpp:30），compile() 新增录制分支：
   pack_i64/pack_i32 缓冲按 k+1 放大，录制输入置
   `sample_all_positions=true`；
3. `get_compiled` 的 decode-only 检查（`input_ids->size(1) != batch`）
   改为分支：命中 (b,k+1) 表走 verify 图；
4. hidden_states 不进图输出时 `forward_raw`（infer_engine.py）需容忍
   null（runner 只消费 output_ids）；
5. 隐性依赖：录制时冻结的 host 标量 max_query_length=k+1 没问题，
   max_sequence_length 随 ctx 变化——需确认 FA varlen 路径不消费它做
   内容相关分支；
6. 配置链：k 需在编译期可知（LLM→EngineConfig→InferEngine→pybind→
   RankWorker→PagedCompiler），或在首次遇到该形状时 lazy 录制。

预期收益：融合前向图化后，0.6B decode 每步成本回到图化单前向量级，
投机的理论收益（w1/w4 类小 batch 延迟敏感负载 2× 级）才完全兑现。

---

## v15（2026-09-01）：bs=1 decode step 的 kernel 级归因

动机：w1 实测（0.6B 1.74ms/tok）对带宽 roofline（~1.2GB/step ÷
1.79TB/s ≈ 0.67ms）看似只有 39% 达成率，需定位缺口再定优化方向。

### 方法

nsys `-t cuda --cuda-graph-trace=node` 采集 w1（graph 模式）；
`graphNodeId IS NOT NULL` 过滤图内 kernel，按 >100µs 时间间隙切
replay（两模式 342 kernel/replay，稳态 254 个 replay），逐步中位数
统计。分析脚本 `/root/step_breakdown{,2..5}.py`（5090），profile：
`/root/prof_w1_06b_n.nsys-rep`、`/root/prof_w1_17b.nsys-rep`。

### 0.6B decode step（wall 1390µs，busy 1357µs，图内 idle 仅 2%）

| kernel | µs/step | n/step | µs/call | 推断调用点 | 字节 | 带宽达成 |
|---|---|---|---|---|---|---|
| gemvx g(768) | 277.7 | 28 | 9.92 | gate_up [1024→6144] | 12.6MB | 1.27 TB/s (71%) |
| gemvx g(256) | 272.4 | 56 | 4.86 | o + down (N=1024) | 4.2/6.3MB | ~1.08 TB/s (60%) |
| gemvx g(512) | 203.6 | 28 | 7.27 | qkv [1024→4096] | 8.4MB | 1.15 TB/s (64%) |
| gemvx g(18992) | 189.9 | 1 | 189.9 | lm_head [1024→151936] | 311MB | **1.64 TB/s (92%)** |
| add_rmsnorm | 104.7 | 56 | 1.87 | residual+norm | ~KB 级 | 纯延迟 |
| paged splitkv cta+combine | 155.7 | 56 | — | decode attention | KV 小 | — |
| pagedCaching | 42.3 | 28 | 1.51 | KV 写入 | 小 | 纯延迟 |
| rmsNormRope ×2 | 70.7 | 56 | 1.2~1.3 | q/k norm+rope | 小 | 纯延迟 |
| SwiGLU | 33.8 | 28 | 1.21 | MLP 激活 | 小 | 纯延迟 |
| sampling/embedding | ~6.5 | — | — | 含伴随图 argmax | — | — |

### 1.7B decode step（wall 2815µs，busy 2785µs）

GEMV 合计 2353µs / 3.44GB = **1.46 TB/s (82%)**：gate_up 89%、
qkv 75%、o+down 71%、lm_head 92%。小 kernel 合计 ~432µs (15%)。

### 归因结论（修正粗估）

1. **GEMV 并不烂**：decode 用 cuBLAS gemvx，大矩阵贴近实测峰值
   （lm_head 92% ≈ 实用上限 1.65TB/s）；小矩阵（o/down/qkv）60~75%，
   是 launch ramp + 矩阵太小的物理下限，可榨空间 ~10~16%（0.6B）。
2. **0.6B 的三个真实资金池**（按大小）：
   - 小 kernel 延迟 ~413µs（30%）：225 个 1~2µs 级 kernel 的 dispatch
     地板。融合方向：add_rmsnorm 进 GEMV epilogue、pagedCaching 进
     rope/attention、SwiGLU 并进 down epilogue。预期回收 ~200µs。
   - 图外 CPU ~0.35ms/step（20%，bench 1.74ms − 图 1.39ms）：打包
     H2D + graphLaunch + Python 调度。结构性解法 = async scheduling
     （step N 执行时跑 step N+1 的调度/更新，vLLM 0.9 同款）。
   - GEMV 70→85~92%：~150~220µs，需自研 split-K GEMV 或 cuBLASLt
     启发式调优，收益上限最小。
3. **1.7B 空间更小**：GEMV 已 82%，小 kernel 15%——bs=1 小模型的
   常数项随模型变大被摊薄，**kernel 微优化路线天花板清晰可见**。
4. 真正的数量级杠杆仍在 **FP8 权重/KV**（带宽减半，与上述乘算；
   lm_head 一项即占 14% step，FP8 后立省 ~95µs@0.6B）与**投机采样**
   （decode 2× 级）。两者基建已在仓库（quantization_method /
   kv_cache_k_scale / draft_model_path / speculative_cache_ops）。

---

## v14（2026-09-01）：v13 5090 复核 + w6 decode-stall 负载

动机：v13（chunked prefill + prefill/decode 混排）提交时仅有本机
Python 单测，GPU 侧未验证；且验收负载 w5 是「同时到达」形态，测
不出混排的真实收益（decode 流被长 prefill 队头阻塞的场景）。

### 部署与正确性（5090，hybrid+graph，64blk，prefix caching 开）

远端 `/root/src/InfiniLM-hybrid` 为 v12 等价树（md5 逐文件比对），
v13 纯 Python，推 6 文件复用 C++ binary。llm.py 上 v11 遗留的
stepprof 插桩 patch 被 v13 版覆盖（已备份 llm.py.stepprof.bak）。

- gate 探针确认：`INFINILM_ENABLE_CHUNKED_PREFILL=1` 时
  `enable_chunked_prefill=True`（注意 bench 下 logging 不出 INFO，
  「Chunked prefill enabled」日志不可见，只能用探针确证）。
- 0.6B 全负载矩阵（w1~w5，budget=1024 强制切块 vs 关 vs 默认
  budget）：三组两两对拍均 43/43 请求逐 token exact。
- 1.7B w2+w5（budget=1024，on vs off）：9/9 exact。
- 远端 test_chunked_prefill.py 4 用例全绿。

### w6_decode_stall：decode 长流 + 中途注入长 prefill

形态：8 条短 prompt 各 decode 512 tok，第 64 步注入 1 条 ~6.5k tok
prompt（nonce 头避开 prefix caching），逐步计时。budget=1024。

| 指标 | 0.6B 关 | 0.6B 开 | 1.7B 关 | 1.7B 开 |
|---|---|---|---|---|
| decode 步均值（注入前） | 2.4~2.5ms | 2.6~2.8ms | 4.0~4.1ms | 4.1ms |
| 注入后最大步长 | 75.9/74.9ms | 21.8/19.4ms | 138.6ms | 31.3ms |
| 注入后 p90 步长 | 2.9ms | 3.2ms | 4.4ms | 4.4ms |
| 注入请求 TTFT | 75.9ms | 106.1ms | 138.6ms | 185.2ms |
| e2e | 1.56s | 1.61/1.62s | 2.40s | 2.41s |

结论：chunked 把 decode 流最坏 ITL 尖峰削 **3.5×（0.6B）/ 4.4×
（1.7B）**，p90 几乎不动；代价是注入请求 TTFT +40%/+34% 与 e2e
~3%（切块后小 kernel + 混排 eager 步的开销）。w5（同时到达形态）
ABBA 下 e2e 反而亏 ~5%（无 decode 流量可保护）——chunked 的价值
在在线服务的延迟平稳性，不在离线批量吞吐。

### 1.7B w6 的 on/off 分歧归因（非逻辑 bug）

on vs off 出现 5~6/9 exact、token 80~108 分叉。排查链：off×3 两两
9/9 exact；on×2（同配置）9/9 exact（on 模式确定性）；关 v12 ctx
路由（INFINILM_DECODE_CTX_THRESHOLD=999999）后 on vs off 仍
5/9@80；路由开/关的 on 两跑互对 5/9@80。结论：混排步中 decode
token 走 eager varlen（而非 decode 图），叠加 6.5k ctx 触发 v12
FA 路由，kernel 归约顺序差的 epsilon 被贪心放大——与 v9/v11 已
接受的 kernel 切换分歧同类。0.6B 全 exact；1.7B w5（纯 prefill
切块，无 decode 混排）9/9 exact 说明切块逻辑本身无数值偏差。

数据归档：`results/5090_v13_chunked/`（23 份：全矩阵×4、w5
ABBA×4、1.7B w2/w5×2、w6 0.6B×4、w6 1.7B×5）。远端遗留
run_bench.py / probe_gate.py / llm.py.stepprof.bak。

---

## v12（2026-08-29）：decode 按 ctx 长度自适应路由 —— w2@1.7B 倒退根治

动机：v9 遗留「hybrid 在长 ctx decode 输给 FA kvcache」（w2@1.7B 0.75
vs 0.62，交叉点 1k~3.3k 未定）。本轮先做 ctx 扫描把交叉点定死，再落
路由。

### ctx 扫描（新增 `bench.py --ctx-sweep`，单请求 decode 128 tok，eager）

`_BASE_PARA×k` ≈ 81k token 提示词，两后端 prefill 同为 FA2，e2e 差即
decode kernel 差。ms/tok（越低越好）：

| ctx | 0.6B hybrid | 0.6B FA | 1.7B hybrid | 1.7B FA |
|---|---|---|---|---|
| 162 | **2.79** | 4.25 | **3.39** | 4.24 |
| 1296 | **2.80** | 4.24 | **4.25** | 4.33 |
| 1944 | **3.13** | 4.35 | 4.70 | **4.38** |
| 2592 | **3.51** | 4.32 | 5.11 | **4.45** |
| 3240 | **3.94** | 4.28 | 5.57 | **4.55** |
| 3888 | 4.36 | **4.29** | 5.96 | **4.58** |
| 5184 | 5.26 | **4.42** | 6.92 | **4.73** |

两个结构性发现：

1. **FA kvcache 随 ctx 几乎平坦**（1.7B 斜率 ~0.1µs/tok），splitkv 线性
   增长（0.6B ~0.49、1.7B ~0.70 µs/tok）——FA kernel 内对 KV 长度方向
   并行得好，splitkv 在 bs=1 并行度不够。交叉点 **0.6B≈3.4k /
   1.7B≈1.5k**，随模型几何移动，不能写死全局常数。
2. **FA decode 的固定底线与模型大小无关**（0.6B/1.7B 同為 4.24ms）——
   短 ctx 下 FA 路径是固定开销（eager launch/边界）主导而非带宽主导。
   上表是 eager 数据；graph 模式吃掉固定开销后交叉点会移动，表内阈值
   用于 graph 选图是偏保守的（1.7B w2 实测改善即在 graph 模式拿到）。

### 实现（本分支工作区）

- `attention_layer.{hpp,cpp}`：`decode_fa_ctx_threshold()`——env
  `INFINILM_DECODE_CTX_THRESHOLD` 优先；否则按 (hidden, layers,
  kv_heads) 查实测表：Qwen3-0.6B(1024/28/8)→3400、
  Qwen3-1.7B(2048/28/8)→1500；未测几何 SIZE_MAX（**不路由=现状**）。
  `HybridAttentionImpl::forward` decode 分支：
  `max_sequence_length > 阈值` → `flash_->forward`（mha_kvcache_）；
  两 kernel 读同一块 BSHD paged cache（splitkv 走 permute 视图），
  **切换零拷贝**。
- `infer_engine.cpp`：decode 步也填 `max_sequence_length`（host 侧 max
  over CPU int32 tensor，无同步开销；非 CPU/I32 返回 0=不路由）。
- `paged_compiler.{hpp,cpp}`：路由启用时每个 batch 档**录双图**——假
  msl=1 录 splitkv 版、假 msl=阈值+1 录 FA 版（录制时算子只登记不执行，
  假值只冻结 kernel 选择，replay 读真实 pack_i32）；`get_compiled` 按
  host 侧 max ctx 选图（非 CPU/I32 保守走短 ctx 版）；采样伴随图每变体
  各录一份，`get_sampling_compiled` 跟随 `last_served`。两变体共享
  block_tables_holder_，block table 更新一处生效。

### 正确性与性能（5090，hybrid+graph，--dump-outputs）

- **0.6B 全矩阵对 v11 构建逐 token 全 exact**（阈值 3400，w2 峰值 ctx
  3368 不触发路由；双图重构未改变短路径）。
- **1.7B**：w1/w2/w4 对 v10 dump 全 exact、w3 31/32@109（既有批噪声底
  同类）；**w2 路由后输出与 flash-attn 后端逐 token 全 exact**（graph
  与 eager 均验证）——路由前后 kernel 归约顺序差在该 prompt 的 128
  token 内未翻牌，且路由结果确实落在 FA kernel 上。
- **性能**：1.7B w2 e2e **0.74→0.55s（-26%**，graph）/ 0.75→0.66s
  （eager），优于 flash-attn 后端自身的 0.64s（eager）；w1/w3/w4 与
  0.6B 全负载持平 v11（噪声内）。

### 边界与后续

- 阈值是 bs=1 eager 定的；**bs×ctx 第三象限（大 batch × 长 ctx）无数据**
  ——bs=32 时 batch 方向自带 32 倍并行度，splitkv 短板被补，交叉点可能
  大幅后移，agent 高并发场景「谁赢」待网格扫描（注意显存：两模型 KV
  均 112KB/token，32GB 上 bs32×16k≈57GB 不可行，网格上限 bs32×8k 或
  bs8×16k）。
- 与 vLLM 的对照最大只测到 3.4k ctx；agent 主战场（10k+ 多并发）的
  立项差距完全未知。
- 中途切 kernel 引入与「换后端」同量级的 ulp 扰动（v9 已记录），greedy
  可能翻接近的 top-2；对未测几何默认关闭，零回退风险。

### 待办更新

- [x] ~~decode 按 ctx 长度自适应路由~~ —— 本轮落地（w2@1.7B 0.74→
      0.55s），阈值为 bs=1 eager 实测表 + env 覆盖，未测几何不路由。
- [ ] bs×ctx 网格扫描（bs 1/8/32 × ctx 1k/4k/8k + bs8×16k，graph 模式）
      ——回答 agent 象限 splitkv/FA 谁赢，并校准 graph 模式阈值。
- [ ] 长 ctx 多并发对 vLLM 对照（agent 主战场的立项差距）。
- [ ] hybrid+graph 设为默认的评估（沿用 v10）。
- [ ] （可选）5060 Ti 复核；hybrid 推广其他模型族；flashinfer 采样器。

数据归档：`results/5090_graphopt/`（0.6B `..._0829_154455`、1.7B
`..._154513`、1.7B FA 参考 `..._154523`、eager 路由 `..._154645`）。

---

## v11（2026-08-29）：图外开销优化落地 —— 打包 H2D + 采样伴随图

动机：v10 归因指出 graph 模式每步仍有图外开销——6 个小输入各一次
H2D copy、采样段 32 请求 × 3 launches（cub ArgMax×2 + cast）。本轮把
这两处削掉并实测收益。

### 改动（本地未提交，csrc/engine/{compiler/*,rank_worker.cpp}）

1. **打包 H2D**：`make_decode_input` 把 input_ids / position_ids /
   slot_mapping 打进一条连续 int64 缓冲（pack_i64），total_seq_lens /
   input_offsets / cu_seqlens 打进一条连续 int32 缓冲（pack_i32），
   图输入做成两条缓冲的视图；`get_compiled` 快速路径两次 memcpyH2D
   代替原 5+1 次 copy_from；block table 在 block_per_req==编译宽度时
   跳过 -1 填充。
2. **采样伴随图**：`compile()` 对 b≤64 的每个 decode 图录一个
   companion graph（逐请求 argmax：cub DeviceReduce ArgMax + 索引
   cast），greedy（top_k==1 或 temperature==0）且 decode 图命中时
   replay 它，代替每步 32×3 次 launch。kill switch：
   `INFINILM_DISABLE_SAMPLING_GRAPH=1`；调试：
   `INFINILM_DEBUG_SAMPLING=1`（命中时打印一次）。

### 关键坑（排查记录，后续接图的人必读）

infinicore 的 `Graph` **不是裸 CUDA stream capture，是算子序列录制**：
只有经 `INFINICORE_GRAPH_OP_REGISTER_*` 注册的算子类才会在录制时进入
`op_list_`（录制模式下算子只登记不执行；`instantiate()` 先跑 5 遍
warmup 再按 capture 安全性切段捕获）。`random_sample` 是纯 dispatcher
注册（`random_sample_infiniop.cc` 无 graph 钩子），greedy 路径又被
`tryGreedyWithInfiniOps`（infiniops Argmax）在 dispatcher 之前拦截——
直接 `startGraphRecording() + random_sample_` 录到的是 **operators=0
的空图**，replay 等于空操作，读到的是编译期陈旧 logits 的 argmax 结果
（编译用全零退化输入，total_seq_lens=0 → logits 为 NaN 垃圾 → 越界
token id，`tokenizer.decode` 抛 OverflowError）。修复：自定义
`SamplingLoopOperator`（`GraphOperator` 子类）包装整个采样循环，
`context::addGraphOperator` 手动入列，由 `instantiate()` 统一 warmup+
捕获。第二坑：包装 lambda 必须**按值捕获**张量（操作符比 compile()
作用域活得久，且持锁张量内存）。

### 正确性

hybrid+graph 全矩阵 `--dump-outputs`（`..._0829_145012.json`）与修复前
已验证版本（`5090_graphopt/..._0829_141752.json`）**w1/w2/w3/w4 逐
token 全 exact**（32/32 请求全对）。采样图与 eager argmax 语义一致。

1.7B 复验（w3_batch32，同构建采样图 ON vs kill switch）：31/32 exact
@token90；同配置两个进程互跑（采样图 ON vs ON）也是 30/32 @token90
——偏差来自 forward 自身的运行间抖动（v10 记录的 1.7B bs32 批噪声底
同类），与采样路径无关。1.7B w3 吞吐 6214~6269 tok/s，与 v10 graph
基线（6211）持平。

### 性能（同构建 ABBA，kill switch 交替，0.6B w3_batch32）

| 轮次 | A=采样图 ON | B=eager 采样 |
|---|---|---|
| 1 | 11214 | 9530 |
| 2 | 9547 | 9499 |
| 3 | 9526 | 10275 |

中位数 9547 vs 9530 tok/s —— **差 ~0.2%，在 5090 热双态噪声内**。
STEP_PROF 分段（A vs B，step 64/128 均值）：forward 段 1.67 vs 1.68ms
（稳定省 ~10µs/step），sched/post/gap 不变。

### 结论

1. **图外开销在 graph 模式本就被异步隐藏**：采样 launch 在 forward 图
   仍在 GPU 执行时就已发出（gap≈0），削掉它 e2e 不动。本轮改动保留
   （每步省 ~10µs CPU + 消除采样段分配搅动，对更大 batch 或 CPU-bound
   场景仍有意义；正确性已验证，且有 kill switch）。
2. **瓶颈回到 forward 图本身**：bs32 每步 ~1.6~2.1ms 是图内 GPU 时间。
   下一步方向：图内 kernel 归因（GEMM / hybrid attention / mamba 段在
   bs=32 的占比）、以及与 vLLM 的调度层差距（continuous batching 的
   step 内重叠），图外已无可削。

---

## v10（2026-08-29）：5090 全栈对照 —— InfiniLM(hybrid) vs vLLM 0.28

动机：v8/v9 遗留的「立项决策最后一块」——此前全部 vLLM 数据出自
5060 Ti（WSL2）与 8GB 病态机，vLLM 从未在健康平台跑过。本轮在 5090
建起 vLLM 独立 venv 并跑全负载矩阵，InfiniLM 侧用 v9 的 hybrid 基线。

环境：

- **vLLM 0.28.0 + torch 2.13.0+cu130**（`/root/.venv-vllm`，uv 0.9.9
  经阿里云镜像安装）：v1 默认（CUDA graph on、prefix caching on、
  chunked prefill on），gpu_memory_utilization=0.85，attention 自动选
  **FLASH_ATTN**（vllm 内置 FA2，sm_120 候选序列首位）。两个坑见文末
  「5090 装 vLLM 工程记录」。
- **InfiniLM hybrid**：64blk、no-graph（沿用 v1~v9 对比惯例），另加测
  一组 `--enable-graph`（hybrid×graph 首次实测，顺带验证兼容性）。
- 同树同机同日：hybrid ×2 轮、hybrid+graph ×1 轮、vLLM ×2 轮，全部
  `--dump-outputs`；hybrid 同日两轮与 v9（0827）数值互验一致。

### 性能（两轮均值；vLLM Δ 列为对 hybrid / 对 hybrid+graph）

**0.6B**（hybrid 峰值 4.1GB，vLLM 28GB@0.85 档）：

| 负载 | hybrid | hybrid+graph | vLLM | Δ vs hybrid | Δ vs hy+graph |
|---|---|---|---|---|---|
| w1 单请求 decode | 2.92 ms/tok | 1.76 | **1.63** | **-44%** | -7% |
| w2 长 prefill + 128 decode | 0.53 s | 0.53 | **0.26 s** | **-50%** | **-50%** |
| w3 batch32 | 8621 tok/s | 9479 | **15510** | **+80%** | **+64%** |
| w4 长 decode | 2.93 ms/tok | 2.03 | **1.60** | **-45%** | **-21%** |

**1.7B**（hybrid 峰值 7.2GB，vLLM 27.6~28.2GB）：

| 负载 | hybrid | hybrid+graph | vLLM | Δ vs hybrid | Δ vs hy+graph |
|---|---|---|---|---|---|
| w1 | 3.30 ms/tok | 3.20 | **3.10** | **-6%** | -3% |
| w2 | 0.75 s | 0.74 | **0.48 s** | **-36%** | -35% |
| w3 | 6162 tok/s | 6211 | **8767** | **+42%** | +41% |
| w4 | 3.45 ms/tok | 3.50 | **3.06** | **-11%** | -12% |

（加载耗时参考：hybrid ~6s；vLLM 冷启 46s / 编译缓存命中后 18~22s。）

### 结论

1. **健康平台上 vLLM 全面领先**，5060 Ti 的「decode 持平、仅 prefill
   差 1.3~1.5×」结论不能外推。差距结构：
   - **w3（batch32 吞吐）差距最大**（+42%~+80%），其次 **w2**
     （-36%~-50%）——指向批处理调度、chunked prefill 与全图捕获的
     decode 段，而非单个 kernel；
   - 单请求 decode（w1/w4）1.7B 仅差 6%~11%（hybrid 无 graph 对
     vLLM 有 graph），0.6B 差 44%~45%（小模型 launch-bound 占比高）。
2. **graph 不是差距主因**：hybrid+graph 在 0.6B decode 上 -31%~-40%
   （0.6B launch 开销占比确实大），但补不回 w3/w2 的差距（仍落后
   41%~64%）；1.7B 上 graph 收益在噪声内（±3%）。hybrid×graph 首次
   实测无回退，输出与 no-graph 对照 0.6B 四负载全 exact、1.7B 仅
   w3 30/32（@93，既有批噪声底同类），作为默认配置可行。
3. **v9 的内部结论不受影响**：hybrid 仍是 InfiniLM 三后端中最优单配置；
   但对 vLLM 的立项差距在健康平台上重新打开，下一轮优化目标应转向
   w3/w2 的归因（见待办）。

### w3 差距归因（nsys，0.6B，同日补抓）

方法：nsys 2024.6 抓 `--only w3_batch32` 全 trace，sqlite 导出后按 w3
窗口切片聚合（脚本 VM `/root/slice_w3.py`，trace 存 `/root/prof/v10/`）。
w3 = 32×~15tok prefill + 128 步 batch32 decode，**decode 段主导**。
注意：nsys 对 CUDA graph replay 内的 kernel 记录不全（vLLM 侧 window
busy 仅 9%，物理上不可能——0.6B×bs32 单权重读取就需 ~0.7ms/step），
vLLM kernel 级数字仅作参考，wall/launch 数可信；hybrid 侧记录完整。

**hybrid no-graph**（w3 窗口 570ms）：

- **每 decode step ~435 次 kernel launch**：gemm 4 次/层（qkv/o/gate_up/
  down 已是融合 gemm）+ 2 次 cublas splitk 伴随 + add_rms_norm 2 +
  rms_norm_rope 2 + swiglu 1 + cache 1 + attn 2 + ~4 次小 elementwise；
- **GPU busy 仅 306ms（54%）**；未 profile 口径 wall 3.7ms/step vs busy
  ~2.4ms/step → **~35% 的 wall 是 launch 空泡**；
- kernel 时间大头是 gemm（窗口内 248ms）：bs=32 的 gemm 平均 13.7µs，
  已在 latency floor，kernel 本身无多少油水。

**hybrid+graph**（e2e 0.43s，仅 +10%）：

- `PagedCompiler` 对 bs=1..64 逐档捕获 decode-only 图，bs=32 在列——
  w3 decode 确实走了 graph replay，launch 空泡大头已消除；
- 但每步图外仍有残余开销：5×D2D 输入拷贝 + 每次 replay 前的 reset
  memset（`paged_compiler.cpp:165` 注释自承「still pays a memset before
  every graph replay」）+ 采样 kernel + Python 调度，合计 ~0.7ms/step；
- 图内 GPU 时间 ~2.4ms/step 不变 → wall 3.1ms/step，**从 launch-bound
  转为 kernel-time-bound**。

**vLLM**（e2e 0.26s，2.0ms/step）：整步单次 graph replay + inductor
combo/fused kernel（triton 融合 norm/silu、combo gemm），图外仅少量
采样 kernel。

**归因结论**：w3 的 +80% 差距 = ①launch 空泡（hybrid 有 graph 但图外
残余未削）+ ②图内 kernel 时间本身（bs=32 gemm 贴 latency floor，靠
数量取胜——vLLM 融合度更高）。后续方向：a) 削图外 per-step 开销（合并
5 次输入拷贝、去掉 per-replay memset、采样入图）；b) 评估跨 gemm 的
进一步融合/grouped gemm。w2 的账独立于本次（v9 已定位：3.3k ctx 下
decode 段变慢 + 无 graph），需要补 trace 时另抓。

### 正确性（`compare_outputs.py`，双轮互验）

- **同引擎跨轮噪声底**：hybrid 0.6B 全 exact、1.7B w3 30/32（@91，
  与 v9 底同位置）；vLLM 自身 w3 也非确定（0.6B 27/32、1.7B 25/32，
  批处理数值噪声）。
- **跨引擎**：1.7B w1/w2 全 exact，w4 @190 与 v4/v5 历史分叉位置
  完全重合；0.6B w2 exact，w1 @0 首 token 翻牌（hybrid「这个问题可能
  引发一些人对猫的误解…」连贯，vLLM 落入重复循环——0.6B 贪心退化的
  常见形态，双侧均为模型自身质量范围内输出），w4 @32 与 v9
  hybrid-vs-fa 的 @32 重合；w3 22~27/32 落在双方噪声底交集内。
- 判定：**无功能性错误**，全部为 bf16 并列翻牌 × 自回归放大。

### 5090 装 vLLM 工程记录（复现要点）

- **无外网**：NGC 镜像自带 `/etc/pip.conf` + `/etc/xdg/pip/pip.conf`
  配了 pypi.org 主 index + `pypi.ngc.nvidia.com` extra-index，`-i` 只
  覆盖主 index，包查询全卡在 ngc 的 TCP 超时——已将两个文件挪为
  `.bak`。pip 单连接仍被限速 ~200KB/s（同文件 curl 直连 11MB/s，
  aliyun/tuna 同速），改用 **uv**（并发下载）安装。
- vLLM 0.28 运行需 `CUDA_HOME=/usr/local/cuda-12.8` + venv 内 ninja
  （README 已有此条）。
- **flashinfer 采样器 JIT 在 sm_120 上要求 nvcc≥12.9**（编译
  `compute_120f`），VM 工具包为 12.8 → 引擎 warmup 直接报
  「FlashInfer requires GPUs with sm75 or higher」（TARGET_CUDA_ARCHS
  为空后的误导性报错）。对策：`VLLM_USE_FLASHINFER_SAMPLER=0` 走
  原生采样器；贪心解码（temperature=0/top_k=1）不受影响。attention
  不受影响——Qwen3 在 sm_120 默认 FLASH_ATTN。
- 堡垒机 scp 逐文件重新鉴权，批量拉取需在 VM 侧先打 tar 包。

### 待办更新

- [x] ~~w3/w2 对 vLLM 差距的 nsys 归因~~ —— w3 已由 v10 补抓完成：
      launch 空泡（图外残余 per-step 开销 ~0.7ms/step）+ 图内 kernel
      时间（bs=32 gemm 贴 latency floor）。派生优化项：削图外开销
      （合并输入拷贝/去 per-replay memset/采样入图）与 gemm 融合度。
      w2 归因沿用 v9 结论（长 ctx decode + 无 graph），需要时补 trace。
- [ ] hybrid+graph 设为默认的评估（0.6B decode -31%~-40%、四负载无
      回退；需补更多模型的正确性对拍）。
- [x] ~~decode 按 ctx 长度自适应路由~~ —— 已由 v12 落地（w2@1.7B
      0.74→0.55s，阈值查表 + env 覆盖，未测几何不路由）。
- [ ] （可选）5060 Ti 复核 hybrid；hybrid 推广到其他模型族（沿用 v9）。
- [ ] （可选）flashinfer 采样器恢复：pip 侧装 `nvidia-cuda-nvcc-cu13`
      或升级工具包至 ≥12.9；当前 FLASH_ATTN + 原生采样器已够用。

数据归档：`results/5090_vllm/`（10 份：hybrid×2、hybrid+graph×1、
vLLM×2，双模型）；v9 的 15 份同步拉回 `results/5090_hybrid/`。VM 侧
venv `/root/.venv-vllm`，运行环境变量：
`HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-12.8 VLLM_USE_FLASHINFER_SAMPLER=0`。

---

## v9（2026-08-27）：hybrid 后端落地 —— prefill 走 FA2 / decode 走自研 splitkv

动机：v8 发现 5090 上 FA2 只赢 prefill、decode 慢自研 splitkv 23%~54%，
「flash-attn 设为默认」不成立。本轮把分离路由做成一个后端：
`--attn-backend hybrid`。

实现（全部在本分支工作区，零新增文件）：

- `attention_backends.hpp`：新增 `AttentionBackend::HYBRID` + 字符串解析
  `"hybrid"`；Python 侧 `llm.py`/`bench.py` 透传（pybind 原样走
  `parse_attention_backend`，无需其他改动）。
- `attention_layer.{hpp,cpp}`：新增 `HybridAttentionImpl`（内含一个
  `FlashAttentionImpl`）。`is_prefill` 判定与既有 impl 相同（展平 paged
  模式下纯 decode 步恰为每序列 1 个 query token）；prefill/混合 batch →
  `flash_->forward`（mha_varlen），纯 decode → 复用
  `flash_->do_kv_cache_update`（paged_caching_ 经 permuted view 写 BSHD
  cache）后，以 `k_total->permute({0,2,1,3})` 的逻辑 BHSD 视图直接调
  `paged_attention_`。
- `kv_cache.cpp`：HYBRID 与 FLASH_ATTN 同走 BSHD 物理布局；
  `infinilm_model.cpp`：HYBRID 落入 paged cache 分配分支（Qwen3 因此
  走 `forward_paged_`，自动带上 v5 的 rms_norm_rope 融合）。

strides 安全性（hybrid 成立的命门，InfiniCore 侧核实）：

- `paged_attention` descriptor（`info.h`）从 tensor descriptor 逐维取
  stride，仅要求 `stride(3)==1`（head_dim 连续）——permuted BSHD 视图
  满足（stride(0)=BS·H·D, stride(1)=D, stride(2)=H·D, stride(3)=1）。
- 实际使用的 `kernel_v2.cuh`（所有 nvidia launcher 都 include 它）按
  `k_row_stride` 逐 token 寻址；旧 `kernel.cuh` 里 `token*HEAD_SIZE` 的
  硬编码路径未被任何 launcher 引用。
- infinicore wrapper 的 FA 快速通道（`canUseFlashAttention`）两条后端
  条件一致，且该通道本来就总被喂 permuted 视图；本平台实证 decode 走
  的是 kernel_v2（见下，w1/w4 hybrid≡paged 而远快于 FA）。

性能（5090，64blk，no-graph，同树同 binary 交错 ABBA）：

0.6B hybrid(A) vs flash-attn(B)，e2e 秒：

| 负载 | A1 | A2 | B1 | B2 | e2e Δ |
|---|---|---|---|---|---|
| w1 单请求 decode | 0.33 | 0.36 | 0.51 | 0.53 | **-33.7%** |
| w2 长 prefill + 128 decode | 0.51 | 0.53 | 0.56 | 0.57 | **-8.0%** |
| w3 batch32 | 0.39 | 0.42 | 0.65 | 0.64 | **-37.2%** |
| w4 长 decode | 2.68 | 2.81 | 4.27 | 4.33 | **-36.2%** |

0.6B hybrid(C) vs paged-attn(D)，e2e 秒：

| 负载 | C1 | C2 | D1 | D2 | e2e Δ |
|---|---|---|---|---|---|
| w1 | 0.35 | 0.36 | 0.35 | 0.36 | 持平 |
| w2 | 0.53 | 0.53 | 0.64 | 0.64 | **-17.2%** |
| w3 | 0.45 | 0.46 | 0.44 | 0.47 | 持平 |
| w4 | 2.85 | 2.93 | 2.87 | 2.89 | 持平 |

1.7B（HY/FA 为 ABBA，PG 单跑 + v8 复测值 0.86s 佐证）：

| 负载 | HY A1/A2 | FA B1/B2 | PG | HY vs FA | HY vs PG |
|---|---|---|---|---|---|
| w1 decode | 3.26/3.26 ms | 4.20/4.37 ms | 3.23 ms | **-22.5%** | 持平 |
| w2 e2e | 0.75/0.75 s | 0.61/0.63 s | 0.84 s | **+19%（倒退）** | **-10.7%** |
| w3 e2e | 0.66/0.66 s | 0.61/0.68 s | 0.66 s | 持平 | 持平 |
| w4 decode | 3.54/3.57 ms | 4.24/4.38 ms | 3.55 ms | **-18.4%** | 持平 |

结论：

1. **设计目标达成**：hybrid decode ≡ paged decode（w1/w4 逐项持平），
   prefill 保留 FA2（w2 对 paged -17%）；0.6B 上对 flash-attn 全面
   -8%~-37%，把 v8 发现的 FA2 decode 回退全部吃回。
2. **w2@1.7B 是唯一倒退项**（0.75 vs FA 0.62，可复现、非漂移：
   A1≡A2/B1≡B2）。分解：5090 上 1.7B 3240-token prefill 仅 ~70ms，
   w2 e2e 由 decode 段（ctx 3240→3368）主导；该 ctx 下 paged kernel
   不再赢 FA kvcache——而 w4（ctx≤~1k）hybrid≡paged 仍快 FA 18%。
   **交叉点在 1k~3.3k ctx 之间**（0.6B 在 3.3k 处 hybrid 仍赢 FA，
   与 kv-head 数/几何相关）。v8 的「FA decode 慢 23~54%」因此应限定为
   短/中 ctx。精确分相计时留待 nsys。
3. 后续方向：decode 按 ctx 长度自适应选 kernel（短 ctx→paged splitkv，
   长 ctx→FA kvcache），vLLM 式调度；当前 hybrid 已是严格优于
   paged-attn 全负载、优于 flash-attn 3/4 负载的单配置。

正确性：

- **决定性**：0.6B A1≡A2、C1≡C2 四负载全 exact；1.7B A1≡A2 仅 w3
  31/32（@91，batch 调度时序噪声，与 v5 观察到的 batch 噪声底同类）。
- 对两参考后端的交叉对照：0.6B w2 三方全 exact；w1 hybrid-vs-fa @8
  = fa-vs-pg 噪声底 @8；w3 27~28/32 ≈ 底 27/32；w4 单请求 @32 并列
  翻牌（「人们如何交流」vs「人们如何进行交流」，两侧 1024 token 全程
  连贯枚举）——hybrid 对 fa 与 pg 同 token 分叉而 fa≡pg，说明 strided
  读引入 ~ulp 级 logit 差（kernel 内不同访存路径的归约顺序差），
  量级与换后端同。1.7B：w1/w2 对 paged 全 exact；w3 24~26/32 vs 底
  25/32；w4 @160/@190 = 底 @160。

数据归档：`results/5090_hybrid/`（15 份：0.6B ABBA×8、1.7B 单跑×3、
1.7B ABBA×4）。远程树 `/root/src/InfiniLM-hybrid`（本工作区快照，
非 git）；构建 `XMAKE_ROOT=y INFINI_ROOT=/root/.infini-fa xmake f -c
-m release && xmake build -j32 _infinilm && xmake install _infinilm`；
运行脚本 `/root/run_hybrid.sh`、`/root/run_hybrid17.log`。

### 待办更新

- [x] ~~decode 按 ctx 长度自适应路由~~ —— 已由 v12 落地（交叉点经
      ctx 扫描实测：0.6B≈3.4k / 1.7B≈1.5k，bs=1 eager）。
- [x] ~~InfiniLM-vs-vLLM 健康平台全栈对照~~ —— 已由 v10 完成：5090 上
      vLLM 0.28 全面领先 hybrid（0.6B decode -44%、w3 +80%；1.7B
      decode -6%~-11%、w3 +42%），新归因待办见 v10。
- [ ] （可选）5060 Ti 上复核 hybrid（v4 平台上 FA decode 无回退，
      hybrid 预期与 paged 持平）。
- [ ] hybrid 推广到其他模型族（当前仅 Qwen3 paged 路径带融合；
      hybrid 本身对任意走 forward_paged_ 的模型可用）。

---

## v8（2026-08-27）：5090 全栈合流 —— FA2 + rms_norm_rope 融合

动机：v4（FA2）与 v5（rms_norm_rope 融合）此前分别在 16GB 5060 Ti 和
8GB 病态机上验证，从未在同一健康平台上叠加。本轮在 5090 上构建
FA 版 InfiniCore（aten=y，装到 `/root/.infini-fa`），两份 InfiniLM
扩展（fused/unfused）重建对准该库，跑 flash-attn 后端的 ABBA。

构建要点（在 v6 复现要点之上新增）：

- FA 源码用 **pypi sdist**（`pip download flash-attn==2.7.4.post1
  --no-binary :all:`，阿里云镜像可达）：自带裁剪版 cutlass，够 FA 用。
- FA 的 cute 头文件包含路径：VM 侧给 `xmake/nvidia.lua` 的
  flash-attn-nvidia target 补了一行 `FLASH_ATTN_ROOT/csrc/cutlass/include`
  （上游 FA target 只加 csrc/flash_attn，此前依赖外部 CUTLASS_ROOT——
  值得上游化）。
- **不要** export 空的 `CUTLASS_ROOT`（`os.getenv` 返回 "" ≠ nil，会误开
  ENABLE_CUTLASS_API，scaled_mm 在裁剪版 cutlass 上编译不过）。
- FA bwd kernel 单文件 nvcc 峰值内存大：-j24 触发 OOM（cicc signal 9），
  **-j6** 通过；全量约 25min（含 FA 84 .cu）。
- aten=y 后 InfiniLM 两份扩展需对准 `/root/.infini-fa` 重建（C++ ABI
  一致性）。

### 0.6B flash-attn 交错 ABBA（e2e 秒）

| 负载 | A1 未融合 | A2 未融合 | B1 融合 | B2 融合 | e2e Δ |
|---|---|---|---|---|---|
| w1 单请求 decode | 0.59 | 0.60 | 0.53 | 0.54 | **-10.1%** |
| w2 长 prefill + 128 decode | 0.65 | 0.68 | 0.58 | 0.58 | **-12.8%** |
| w3 batch32 | 0.68 | 0.69 | 0.65 | 0.65 | **-5.1%** |
| w4 长 decode | 4.73 | 4.79 | 4.34 | 4.38 | **-8.4%** |

### 1.7B flash-attn 单跑对照

| 负载 | 未融合 | 融合 | Δ |
|---|---|---|---|
| w1 单请求 decode | 4.77 ms/tok | 4.33 ms/tok | **-9.3%** |
| w2 长 prefill + 128 decode | 0.71s | 0.63s | **-11.3%** |
| w3 batch32 | 5745 tok/s | 6006 tok/s | **+4.5%** |
| w4 长 decode | 4.81 ms/tok | 4.38 ms/tok | **-8.8%** |

**融合收益在 FA 栈下比在 paged-attn 栈下更大**（对比 v6：0.6B
-4%~-8% → 本轮 -5%~-13%；1.7B -1%~-4% → -9%~-11%）。机制自洽：
attention 被 FA 压快后，elementwise 链占比上升，融合的相对收益放大。

### 同树双后端对照（fused 树，5090 新现象）

| 负载 | 0.6B paged | 0.6B flash | 1.7B paged | 1.7B flash |
|---|---|---|---|---|
| w1 decode | **2.76 ms/tok** | 4.17 | **3.30** | 4.33 |
| w2 prefill e2e | 0.64s | **0.58s** | 0.86s | **0.63s** |
| w3 batch32 | **9049 tok/s** | 6285 | **5859** | 6006（≈持平） |
| w4 decode | **2.77 ms/tok** | 4.24 | **3.57** | 4.38 |

**FA2 在 5090 上 decode 慢 23%~54%**（mha_fwd_kvcache 的 sm80 时代
kernel 在 Blackwell 上效率不佳；v4 在 5060 Ti 上两者基本持平），只赢
prefill（1.7B w2 -27%）。含义："flash-attn 设为默认后端"在本平台
不成立；合理方向是 **prefill 走 FA、decode 走自研 splitkv 的分离路由**
（vLLM 即此类设计）。5060 Ti 上该结论需复核（v4 数据是融合前的）。

### 正确性（--dump-outputs + compare_outputs.py）

- FA 栈同 binary 噪声底 = 0（A1 vs A2 四负载全 exact）。
- fused vs unfused（FA 栈）：0.6B w2/w4 全 exact，w1 @8、w3 28/32；
  1.7B w2 exact，w1 @40、w3 28/32、w4 @190。
- 关键对照：同树换后端（paged↔flash，代码不变）的噪声底分叉位置
  **与 fused-vs-unfused 重合**（0.6B w1@8、1.7B w1@40、w4@160~190）——
  融合引入的数值扰动与换一个 attention 实现同量级，非功能错误。

数据归档：`results/*_rtx5090_fa_{fused,unfused}.json`（FA ABBA 八轮中的
四+1.7B 两轮）与 `*_rtx5090_paged_fused.json`（同树 paged 参考）。

### 待办更新

- [ ] InfiniLM-vs-vLLM 健康平台全栈对照（5090：vLLM 独立 venv 待建）——
      立项决策表的最后一块。
- [ ] prefill=FA / decode=splitkv 分离路由的引擎支持评估（v8 新方向）。
- [ ] （可选）5060 Ti 上复核 FA decode 回退现象。

---

## v7（2026-08-27）：elementwise 链闭环 —— nsys kernel 级证据

动机：v3 的 nsys 归因（"rmsnorm+rope+swiglu 未融合，~157ms/prefill，
~590 次小 kernel"）留下的第二优化项，在 v5 融合 rms_norm_rope 后还剩
多少？v6 期间代码走读发现 **swiglu 与 add_rms_norm 其实早已融合**：

- `Qwen3MLP = layers::MLP`（`qwen3_for_causal_lm.hpp:7`）→
  `csrc/layers/mlp/mlp.cpp:34` 调 `infinicore::op::swiglu`（InfiniCore
  的 NVIDIA 融合 kernel，2025-07 起就在上游 main）——v3 旧树同样如此；
- paged 路径 `TextDecoderLayer::forward(positions, hidden, residual)`
  走 `RMSNorm::forward_inplace(x, residual)` → NVIDIA 上
  `op::add_rms_norm_inplace`（`InfiniCore src/infinicore/nn/rmsnorm.cc:37`）。

即 v3 口径里的"swiglu 未融合"不成立（当时已是单 kernel），剩余项只有
q/k norm+rope（v5 已融合）。本轮在 5090 上用 nsys 对 w2
（3240 tok prefill + 128 decode，含一轮 warmup，即两次前向）做
kernel 级 A/B 实证：

| 成分 | 未融合（n / GPU 时间） | 融合（n / GPU 时间） |
|---|---|---|
| q/k norm+rope | rmsnormKernel 14592 / 39.4ms + ropeThreadPerItem 14336 / 28.4ms | **rmsNormRopeKernel 14336 / 21.3ms** |
| 残差+norm（两侧均融合） | add_rmsnormKernel 14336 / 31.5ms | 14336 / 31.1ms |
| MLP swiglu（两侧均融合） | SwiGLUCuda 7168 / 10.5ms | 7168 / 10.4ms |
| **全 trace kernel 总数** | **101,834** | **87,498（-14%）** |

（两侧 trace 均含 warmup+计时两次 w2；每次前向的 q/k norm+rope 从
28 层 × 4 launch 降到 ×2。）

对账：q/k norm+rope GPU 时间 67.8ms→21.3ms（两次前向合计省 ~46ms，
单次 ~23ms），叠加 launch 延迟节省，与 v6 的 w2 e2e -4.5%（0.67→
0.64s，省 ~30ms）量级吻合。

结论与待办更新：

1. **v3 的 elementwise 链项至此闭环**：paged 路径三项（残差+norm、
   q/k norm+rope、swiglu）全部单 kernel 化。"swiglu 融合"待办销项——
   无需新算子，上游既有实现。
2. trace 中剩余的大头：decode 段 paged-attn splitkv（286.7ms/两次）与
   gemm（259ms，28,704 次小 gemm —— decode 每 token 每层 5 个投影
   gemm，微 batched 化是潜在方向但收益待估）；prefill 段自研
   PagedAttentionPrefill 在 0.6B 上 53ms/次（3240 tok），v4 已证 FA2
   可再压一个量级——5090 上落 FA 是下一个候选动作。
3. 分析脚本：profile 采集与聚合命令见 v6 复现要点 + 本轮
   `nsys profile -t cuda` + 自研聚合脚本（分类统计 kernel 名）。

---

## v6（2026-08-27）：5090 复测 —— rms_norm_rope 收益确认，量级 -4%~-8%

动机：v5 在 8GB WSL2 病态平台（显存驻留超 ~3~4GB 后带宽崩塌至
1~11 GB/s）测得 -4%~-23%，需在健康平台复测量级。本轮在租用
RTX 5090 32GB（Gitee AI 容器，CUDA 12.8 / driver 610.43.02 /
384 核 x86_64）完成。

**backend 差异注意**：本轮为 **paged-attn**（5090 上尚未构建 FA 版
InfiniCore；v5 主数据为 flash-attn）。融合点在 attention 之前的 q/k
norm+rope，与 attention 后端无关，但绝对数字不可与 v5 直接比较。
配置：64 blocks、no-graph、贪心解码、逐字节同 prompt（同 v5）。

### 0.6B 交错 ABBA（paged-attn，e2e 秒）

| 负载 | A1 未融合 | A2 未融合 | B1 融合 | B2 融合 | e2e Δ |
|---|---|---|---|---|---|
| w1 单请求 decode | 0.36 | 0.39 | 0.36 | 0.36 | -4.0%（A 侧自身波动同量级，边际） |
| w2 长 prefill + 128 decode | 0.67 | 0.67 | 0.64 | 0.64 | **-4.5%** |
| w3 batch32 | 0.49 | 0.48 | 0.45 | 0.45 | **-7.2%** |
| w4 长 decode | 3.14 | 3.08 | 2.87 | 2.85 | **-8.0%** |

本机无"越跑越慢"漂移（A1≈A2；ABBA 仅作保险）。方向与 v5 一致，
量级收窄——launch 开销在强 CPU + 健康显存平台上占比下降。

### 1.7B 单跑对照（paged-attn，各一轮，量级仅作参考）

| 负载 | 未融合 | 融合 | Δ |
|---|---|---|---|
| w1 单请求 decode | 3.32 ms/tok | 3.28 ms/tok | -1.2% |
| w2 长 prefill + 128 decode | 0.90s | 0.86s | -4.4% |
| w3 batch32 | 5795 tok/s | 5637 tok/s | -2.7%（疑噪声，未复跑） |
| w4 长 decode | 3.66 ms/tok | 3.60 ms/tok | -1.7% |

### 正确性（--dump-outputs + compare_outputs.py）

- **噪声底 = 0**：同 binary 跨轮（A1 vs A2）四负载全 exact——本机上
  unfused 完全确定，因此下述分叉全部可归因于融合 kernel 的 fp32 归约
  顺序差异（算子级 ≤3ulp 已在 v5 证明）。
- 0.6B fused vs unfused：w2、w4 全 exact；w1 @8 分叉；w3 29/32
  exact。分叉处两侧文本均连贯（如 w3 req9："描述一个没有重力的世界"
  vs "描述一个有重力的世界"），属 bf16 logit 并列翻牌。
- 1.7B fused vs unfused：w1、w2 全 exact；w3 23/32（含一处 @0 首
  token 翻牌）；w4 @160 分叉（"从人类学角度分析" vs "从科技角度分析"，
  两侧连贯）。分叉形态与 v5（8GB 机）一致。

### 结论

1. rms_norm_rope 融合在健康平台确认有效：0.6B ABBA e2e **-4%~-8%**
   （prefill/批处理越重收益越大），1.7B -1%~-4%（单跑）。v5 的 -23%
   量级含 8GB 机病态放大；收益随平台算力/CPU 性能上升而收窄，数据中心
   卡上预期也是这个量级。
2. 正确性证据链闭环：算子级 ≤3ulp（v5）+ 双模型 e2e 并列翻牌形态
   双平台一致（v5/v6）。
3. 数据归档：`results/*_rtx5090_{fused,unfused}.json`（0.6B ABBA 四轮
   + 1.7B 各一轮，均含 dump-outputs）。

### 5090 租用机复现要点（Gitee AI 容器）

- SSH 经堡垒机：原 `dev_perf/vm_ssh.py` / `vm_scp.py`（pexpect 状态机，
  密码从 `VM_PASSWORD` 环境变量读取）因脚本内含明文 endpoint
  （IP+端口+账号）已从仓库移除，复现时需自备等价脚本。
- GitHub 不可达：xmake 从 gitee 源码构建（`gitee.com/tboox/xmake`，
  子模块为相对 URL，clone 时自动落在 gitee）；xmake-repo 预置
  `gitee.com/tboox/xmake-repo` 到 `~/.xmake/repositories`（apt 的
  xmake 2.8.7 与现版仓库不兼容：on_source nil）。git 在 pty 下会开
  pager 卡住自动化，需 `git --no-pager`。
- pip 用 `-i https://mirrors.aliyun.com/pypi/simple/`（pypi.org DNS
  只回 IPv6）；HF 下载需 `HF_HUB_DISABLE_XET=1`（hf-mirror 的 xet
  通道 401）。
- 容器仅 /root、/data 持久化；macOS 侧打包后需
  `find -name '._*' -delete`（bsdtar 的 AppleDouble 文件会混进编译）。
- InfiniCore 构建：`--cudnn=y`（cudnn=n 在本 HEAD 上 avg_pool3d 编译
  不过）；删除空的 third_party/cutlass 目录（否则 ENABLE_CUTLASS_API
  打开后 scaled_mm 找不到 cute 头）；`xmake build` 后需显式 build+install
  `infiniccl`、`infinicore_cpp_api`、`_infinicore`（非默认 target）。
- unfused 快照（2366377）的 bench.py 有 /home/yyy 硬编码路径，VM 上
  sed 成自身 checkout 路径（v5 已去硬编码，仅影响旧快照）。

---

## v5（2026-08-24）：第二优化项之一落地 —— 融合 per-head RMSNorm+RoPE

动机：v4 遗留的第二优化项——未融合 elementwise 链（~137ms/prefill，
1.7B 口径；v3 nsys：rmsnorm+rope+swiglu 合计 ~157ms / ~590 次小
kernel）。v5 先融合 attention 内的 q/k per-head RMSNorm + RoPE：每个
张量 2 次 kernel launch + 2 遍显存往返 → 1 次，decode 每 token 省
28 层 × 4 = 112 次 launch。

**平台注意**：本轮实验在另一台机器（8GB 卡 + WSL2；实测显存驻留超
~3~4GB 后带宽崩塌至 1~11 GB/s）完成，与 v1~v4 的 16GB 5060 Ti 不是
同一平台，绝对数值不可直接比较。安排：计时 A/B 用 Qwen3-0.6B（小模型
launch 开销占比更高，对融合更敏感），正确性对拍用 Qwen3-1.7B（慢速区
不影响数值）。**收益量级需回 16GB 机复测确认。**

做法：

- InfiniCore 新增 `rms_norm_rope` 融合算子（11 个新文件）：C 层
  `src/infiniop/ops/rms_norm_rope/`（CUDA kernel `cuda/kernel.cuh`：
  fp32 归约 + 逐变体复刻 rope 舍入路径，GPT_J/NEOX × half/bf16）；
  C++ 桥接 `include/infinicore/ops/rms_norm_rope.hpp`。约束：full-rotary
  （head_dim = 2 × table_dim）、pos I32/I64、sin/cos F32。
- InfiniLM 接线（`csrc/models/qwen3/qwen3_attention.cpp`）：
  `forward_paged_` 的 q_norm/k_norm + 两次 rope 共 4 处调用 → 两次
  in-place `rms_norm_rope_`（prefill/decode 共用路径）；
  `forward_static_` 保持未融合链不变。当前仅覆盖 Qwen3 paged 路径。

性能（0.6B，flash-attn，交错 ABBA 控漂移——本机存在"越跑越慢"漂移，
A2 全面慢于 A1，单次先后 A/B 会系统性扭曲差值）：

| 负载 | A1 未融合 | A2 未融合 | B1 融合 | B2 融合 | e2e Δ |
|---|---|---|---|---|---|
| w1 单请求 decode | 1.00s | 1.05s | 0.92s | 1.05s | **-3.9%** |
| w2 长 prefill + 128 decode | 1.49s | 1.72s | 1.20s | 1.26s | **-23.4%** |
| w3 batch32 | 1.56s | 1.77s | 1.33s | 1.35s | **-19.5%** |
| w4 长 decode | 9.06s | 9.65s | 8.01s | 8.14s | **-13.7%** |

机制自洽性：decode 为 launch-bound，每 token 省 112 次 launch × ~10µs
≈ 1.1ms，与 w4 的 -13.7%（8.8→7.9 ms/tok）吻合；w2/w3 的更大收益叠加
了 prefill/批处理段 elementwise 链融合。首轮未控漂移数据方向一致
（paged-attn：w1 -12.9%、w2 -7.9%、w3 -20.0%、w4 -10.8%；flash-attn
w4 -25.0%）。

正确性（三层证据）：

1. **算子级**（`dev_perf/op_check_rms_norm_rope.cpp`：bf16 输入下融合
   kernel vs `rms_norm_` + `rope_` 参考链逐元素对比）：**>99.999% 逐位
   一致，最差 ~3ulp**——差异仅来自融合 kernel 内 fp32 归约的分块顺序。
2. **0.6B e2e 逐 token 对拍**（fused vs unfused，双后端，
   `compare_outputs.py`）：分叉率与"同二进制换后端"噪声底同量级
   （w2 两侧均 exact；w3 28~29/32 vs 底 30/32；w4 分叉为单请求
   late-token 并列翻牌）。
3. **1.7B e2e 逐 token 对拍**（双后端）：分叉位置与噪声底互有先后——
   w2-paged @59、w4-flash @190 与底完全重合，w1-paged 与 w2-flash 的
   fused 全 exact 而底自身分叉（@92 / @59）；目检 w4 分叉处两侧文本
   均为连贯枚举（"从文学角度分析" vs "从人类社会角度分析"），1024
   token 全程连贯。结论：bf16 logit 并列翻牌经自回归放大，非功能错误。

结论与剩余项：

1. rms_norm_rope 融合在 0.6B 上 e2e 收益 -4%~-23%（负载相关，prefill/
   批处理越重收益越大），正确性三层证据齐备；**16GB 机上量级待复测**。
2. 未覆盖：swiglu 融合（v3 口径中 elementwise 链的另一半）、Qwen3
   以外模型、`forward_static_` 路径。
3. 工程配套：bench.py/README 去硬编码路径（INFINI_ROOT 默认
   `~/.infini`，INF_MAIN_PYTHON 可指向其他 checkout 的构建产物）。

复现：

```bash
# 前提：FA 版 InfiniCore（见 v4，rms_norm_rope 已在库中）；增量重建 InfiniLM 扩展
cd <InfiniLM-perf> && xmake build _infinilm && xmake install
HF_HUB_OFFLINE=1 INFINI_ROOT=$HOME/.infini-fa PYTHONPATH=<InfiniCore>/python \
  python dev_perf/bench.py --engine infinilm --num-blocks 64 \
  --attn-backend flash-attn --dump-outputs
python dev_perf/compare_outputs.py <unfused.json> <fused.json>
# 算子级校验的构建/运行命令见 dev_perf/op_check_rms_norm_rope.cpp 头部注释
```

---

## v4（2026-08-24）：B 方向落地 —— prefill 接 FlashAttention-2

按 v3 末尾的 kernel 定位（自研 `PagedAttentionPrefill` 比 FA2 慢 13.5×），
走"直接调 FA2"路线：重建 InfiniCore（`aten=y` +
`--flash-attn=<FA repo>`，FA 取 **v2.7.4.post1**，其 `mha_varlen_fwd` /
`mha_fwd_kvcache` 与 InfiniCore `flash_attention_adaptor.hpp` 逐参数匹配，
原生支持 paged block_table），装到 `~/.infini-fa`；InfiniLM 侧用已有的
`FlashAttentionImpl`（`--attn-backend flash-attn`：prefill 走
`flash::mha_varlen_fwd`，decode 走 `flash::mha_fwd_kvcache`）。

同机同批对照（Qwen3-1.7B，64blk，no-graph；paged 与 flash 跑在同一个
FA 版 InfiniCore 库上，对照干净）：

| 负载 | paged-attn | flash-attn | vLLM（0.85 档） |
|---|---|---|---|
| w1 单请求 decode | 10.86 ms/tok | 11.62 ms/tok | 8.95 ms/tok |
| w2 长 prefill + 128 decode | 2.48s | **2.14s**（单跑复测 1.95s） | 1.50s |
| w3 batch32 总吞吐 | 2191 tok/s | 2127 tok/s | 2957 tok/s |
| w4 长 decode | 11.07 ms/tok | 11.16 ms/tok | 9.71 ms/tok |

Qwen3-4B（64blk，no-graph）w2：paged 5.41s → flash **4.04s**（-25%），
vLLM 参考值 3.84s → 差距收敛到 ~1.05×。

正确性（贪心解码逐 token 对拍，`compare_outputs.py`）：

- **w2（FA varlen paged 路径压力最大的负载）输出与 paged-attn 完全一致，
  也与 vLLM 完全一致**——1.7B、4B 均如此。
- 其余负载的分叉率与 vLLM-vs-paged 的分叉率同量级（w3：24/32 vs 23/32
  exact；w4：两家同在一处 late-token 分叉；目检文本均连贯）——属 bf16
  规约顺序差异，非功能错误。

结论：

1. **prefill 主差距已被 FA2 消化**：w2 e2e 1.7B -14~30%、4B -25%；4B 上
   与 vLLM 基本持平（1.05×）。与 v3 预测（attention 491ms→~36ms）吻合。
2. 剩余差距（1.7B w2 flash 2.14s vs vLLM 1.50s）主要在：未融合
   elementwise 链（~137ms/prefill）与 decode 段（~10%），即 v3 已列的
   第二优化项。
3. 注意：vLLM 0.85 档本次实测峰值 15.9GB，贴 97% 看门狗线，复测建议
   0.72~0.8 档。

复现：

```bash
# 一次性：重建 InfiniCore（约 28min，FA 84 个 .cu 全量编译）
cd /home/yyy/src/InfiniCore   # flash-attention repo 在 /home/yyy/src/flash-attention @ v2.7.4.post1
xmake f -c --aten=y --flash-attn=/home/yyy/src/flash-attention --graph=y \
  --cudnn=n --ccl=n --nv-gpu=y --cpu=y --omp=y \
  --cuda=$HOME/.local/cuda-13.2 --cuda_arch=sm_120 -m release -k shared
xmake build -j6 && xmake install -o ~/.infini-fa

# 运行（INFINI_ROOT 指向 FA 版库，勿覆盖 V4 线在用的 ~/.infini-dsv4）
INFINI_ROOT=$HOME/.infini-fa HF_HUB_OFFLINE=1 \
  /home/yyy/src/InfiniLM/.venv/bin/python dev_perf/bench.py \
  --engine infinilm --model Qwen/Qwen3-1.7B --num-blocks 64 \
  --attn-backend flash-attn --dump-outputs
```

---

## v3（2026-08-24）：num_blocks 扫描与膝点定位

模型 Qwen3-1.7B（bf16），RTX 5060 Ti 16GB（WSL2），全部 no-graph。

| num_blocks | 加载后显存 | w1 ms/tok | w2 e2e | w3 tok/s | w4 ms/tok |
|---|---|---|---|---|---|
| 64 | 7.9GB（48%） | 11.9 / 12.6（两次） | 2.70 / 2.80 | 1995 / 1870 | 12.3 / 14.1 |
| 128 | 9.8GB（60%） | 13.3 | 2.81 | 1852 | 13.9 |
| 256 | 13.3GB（82%） | 13.3 | 2.82 | 1816 | 13.8 |
| 320 | 15.4GB（94%） | 13.5 | 2.81 | 1850 | 13.5 |
| 512（v1 数据） | 16.0GB（98%） | **33.2** | **42.0s** | **42** | **47.9** |

run 间波动约 ±10%（64blk 两次复测所得），64~320 之间的差异在波动范围内。

### v3 结论

1. **劣化是 98% 极端饱和处的悬崖，不是渐变**：48%~94% 全区间性能持平，
   只有 512blk（98%）坠崖。排除"engine 内 O(num_blocks) 的 per-step 开销"
   假设，指向显存近满时分配慢路径/驱动行为（WSL2）。具体机制需 profile
   512blk 配置确认，但该配置会触发本机 97% 显存看门狗，暂被阻塞。
2. **CUDA graph 收益 ~10%**（v2 的 64blk 对照：w1 11.9→10.7、w4 12.3→11.1、
   w3 +7%、w2 -5%），与 num_blocks 无关。值得默认开启，但不是量级差距。
3. **工程缺陷确认**：默认 num_blocks=512 在 16GB 卡上必踩悬崖，且全程无
   告警。可立项方向：cache 预分配按显存自适应（预留 ≥5~10% headroom）或
   近饱和时显式告警。
4. v1 的全部四条差距假设（launch 开销、prefill 路径、批处理串行、decode
   随长度劣化）均为该悬崖的表现，逐条推翻，详见 v1 存档节。

### 机制分析（代码侧，主 checkout 调查结论）

- 分配链：`llm.py` num_blocks → `PagedKVCacheConfig` → 逐层
  `Tensor::zeros({2, num_blocks, 256, kv_heads, head_dim})`
  （`csrc/cache/kv_cache.cpp:142`）。Qwen3-1.7B 为每层 512MiB × 28 = 14GiB，
  与实测吻合。底层是 InfiniCore `PinnableBlockAllocator`——裸 cudaMalloc +
  尺寸分级 free-list 缓存。
- **稳态 decode 每步零新设备分配**：8 个 CPU 输入 tensor 逐个 H2D、各层
  attention workspace、采样 workspace 全部命中分配器缓存；没有任何
  per-step 开销随 num_blocks 增长。512 vs 320 的 3~50× 劣化**不可能是引擎
  算法开销**——代码侧排除了引擎内因素。
- 分配器失败行为是"报错即死"（throw → exit(137)），无重试、无碎片整理、
  无自动 trim；近饱和**全程无任何告警**，只有启动时一行
  `Using Paged KV Cache with num_blocks=512`。
- 加载耗时 59.5s（512blk）vs 3.9s（64blk）跑的是完全相同的代码路径
  （28 次 cudaMalloc(512MB) + 设备清零 + 权重 H2D），15× 差距只能来自
  cudaMalloc/kernel 执行本身，即驱动/内存子系统。
- 综合判断：悬崖在引擎之下——WSL2（dxgkrnl）显存近满时的
  paging/eviction/residency 抖动是最可疑机制，可统一解释"含稳态零新分配
  的 decode 在内全部变慢"。最终确认需 nsys/driver 计数器（被看门狗阻塞）。

### Qwen3-4B 三配置复测（2026-08-24，与 v3 同机）

InfiniLM 均为 64blk；vLLM 为 `--gpu-mem-util 0.72`（13.97GB，86%）。
带宽口径：4B bf16 权重 ~8GB，5060 Ti 理论 decode 上限 ~56 tok/s。

| 负载 | InfiniLM no-graph | InfiniLM graph | vLLM |
|---|---|---|---|
| w1 单请求 decode | 25.6 ms/tok（39.1 tok/s） | 24.3（41.2） | 23.0（43.4） |
| w2 长 prefill + 128 decode | 5.94s | 5.79s | **3.84s** |
| w3 batch32 总吞吐 | 891 tok/s | 944 tok/s | 841 tok/s |
| w4 长 decode | 26.7 ms/tok | 25.0 | 23.7 |
| 加载耗时 / 显存 | 18.0s / 14.86GB（91%） | 37.0s / 14.82GB | 43.3s / 13.97GB |

结论：

1. **"持平"在 4B 上成立**：decode 三家都在带宽上限的 74~77%，w3 InfiniLM
   略优，w4 基本持平。
2. **唯一持续存在的真实差距是长 prefill**：vLLM 比 InfiniLM 快 ~1.5×
   （1.7B 时 ~1.3×）。vLLM 开了 chunked prefill（max_num_batched_tokens=
   8192），这是下一个值得 profile 的点，但量级是 1.5× 而非数量级。
3. graph 在 4B 上收益收窄到 ~5%（w1 25.6→24.3，w4 26.7→25.0）。
4. 4B 64blk 已占 91% 显存仍无悬崖，再次印证悬崖只在 ~98% 极端饱和处；
   vLLM 0.85 档在 16GB 卡上会撞 97% 看门狗（w4 中途被 SIGTERM），0.72 正常。

### prefill 差距的 kernel 级定位（nsys，2026-08-24）

对 w2（3240 tok prefill + 128 decode）在 1.7B / 64blk 下分别抓 InfiniLM
（no-graph）与 vLLM 的 CUDA 轨迹（`results/prof/*.nsys-rep`，分析脚本
`results/prof/slice.py`）。prefill 窗口内 GPU busy 均 ~100%——差距在
kernel 内部，不在调度/Python 开销。

prefill 前向一次的 kernel 时间构成（3240 tokens）：

| 成分 | InfiniLM | vLLM | 倍数 |
|---|---|---|---|
| prefill attention | **491ms**（PagedAttentionPrefillHd128WarpCta8Pipe，26 层 × ~19ms） | 36ms（FA2 splitkv，28 层 × 1.3ms） | **13.5×** |
| rmsnorm + rope + swiglu | ~157ms（未融合，~590 次小 kernel） | ~20ms（triton 融合 kernel） | 7.8× |
| gemm（qkv/o/gate/up/down） | 185ms | 212ms | 持平（略快） |
| **合计** | **888ms / 1655 次调用** | **305ms / 706 次调用** | **2.9×** |

结论：prefill 差距的第一来源是自研 `PagedAttentionPrefill` kernel 比
FlashAttention-2 慢一个数量级（每层 19ms vs 1.3ms），第二来源是
elementwise 链未融合。这是边界清晰、可度量的 kernel 优化目标：
把 prefill attention 换成/优化到 FA2 量级，w2 的 prefill 段理论上可从
~1.2s 压到 ~0.6s（1.7B 口径），e2e 差距从 1.3~1.5× 收敛到接近 1。

### 待办

- [ ] （需看门狗临时放宽）nsys 抓 512blk 的 w1，直接观察 98% 悬崖机制。
- [x] ~~prefill attention kernel 优化立项~~ —— 已由 v4 完成：接 FA2
      （`--attn-backend flash-attn` + FA 版 InfiniCore），w2 的 prefill 段
      差距收敛到 ~1.05~1.3×，正确性逐 token 对拍通过。
- [x] ~~（可选）elementwise 融合（rmsnorm/rope）作为第二优化项~~ ——
      已由 v5 完成：`rms_norm_rope` 算子接入 Qwen3 paged 路径，0.6B
      ABBA e2e -4%~-23%，三层正确性证据齐备；跨平台复测已由 v6 完成
      （5090 上 0.6B ABBA -4%~-8%，正确性形态一致）。
- [x] ~~（可选）swiglu 融合~~ —— 销项：上游 swiglu/add_rms_norm 早已
      融合，Qwen3 paged 路径一直在用；v7 nsys 实证 elementwise 链已
      全部单 kernel 化。
- [ ] （可选）flash-attn 设为默认 attention backend 的评估：需在更多模型
      上补正确性对拍，并确认 FP8/滑窗/softcap 模型的回退路径。
- [ ] Qwen3-8B-FP8 下载完成后，可作为 8B 级 + FP8 口径的复测对象。

---

## v2 存档（2026-08-24）：64blk 对照实验原始表

| 负载 | InfiniLM 512blk no-graph | InfiniLM 64blk no-graph | InfiniLM 64blk graph | vLLM（v1 默认） |
|---|---|---|---|---|
| w1 单请求 decode | 33.2 ms/tok | 11.9 ms/tok | **10.7 ms/tok** | 12.0 ms/tok |
| w2 长 prefill + 128 decode | 42.0s | 2.70s | 2.56s | 2.0s |
| w3 batch32 总吞吐 | 42 tok/s | 1995 tok/s | 2136 tok/s | 2166 tok/s |
| w4 长 decode | 47.9 ms/tok | 12.3 ms/tok | 11.1 ms/tok | 12.2 ms/tok |
| 加载耗时 | 59.5s | 3.9s | 11.6s（含 graph 编译） | 24.0s |
| 加载后显存 | 15.96GB | 7.89GB | 8.07GB | 15.89GB |

---

## v1 存档（2026-08-23/24，已被推翻）

数据：`results/infinilm_..._0823_235223.json`、`results/vllm_..._0824_002522.json`。
配置：InfiniLM no-graph、paged-attn、prefix caching on、**num_blocks=512**；
vLLM v1 默认（CUDA graph on、prefix caching on）、gpu_memory_utilization=0.85。

| 负载 | InfiniLM | vLLM | 差距 |
|---|---|---|---|
| w1 单请求 decode | 33.2 ms/tok（30.2 tok/s） | 12.0 ms/tok（83.2 tok/s） | 2.8× |
| w2 长 prefill（3240 tok）+ 128 decode | 42.0s e2e | 2.0s e2e | 21× |
| w3 batch32 × 128 tok | 42.0 tok/s | 2166.0 tok/s | 52× |
| w4 长 decode 1024 tok | 47.9 ms/tok（20.9 tok/s） | 12.2 ms/tok（82.3 tok/s） | 3.9× |

v1 归因假设与判决：

1. ~~单请求 decode 低是缺 CUDA graph 的 launch 开销~~ —— 推翻：no-graph
   64blk 已达 11.9 ms/tok，graph 只再提速 ~10%。
2. ~~w2 是 prefill 路径异常（纯 prefill 估算差距 ~75×）~~ —— 推翻：
   64blk 下 w2 e2e 2.7s，与 vLLM 同量级。
3. ~~w3 批处理存在 per-request 串行开销~~ —— 推翻：64blk 下 w3
   1850~2136 tok/s，扩展性正常。
4. ~~w4 decode 随上下文长度劣化~~ —— 推翻：64blk 下 w4 与 w1 持平。

## 立项书模板（方向定稿后填写）

"在 [模型] + [硬件] + [负载矩阵] 下，[TTFT/TPOT/吞吐] 从 X 提升到 Y（≥Z%），
精度（ceval/mmlu/ppl）不降级，单测与 CI 全绿，benchmark 脚本入库、
他人可复现。"

风险提示：5060 Ti 的瓶颈结构 ≠ 数据中心卡，立项报告中必须注明平台，
收尾时应在数据中心卡（或训练营国产平台）上复测。
