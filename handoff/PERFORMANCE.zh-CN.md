# 性能与可复现性

## 两种计量口径

历史“固定 512”测试对每条请求持续执行 512 个输出位置，可能包含第一个 EOS 之后的诊断计算。新版有效输出测试在第一个 EOS 或输出上限处结束请求，并剔除 EOS 和结束后的输出。两者的输入、输出分布、batch 调度和计时开销也可能不同，不能用两张表直接计算性能回退比例。

旧版数字具有已完成实验和审计记录，代码及构建入口在交接包中保留。**新服务器能否复现相近速度，需要按同一权重、SDK、精度、prompt、batch、计时范围和硬件重新测量。** 旧版 QA 只覆盖原实验声明的任务、重复和检查点，不代表广泛质量评测或生产服务验收。

## 历史固定输出结果

均为单实例、512 token 输入、固定 512 个输出位置；decode 不含 prefill，含 prefill 吞吐计入首次处理输入的时间。排除权重加载、tokenizer 和 HTTP。Kimi、GLM 的 global batch 是排队请求总数，micro batch 是每个流水线包内的请求数。

| 模型 | 配置 | decode 总吞吐 tok/s | 含 prefill 总吞吐 tok/s | decode tok/s/卡 | 对应实现及记录 |
| --- | --- | ---: | ---: | ---: | --- |
| Kimi-K3 | 32 卡，TP8×PP4，G64/M16 | 116.1775 | 93.3510 | 3.6305 | `r963` token-only CAPI，运行 `910380.4`，审计 `r1010` |
| GLM-5.3 | 32 卡，TP8×PP4，G64/M16 | 63.1500 | 40.1335 | 1.9734 | `r927` KV prefix view，运行 `910374.87`，审计 `r928/r929` |
| DeepSeek-v4-Flash | 8 卡，TP8，B16 | 25.3164 | 19.9828 | 3.1646 | `r815` all-selected，运行 `910373.442`，审计 `r854` |

Kimi 116.18 使用四条固定 prompt 重复填充请求；其 token-only 优化跳过大部分 GPU logits 到 CPU 的复制，GPU 采样和数学运算保持原样。同进程对照为 110.7358/89.8090，decode 提升 4.91%。不能把它与另一组 64 条不同 prompt 的 107.6793/87.1862 直接比较并归因于内核优化。旧版 sort QA 曾明确允许经过词表验证的单个空格 token 220 差异，不能将其描述为所有 token 完全一致。

GLM 的 52.50→63.15 历史对比来自不同节点、不同运行，不能直接当作同进程因果收益。DeepSeek 的 all-selected 路径改变 attention 的累加顺序，采用该数值变体自己的重复参考，并未证明与更早版本所有 logits 逐位相等。

## 当前版本已通过审计的有效输出结果

GLM 均为 4 节点、32 卡、TP8×PP4、512 token 输入，两条冻结 prompt 重复填充。请求按实际 EOS 或上限结束；native 加验证 harness，排除加载、tokenizer、reset、HTTP。下列范围来自已完成运行；不是整集群吞吐，也不是硬件极限。

| G/M | 每请求输出上限 | decode 总吞吐 tok/s | 含 prefill 总吞吐 tok/s | decode tok/s/卡 | 审计 |
| --- | ---: | ---: | ---: | ---: | --- |
| 64/8 | 512 | 34.9500–35.0802 | 26.5202–26.6126 | 1.0922–1.0963 | `glm-cohort-r3098-audit.json` |
| 128/8 | 128 | 40.9251–41.2904 | 17.9200–17.9949 | 中位数 1.2846 | `glm-g128-r3239-audit.json` |
| 128/32，plan cache ON | 128 | 42.2873–42.4716 | 18.1214–18.1410 | 中位数 1.3244 | `glm-g128-r3272-audit.json` |
| 32/8，fastpath ON | 128 | 38.3447–38.4114 | 13.8790–13.9481 | 中位数 1.1993 | `glm-fastpath-r3330-audit.json` |
| 64/8，fastpath ON | 128 | **48.1930–48.2105** | 17.4808–17.5232 | 中位数 **1.5063** | `glm-fastpath-r3330-audit.json` |

G128/M32 plan cache ON 的 decode 中位数为 **42.3794 tok/s**，含 prefill 中位数为 **18.1312 tok/s**。同进程 cache OFF/ON/ON/OFF 的 decode 中位数从 34.1216 到 42.3794，提高 24.20%。128 和 512 输出上限的端到端吞吐不能直接横向比较。

fastpath 的 G64/M8 decode 中位数为 **48.2018 tok/s**，含 prefill 中位数为 **17.5020 tok/s**；同进程 OFF/ON/ON/OFF 从 35.3797 到 48.2018，提高 **36.24%**。两种策略的完整 token/EOS、相同 micro batch 的 raw 重复、340 个 CPU 全词表 raw 文件和 32 个 worker 正常关闭通过独立审计。固定开启 plan cache，优化 bounded metadata、只读 prefix 和 decode leases，GPU 数学路径不变。与 G128/M32 是不同配置；正在组合两种优化，收益不能相乘推断。

当前 Kimi 尚无通过完整多 batch 数值检查的新总吞吐；DeepSeek 已新增严格检查通过的 B4 结果，见后文。其已通过单请求检查的 decode 范围分别为 6.7605–6.8213 tok/s（32 卡）和 2.3661–2.4362 tok/s（8 卡）。GLM 单请求为 4.24595–4.32029 tok/s（32 卡）。这些均为 512 输入、最多 512 输出；不能从旧版固定 batch 总吞吐推导新版单请求速度。

## 本次重新测量

本次启动了原始旧版 Kimi、GLM、DeepSeek 固定 512 重放，并安排实际 EOS 场景复测。最终结果只在 Slurm 结束、独立审计及正常资源释放通过后写入 `evidence/recheck-status.json`。该文件没有合格结果时，不把历史数字称为“本次已复现”。

### 旧版 DeepSeek 固定 512 本轮结果

生产任务 `910384.251` 和独立 CPU 审计 `910377.419` 均 COMPLETED/0。B16 两轮 decode 为 **25.8244、25.5249 tok/s**，含 prefill 为 **20.3491、20.1239 tok/s**；B8 为 14.6126、14.5371 tok/s，含 prefill 为 12.6706、12.6022 tok/s。固定 512 的吞吐接近历史记录。

这是原版 `r817` 审计口径：已知回答 ID/EOS、相同几何的全长重复、raw SHA/CPU argmax、SDK/源代码、计数、owner 和退出检查通过；原版允许重复行 logits 差异，B16 decode 的检查点确实存在该差异。它既不是按实际 EOS 的有效吞吐，也不能当作当前严格重复行 raw gate 已修复。保留完整原始审计 `evidence/deep-allselected-r817-910384.251-summary.json`；实际 EOS 复测另行验收。

### 旧版 DeepSeek 长文本、按实际 EOS/上限结束

生产任务 `910407.98` 与独立 CPU 审计 `910377.421` 均 COMPLETED/0。两条冻结长文 prompt（480–512 输入）重复填充，8 卡，最多生成 512，第一 EOS 即结束请求：

| batch | 有效 decode 总吞吐 tok/s | 含 prefill 总吞吐 tok/s | decode tok/s/全部服务卡 |
| --- | ---: | ---: | ---: |
| 8 | 14.4647–14.6060 | 12.5440–12.6558 | 1.8081–1.8257 |
| 16 | **25.6829–25.8375** | **20.1960–20.3363** | **3.2104–3.2297** |

本组每条输出均达到 512 上限，未出现提前 EOS，尚未覆盖混合长短输出和提前结束后的动态补批。完整 ID、相同几何重复、raw 检查点、CPU argmax、已知短回答 EOS、owner 与退出已审计；仍沿用历史数值口径，不能据此宣布当前严格重复行数值问题已修复。首 token 约 43–44 秒（B8）；B16 的前后两组约 44/87 秒，尚无在线延迟 SLA 或 HTTP 服务验收。

独立审计采用所有请求首 token 都完成后的共同 decode 区间，按区间内实际有效 token 计数。原报告曾以最早首 token 开始计时，B16 原始诊断值为 22.62–22.76；原数据保留，规范化后的 25.68–25.84 才是本表 decode 值。含 prefill 全程计时不受该修正影响。生成示例和审计在 `evidence/legacy-deep-eos-r3453-audit.json`。

## 统一复测报告

每条结果保存代码提交及覆盖层 SHA、权重与 tokenizer 标识、实际设备、TP/PP、副本数、G/M、输入与输出长度、EOS、精度和开关。至少报告两轮生成的完整 token ID、原始检查点、正常关闭、首 token 时间、decode 总吞吐、含 prefill 总吞吐及所有服务卡数。正式的新旧对比应使用相同 prompt、输出上限、EOS、同一组节点和 ABBA 顺序。

硬件记录中的 BF16 大 GEMM 中位数约 304.405 TF/s、F32 大 GEMM 约 53.882 TF/s、D2D copy 单向 payload 约 674.123 GB/s 都是实测微基准。不能将其直接写成 BW1000 官方物理峰值或整模型可达吞吐上限。

## 10 月 20 日晚：补充复测与有效性能

旧版三模型均已结束并通过各自原版独立审计；保留历史数值限制，不将其等同于新版严格 gate。

| 旧版固定 512 配置 | 本轮 decode tok/s | 本轮含 prefill tok/s | 审计 |
| --- | ---: | ---: | --- |
| Kimi，32 卡，G64/M16 | **116.6805–117.0638** | **93.6540–93.8675** | `kimi-token-only-r965-audit-910375.387.json` |
| GLM，32 卡，G64/M16 | **62.8892–62.9079** | **39.9473–39.9730** | `glm-fp8-cached-model-r929-910380.228-audit.json` |
| Deep，8 卡，B16 | **25.5249–25.8244** | **20.1239–20.3491** | `deep-allselected-r817-910384.251-summary.json` |

Kimi 同进程 full-CPU-logits 对照为 109.9676–110.0707 decode，token-only 中位收益约 6.23%，属于输出复制/计量路径优化。旧版 Kimi/GLM 的实际 EOS 长文复测另行审计，不能直接沿用固定 512 值作为在线服务吞吐。

**当前锁定 `3e0d89fe` 的新增有效结果：**

| 模型与配置 | decode tok/s | 含 prefill tok/s | decode tok/s/全部卡 | 范围 |
| --- | ---: | ---: | ---: | --- |
| GLM，32 卡，G128/M8，最多 512 输出 | **40.7437–40.8900** | **30.8360–30.9098** | 中位数 **1.2755** | `glm-g128-r3340-audit.json` |
| Deep，8 卡，B4，最多 512 输出 | **7.2263–7.3813** | **6.7120–6.8528** | **0.9033–0.9227** | `deep-attention-r3494-B4-audit.json` |

输入约 512，两条冻结长文 prompt 重复填充，所有输出都达到 512 上限；native 加验证 harness，排除加载、tokenizer、reset、HTTP。GLM 同进程 G64→G128 decode 中位数 35.2727→40.8169，增加 15.72%；TTFT 约 299/520 秒，不能据此承诺低延迟在线服务。Deep B4 TTFT 21.79–22.27 秒，12 项冻结 B1 前后 QA、12 项 B4 QA、完整重复行 raw、重复生成、CPU 检查和 8 个 owner 正常关闭通过；Model `r3443` mode2 attention F64、SDK Cast64 `r3315`、Common `r2845`。**B2 仍失败，B4 的通过不覆盖其他 batch 或动态请求。**

GLM 专家稳定排序在长输出 G64/M8 的 decode 没有改善，含 prefill 约提升 1.65%；短输出 M8 含 prefill 约提升 4.1%，decode 增益低于 0.2%。prefill cache `r3187` 完整审计显示端到端基本持平。保留这些有限收益，不将组件加速写成整模型加速。

以上均不是 `c3deba4` 的性能，也不是混合长度、动态补批或 HTTP 服务的实测上限。

## 最新追加：GLM 组合优化与旧版 Kimi 长文

GLM fastpath 与 plan cache 固定开启的组合已完成 32 卡、G64/G128、M32 的同进程 OFF/ON/ON/OFF 对照。512 输入、最多 128 输出，首 EOS 结束。独立审计 `glm-fastpath-r3440-audit.json` 及 CPU 作业均 COMPLETED/0，268 个全词表 raw、完整 ID/EOS、相同 M 重复和 32 个 worker 关闭通过。冻结 B1 与 B8 QA 通过；该矩阵没有另外跑 B32 的已知答案 QA，M32 的长输出矩阵正在补充该覆盖。

| GLM 配置 | decode 中位 tok/s | 含 prefill 中位 tok/s | decode tok/s/全部32卡 | 同进程 decode 增益 |
| --- | ---: | ---: | ---: | ---: |
| G64/M32 | 42.9181 | 16.8193 | 1.3412 | 44.22% |
| G128/M32 | **60.5148** | **20.8333** | **1.8911** | **43.57%** |

G128 两轮 candidate decode 为 60.4585–60.5711，含 prefill 为 20.8263–20.8403；TTFT 约 518 秒。这是当前已审计的最高 GLM 有效 decode 总吞吐，不能与固定输出 63.15 直接相除判断回退，也不能将它套到最多 512 输出上。

旧版 Kimi 长文复测及 CPU 审计 `legacy-kimi-eos-r3512-audit.json` 均 COMPLETED/0，32 卡，480–512 输入、最多 512 输出，两条 prompt 重复填充，实际全部达到上限：

| 配置 | decode tok/s | 含 prefill tok/s | decode tok/s/全部32卡 |
| --- | ---: | ---: | ---: |
| G16/M16，serial | 30.6626–30.7419 | 25.3152–25.3656 | 0.9582–0.9607 |
| G64/M16，PP | **68.5333–68.7017** | **59.8882–60.0999** | **2.1417–2.1469** |

已验证 44 项冻结单请求 QA、12 项 B16 QA、首4和后续检查点 raw、CPU argmax、完整 ID/自身重复、输入 SHA、库版本及 32 个 owner 正常关闭；保留历史 sort 独立空格220兼容口径。固定 512 的约117 与此处约69 使用不同 prompt 与 EOS 检查/反馈调度，不能直接归因为模型内核回退。当前尚未完成将这些开销逐项拆出的同 prompt 对照，也未覆盖 HTTP、提前 EOS 混合输出与动态补批。

## 追加：独立审计完成的参考结果

| 版本/模型 | 配置 | decode tok/s | 含 prefill tok/s |
| --- | --- | --- | --- |
| current3e0 Kimi，F64参考 | 32卡，G32/M8，512输入/最多128输出，round-feedback | 46.37–46.52 | 30.65–30.72 |
| current3e0 Deep，F64参考 | 8卡，B8，prefill micro4，512输入/最多512输出 | 17.28–17.42 | 14.85–14.95 |
| current3e0 Deep，F64参考 | 8卡，B16，prefill micro4，512输入/最多512输出 | 28.23–28.48 | 21.80–21.96 |
| legacy GLM，实际EOS协议 | 32卡，G64/M16，512输入/最多512输出 | 36.57–36.69 | 27.43–27.48 |

各行均通过各自独立审计，详见 `evidence/qualified-performance-r3578.json`。自然长文测试仍为冻结输入、静态队列，所有这些长文输出到达上限；不是混合请求 HTTP 生产吞吐。用户最新明确仅要求完整 token 序列一致，logits 位模式差异只诊断；Kimi/Deep 的原生低精度重测已启动，以上 F64 数字仅作参考，不是默认性能路线。GLM 当前长输出使用 Socket-over-IB，RDMA 对照仍在测，见 [实际通信](NETWORK.zh-CN.md)。


## 10月21日追加：已审计旧3e0结果

| 模型/精度 | 配置 | decode tok/s | 含prefill tok/s |
| --- | --- | --- | --- |
| Kimi，F64参考 | 32卡，G64/M8，512输入/最多128输出 | 57.73–58.26 | 38.16–38.39 |
| GLM，fastpath | 32卡，G128/M8，512输入/最多512输出 | 中位54.47 | 中位38.10 |
| Deep，attention FP32归约 | 8卡，B8，prefill micro4，512输入/最多512输出 | 17.60–17.74 | 15.01–15.11 |

这些结果均不是最新06d性能。Kimi一行仅作F64参考；Deep此次完整输出token与冻结F64参考相同，未使用F64AllReduce。不同节点/轮次的结果不构成因果加速比较。均为静态冻结输入及原生计时，不是HTTP生产吞吐或硬件极限。独立审计摘要见 `evidence/qualified-performance-r3649.json`。


## 最新06d原生路径已验收（与5f91在测结果分开）

| 模型 | 配置 | decode tok/s | 含prefill tok/s |
| --- | --- | --- | --- |
| GLM-5.3，fastpath | 32卡，G1/M1，512输入/最多128输出 | 5.32 | 3.40 |
| GLM-5.3，fastpath | 32卡，G8/M8，同上 | 19.53 | 6.51 |
| GLM-5.3，fastpath | 32卡，G16/M8，同上 | 30.85 | 10.32 |
| GLM-5.3，fastpath | 32卡，G16/M16，同上 | 25.14 | 9.60 |
| Deep-v4-Flash，attention FP32 SUM | 8卡，B8，prefill micro4，512输入/最多512输出 | 18.26–18.54 | 15.42–15.65 |

GLM为同进程ABBA候选中位数，G16/M8相对同配置基线22.56提高36.8%；完整多batch输出与冻结参考一致，B1自然长输出仅同配置OFF/ON全ID对照，已知QA使用外部参考。Deep完整输出匹配冻结F64参考，FFN/indexer保持原生；不使用F64 AllReduce。这些是06d/046/f998/b413/6f32栈，不是5f91栈。512输入不代表每例都自然EOS：Deep所测长例都触及512输出上限。均为静态输入原生+harness计时，不是HTTP、混合到达或硬件极限。不同batch、输出长度或节点轮次的数字不能直接解释为版本回退。详见 `evidence/qualified-06d-r3707.json`。


最新5f91 Deep B8已独立通过：18.27–18.44 decode / 15.43–15.56 含prefill tok/s，8卡；512输入/最多512输出，attention FP32归约，其余原生，完整ID/EOS/CPUargmax/8owner/Slurm均验收。原06d结果仍按06d保留，不能混标。详见 `evidence/qualified-latest-Deep-r3729.json`。
