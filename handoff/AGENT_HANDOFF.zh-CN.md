# Agent 接手文档

## 最新补充（2026-10-21T00:58:41.529143+00:00，覆盖历史进行中状态）

Kimi选择性投机六组全部完整token不同，无可采用加速；真实投影回放已证明M1/M2本地GEMM差异在归约前存在，FP32/F64归约不能解决此处。GLM容量prefill缓存整模型收益接近零，默认关闭；PP3 rows64实际24卡容量通过，一次加载多batch测试与独立审计进行中。详见 [本轮实验与迁移补充](EXPERIMENT_UPDATE_20261021.zh-CN.md)。所有原有主指标资格保持，投机排除主目标。

## 新增实验进展（2026-10-20T23:43:10.529633+00:00，覆盖后文旧状态）

Kimi 的两种块验证修复模式均完成：32卡TP8PP4、B1/2/4/8/16 × draft上限1/3/7，共30个配置比较，**全部完整token与普通解码不同，而且更慢，合格加速结果为0组**。正常32owner退出、Slurm0；不是运行报错。B16/K1模式2诊断值为普通21.54、投机13.67 tok/s，接受率20.36%；token失败，该速度不作为可用结果或主目标。首个真实接受块已完成追踪与两次未追踪重放，均在第26个生成token分歧；串行完整ID相同。末尾脚本把二维generated_ids直接与一维参考比较，Slurm FAILED1；3836事后独立核对实际前后AR完整ID相同，32worker报告关闭，但只有8项beforeQA，afterQA与末尾affinity未执行，不当正常模型验收。原失败源码/终态保留，3837仅修索引后准备与下轮诊断合并，不为末尾脚本单独重载。首KDA0的8卡投影/活动初态/提交后卷积与SSM状态全同；最早generic raw差异为layer1 attention首token一个BF16元素，次token首次在该层ffn输出430元素；这些是诊断，不是raw gate或已确认根因。实际配置layer1为第二个KDA、layer3为首个MLA。下一轮同次加载抓KDA1详细状态、MLA3 W1/W2，另测n3/K3、n4/K1选择性草稿B1/8/16，前后AR与投机各两次；需实际有草稿并接受、完整ID/EOS/重复/owner/Slurm全部通过才列可用实验结果，仍排除主指标。3841只改host追踪并原样复用旧GPU对象，3847队列等真实成功构建再提交。只比较最终完整token；逐层raw差异用于定位。证据 `evidence/Kimi-spec-M1-complete-negative-r3760.json`、`evidence/Kimi-spec-M2-complete-negative-r3760.json`。

Deep最新5f91、8卡B16已通过独立3799审计，512输入/最多512输出、prefill micro4，decode **31.40–32.29**、含prefill **23.58–24.33 tok/s**；每卡分别3.925–4.036及2.948–3.042。完整ID与冻结F64参考相同、有限值/CPUargmax/正常8owner/Slurm通过，实际仅attention FP32 SUM，无F64 AllReduce。两类重复prompt四阶段均到512上限，不是HTTP/混合流量。

GLM容量prefill计划缓存完成8卡304组完整矩阵新旧对照、修改权重/scale检查及912个独立CPU样本；768MiB/16LRU边界、owner/stream/释放通过。相对每次新建销毁计划，组件host计时比中位2.91，stream计时比3.19，包含分配与调用间隙，**不是整模型加速或纯kernel峰值**。首次16GiB主机内存OOM，96GiB同代码重试通过；Model3816首次host编译const输出指针失败，3822仅按现有809约定补显式mutable输出cast，失败源保留。默认关闭的3822实际Model源码已加入；两/四节点的同次加载B8/M8、B16/M8各20阶段缓存ABBA已提交，decodebundle固定ON，完整token/owner/网络/性能审计已排队。不能继承组件加速比。

GLM128行扫描已独立验收，M8/M16全部冻结完整ID通过，M4仍淘汰；最新B128/M16为70.1499 decode /21.8239含prefill tok/s，每全部32卡2.19218/.68200，prefill输入126.27tok/s，512输入/max128，两次正反序中位。大batch配置收益不能直接称代码因果加速。共享完整1M RoPE的PP2/PP4均已完整验收；同B16/M8，2节点总25.3407、每16卡1.58379，4节点总30.8713、每32卡.96473。不同节点和容量下的描述性比较，未证明最优机器数。3790旧scope字符串与all32字段名沿用32卡字样，但PP2实际循环/normal_workers_closed/服务卡数为16，速率除16；保留原证据并在此澄清。当前目标仍是三模型有效总吞吐/全部服务卡数，投机排除主指标；最优节点数、纯TP/EP、HTTP和硬件极限未完成。

## 最新进展（2026-10-20T22:24:21.619999+00:00，覆盖后文旧状态）

最新5f91栈GLM的32卡TP8PP4、512输入/最多128输出、640上下文、96行容量完整扫描已结束；32owner正常关闭、Slurm0，3797独立审计后M8/M16全部QA及自然生成完整token与冻结3330相同且重复稳定。**M4 QA真实token不同，整套M4几何淘汰，不采用其速度**。B96/M16 decode中位64.8568 tok/s，含prefill20.5901，每全部32卡为2.0268/0.64344；prefill输入吞吐约120.23 input tok/s。是静态两类重复prompt、两次正反序测量，不是HTTP/混合流量，也不能和旧max512输出数值直接算回归或加速。详见 `evidence/qualified-GLM-profiles-r3797.json`。

GLM PP2旧试验在加载前尾段仅剩约3.1GiB，16上下文正常退出，3795独立审计；旧3773误按性能schema读取而失败，保留。idle trim实际释放0字节。新3783方案在每个Runtime内按head64/128、完整1M长度、theta8e6等完整键共享只读RoPE；弱池不留住RoPE/Runtime，组件3788全部8卡/双Runtime正常关闭。旧3783包装器漏CUDA include，在编译前失败，3788仅修包装器。模型数学、权重及原始1M RoPE表范围不变；本轮服务KV仍640。PP2同节点rank0构造后head实省5.875GiB、tail实省6GiB，tail空闲3.1992→9.1992GiB。加载前留8GiB（6GiB运行余量+2GiB加载事务），运行余量仍6GiB。3789调度同一3784 harness的PP2/PP4已加载并在QA；不能声称两节点性能已合格。3790排队独立验收。

Kimi3759修复后已通过8项QA与B1/B2等冻结AR，但M1/M2 B1投机K1仍同样在索引41出现6130对2762；拒绝重放token差异为0，整作业仍在更大batch，尚无投机加速可采用。需要同一前缀上首个接受块与串行的逐层定位；Common dense-prefill F32仅在rows>=512启动，本轮验证器N<=64，不能把它当已证实原因。3775把同池KDA接到实际Runtime并默认关闭，开关0/1、正确/错误ambient device共288组对照通过且3791审计；129状态槽算子每次可省48.375MiB，实际32行Model状态池33槽则每次12.375MiB，尚无整模型加速结果。

Kimi3761大B在加载前因128行同时预留516KV页，头段2.7–2.8GiB低于原3GiB门槛失败。不要放低门槛或当已跑通；若压缩KV页，须同时解耦当前KDA池num_blocks/4计算、保留129成对状态槽及物理page边界。DSpark在线特征/PP传输/增量draftKV仍未完成，投机排除主目标。

Deep最新B8已审计18.27–18.44 decode/15.43–15.56含prefill，8卡。3798在相同已验收3688模型/SDK/Common上新增B16整模型试验，512输入/最多512输出、micro4，逐行对重复prompt对应的冻结F64完整ID；3799审计已排队，不能继承旧B16结果。新服务器仍须重建各层依赖、正常owner/数值/网络/性能验收；本轮没有最优节点数、纯TP/EP、HTTP或硬件极限完成结论。

## 最新状态（2026-10-20T21:21:11.524499+00:00，覆盖后文历史进行中描述）

Kimi 投机首轮 B1/2/4/8/16 × K1/3/7 的15组均完整 token 不同且更慢，合格加速结果为0。投机独立于主目标。实际首次前向异常已定位 CausalConv1d；SDK-only Model3712越过该处后，原私有KDA调用又触发设备检查：新Runtime会恢复线程之前设备，raw helper不能假设 ambient CUDA 仍为owner。两个已确认报错的步骤及其PP等待进程已停止，父申请和健康作业未取消；不是正常owner关闭或性能验收。

3755隔离构建给raw KDA调用加显式owner DeviceGuard，保留原数值、设备与stream检查及调用者设备恢复。实际 Runtime 的正确/错误 ambient device 差分组件已通过（288 组），整模型仍待完整验收。 对应3759两组模式为KDA逐token舍入、KDA+MLA decode投影；B1–16、AR三次/投机两次。3761大batch B16/32/64要先等实际QA+冻结B1完整AR重复通过，3760/3762只在实际终态后独立审计。旧3736只作准备稿，不是已通过的性能版本。

GLM最新96/128行、640上下文的G64/96/128 × M4/8/16正在做正反序重复；3756全32 owner、完整ID/EOS/CPUargmax/库SHA/Slurm独立审计已排队，暂不采用未终态速度。Deep最新5f91的已审计结果保持8卡B8 decode18.27–18.44、含prefill15.43–15.56 tok/s，512输入/最多512输出。

DSpark仍只有离线组件，最新目标五层特征、PP传输、增量draft KV与在线接受率尚未接入。下一步先完成验证器完整token语义，再接训练草稿；不能把离线命中率当在线加速。KDA同池288组算子证明可省48.375MiB快照，但尚未用于Model。详细入口：`evidence/kimi-device-guard-r3765.json`、`drivers/*r3755*`、`drivers/*r3759*`、`drivers/*r3761*`。

## 最新用户验收要求（优先于下文历史 raw gate）

用户明确要求：**不需要所有数值逐位一致，保证最终输出的 token 一致即可。** 完整生成 token ID 序列（含 EOS）与同输入冻结参考一致是数值验收标准；logits 在重复行、rank、重复运行或精度变体间的逐位差异仅作诊断。仍验证有限值、CPU argmax 与实际所选 GPU token 一致、EOS、同配置重复、owner/设备/stream、正常退出和 Slurm 终态。raw SHA 用于文件完整性，不能据此要求模型数值逐位相同。

原生 BF16/FP32 为优先性能路径，F64 仅作参考和对照。F64 相对 BF16 的通信负载字节数为 4 倍，整模型慢多少必须实测。历史 raw-only 失败保留并标注旧规则；仅 raw 不同不再淘汰候选。完整 token ID 不同（包括空格 token）仍是失败，不做文本归一化放宽。

Kimi 已安排原生 BF16、FP32 两组完整模型试验，对冻结 F64 序列检查完整 token，并用相同 round-feedback 调度作 F64/候选/候选/F64 同进程对照。Deep 原生与匹配同步模式也按 token 标准重测。新精度路径尚无审计后的性能数字。具体代码见 `drivers/*r3560*`、`drivers/audit_kimi_token_r3562.py`、`drivers/*r3564*`，规则见 `evidence/token-acceptance-policy.json`。

## 任务与优先级

用户要求迁移三模型适配及优化工作，并交接历史高吞吐版和最新重构版。当前优化目标为**多 batch 下的有效总吞吐/全部服务卡数**，同时报告含 prefill 吞吐；不再以单用户 15 tok/s 为准入条件。旧版固定 512 数字必须在实际 EOS 场景下复测。完整 token 序列一致是最新数值验收标准；模型性能仍需有限值、EOS、正确计时、实际 owner 关闭和独立终态审计。

先读 `pins.json`、`PERFORMANCE.zh-CN.md`、`NUMERICS.zh-CN.md` 和 `evidence/recheck-status.json`。原集群任务与访问细节在用户本地私有附录，GitHub 不包含凭据。新服务器的分区、账号、DCU 类型、驱动、网络和模型路径须重新发现；不能继承旧集群 40 节点/320 卡配额或节点名单。

## 稳定与实验代码

| 模型 | 当前稳定构件 | 未完成或失败候选 |
| --- | --- | --- |
| GLM | 最新 Ops/Common `r2844/r2845`；Model 原始 `r2856`；多 batch host cache `r3072/r3073` | `r3323` fastpath 已审计通过 48.20 tok/s；待验证：`r3148` prefill cache、`r3348` 专家排序、长输出 G128/M8、`r3439` fastpath+G128/M32 |
| Kimi | 最新 Ops/Common `r2844/r2845`、Model `r2863` 的 B1 | `r3261/r3266` FP32 归约整模型失败；`r3393/r3395` F64 Common 已构建，等待完整模型验证 |
| Deep | 最新 Ops/Common `r2844/r2845`、Model `r2856` 的 B1 | `r2904` 多请求 FP32 归约仍失败；`r3443` 匹配同步控制与 attention/FFN/indexer F64 分级候选已安排隔离构建/组件测试 |

历史版本分别为 Kimi `r963`、GLM `r927`、Deep `r815`。它们依赖各自旧 SDK、旧 Common 和 BLAS policy，不能只复制一个 `.so` 到新版环境。

## 必须保持的工程约束

- 每个 Runtime、stream、张量与通信资源都由正确的 owner 线程操作。验证真实设备和 CPU affinity，不靠环境变量声明代替读取结果。
- 容器内 `/local` 必须绑定节点本地 tmpfs，Python cache 加 PID。共享 NFS 下的 import/pycache 曾造成长时间停顿；不能删除仍被活任务使用的缓存。
- 切换 SDK/Common/Model 使用新目录、新进程和加载映射 SHA，不能热替换运行中的库。generated public header 顺序不同曾导致检查失败，应验证语义与实际 ABI 后在隔离前缀处理。
- Kimi 私有 Model CommunicationGroup 与 Common 的布局不同。保留 `r3261` 修复：私有构造方法留在 Model，仅自由归约函数调用 Common，避免请求 Scope/TLS 分裂。
- 保留负例、失败运行与原始输出；旧 raw-only gate 已被用户的新 token 标准取代。只比较某一步 argmax 不足，必须验证完整序列；插桩结果仍须未追踪重放。
- 有效输出在第一个 EOS 结束，EOS 和结束后诊断输出不计入 useful tokens。decode、含 prefill、TTFT、HTTP 服务吞吐分开记录。
- Slurm `sacct`/`scontrol` 是作业状态依据。SSH、观察器或 launcher 超时不等于 GPU 任务失败，不能因此重复提交或取消仍在运行的任务。
- 不取消长期占用、不 reset GPU、不用无实际目的任务填满资源。旧集群有一张已隔离的故障卡，具体位置在私有附录；不能对该节点直接执行完整 TP8。

## 下一步执行顺序

1. 验证交接清单，恢复两套隔离源目录、模型和参考数据；在新节点完成单卡健康及小算子测试。
2. 分别构建 SDK、Common、GPU helpers 和 Model。记录编译器、ABI、动态库映射、设备与网络。先 1 节点，再 4 节点，不直接全量铺开。
3. 优先完成旧版 fixed512 与实际 EOS 复测，输出同一配置的两种口径。未经本轮完整审计不得称“已复现 116/63/25”。
4. Kimi 优先审计 r3560 原生 BF16/FP32 完整 token 与 F64 参考的一致性和精度 ABBA 吞吐。若 token 真正不同，保存第一处 token 差异，再缩小 partial/reduce/residual 边界；raw 差异只作诊断。
5. GLM 审计已在跑的排序、fastpath、长输出实验。两个旧 cache 矩阵曾在 21,000 秒 shell timeout 结束；不要无条件重跑整个长矩阵，拆成较短 ABBA。
6. Deep 在真实未追踪 B2 重放上分离局部计算与 collective。只有正确性通过再测多 batch 吞吐。
7. 在三个模型各自可用的配置上扫 G/M、副本和节点数；分母计入整个服务部署的全部卡，保留长短输出混合场景。多副本、HTTP、连续批处理和最优节点数尚未建立。

## 交接更新规则

每轮将源代码 SHA、实际终态、完整检查范围、性能口径和剩余问题写入新的 evidence 文件，随后更新 `recheck-status.json`。不要覆盖旧失败记录，也不要把仍在运行的记录标成成功。此交接不表示三模型优化目标已全部完成。

## 10 月 20 日晚追加状态（覆盖前文进行中条目）

1. 旧版 Kimi/GLM/Deep 固定 512 三组本轮复测已审计，性能见更新表；Kimi/GLM 实际 EOS 复测仍需独立审计。
2. 当前 `3e0` GLM 长输出 G128/M8 已通过，40.74–40.89 decode、30.84–30.91 含 prefill tok/s，32 卡。fastpath 长输出 M8、M32 与短输出 G128/M32 组合矩阵在测，不得先采用日志速度。
3. Kimi `r3419` 完整多 batch 失败，`r3486` 已捕获 B2 sort 缺空格 token220；保留原 literal gate 失败。B4/B8 独立完整验收 `r3501` 与 G32/M8 流水线 `r3483` 在测。先做完整 CPU raw/QA/关闭审计，再引用速度。
4. Deep B4 mode2 已通过 `r3494`，7.23–7.38 decode、6.71–6.85 含 prefill tok/s，8 卡。B2 raw 失败；B8 `r3498` 是入口输入约束失败，先修协议再重测。B16 要按 prefill micro8 分块，不能直接提交 8192 输入元素。
5. 新上游 `c3deba4`、Core `14a8bd5`、Ops `bcfe96d` 的 9 套源覆盖已完成，PadConstant→Pad 冲突显式解决并保留私有 guards。三 SDK 的 CPU-only build/load 最终 abort；读完整首错后隔离修复。Common/Model/helpers 须全部重建，禁止新旧混用。
6. 运行时归档 `r3432` 不含后续 Cast64/F64 修复库。`drivers/` 新增 Cast64 构建、Deep mode2 与新测试源码；从 `current` 源覆盖生成隔离候选后重建，不能把旧归档改名冒充修复版。

新服务器启动顺序与验收门槛仍按 MIGRATION。实际作业与节点映射留在本地私有附录；接手前读取最新 sacct，不能按本快照重提仍活跃作业。

更新至本轮末：新上游三个 SDK 已在健康节点通过 GPU 可见加载（每节点8卡小计算检查）；原 CPU-only 加载失败是无可见 ROCm 设备，保留负例。新 Common/Model、算子与吞吐仍须验收。旧版 Kimi 长文已独立审计 68.53–68.70/59.89–60.10 tok/s；当前 GLM G128/M32、最多128输出的组合优化已审计 60.51/20.83 tok/s（decode/含prefill），详见性能末表。Deep B8 分块 prefill 已重新测试；Kimi 流水线 producer 已结束，独立审计待完成。


## 10月21日追加：加载、并行与独立投机实验

最新实际源码补充在 `experimental-latest-06d0c03/`，固定 LM06d0c03/Core046f012/Opsf998e4d；原 legacy/current 两份源快照逐字节保留。这是实验源码附加包，尚无最新06d整模型合格吞吐。新的已审计旧3e0性能、加载限制和Kimi投机试验见 [加载与并行](LOADING_AND_PARALLELISM.zh-CN.md)、[投机实验](SPECULATIVE_EXPERIMENT.zh-CN.md)。投机结果明确排除主目标指标。


## 最新补充：5f91重构依赖与投机修复实验

最新递归提交和实际源码见 [5f91实验接手](experimental-latest-5f91ad7/README.zh-CN.md)。RT/CCL和三模型所有层已重建，新32卡高速通信组件验收通过。06d GLM与Deep已有独立全token验收性能；5f91整模型仍在测，不能混标。Kimi首轮B1–16的15组投机比较均发生真实token差异，未得到可采用的投机加速；KDA/MLA修复两组及B16/32/64流水线第三组继续独立测试，全部排除主指标。


## 本轮终态补充（覆盖前文“在测”）

最新5f91栈Deep-v4-Flash已完成独立审计：8卡B8、512输入/最多512输出、attention BF16->FP32 SUM->BF16，decode18.27–18.44 tok/s，含prefill15.43–15.56 tok/s，完整token与冻结F64参考一致，8个owner正常关闭、Slurm0。不使用F64 AllReduce；四个长例都到512上限，不代表混合真实流量。见 `evidence/qualified-latest-Deep-r3729.json`。

Kimi M1/M2最新两组均在首次前向失败：operator has no active implementation；尚未进入token与性能验收。普通IndexSelect8卡小测试通过；所测include顺序及新旧Embedding注册均相同，不能认定header改动3712已修复根因。已准备一次带过滤异常调用栈的4节点原失败重放3731，复用stage0部分CPU缓存，先定位实际算子再重新加载多组。

大batch流水线3703在加载前因128行上下文仍套用了32行尺寸断言而失败，32owner正常关闭，未取消父任务或任何健康步骤；另外发现继承的大写FROZEN环境变量名需一致。B32/64还须把paired-state shadow偏移由17改65，并验证128行/516页/129状态槽。该流水线没有可用速度，不能按已运行成功交接。旧M1/M2和原始15组token负例均保留。

GLM cap640两组3698因retrieval QA的518输入+固定256输出上限检查失败，属于驱动容量约束；3721把QA上限设为min(256,640-input+1)，仍要求完整EOS和token参考，未放宽质量或6GiB余量。96/128行两组继续验证；没有合格大batch新结果可采用。

KDA129槽disjoint read/write同池测试3733用于判断能否省去每层整池快照，单次约48.4MiB；3728仅因缺动态helper路径在kernel前失败，不能当算子失败。独立算子通过也不代表整模型已修复。训练DSpark在线集成仍未完成，当前没有可采用的投机加速，投机始终排除主目标指标。


流水线准备稿3736已补齐128行/516页/129状态槽、shadow偏移65及继承的大写FROZEN变量；**未提交**，launcher要求实际首次前向根因修复与QA新receipt后才能启动。include顺序改动3712自身没有证明能解决算子故障，不应据此加载整组大矩阵。


KDA同池补充：3733在8卡完成288组129槽对照，CPU独立终态与清单审计3741通过；读1..R、写65..64+R互不重叠时，和克隆初始池/逐token串行的输出及写入状态一致，活动读槽未变，可省每次KDA48.375MiB快照。尚未改Model helper采用同池，也没有整模型速度结果。最新3731异常重放已定位到KimiK3DeltaAttention::forward+0x938，继续解析实际算子，不能称根因已修复。
