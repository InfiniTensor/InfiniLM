# 最新5f91重构栈源码与实验接手

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

本目录追加真实源码，原 legacy/current/06d 目录保持原始字节。不是已验收的第三套稳定性能版本。

固定提交见 `source-provenance.json`：LM5f91ad7、Core2b4f2f2、Ops1ca4777、RTcd1d482、CCL51cc9db。LM普通源码相对06d无变化，但递归依赖已变化；RT修复DeviceGuard等语义，RT/CCL均重新编译，不能复用旧二进制。三套SDK、Common/Runtime/Python、Model和GPU helpers已在原集群独立构建；新32卡通信组件已验收，整模型吞吐仍需新栈完整验收。

`cluster-layout/` 是原工作区相对路径的实际源码，9套Model/Common/Ops与模型helpers、Kimi验证器3683、GLM合法PP2/PP4候选3694、Deep FP32候选3688，以及官方固定RT/CCL源。InfiniLoader沿用06d附加包中固定88d6bac非主分支源码。模型权重和二进制不在包内。

迁移先从固定官方提交恢复递归依赖和vendor，再按对应Model/Common/Ops覆盖目录；补齐 `/refactor`、`/old`、`/shared` 的容器绑定约定。新建本机build/install目录，按 `drivers/build_tip_runtime_r3669_inside.sh`、`configure_tip_sdk_r3672.py`、`build_tip_pipeline_r3672_inside.sh`、`common_tip_r3673_inside.sh`、`build_tip_models_r3675_inside.sh` 构建，最后编译专门候选。脚本是原集群实际配方，需要用本机真实路径、ABI/架构/CPU affinity/网络及源清单生成新的receipt。原构建SHA用于原证据关联，不应直接当新服务器成功凭证，也不能删除检查后继承吞吐。

Kimi投机默认关闭；`INFINILM_KIMI_SPEC_R3632=1`为专用paired-state API，`INFINILM_KIMI_SPEC_MATH_R3676=1/2`分别选择KDA逐token舍入、以及额外MLA投影对齐。原始512输入prefill路径保持。3683修复3676编译类型错误，使用新的隔离prefix；3676负例保留。算子288组通过不等于整模型已修复。协议3679扫B1/2/4/8/16；3703扫B16/32/64，并把每micro反馈改为每轮反馈，AR ABBA单独衡量流水线收益。全部投机结果排除主指标。完整ID、EOS、CPU argmax/finite、owner、正常关闭及Slurm终态仍须验收。

GLM新实验把KV容量收紧为640，覆盖512输入/最多128输出；96/128行分别扫B64/96/128与M4/8/16，保留6GiB余量检查。旧128/1024与160/768矩阵因余量检查失败，局部日志速度不能采用。PP2合法分割为38/40；40/38不能直接用，40层不是full indexer，必须先实现跨PP的indexer状态传递。尚无合格16卡性能。

最新已独立验收的06d性能及限制见 `../evidence/qualified-06d-r3707.json`；最新5f91全模型在测，不把06d/3e0或旧固定512速度移植到新栈。完整进展见 `../SPECULATIVE_EXPERIMENT.zh-CN.md`。


## 本轮终态补充（覆盖前文“在测”）

最新5f91栈Deep-v4-Flash已完成独立审计：8卡B8、512输入/最多512输出、attention BF16->FP32 SUM->BF16，decode18.27–18.44 tok/s，含prefill15.43–15.56 tok/s，完整token与冻结F64参考一致，8个owner正常关闭、Slurm0。不使用F64 AllReduce；四个长例都到512上限，不代表混合真实流量。见 `evidence/qualified-latest-Deep-r3729.json`。

Kimi M1/M2最新两组均在首次前向失败：operator has no active implementation；尚未进入token与性能验收。普通IndexSelect8卡小测试通过；所测include顺序及新旧Embedding注册均相同，不能认定header改动3712已修复根因。已准备一次带过滤异常调用栈的4节点原失败重放3731，复用stage0部分CPU缓存，先定位实际算子再重新加载多组。

大batch流水线3703在加载前因128行上下文仍套用了32行尺寸断言而失败，32owner正常关闭，未取消父任务或任何健康步骤；另外发现继承的大写FROZEN环境变量名需一致。B32/64还须把paired-state shadow偏移由17改65，并验证128行/516页/129状态槽。该流水线没有可用速度，不能按已运行成功交接。旧M1/M2和原始15组token负例均保留。

GLM cap640两组3698因retrieval QA的518输入+固定256输出上限检查失败，属于驱动容量约束；3721把QA上限设为min(256,640-input+1)，仍要求完整EOS和token参考，未放宽质量或6GiB余量。96/128行两组继续验证；没有合格大batch新结果可采用。

KDA129槽disjoint read/write同池测试3733用于判断能否省去每层整池快照，单次约48.4MiB；3728仅因缺动态helper路径在kernel前失败，不能当算子失败。独立算子通过也不代表整模型已修复。训练DSpark在线集成仍未完成，当前没有可采用的投机加速，投机始终排除主目标指标。


流水线准备稿3736已补齐128行/516页/129状态槽、shadow偏移65及继承的大写FROZEN变量；**未提交**，launcher要求实际首次前向根因修复与QA新receipt后才能启动。include顺序改动3712自身没有证明能解决算子故障，不应据此加载整组大矩阵。


KDA同池补充：3733在8卡完成288组129槽对照，CPU独立终态与清单审计3741通过；读1..R、写65..64+R互不重叠时，和克隆初始池/逐token串行的输出及写入状态一致，活动读槽未变，可省每次KDA48.375MiB快照。尚未改Model helper采用同池，也没有整模型速度结果。最新3731异常重放已定位到KimiK3DeltaAttention::forward+0x938，继续解析实际算子，不能称根因已修复。
