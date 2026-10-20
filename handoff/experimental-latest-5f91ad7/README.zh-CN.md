# 最新5f91重构栈源码与实验接手

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
