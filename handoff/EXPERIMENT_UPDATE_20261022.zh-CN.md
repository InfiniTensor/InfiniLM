# 2026-10-22 实验与迁移补充

本页补充原双版本交接。旧版 fixed512 与当前重构版本的依赖和数学路径继续分别保留。测试的 InfiniLM 为 `5f91ad738583db8f5617aeae53b76523f85d1ec6`，Core/Ops/RT/CCL pins 沿用现有 5f91 清单。以下组件结果不能继承为模型吞吐。

## 投机实验的最新用户目标

只以每个用户实际得到的 decode tok/s 判优，排除 prefill。多 batch 仍逐请求报告；草稿生成、目标验证、选择、状态提交/回滚、重放及未隐藏通信全部计时。只数实际提交的新有效 token，不计拒绝提案、重放、EOS 和 EOS 后输出。普通三模型实验仍优化总吞吐/全部服务卡，投机为独立探索。

B1 优先，其次 B2/4/8/16；记录逐请求 TPOT 和非零块提交间隔，避免块内同 timestamp 形成虚假的零延迟。性能分母应为普通 `Engine.forward` AR；使用成对状态快照的诊断验证器不能作为普通 AR 速度基线。

## 完整长草稿权重已就绪（北京时间 10 月 22 日 12:22）

Modal W16与RedHat原生8提案的原始传输均已完成，远端完整SHA通过。Modal的69个BF16权重、真实目标embedding/head已在BW1000单卡实际执行，一次加载覆盖B1/2/4/8/16、W8/W16、前缀32/512的20组测试。20组固定前缀缓存与重算的草稿token相同，重复也一致；前缀推进后部分提案不同，尚需区分数值路径与缓存语义。草稿提案差异本身不要求逐位修齐，最终目标仍是验证后输出token正确及真实接受长度。

B1缓存草稿生成（含head/argmax/回传及设备同步）W8约13.60～13.73ms、W16约13.69～13.90ms；这是**合成特征下草稿组件成本**，没有包含目标验证／状态提交，也没有在线接受率，不能当成用户吞吐或加速比。草稿骨干一次加载13.3秒、目标embedding/head10.8秒；有常驻复用价值。单卡任务已正常结束，缓存释放并同步，末尾保留约33.6MB设备分配记录，不声称显存归零。

修正原生Linear引用接口的新隔离候选已经构建通过：24个对象缓存复用、一个MLA源文件重编和链接约18.49秒，56+112项有效metadata合约及10+17项非法合约通过。32卡验证任务已实际启动，复用一次权重加载比较MLA输入全M1与q_a两token分组路径；完整token/W8/W9/W16/QA/正常退出验收仍待完成。尚无合格在线投机加速。详见[完整草稿组件范围与数据](evidence/draft-full-weights-and-cache-contract-r4485.json)。

下节为本轮较早的测量与原因定位，未完成传输等状态以本节为准。

## 用户 decode 指标与长草稿验证更新

投机实验以**每位用户实际得到的 decode tok/s**为准，计入真实草稿、目标验证、提交／回滚和重放，排除 prefill 与 anchor。先 B1，再逐请求检查 B2/4/8/16，并报告非零块提交间隔 P95。浮点值无需逐位一致；组件原始值只用于定位，最终以完整 token、EOS、QA 和重复验证判定可用。

最新普通 AR 完整验收为 **5.30406 / 5.34375 tok/s/用户**，平均 TPOT **188.53 / 187.13 ms**；32 卡 TP8/PP4、512 输入、最多128输出，计127个有效decode token。完整冻结ID、QA16、32 owner正常结束、Slurm0及IBext通过。此前5.43–5.47是另一轮有效观察，两轮差值不能归因为代码收益。

当前**尚无合格在线投机加速**。MoE控制及原生BF16分块归约后的 W8 仍在输出索引33分歧，W8/W9/W16 teacher检查也未通过。新的通用边界追踪在实际TP0/6上显示：已记录的前3层边界相同，首个剩余差异位于第3层 MLA q_a 投影；8行中有3个BF16元素不同。

三个八卡组件实验已由CPU独立审核：24个正常Runtime、4,800份raw。q_a 按2 token分组在已测几何和正反布局上保留逐token参考；4 token或整块不保留。MoE的routed-up可尝试4 token组，共享gate/up可尝试2 token组；这些只是有限fixture上的算子候选，不是整模型收益。原生strided-batched M1同样未普遍保留参考，不能直接替换。

已准备默认关闭的 MLA 修复候选：比较全部4个输入投影按M1执行，与q_a按2 token、其余3个按M1执行；复用一次整模型加载，检查实际12项dispatch计数、完整生成、W8/W9/W16、QA和正常退出。构建与整模型验证以实际Slurm句柄为准，尚未升级为可用版本。另已建立25个C++源文件的编译缓存，约45.7秒生成与原库SHA完全相同的库；这是缩短开发等待，并非模型加速。

优先 [Modal Kimi-K3 DFlash](https://huggingface.co/modal-labs/Kimi-K3-DFlash) 原生W16/K15，并比较W8；作者B1平均每轮有效输出在GSM8K为7.984、HumanEval为8.358、MT-Bench为4.921、LongBench-v2为3.688，来自8×B300/SGLang。备选 [RedHat DSpark](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark) 原生8提案。二者本地完整SHA已验证，远端传输仍在推进，尚无当前目标的在线接受率。不能把16候选等同于16接受或16倍加速。

以当前约5.3 tok/s普通AR为例，平均每轮真正提交8个有效token时，要达到15 tok/s，整轮需≤533ms；5倍需≤302ms，10倍需≤151ms。它是判断方案是否值得推进的成本预算，不是BW1000实测。下一步优先降低块验证的投影／通信成本和部分接受后的状态重放成本，再比较真实长草稿。详见[本轮精简证据](evidence/speculative-user-decode-frontier-r4474.json)。

下列为之前轮次，以上节为最新结论。

## 最新结果与投机修复（北京时间 10月22日 10:04）

Kimi B1 普通 `Engine.forward` 的最新完整验收为 **5.46865 / 5.43267 tok/s/用户**，前/后测 TPOT **182.86 / 184.07 ms**；32卡 TP8/PP4、512输入/最多128输出，排除 prefill 首个 anchor，计127个有效decode token。完整冻结ID、QA16、32 owner正常关闭、Slurm0和IBext通过。这是普通AR harness基线；与上轮5.25的差值不能归为代码因果收益。当前仍**没有合格的在线投机加速**。

W8 在首层 dense MLP 的M1对照后仍于输出索引41分歧，三次重复一致。八个实际分片的KDA1输入、输出和活跃conv/SSM状态已经相同；已保存TP0/6的最早剩余差异为FFN1输出。新的真实MoE1单层工具只加载该层896个专家、router、投影与共享专家，一份CPU权重供八个owner复用，以分离各子路径，避免反复加载全模型。单层工具尚未验收为模型修复。

原生BF16归约的消息宽度诊断已独立审计：同一节点 Default/强制/强制/Default 四个新进程、20种几何、128个正常Runtime、10240份CPU raw。Tree+LL和Tree+Simple在这组fixture中保留原Default逐行参考；Ring组合不保留。H7168是由实际专家核心分片输出扩展的宽度fixture，并非实际共享专家输出。**随后32卡全局TreeLL整模型测试的普通AR冻结ID失败，方案排除**；局部结果没有推广为全局配置。下一步保留默认环境，仅对宽块验证检查2/4/8行分块原生BF16归约。无GPU F64。

新增[专家核心组件源码](experimental-latest-5f91ad7/kimi-expert-core-r4391/kimi_tiled_batch_r4391.cu)及同目录原生Runtime C++、Python检查、DTK脚本：当前N1 W13/W2算术扩展到T1/2/4/8/16的并行grid。八个实际TP分片、64个选定真实专家、80个输入几何通过400份raw检查，串行/并行activation和output相同，native provider1输出差异为0。T8有效graph host完成1.75868→1.56474 ms，T16为3.47556→3.06984 ms；**只对比原N1核心串行调用，未对比当前N8 provider1性能，也不包含router/shared/TP，不能声称模型收益**。最初4385空graph计时已排除，4391检查了sentinel覆写和输入改变后与新eager输出相同。迁移需现有锁定Common/SDK及4346基线tile.o、模型权重路径，脚本拒绝覆盖已有候选；不是生产替换包。

DeepSeek mHC Mul/四路Sum/Cast融合已完成整模型验收：8卡、B16/M16、512输入/最多512输出，同一次权重驻留、两种布局各OFF/ON/ON/OFF。decode中位 **32.76699→33.56857 tok/s（+2.45%）**，含prefill **24.62292→25.10258**；完整冻结512-ID/EOS/重复、QA24、8 owner及Slurm0/独立CPU审计通过。保留FP32 attention、顺序四项FP32相加及最终BF16舍入，无F64；这是同进程代码收益，并未超过此前不同轮次35.98的绝对观测值。[融合组件源码](experimental-latest-5f91ad7/mhc-collapse-r4367/deep_collapse_r4367.cu)仍为已验收variant0。

GLM 固定B32、M2/4/8/16/32的一次驻留扫描继续，当前首轮观察较小M2/M4和整批M32均慢于M16；待全矩阵、完整冻结3330输出/QA/重复/正常16 owner和独立审计后才采用。此前两节点B32/M16合格decode33.14179、含prefill10.38139仍有效，32卡70.0～70.7另列。

长草稿优先Modal原生W16/K15，比较W8；RedHat原生K8/W9为备选。二者已有本地完整SHA，远端传输仍推进，远端完整SHA/完整GPU draft forward/实际接受率未验收。LightSeek DFlash2为另一种MLA+SWA备选，原生lookahead8，无训练W16证据，暂不增加权重传输。逐用户decode统计包括真实draft/verify/commit/rollback/replay/未隐藏通信，B1优先再B2/4/8/16；实际每轮8个有效token时15tok/s需全轮≤533ms，按本轮普通AR成本，5倍约≤294ms、10倍约≤147ms。预算不等于BW1000实测或承诺。详见[新验收范围与精简证据](evidence/model-and-speculative-progress-r4427.json)。

补充独立验收：真实MoE1的5384个参数共16.10GB，一次CPU读入约32～33秒、八owner原生load/postload各约0.48～0.66秒，16 owner/6336份raw检查通过。Default配置完整复现实际模型TP0/6的serial和packed FFN1输出；路由、专家核心相同，共享专家与routed-up同时按M1控制后八分片所有已捕获中间值恢复串行参考。原共享路径单独控制仍有2个最终FFN元素不同，上投影单独控制也不够。它是单层原因定位，未成为整模型修复或用户吞吐。

原生BF16分块另经8 owner/800份raw审计：chunk4在20种几何保留Default逐行参考；chunk8仅对H3584全部通过，H7168不通过。新的隔离候选据此只对MoE H3584采用chunk8，mode11/W≤16且默认关闭，普通AR、prefill和全局通信环境保持原路径。共享MLP/routed-up M1是正确性控制，后续还需并行化以获得实际decode收益。

GLM完整扫描已正常结束并独立审计：16卡TP8/PP2、B32、512输入/最多128输出，M8与M16 decode中位 **30.06943 / 33.06372 tok/s**，含prefill **10.06827 / 10.37117**，均通过冻结3330完整ID。M2/M4/M32虽有计时且QA通过，长输出冻结ID不符，全部排除，不能列为可用性能。扩大B/M在该服务配置下没有取得提升；此前33.14179与本轮33.06372为轮次复测差异。详见[新增验收证据](evidence/speculative-cause-and-GLM-micro-r4443.json)。

下方按时间保留历史，最新结论以上节为准；目标继续进行中。

## 最新已验收结果与进行中的修复（北京时间 10 月 22 日 08:05）

最新一次驻留 32 卡 TP8/PP4 已正常完成并独立审计：512 输入、最多 128 输出，B1 普通 `Engine.forward` 前/后测 **5.25369 / 5.26322 tok/s/用户**，平均 TPOT **190.34 / 190.00 ms**。排除 prefill 产生的首个 anchor，计 127 个有效 decode token；完整冻结 ID、前后 QA16、CPU argmax、32 owner 正常关闭、Slurm0 和 IBext 通过。它是当前原生 harness 对照，不是 HTTP 或硬件极限，也不能把相对上一轮约 5.05 的差值归因于代码优化。

W8 根测试仍在索引 53 分歧；真实 prompt-lookup K1/K3 也完整 ID 失败且更慢。**尚无合格在线投机加速**。Oracle W8/W9/W16 只用于目标验证正确性，不代表真实草稿接受长度或用户速度。

细查八个 TP 分片的第一层 KDA 输入/输出及活跃卷积、SSM 状态均已对齐，最早已记录剩余差异为唯一的首层 dense FFN。局部候选仅在默认关闭的 mode11、W≤16 下将该 dense MLP 按原生 M1 执行，包括原生 TP 归约；无 F64、无新 GPU 内核。构建及 CPU 合约检查已通过，新的单次 32 卡加载正在进行，待完整输出/QA/普通前后测/正常退出和独立审计。串行控制用于定位，可能更慢，不能当成最终并行验证优化。

长草稿优先 [Modal Kimi-K3 DFlash](https://huggingface.co/modal-labs/Kimi-K3-DFlash) 原生 W16/K15，同时比较 W8；备选 [RedHat DSpark](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark) 原生 K8/W9。Modal 作者 B1 平均每轮输出：GSM8K 7.984、HumanEval 8.358、MT-Bench 4.921、LongBench-v2 3.688；来自 8×B300/SGLang，不能推成海光速度。窗口16不等于每轮接受16。原 Modal 传输已明确失败，保留原证据，确认原进程退出并验证远端已有 1.75 GB 前缀 SHA 一致后，仅续传剩余部分；RedHat 原传输继续观察。两者远端完整 SHA 和完整 GPU forward 尚待完成。

逐用户测 B1、B2/4/8/16；有效提交 token/各自 decode 时间包含真实草稿、验证、选择、状态提交/回滚、重放、未隐藏通信。报告平均每轮输出及 P95 非零出 token 间隔。若每轮有效输出8 token，15 tok/s要求整轮≤533 ms；按本轮约5.258普通基线，5倍要求约304 ms，10倍约152 ms。只是预算，不能当作实测收益。

GLM 两节点 TP8/PP2 的32行容量候选现已通过整模型、完整冻结 ID、QA、重复、正常16 owner/Slurm0和独立审计。原加载8 GiB/运行6 GiB余量保留，默认32通信通道，RoPE按640服务容量生成、control复用world communicator，两个开关默认关闭。512输入/max128、B32/M16 decode中位 **33.14179 tok/s**，含prefill **10.38139**，每卡decode **2.07136**；B32/M8为30.13566。此前B24/M8约28.28是不同容量配置，不能把差值称为单项算子因果收益；32卡约70的结果仍单列。

DeepSeek FP32新分段计时已通过B8/B16、32输出前缀、OFF/ON/ON/OFF、QA及正常8 owner独立审计；CSA/HCA占主要decode时间，mHC/FFN准备也显著。计时插入同步，属于host完成窗口诊断，不能作为纯GPU时间或新的512输出性能。原FP32 B16/M16完整512输出复测34.55～35.98 decode仍有效；FP16虽过QA但长输出ID不同，继续排除。

测试commit完整tree与上轮检查的main一致，仍按锁定commit与递归依赖恢复。详见[本轮精简证据](evidence/speculative-user-decode-current-r4355.json)。整个三模型优化目标仍进行中。

## 历史长草稿接入进展（后续结果以上节为准）

已增加显式、互斥的 ordinary epoch：每次切换重置缓存，普通 `Engine.forward` 使用原始同读写行输入，epoch 内拒绝成对状态验证 API，普通 decode 不进入 SpecScope、快照和重放。实际 32 卡运行的前测为 512 输入、128 输出，排除首个 anchor 后计 127 个有效 decode token，约 **5.2623 tok/s/用户**、平均 TPOT 190.03 ms，完整 ID 与冻结参考相同。它只是本次前测观测值；后续实验失败，缺少后测和正常 Slurm0，因此尚未成为完整对照性能资格，更不能称在线投机收益。

在原模式 10 基础上，将所有 MLA KV 展开投影也按原生逐行 M1 执行的模式 11，首次使“一个 W2 块后继续生成至 128 token”三次重复全部匹配冻结输出；模式 10 仍在索引 41 分歧。这没有推广为任意块长度的修复：随后 K7/W8 的完整序列仍失败两次，K8/W9 又被旧接口最多 8 个验证位置的 guard 拒绝，整个作业 FAILED1，原结果保留。已正常关闭的 32 个 owner 不能替代 Slurm0 或后测。当前无可采用的在线投机加速。

W16 候选另行隔离构建，只扩展输入元数据和 KDA 偏移的宽度 admission，保留总 packed token ≤128、请求数、成对状态、owner、设备和流边界，原设备内核及普通 API 不变。安排构建后八卡 W8/W9/W16 对串行 W1 的输出和最终状态检查，再做独立 CPU raw 审计；组件通过也不能替代尚失败的整模型 W8 完整 ID 校验。下一轮模型测试应先抓长块最早分歧，避免只放大窗口后报告无效速度。

Modal DFlash 权重已开始传往集群；远端完整 SHA 校验尚未完成。官方 `z-lab/dflash` 源码锁定 `07ebd93db9f472af339b644bb70221ad8428328a`，仅抽取已审阅的第一代模型定义。CPU 微型模型 B1/B2、W8/W16 的缓存与重算、独立 causal/window mask、缓存释放检查通过；完整 checkpoint 的 69 个 key/shape 与 meta 模型全部吻合。该结果没有加载完整权重执行 GPU forward，也没有测当前目标的接受长度。详见 [本轮范围与证据](evidence/speculative-metric-progress-r4268.json)。

长草稿优先比较 Modal W8/W16 与 RedHat K8/W9，逐请求计时、计新提交 token。以本次尚待完整验收的 190.03 ms 普通单步为例，若平均每轮真正提交 8 token，要达到 15 tok/s，整轮草稿、验证、提交和回滚必须合计低于约 533 ms；5 倍加速要求约 304 ms，10 倍要求约 152 ms。这是成本预算，不是实测加速。公开 DFlash 论文的超过 6 倍来自其报告模型和任务；BW1000 上必须另测真实接受长度、验证成本和状态推进。

## 8/16 窗口草稿与当前限制（本轮更新）

进一步查到 [Modal Kimi-K3 DFlash](https://huggingface.co/modal-labs/Kimi-K3-DFlash)，锁定 `c192d15a43407bf758b5ae0880d5c72052fef1de`。配置原有 `dflash_config.block_size=16`，不是将旧 Radix 草稿临时改为 16。官方普通 DFlash serving 路径丢弃 anchor 位置的输出，因此 W16 对应 K15 提案；RedHat DSpark 的 `sample_from_anchor=true` 则为 K8/W9，两者不能混用。

发布方 B1 的平均接受长度：Modal block8→16，GSM8K 5.905→7.984，HumanEval 6.013→8.358，MT-Bench 4.202→4.921，LongBench-v2 3.455→3.688。它们来自 NVIDIA B300/SGLang 的作者测试，并非 BW1000 实测，也不是与 RedHat/Radix 同提示、同采样的排名。Modal 权重 5,196,953,912 字节已在本地完整 SHA 校验：`0f0bd39d19c2dd0249de4f8216180f9729963402f902d50f19a156a8d358b733`；尚未在当前目标上验证接受率或在线速度。模型仓库未附 auto_map 指向的 Python 文件，不能直接假设 AutoModel 加载可用；适配须基于已审阅的 serving 实现。

[RedHat Kimi-K3 DSpark](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark) 原生训练 8 提案，revision `38a88101e0d46bb22134b9da340f381b954d40d4`；9,489,807,826 字节权重也已本地完整 SHA 校验，正在传往集群，远端完整校验尚待完成。该包包含目标 embedding/head，不能把包总参数量全部视为额外草稿骨干。其 96Q/16KV、2048 causal SWA、taps `[24,48,72,88,92]`、MASK 163837 与旧 Radix 格式不同，禁止混用旧特征、mask 和非因果 attention。作者公开平均接受长度随任务约 3.1–6.4，不是承诺每轮接受 8 个提案。

判优以每位用户新提交的有用 token / 其 decode 时间为准。稳态近似为 `E[C]/E[Tcycle]`，相对普通 AR 为 `E[C]*T_AR/E[Tcycle]`；例如一轮有效 8 token、整轮耗时为普通单步 2 倍，对应 4 倍，而非 8 倍。草稿越长是否更好取决于真实前缀存活率和整轮成本。先 B1，再 B2/4/8/16、兼容 W8/W9/W16；普通 AR 使用 Engine.forward，同进程驻留权重 ABBA，prefill 排除，EOS/拒绝/重放不计输出。详见 [当前实验策略](evidence/speculative-user-decode-policy-r4218.json)。尚无合格在线投机加速。

## 已通过的草稿组件

完整 BF16 RadixArk Kimi-K3 DSpark，revision `3c5bac301d9cf392706189d82ed947feca6c2f0f`。原训练为 7 提案，8/16 是额外测试；16 是未训练长度外推，不能声称有效接受 16 token。输入为历史 3e0 目标的两个真实 first-block 特征 fixture，不是当前目标的在线接受率证据。

八卡各一次加载，B1/2/4/8 × K4/7/8/16：2,048 条普通对照、1,536 条阶段记录、4,096 条设备图对照，400 份原始 logits 经独立 CPU 有限值/argmax 检查。融合 Markov 算子的 256 组 CPU FP64 抽样通过；完整草稿 ID 与原 Torch 路径相同，正常退出和图/张量释放通过。

B1 草稿 host-completed 耗时，单位 ms：

| 提案数 | 原 Torch 固定前缀 eager | 原 Torch 固定前缀图重放 | 融合 Markov eager | 融合 Markov 图重放 |
|---|---:|---:|---:|---:|
| 4 | 13.78 | 9.61 | 11.25 | 7.21 |
| 7 | 16.84 | 12.56 | 12.47 | 8.37 |
| 8 | 17.73 | 13.31 | 12.74 | 8.52 |
| 16，外推 | 24.51 | 19.77 | 14.73 | 10.19 |

原 Torch 和融合算子各自是独立八卡实验，表中跨实验值为描述性对照，不是同一设备随机化的单项因果倍数。图内含 backbone、词表投影、Markov、confidence、argmax、结果拼接；图捕获与前缀初始化另记。变更 anchor/noise 验证设备输入可更新，固定前缀不变；尚未证明接受前缀推进、目标特征更新、变长、拒绝回滚或在线用户速度。代码、范围和精简证据见 [组件目录](experimental-latest-5f91ad7/speculative-draft-r4165/README.md) 与 [证据](evidence/speculative-decode-components-r4183.json)。

阶段计时显示 B1 backbone 约 8.3 ms，词表投影约 2 ms，Markov/confidence 在 K8/K16 时约 6.55/13.05 ms。融合实现以 wave64 复用 W2，合并 W1 gather、GEMV、BF16 bias 舍入和 base-logit 相加；未修改目标模型，未使用 F64 AllReduce。

## 目标验证器仍未修好

4148/4149/4150 完成实际 32 卡、一次完整权重加载、模式 6/7/8 对照及独立审计；正常 owner、QA、前后 AR 和追踪/未追踪重复通过，但三模式完整 128-token 继续生成均不同于冻结参考。

模式 6 首差异位置 33；只将多 token MLA attention 改为原生逐 query 调用后，模式 7/8 首差异位置均为 41，仍不能采用。22,860 份 CPU raw 诊断指出：首个 MLA 层的 attention/gated 输入已对齐，下一处为 MLA 输出投影，记录的 rank 0 有 4 个 BF16 元素不同。原始 logits 逐位相同只是诊断，失败依据仍是完整输出 token 不同。

只保存了 rank 0/6 的该投影输入，不能臆造其他六份。4173 最初按全八分片准备 fixture，在 GPU 提交前失败；保留失败。4175/4176 改为实际两个分片、每分片四个设备副本的本地 GEMM 回放：M2 差异在归约前存在，逐行 M1 精确复现串行参考。没有完整 TP8 归约或模型修复资格。

4179/4180/4181 已完成实际 32 卡测试与独立审计，模式 7/9/10 完整 token 均失败，首差异分别为 41/49/41；32 owner 正常关闭，前后 AR/QA/重复通过。它们没有在线性能资格。后续 22,860 份原始记录分析将模式 10 的更早剩余差异定位到局部 MLA KV 展开投影。4214/4215 用真实 layer7 输入、仅两个已保存分片 0/6、每分片四个硬件副本完成 8 卡回放，104 份 CPU 原始输出独立检查通过；分片 6 普通 M2 与逐行 M1 相差 2 个 BF16 元素，逐行 M1 恢复串行参考。此为本地投影诊断，没有假设其他六个分片、没有完整 TP8 归约或整模型修复资格；不能仅因 raw 差异采取模型结果，仍以最终完整 token 为门槛。

## GLM 两节点与 DeepSeek 长输出

GLM 16 卡 TP8/PP2，原层边界 `[0,38,78]`、640 context：四组构造/通信/只读缓存分类试验和独立审计通过。尾段最小空闲显存 B8 11.95 GiB，B16 9.07，B24 9.07，B32 3.33；B24 通过 8 GiB 加载门槛。逻辑缓存大小与实际分配不是同一口径，B16/B24 相同空闲量可能与分配器容量分档有关，不能从逻辑字节直接推断物理节省。

B32 尾段投影 K/V 各 3.125 GiB，latent 0.781、rotary 0.098、indexer 0.049 GiB；逻辑权重约 46.917 GiB。实际网络均为 IBext_v8，仍有 GDR Disabled 日志，未声称网络峰值。4167/4168 已完成原边界 B16/B24、M8/M16 整模型扫描并独立通过完整冻结 ID、QA、重复、16 owner/Slurm0/IB 验收；B16/M8、B24/M8、B16/M16、B24/M16 的 decode 中位分别 25.2770、28.2840、25.1255、27.2881 tok/s，含 prefill 分别 8.4914、9.4605、8.4581、9.3487。此为 512 输入/max128 的配置复测，不是代码因果收益。保留 8 GiB 加载及 6 GiB 运行余量。此前切点 40 为复用 indexer 层而被拒绝，不能重用其不合法方案。

另一个 B32 容量候选通过 persistent cache 精确分配将尾段空闲由 3.33 提到 7.55 GiB，结合 8 个通信通道达约 8.20 GiB并通过构造加载门槛。但其后实际 16 卡整模型 4199 在加载后 QA 发生真实 token 分歧，Slurm FAILED1、4200 队列拒绝，未达到正常全模型验收，不采用任何速度。尚未分离精确分配、通道数与数值路径的因果影响，不能认定内存容量证明等于模型可用。

DeepSeek-v4-Flash 最新 5f91、8 卡、512 输入/最多 512 输出、decode cohort16：B16/B32/B64 decode 中位数分别 **32.7598 / 35.0566 / 37.2190 tok/s**，含 prefill 分别 **24.7171 / 26.5720 / 27.9915**。各四次完整冻结 512-ID/EOS/重复/有限值/CPUargmax/正常 owner/Slurm0 与独立审计通过。B32/B64 是多个 16 行 cohort 顺序执行，原宽 N32/64 路径仍 token 失败。此为长度和配置复测，不能当成单项代码加速或在线混合流量成绩。

当前还没有合格的在线 Kimi 投机加速、三个模型的完整硬件极限、最优 TP/PP/EP 组合或 HTTP 混合流量结果，目标保持进行中。接手者先读私有最新进度检查点和实际 Slurm 句柄；观察超时不等于生产者终态，不可因此重复加载。
