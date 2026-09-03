#!/usr/bin/env python3
"""
InfiniLM 路线 B —— GGUF -> InfiniLM 权重映射表（打包器与审计脚本的单一事实源）。

所有条目都由阶段 0 审计实测得出，不是推测：
  * InfiniLM 侧参数键/shape/取向：scripts/gguf_routeb_probe_params.py 在 CPU 上
    构造 mini qwen3_5 引擎导出的 state_dict（121 键），取向为 [out, in]，与 GGUF
    blob 的行主序一致 -> 打包不需要转置。
  * GGUF 侧键与 shape：scripts/gguf_routeb_audit.py D 节对 866 个张量实测。
  * transform 依据 llama.cpp conversion/qwen.py（行号为该文件实测）：
      388  A_log        -> -exp(A_log)              （故打包需反解 log(-x)）
      391  dt_bias      -> 改名 dt_proj.bias，值不变  （故 ssm_dt.bias 原样用）
      394  *.norm.weight -> w + 1（linear_attn.norm 除外）
                                                （故打包不得再加 1）
      571-605 _LinearAttentionVReorderBase.modify_tensors：需逆重排的集合是
            in_proj_qkv(仅 V 行段) / in_proj_z / in_proj_a / in_proj_b(head_dim=1) /
            A_log / dt_bias(head_dim=1) / conv1d(仅 V 通道段)；
      609  out_proj 重排的是 **列(in 维)** —— 本方案改为运行时对激活做 head gather，
            权重保持逐字节不变，故此处不标 transform。
            `linear_attn.norm`(=ssm_norm) 不在重排列表内，确认无需重排。
      615  注释 "Qwen3.5 always applies interleaved MRoPE" -> mrope_interleaved 必为 True
      619  写入 GGUF 的 mrope_section 是 4 元素 [11,11,10,0]，而 InfiniLM
            qwen3_5_attention.cpp:65 硬性要求 3 元素 -> 打包时去掉尾 0。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# transform 语义
# ---------------------------------------------------------------------------
T_NONE = ""  # 原样搬运（blob 逐字节 / dense 仅换 dtype）
T_VROWS = "vrows"  # 沿 out 维按 V 头分块整块搬回 grouped 序（blob 可行级置换）
T_VELEM = "velem"  # 1-D、每头 1 个元素：T_VROWS 的 head_dim=1 退化形式（同一实现）
T_ALOG = "alog"  # A_log = log(-ssm_a)，再置换
T_DENSE = "dense"  # 反量化为 BF16（框架不支持该参数走量化路径）

# V 头置换在 dim0 上的作用域：
#   all    = 整个 dim0 都是 value 头（in_proj_v / in_proj_z / in_proj_a / in_proj_b / A_log / dt_bias）
#   v_tail = 只有末尾 value_dim 个元素是 value 段（conv1d 的 [q|k|v] 通道拼接）
VPERM_ALL, VPERM_TAIL = "all", "v_tail"

# blob 参数在产物 / 框架里的名字后缀。阶段 2 的 get_param_layout 必须用同名，
# 否则 load_state_dict(strict=False) 会把 400 个权重静默丢掉。
BLOB_SUFFIX = "weight_bytes"

# 两者共用一份置换实现：每头几个元素由条目 shape 推出来（见 gguf_transforms.vperm_head_dim），
# 48 个元素 / 48 个头 = 1 ⇒ 自然就是逐元素置换，不需要第二套代码。
VPERM_TRANSFORMS = (T_VROWS, T_VELEM)


def needs_vperm(e: "Entry") -> bool:
    return bool(set(e.transforms) & set(VPERM_TRANSFORMS))


# ---------------------------------------------------------------------------
# GGML 类型名（数值见 ggml.h；本文件不依赖 gguf-py，避免脚本互相 import 拉环境）
# 实测本 GGUF 出现的类型集合由 scripts/gguf_routeb_shape_contract.py 断言。
# ---------------------------------------------------------------------------
F32, Q8_0, Q4_K, Q5_K, Q6_K = "F32", "Q8_0", "Q4_K", "Q5_K", "Q6_K"
IQ4_NL, IQ4_XS = "IQ4_NL", "IQ4_XS"

# 阶段 3 v1 必须实现的 block 类型（实测本文件主模型只出现这 4 种）。
# Q4_K 不跟 IQ4 一起延期：它与 Q5_K 同族（144B/256，只差第 5 bit 平面），
# Q5_K 本来就要写，多支持 Q4_K 接近零成本，而它占 2 个张量 45 MiB。
NATIVE_BLOB_TYPES = (Q8_0, Q4_K, Q5_K, Q6_K)
# v1 稠密化的 i-quants（执行方案 §2.4 决策：量小、需查码表，上原生 kernel 推到阶段 6）。
# 实测共 5 个张量 0.23 GiB，稠密化后占 0.82 GiB，代价 +0.60 GiB（预算仍 ≤ 24 GiB）。
V1_IQUANT_DENSE = (IQ4_NL, IQ4_XS)
DENSE_SRC_TYPES = (F32, Q8_0, Q6_K) + V1_IQUANT_DENSE


def apply_v1_exceptions(plan, gguf_types, enabled=True):
    """把 v1 不打算写 kernel 的 i-quants 条目就地转为稠密化。

    gguf_types: {张量名: GGML 类型名}，由调用方从真文件采集（本模块不依赖 gguf-py）。
    阶段 6 上了 IQ4 码本后传 enabled=False 即可全部回到逐字节路径。
    """
    n = 0
    if enabled:
        for e in plan:
            if e.blob and gguf_types.get(e.gguf) in V1_IQUANT_DENSE:
                e.blob = False
                e.transforms = e.transforms + (T_DENSE,)
                e.note = (
                    e.note + "；" if e.note else ""
                ) + "v1 稠密化例外（源 %s），阶段 6 上原生 kernel 后取消" % gguf_types[
                    e.gguf
                ]
                n += 1
    return n


@dataclass
class Entry:
    """一条 GGUF 张量 -> 一个 InfiniLM 参数。"""

    infinilm: str  # InfiniLM 参数名（含 model.language_model. 前缀）
    gguf: str  # GGUF 张量名
    shape: tuple  # InfiniLM 期望 shape（未 TP 切分的全量），取向 [out, in]
    blob: bool  # True = 保留 GGUF 原始 block 字节（U8 [out, row_bytes]）
    transforms: tuple = ()
    types: tuple = ()  # 允许的 GGUF 源类型名；() = 不限（由 contract 脚本报告实际值）
    slices: tuple = ()  # 沿 out 维占用的 [start, end)；共用同一 gguf 的条目做覆盖校验
    vperm: str = VPERM_ALL  # T_VROWS 的作用域（仅当 transforms 含 T_VROWS 时有意义）
    # 该条目的权重需要置换的是**列（in 维）**而不是行：块量化沿 in 维分块
    # （Q4_K/Q5_K/Q6_K block_size=256），打包期置换列 = 跨块重排 = 必须重量化，做不到。
    # 于是只能在运行时置换喂给它的输入激活，规则由 activation_vperm_rules() 导出进 config。
    # 故意不放进 transforms：那个元组描述的是「打包期对字节做的事」，混进去会污染字节路径。
    act_vperm: bool = False
    note: str = ""


@dataclass
class Dims:
    hidden: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    ffn: int
    lin_k_heads: int
    lin_v_heads: int
    lin_k_dim: int
    lin_v_dim: int
    conv_kernel: int
    vocab: int
    n_layers: int
    interval: int
    mrope_section: tuple = (11, 11, 10)
    rope_theta: float = 1e7
    partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144
    architectures: str = "Qwen3_5ForConditionalGeneration"

    # --- 派生量 ---
    @property
    def q_rows(self) -> int:  # q_proj 行数 = heads * head_dim * 2（q 与 gate 每头交错）
        return self.n_q_heads * self.head_dim * 2

    @property
    def kv_rows(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def o_in(self) -> int:
        return self.n_q_heads * self.head_dim

    @property
    def key_dim(self) -> int:
        return self.lin_k_heads * self.lin_k_dim

    @property
    def value_dim(self) -> int:
        return self.lin_v_heads * self.lin_v_dim

    @property
    def qkv_rows(self) -> int:  # q | k | v 融合（与 GGUF attn_qkv 一致）
        return self.key_dim * 2 + self.value_dim

    @property
    def conv_channels(self) -> int:
        return self.qkv_rows

    @property
    def v_per_k(self) -> int:
        return self.lin_v_heads // self.lin_k_heads

    def layer_types(self) -> list:
        """与 C++ prepare_qwen3_5_model_config 的推导完全一致：(i+1) % interval == 0。"""
        return [
            "full_attention" if (i + 1) % self.interval == 0 else "linear_attention"
            for i in range(self.n_layers)
        ]


REAL = Dims(
    hidden=5120,
    n_q_heads=24,
    n_kv_heads=4,
    head_dim=256,
    ffn=17408,
    lin_k_heads=16,
    lin_v_heads=48,
    lin_k_dim=128,
    lin_v_dim=128,
    conv_kernel=4,
    vocab=248320,
    n_layers=64,
    interval=4,
)

MINI = Dims(
    hidden=512,
    n_q_heads=2,
    n_kv_heads=1,
    head_dim=256,
    ffn=1024,
    lin_k_heads=2,
    lin_v_heads=6,
    lin_k_dim=128,
    lin_v_dim=128,
    conv_kernel=4,
    vocab=1024,
    n_layers=8,
    interval=4,
)


PREFIX = "model.language_model."


def layer_entries(d: Dims, i: int, role: str) -> list:
    """第 i 层的映射条目。role ∈ {'linear_attention', 'full_attention'}。

    注：源类型不在表中写死（同一后缀在不同层就用过 Q4_K/Q5_K/Q6_K/Q8_0/IQ4_*），
    由 contract 脚本从真文件采集后比对 NATIVE_BLOB_TYPES / DENSE_SRC_TYPES。
    """
    L = f"{PREFIX}layers.{i}."
    G = f"blk.{i}."
    kd, vd = d.key_dim, d.value_dim
    out = [
        Entry(
            L + "input_layernorm.weight",
            G + "attn_norm.weight",
            (d.hidden,),
            False,
            (T_DENSE,),
            note="GGUF 已 baked +1，打包不得再加",
        ),
        Entry(
            L + "post_attention_layernorm.weight",
            G + "post_attention_norm.weight",
            (d.hidden,),
            False,
            (T_DENSE,),
            note="同上",
        ),
        Entry(
            L + "mlp.gate_proj.weight", G + "ffn_gate.weight", (d.ffn, d.hidden), True
        ),
        Entry(L + "mlp.up_proj.weight", G + "ffn_up.weight", (d.ffn, d.hidden), True),
        Entry(
            L + "mlp.down_proj.weight", G + "ffn_down.weight", (d.hidden, d.ffn), True
        ),
    ]
    if role == "full_attention":
        out += [
            Entry(
                L + "self_attn.q_proj.weight",
                G + "attn_q.weight",
                (d.q_rows, d.hidden),
                True,
                (),
                note="行数含 q|gate 每头交错，与 Qwen35FusedQKVLinear 一致",
            ),
            Entry(
                L + "self_attn.k_proj.weight",
                G + "attn_k.weight",
                (d.kv_rows, d.hidden),
                True,
            ),
            Entry(
                L + "self_attn.v_proj.weight",
                G + "attn_v.weight",
                (d.kv_rows, d.hidden),
                True,
            ),
            Entry(
                L + "self_attn.o_proj.weight",
                G + "attn_output.weight",
                (d.hidden, d.o_in),
                True,
            ),
            Entry(
                L + "self_attn.q_norm.weight",
                G + "attn_q_norm.weight",
                (d.head_dim,),
                False,
                (T_DENSE,),
                note="GGUF 已 baked +1",
            ),
            Entry(
                L + "self_attn.k_norm.weight",
                G + "attn_k_norm.weight",
                (d.head_dim,),
                False,
                (T_DENSE,),
                note="GGUF 已 baked +1",
            ),
        ]
    else:
        out += [
            Entry(
                L + "linear_attn.in_proj_q.weight",
                G + "attn_qkv.weight",
                (kd, d.hidden),
                True,
                (),
                slices=((0, kd),),
                note="attn_qkv 行 [0:kd]",
            ),
            Entry(
                L + "linear_attn.in_proj_k.weight",
                G + "attn_qkv.weight",
                (kd, d.hidden),
                True,
                (),
                slices=((kd, 2 * kd),),
                note="attn_qkv 行 [kd:2kd]",
            ),
            Entry(
                L + "linear_attn.in_proj_v.weight",
                G + "attn_qkv.weight",
                (vd, d.hidden),
                True,
                (T_VROWS,),
                slices=((2 * kd, 2 * kd + vd),),
                note="attn_qkv 行 [2kd:]，V 头 tiled->grouped",
            ),
            Entry(
                L + "linear_attn.in_proj_z.weight",
                G + "attn_gate.weight",
                (vd, d.hidden),
                True,
                (T_VROWS,),
                note="qwen.py:583 行重排（head_v_dim）",
            ),
            Entry(
                L + "linear_attn.in_proj_a.weight",
                G + "ssm_alpha.weight",
                (d.lin_v_heads, d.hidden),
                False,
                (T_DENSE, T_VROWS),
                note="实测源为 Q8_0；框架该参数不走量化路径 -> 稠密化；"
                "qwen.py:587 行重排 head_dim=1",
            ),
            Entry(
                L + "linear_attn.in_proj_b.weight",
                G + "ssm_beta.weight",
                (d.lin_v_heads, d.hidden),
                False,
                (T_DENSE, T_VROWS),
                note="同上",
            ),
            Entry(
                L + "linear_attn.A_log",
                G + "ssm_a",
                (d.lin_v_heads,),
                False,
                (T_ALOG, T_VROWS),
                note="GGUF 存的是 -exp(A_log)，需 log(-x) 反解",
            ),
            Entry(
                L + "linear_attn.dt_bias",
                G + "ssm_dt.bias",
                (d.lin_v_heads,),
                False,
                (T_VELEM,),
                note="qwen.py:589 逐头置换，值不变",
            ),
            Entry(
                L + "linear_attn.conv1d.weight",
                G + "ssm_conv1d.weight",
                (d.conv_channels, 1, d.conv_kernel),
                False,
                (T_DENSE, T_VROWS),
                vperm=VPERM_TAIL,
                note="GGUF 已 squeeze 成 [C,K] -> 补回中间维；仅末尾 V 通道段重排",
            ),
            Entry(
                L + "linear_attn.norm.weight",
                G + "ssm_norm.weight",
                (d.lin_v_dim,),
                False,
                (T_DENSE,),
                note="不在 qwen.py 重排列表内；两侧都不加 1",
            ),
            Entry(
                L + "linear_attn.out_proj.weight",
                G + "ssm_out.weight",
                (d.hidden, vd),
                True,
                (),
                act_vperm=True,
                note="qwen.py:609 重排的是列(in 维)，blob 不能跨块置换 -> "
                "运行时对输入激活做 grouped->tiled（见 config 的 activation_vperm）",
            ),
        ]
    return out


def build_plan(d: Dims) -> list:
    """全模型映射条目（含顶层）。"""
    entries = [
        Entry(
            PREFIX + "embed_tokens.weight",
            "token_embd.weight",
            (d.vocab, d.hidden),
            False,
            (T_DENSE,),
            note="实测 GGUF 为 Q6_K -> 反量化",
        ),
    ]
    for i, role in enumerate(d.layer_types()):
        entries += layer_entries(d, i, role)
    entries += [
        Entry(
            PREFIX + "norm.weight",
            "output_norm.weight",
            (d.hidden,),
            False,
            (T_DENSE,),
            note="GGUF 已 baked +1",
        ),
        Entry(
            "lm_head.weight",
            "output.weight",
            (d.vocab, d.hidden),
            False,
            (T_DENSE,),
            note="实测 GGUF 为 Q8_0 -> 反量化",
        ),
    ]
    return entries


def activation_vperm_suffix(e: "Entry") -> str:
    """条目对应的 checkpoint stem 后缀（剥掉层号、含结尾 '.'），供 C++ 做尾匹配。

    C++ 递来的 stem 形如 `layers.7.linear_attn.out_proj.`（挂在前缀下的相对形态，
    见 gguf.cpp 的 key_prefix_ 裁剪），所以这里必须同时去掉 PREFIX 和 `layers.<i>.`。
    """
    name = re.sub(r"^" + re.escape(PREFIX) + r"layers\.\d+\.", "", e.infinilm)
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    return name + "."


def activation_vperm_rules(d: "Dims", plan: list) -> list:
    """从映射表派生「运行时要对输入激活做的 V 头置换」清单（写进 quantization_config）。

    为什么必须有这件事：conversion/qwen.py:607-609 在导出 GGUF 时把 out_proj 的**列**从
    grouped 换成了 tiled；而 GDN kernel 期望/产出的 v 头序是 grouped（InfiniCore
    chunk_gated_delta_rule/cuda/kernel.cuh:112 `key_head_idx = value_head_idx /
    value_heads_per_key_head`）。打包期我们把 in_proj_v 等**行**向条目逆置换回 grouped，
    但 out_proj 的列置换不掉（跨块），所以只能把激活置换过去：grouped -> tiled。
    规则在这里派生、C++ 只照单执行，两边不各抄一份（§6.0 纠正 2 的同一原则）。
    """
    n_k, r, hd = d.lin_k_heads, d.v_per_k, d.lin_v_dim
    rules, seen = [], set()
    for e in plan:
        if not e.act_vperm:
            continue
        in_dim = int(e.shape[1])
        if in_dim != n_k * r * hd:
            raise ValueError(
                "%s: 条目 in 维 %d != num_k_heads*num_v_per_k*head_dim = %d，"
                "无法按头分块置换" % (e.infinilm, in_dim, n_k * r * hd)
            )
        suffix = activation_vperm_suffix(e)
        if suffix in seen:
            continue
        seen.add(suffix)
        rules.append(
            {"suffix": suffix, "num_k_heads": n_k, "num_v_per_k": r, "head_dim": hd}
        )
    return rules


def expected_keys(d: Dims) -> list:
    return [e.infinilm for e in build_plan(d)]


# 打包期需丢弃的 GGUF 张量。实测 blk.64 共 15 个张量 =
# 11 个普通 full-attention 层张量（attn_norm/attn_q/attn_k/attn_v/attn_q_norm/
# attn_k_norm/attn_output/post_attention_norm/ffn_gate/ffn_up/ffn_down）
# + 4 个 nextn.*（eh_proj/enorm/hnorm/shared_head_norm），共 0.327 GiB。
# 推论：主模型的 full-attn 层是 16 个（blk.3,7,...,63），而带 attn_q 的 block
# 共 17 个 —— 多出的那个就是 MTP block，不要误当成第 17 个注意力层。
DROP_PREFIXES = ("blk.64.",)
MTP_BLOCK = 64


def compress(shape: tuple) -> tuple:
    """去掉长度为 1 的维。GGUF 写入时对 conv1d 做过 squeeze（qwen.py:393），
    比对形状时需同样处理，否则 (C,1,K) vs (C,K) 会误报。"""
    return tuple(int(x) for x in shape if int(x) != 1)


# ---------------------------------------------------------------------------
# 派生工具：产物参数名、type 表键、行字节、config.json
# —— 打包器 / 阶段 2 C++ / 契约脚本都必须走这里，不得各抄一份
# ---------------------------------------------------------------------------
def ckpt_name(e: "Entry") -> str:
    """写进 safetensors（以及框架 state_dict）的参数名。"""
    if e.blob and e.infinilm.endswith(".weight"):
        return e.infinilm[: -len(".weight")] + "." + BLOB_SUFFIX
    return e.infinilm


def type_table_key(name: str) -> str:
    """config.json:quantization_config.ggml_types 的键 = checkpoint 张量名原文。

    曾经用过“去 model.language_model. 前缀 + .weight_bytes 归一回 .weight”的压缩写法，
    但那要求阶段 2 的 C++ 把同一套规则逐字符重实现一遍，拼错不会报错只会静默走
    稠密路径（能加载、显存暴涨、结果错）。现在 key 就是 safetensors 里的张量名，
    C++ 只递 stem（如 `layers.0.mlp.gate_proj.`）再探 `stem+"weight_bytes"` /
    `stem+"weight"`，命中 0 个或 2 个都抛错；挂载前缀由 quantization_config.key_prefix
    告知，不在 C++ 里硬编码。详见执行方案 §6.0 纠正 2。
    """
    return name


def row_bytes(n_in: int, block_size: int, type_size: int) -> int:
    """blob 一行的字节数。本模块不依赖 gguf-py，故 (block_size, type_size) 由调用方给。"""
    if n_in % block_size:
        raise ValueError("in=%d 不能被块大小 %d 整除" % (n_in, block_size))
    return n_in // block_size * type_size


def make_text_config(d: "Dims") -> dict:
    """config.json 的 text_config 段。

    键名集合以 scripts/gguf_routeb_probe_params.py::CFG 为准 —— 那份 config 已被
    InferEngine 实测接受（121 键全对齐），不要再引入未验证的键（如 architectures /
    layer_types：layer_types 由 qwen3_5_for_causal_lm.cpp:72-87 从 interval 推导）。
    """
    return {
        "model_type": "qwen3_5_text",
        "hidden_size": d.hidden,
        "num_hidden_layers": d.n_layers,
        "num_attention_heads": d.n_q_heads,
        "num_key_value_heads": d.n_kv_heads,
        "head_dim": d.head_dim,
        "intermediate_size": d.ffn,
        "rms_norm_eps": d.rms_norm_eps,
        "max_position_embeddings": d.max_position_embeddings,
        "vocab_size": d.vocab,
        "full_attention_interval": d.interval,
        "linear_num_key_heads": d.lin_k_heads,
        "linear_num_value_heads": d.lin_v_heads,
        "linear_key_head_dim": d.lin_k_dim,
        "linear_value_head_dim": d.lin_v_dim,
        "linear_conv_kernel_dim": d.conv_kernel,
        "attention_bias": False,
        "rope_parameters": {
            "rope_type": "mrope",
            "rope_theta": d.rope_theta,
            "partial_rotary_factor": d.partial_rotary_factor,
            # 必须 3 元素：qwen3_5_attention.cpp:65 硬校验；且
            # position_id_axes = len(mrope_section)（qwen3_5_for_causal_lm.cpp:52-64）
            "mrope_section": list(d.mrope_section),
            # 无默认值，缺键即抛；conversion/qwen.py:615 注释已确认恒为交错
            "mrope_interleaved": True,
        },
    }


def make_root_config(d: "Dims", ggml_types: dict, act_vperm: list = None) -> dict:
    """config.json 根段。

    ★ quantization_config 必须在**顶层**：ModelConfig ctor 只读
    `config_json["quantization_config"]`（model_config.cpp:5/16），而
    prepare_qwen3_5_model_config 的 text_config -> root 合并发生在 ctor **之后**；
    写在 text_config 里会得到 null => NoneQuantization 的静默降级。
    """
    return {
        "model_type": "qwen3_5",
        "torch_dtype": "bfloat16",
        # BF16 logits collapse close top candidates into exact ties.  Keep the
        # dense BF16 head weights, but accumulate/write its output in FP32.
        "lm_head_output_dtype": "float32",
        "tie_word_embeddings": False,
        "text_config": make_text_config(d),
        "quantization_config": {
            "quant_method": "gguf",
            # C++ 侧的表 key = 本表 key 去掉这段前缀（层级以下的模块不知道自己挂在
            # model. 下）；由打包器写入，不在 C++ 里硬编码
            "key_prefix": PREFIX,
            "ggml_types": ggml_types,
            # 运行时激活 V 头置换规则（见 activation_vperm_rules）。空列表 = 该产物没有
            # 列向置换的条目；C++ 缺这个键会直接拒启，避免旧 config 静默跑出错位权重。
            "activation_vperm": act_vperm or [],
        },
    }
