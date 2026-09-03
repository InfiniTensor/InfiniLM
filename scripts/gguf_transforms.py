#!/usr/bin/env python3
"""
InfiniLM 路线 B —— 打包期变换（纯 numpy，不依赖 gguf-py / torch / InfiniCore）。

为什么单独一个文件：审计脚本 `gguf_routeb_audit.py` C 节要**证明**这些置换自等/可行，
打包器 `gguf_to_infinilm.py` 要**执行**同一份置换。两处各写一遍正是阶段 0 踩过坑
（同一事实两份定义 -> 两套互相矛盾的预算数字），故这里只有一份实现。

置换方向约定（依据 llama.cpp conversion/qwen.py:571-605 与 §2.7）：
    HF / InfiniLM 序 = grouped，索引 [k][v]
    GGUF       序 = tiled  ，索引 [v][k]      （dst[v*n_k + k] = src[k*n_v_per_k + v]）
    ⇒ llama.cpp 写入 = grouped -> tiled = reorder_v
    ⇒ 本方案打包   = tiled -> grouped = reorder_v_inverse
方向本身仍属阶段 4 的 A/B 项（作用域已钉死，方向未闭环），故打包器暴露
`--vperm {inv,fwd,none}` 三个取值，默认 inv。
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# V 头置换
# ---------------------------------------------------------------------------

def reorder_v(t: np.ndarray, n_k: int, n_v_per_k: int, hd: int) -> np.ndarray:
    """grouped -> tiled，与 llama.cpp `_reorder_v_heads` 同语义（沿 dim0 的整头/整元素置换）。

    支持任意尾部维度：1-D（A_log/dt_bias，hd=1）、2-D（权重行）、3-D（conv1d [C,1,K]）。
    """
    rest = t.shape[1:]
    return (t.reshape((n_k, n_v_per_k, hd) + rest)
             .transpose((1, 0, 2) + tuple(range(3, 3 + len(rest))))
             .reshape((n_k * n_v_per_k * hd,) + rest))


def reorder_v_inverse(t: np.ndarray, n_k: int, n_v_per_k: int, hd: int) -> np.ndarray:
    """逆变换 = 两个轴参数对调后再调用一次。"""
    rest = t.shape[1:]
    return (t.reshape((n_v_per_k, n_k, hd) + rest)
             .transpose((1, 0, 2) + tuple(range(3, 3 + len(rest))))
             .reshape((n_k * n_v_per_k * hd,) + rest))


_VPERM = {"inv": reorder_v_inverse, "fwd": reorder_v, "none": None}


def vperm_head_dim(e, dims) -> int:
    """一条映射条目里每个 value 头占多少元素。

    `in_proj_a/b`、`A_log`、`dt_bias` 是 head_dim=1 的退化形式（每头一个标量），
    其余（in_proj_v / in_proj_z / conv1d 的 V 段）是 lin_v_dim 个。判据用 shape 而不是
    键名匹配，避免打包器里再写一张名字表。
    """
    n_heads = dims.lin_v_heads
    rows = int(e.shape[0]) if e.vperm == "all" else dims.value_dim
    if rows % n_heads:
        raise ValueError("%s：作用域行数 %d 不能被 value 头数 %d 整除"
                         % (e.infinilm, rows, n_heads))
    return rows // n_heads


def apply_vperm(arr: np.ndarray, e, dims, direction: str = "inv") -> np.ndarray:
    """按条目的作用域（all / v_tail）对 dim0 做 V 头置换。"""
    fn = _VPERM[direction]
    if fn is None:
        return arr
    n_k, hd = dims.lin_k_heads, vperm_head_dim(e, dims)
    v_per_k = dims.lin_v_heads // n_k
    if int(dims.lin_v_heads) % n_k:
        raise ValueError("lin_v_heads %d 不能被 lin_k_heads %d 整除"
                         % (dims.lin_v_heads, n_k))
    if e.vperm == "v_tail":
        n_v = n_k * v_per_k * hd
        if arr.shape[0] < n_v:
            raise ValueError("%s：dim0=%d 小于 value 段长度 %d"
                             % (e.infinilm, arr.shape[0], n_v))
        out = np.asarray(arr, dtype=arr.dtype)
        return np.concatenate([out[:-n_v], fn(out[-n_v:], n_k, v_per_k, hd)], axis=0)
    return fn(np.asarray(arr, dtype=arr.dtype), n_k, v_per_k, hd)


# ---------------------------------------------------------------------------
# 其它变换
# ---------------------------------------------------------------------------

def alog_from_ssm_a(a: np.ndarray) -> np.ndarray:
    """A_log = log(-ssm_a)。

    GGUF 存的是 `-exp(A_log)`（conversion/qwen.py:388），而 InfiniCore
    fused_gated_delta_net_gating 自己算 -expf(A_log) ⇒ 它要 HF 约定。
    实测本文件 48 个值全为负；出现非负值说明源不是这个约定，必须炸出来而不是静默 NaN。
    """
    a = np.asarray(a, dtype=np.float32)
    if not np.all(a < 0):
        raise ValueError("ssm_a 存在非负值（min=%g），无法取 log(-x)；"
                         "请核对 conversion/qwen.py 的 A_log 约定" % float(a.min()))
    return np.log(-a)


def gguf_meta(reader, suffix: str):
    """元数据键带架构前缀（qwen35.*），允许传短名；contents() 对单元素返回标量，统一成列表。"""
    for key in ("qwen35.%s" % suffix, "general.%s" % suffix, suffix):
        if key in reader.fields:
            v = reader.fields[key].contents()
            return v if isinstance(v, (list, tuple, np.ndarray)) else [v]
    raise KeyError("GGUF 元数据缺少：%s（qwen35./general. 前缀均未命中）" % suffix)


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 的位模式（uint16）。

    只做 round-to-nearest-even 的截断，与 torch 的 `.to(torch.bfloat16)` 等价；
    打包器实际写盘用 torch 做 cast，这里留给校验路径把 BF16 张量按位比回来。
    全程 uint32 而不升 uint64：进位只丢失 bit32，不影响要取的 bit16..31，
    内存却减半（lm_head 这类亿级张量上 uint64 会直接 OOM）。
    """
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u + bias) >> np.uint32(16)).astype(np.uint16)
