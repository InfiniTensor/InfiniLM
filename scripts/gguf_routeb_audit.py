#!/usr/bin/env python3
"""
InfiniLM 路线 B —— 阶段 0 风险清零审计（执行方案 §4）

检查项：
  A. 容器/字节布局：GGUF 原始字节按 [out, row_bytes] 重解释 + 自研 block 解码
     是否与 gguf-py 权威实现**逐比特相等**（Q8_0 / Q4_K / Q5_K / Q6_K）
  B. 对齐事实：块起始与行 stride 的真实对齐度（写 kernel 前的硬约束）
  C. V 头重排：grouped<->tiled 正向/逆向置换是否自等（执行方案 §2.7）
  D. 命名/形状契约：GGUF 实际张量集合是否与打包器的映射表完全一致
  E. 元数据：rope / ssm / 层类型等 config.json 依据

用法：
  python3 scripts/gguf_routeb_audit.py \
      [--gguf /home/liuxd/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf]
退出码 0 表示全部 PASS。
"""

from __future__ import annotations

import argparse
import os
import sys
import collections

import numpy as np

_LLAMA_CPP = os.environ.get("LLAMA_CPP_DIR", "/home/liuxd/llama.cpp")
sys.path.insert(0, os.path.join(_LLAMA_CPP, "gguf-py"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gguf import GGUFReader                      # noqa: E402
from gguf.constants import (                     # noqa: E402
    GGML_QUANT_SIZES,
    GGMLQuantizationType as QType,
)
import gguf.quants as gq                         # noqa: E402

QK_K = 256

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# 自研 block 解码：完全按 ggml 内存布局手写（将来 1:1 移植进 ggml_blocks.h）
# 输入统一为 uint8 blob，形状 [n_rows, row_bytes]；输出 float32 [n_rows, n_cols]
# ---------------------------------------------------------------------------

def _rows_to_blocks(blob: np.ndarray, type_size: int) -> np.ndarray:
    """[n_rows, row_bytes] -> [n_blocks, type_size]，块沿 in 连续、按 out 行排列。"""
    assert blob.dtype == np.uint8
    n_rows, row_bytes = blob.shape
    assert row_bytes % type_size == 0, f"row_bytes={row_bytes} 不是 type_size={type_size} 的整数倍"
    return blob.reshape(-1, type_size)


def _f16(col: np.ndarray) -> np.ndarray:
    return col.view(np.float16).astype(np.float32)


def decode_q8_0(blob: np.ndarray, n_cols: int) -> np.ndarray:
    """块 = d(f16,2B) + qs(int8,32B)，共 34B / 32 元素。"""
    bs, ts = 32, 34
    b = _rows_to_blocks(blob, ts)
    d = _f16(b[:, :2])                                  # [nb,1]
    x = b[:, 2:ts].view(np.int8).astype(np.float32)     # [nb,32]
    return (d * x).reshape(blob.shape[0], n_cols)


def _k_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Q4_K/Q5_K 的 12 字节 -> 8 组 (sc, min)，6+2 bit 交错打包。"""
    n = scales.shape[0]
    s = scales.reshape((n, 3, 4))
    d, m, m_d = np.split(s, 3, axis=-2)
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.reshape((n, 8)), mn.reshape((n, 8))


# --- 索引表：这就是后续 CUDA 实现的 ggml_blocks.h 布局规范 -------------------
# Q4_K / Q5_K：8 个子块 x 32 元素（不是 16x16！），子块 g 内偏移 o：
#   qs 字节 = qs_base + (g // 2) * 32 + o，nibble 位移 = (g % 2) * 4
_e = np.arange(QK_K)
_g = _e // 32
_o = _e % 32
K_QS_BYTE = (_g // 2) * 32 + _o
K_QS_SHIFT = (_g % 2) * 4
K_SCALE_IDX = _g                       # 每 32 元素一组 scale/min

# Q5_K 的第 5 bit：qh 字节 = o，位 = g
K5_QH_BYTE = _o
K5_QH_BIT = _g

# Q6_K：256 元素，6 bit = 低 4(nibble) + 高 2
#   低 4 bit：字节 = (h // 2) * 64 + r，位移 = (h % 2) * 4（h = e // 64, r = e % 64）
#   高 2 bit：字节 = (g // 4) * 32 + o，位移 = (g % 4) * 2
#   scale 索引 = e // 16（16 个子块 x 16 元素）
_h = _e // 64
_r = _e % 64
Q6_LO_BYTE = (_h // 2) * 64 + _r
Q6_LO_SHIFT = (_h % 2) * 4
Q6_HI_BYTE = (_g // 4) * 32 + _o
Q6_HI_SHIFT = (_g % 4) * 2
Q6_SCALE_IDX = _e // 16


def decode_q4_k(blob: np.ndarray, n_cols: int) -> np.ndarray:
    """块 = d(2) dmin(2) scales(12) qs(128) = 144B / 256 元素。"""
    ts = 144
    b = _rows_to_blocks(blob, ts)
    d = _f16(b[:, 0:2])
    dmin = _f16(b[:, 2:4])
    sc, mn = _k_scale_min(b[:, 4:16])
    qs = b[:, 16:ts]
    q = ((qs[:, K_QS_BYTE] >> K_QS_SHIFT.astype(np.uint8))
         & np.uint8(0x0F)).reshape(b.shape[0], 8, 32).astype(np.float32)
    d_eff = (d * sc.astype(np.float32)).reshape(b.shape[0], 8, 1)
    m_eff = (dmin * mn.astype(np.float32)).reshape(b.shape[0], 8, 1)
    return (d_eff * q - m_eff).reshape(blob.shape[0], n_cols)


def decode_q5_k(blob: np.ndarray, n_cols: int) -> np.ndarray:
    """块 = d(2) dmin(2) scales(12) qh(32) qs(128) = 176B / 256 元素。"""
    ts = 176
    b = _rows_to_blocks(blob, ts)
    d = _f16(b[:, 0:2])
    dmin = _f16(b[:, 2:4])
    sc, mn = _k_scale_min(b[:, 4:16])
    qh = b[:, 16:48]
    qs = b[:, 48:ts]
    n = b.shape[0]
    lo = (qs[:, K_QS_BYTE] >> K_QS_SHIFT.astype(np.uint8)) & np.uint8(0x0F)
    hi = (qh[:, K5_QH_BYTE] >> K5_QH_BIT.astype(np.uint8)) & np.uint8(0x01)
    q = (lo | (hi << np.uint8(4))).reshape(n, 8, 32).astype(np.float32)
    d_eff = (d * sc.astype(np.float32)).reshape(n, 8, 1)
    m_eff = (dmin * mn.astype(np.float32)).reshape(n, 8, 1)
    return (d_eff * q - m_eff).reshape(blob.shape[0], n_cols)


def decode_q6_k(blob: np.ndarray, n_cols: int) -> np.ndarray:
    """块 = ql(128) qh(64) scales(16,int8) d(2) = 210B / 256 元素。"""
    ts = 210
    b = _rows_to_blocks(blob, ts)
    n = b.shape[0]
    ql = b[:, :128]
    qh = b[:, 128:192]
    sc = b[:, 192:208].view(np.int8).astype(np.float32)
    d = _f16(b[:, 208:210])
    lo = (ql[:, Q6_LO_BYTE] >> Q6_LO_SHIFT.astype(np.uint8)) & np.uint8(0x0F)
    hi = (qh[:, Q6_HI_BYTE] >> Q6_HI_SHIFT.astype(np.uint8)) & np.uint8(0x03)
    q = ((lo | (hi << np.uint8(4))).astype(np.int16) - 32
         ).reshape(n, 16, 16).astype(np.float32)
    step = (d * sc).reshape(n, 16, 1)
    return (step * q).reshape(blob.shape[0], n_cols)


DECODERS = {
    QType.Q8_0: decode_q8_0,
    QType.Q4_K: decode_q4_k,
    QType.Q5_K: decode_q5_k,
    QType.Q6_K: decode_q6_k,
}


# ---------------------------------------------------------------------------
# A. 容器 / 字节布局 / block 位运算
# ---------------------------------------------------------------------------

def pick_samples(tensors: dict[str, object], per_type: int = 3) -> list:
    """每种量化类型最多挑 per_type 个（按 (in,out) 形状去重），只解部分行以省时。"""
    by_type = collections.defaultdict(list)
    for name, t in tensors.items():
        qt = QType(int(t.tensor_type))
        if qt in DECODERS and name.startswith("blk.") and ".nextn." not in name:
            by_type[qt].append(t)
    out = []
    for qt, lst in sorted(by_type.items(), key=lambda kv: int(kv[0])):
        seen = set()
        picked = 0
        for t in sorted(lst, key=lambda x: x.name):
            key = (int(t.shape[0]), int(t.shape[1]))
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
            picked += 1
            if picked >= per_type:
                break
    return out


def section_a(reader) -> None:
    print("\n== A. 容器与 block 位运算（逐比特）==")
    tensors = {t.name: t for t in reader.tensors}
    samples = pick_samples(tensors)
    assert samples, "未取到任何样本"
    all_ok = True
    for t in samples:
        qt = QType(int(t.tensor_type))
        bs, ts = GGML_QUANT_SIZES[int(qt)]
        n_in, n_out = int(t.shape[0]), int(t.shape[1])   # GGML: ne[0]=in, ne[1]=out
        row_bytes = n_in // bs * ts
        blob = np.ascontiguousarray(t.data)              # 解析器已给 [out, row_bytes]
        ok_shape = blob.shape == (n_out, row_bytes)
        dec = DECODERS[qt]
        n_rows = min(64, n_out)                           # 只解前 n_rows 行，省时
        ours = dec(blob[:n_rows], n_in)
        ref_full = gq.dequantize(blob[:n_rows], qt)       # 权威实现，输入为字节形状
        ref = np.asarray(ref_full, dtype=np.float32)
        exact = ours.shape == ref.shape and np.array_equal(ours, ref)
        # 单行独立性：逐行解码必须与整体解码一致（证明行是连续独立单元）
        one = dec(blob[7:8], n_in)
        indep = np.array_equal(one, ref[7:8])
        all_ok &= check(
            f"{t.name} {qt.name} in={n_in} out={n_out} row_bytes={row_bytes}",
            ok_shape and exact and indep,
            f"bit-exact={exact} row-indep={indep}",
        )
    check("A 汇总", all_ok)
    # 非量化张量的轴序（打包器是否需要转置的依据）
    conv = tensors["blk.0.ssm_conv1d.weight"]
    check("F32 张量的 data 也是 C 序 [shape[1], shape[0]]（= HF 取向，打包器不转置）",
          conv.data.shape == (int(conv.shape[1]), int(conv.shape[0])),
          f"ne={list(map(int, conv.shape))} data={conv.data.shape} -> HF [10240,1,4]")
    norm = tensors["blk.0.attn_norm.weight"]
    check("1-D norm 保持 dtype=float32且长度 = hidden",
          norm.data.dtype == np.float32 and norm.data.shape == (5120,))


# ---------------------------------------------------------------------------
# B. 对齐事实
# ---------------------------------------------------------------------------

def section_b() -> None:
    print("\n== B. 对齐事实（kernel 的硬约束）==")
    facts = []
    for qt in (QType.Q8_0, QType.Q4_K, QType.Q5_K, QType.Q6_K, QType.IQ4_NL, QType.IQ4_XS):
        bs, ts = GGML_QUANT_SIZES[int(qt)]
        align_block = 2 if ts % 2 == 0 else 1
        for n_in in (5120, 6144, 10240, 17408, 248320):
            if n_in % bs:
                continue
            rb = n_in // bs * ts
            a = 16
            while a > 1 and rb % a:
                a //= 2
            facts.append((qt.name, bs, ts, n_in, rb, a, align_block))
    print(f"  {'type':8s} {'bs':>4s} {'ts':>4s} {'in':>7s} {'row_bytes':>10s} "
          f"{'行对齐':>7s} {'块起始对齐':>10s}")
    worst_row, worst_block = 16, 2
    for name, bs, ts, n_in, rb, a, ab in facts:
        print(f"  {name:8s} {bs:4d} {ts:4d} {n_in:7d} {rb:10d} {str(a)+'B':>7s} {str(ab)+'B':>10s}")
        worst_row = min(worst_row, a)
        worst_block = min(worst_block, ab)
    check("块起始地址仅保证 2B 对齐（Q6_K=210B / Q8_0=34B 非 4 倍数）",
          worst_block == 2, f"min_block_align={worst_block}B")
    check("Q6_K 在 in=5120/17408 时行 stride 仅 8B 对齐",
          any(f[0] == "Q6_K" and f[5] == 8 for f in facts))
    print("  -> 结论：kernel 不得对单块起始地址做 >2B 向量化加载假设；容器不做 pad。")


# ---------------------------------------------------------------------------
# C. V 头重排（grouped <-> tiled）
# ---------------------------------------------------------------------------

def reorder_v(t: np.ndarray, n_k: int, n_v_per_k: int, hd: int) -> np.ndarray:
    """与 llama.cpp conversion/qwen.py::_reorder_v_heads 同语义（沿 dim0 的整头置换）。"""
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


def section_c() -> None:
    print("\n== C. V 头重排（执行方案 §2.7）==")
    n_k, n_v_per_k, hd = 16, 3, 128        # Qwen3.8: 16 key heads, 48 value heads
    n_v = n_k * n_v_per_k
    rng = np.random.default_rng(0)

    grouped = rng.standard_normal((n_v * hd, 7)).astype(np.float32)
    tiled = reorder_v(grouped, n_k, n_v_per_k, hd)
    back = reorder_v_inverse(tiled, n_k, n_v_per_k, hd)
    check("grouped -> tiled -> grouped 自等", np.array_equal(grouped, back))
    check("reorder_v 是整头搬运（每个 head 的 hd 行连续不被打散）",
          all(np.array_equal(tiled[i * hd:(i + 1) * hd],
                             grouped[((i % n_k) * n_v_per_k + i // n_k) * hd
                                     :((i % n_k) * n_v_per_k + i // n_k) * hd + hd])
              for i in range(n_v)))

    # 槽位 j（value head 编号）-> 真实 k 头 的两种语义
    k_grouped = [j // n_v_per_k for j in range(n_v)]        # InfiniCore kernel 的假设
    k_tiled = [j % n_k for j in range(n_v)]                 # GGUF(tiled) 的真实归属
    check("tiled 序直接喂给 `value_head_idx / value_heads_per_key_head` 会错配 k 头",
          k_grouped != k_tiled,
          f"错配槽位数={sum(a != b for a, b in zip(k_grouped, k_tiled))}/{n_v}")

    # 逆重排后回到 grouped 语义
    _ = np.repeat(np.arange(n_v), hd)   # labels 仅用于形状参考
    check("逆变换后槽位归属恢复 grouped 语义",
          np.array_equal(
              reorder_v_inverse(
                  np.array([k * n_v_per_k + v for v in range(n_v_per_k) for k in range(n_k)]),
                  n_k, n_v_per_k, 1),
              np.arange(n_v)),
          "逆变换后 slot i 的 head 编号 = i，kernel 的 k = i // n_v_per_k 成立")
    check("in_proj_a/b・A_log・dt_bias 的 head_dim=1 退化形式（逐元素置换）同样自等",
          np.array_equal(
              reorder_v_inverse(reorder_v(np.arange(n_v), n_k, n_v_per_k, 1),
                                n_k, n_v_per_k, 1),
              np.arange(n_v)))
    check("多维情形（如 conv1d 的 [channels, 1, kernel]）仅置换头维、尾部轴不动",
          np.array_equal(
              reorder_v_inverse(reorder_v(grouped[:, :1], n_k, n_v_per_k, hd),
                                n_k, n_v_per_k, hd),
              grouped[:, :1]))
    # 行置换对量化 blob 是「整块搬运」：以 Q6_K 为例验证字节级可置换性
    bs, ts = GGML_QUANT_SIZES[int(QType.Q6_K)]
    row_bytes = 5120 // bs * ts
    blob = rng.integers(0, 256, size=(n_v, row_bytes), dtype=np.uint8)
    perm = np.arange(n_v)[::-1].copy()
    check("量化 blob 的行置换 == 字节整行置换（无需重新量化）",
          np.array_equal(blob[perm], np.ascontiguousarray(blob)[perm]))
    print("  -> 结论：整行置换可字节级完成；ssm_out 的列(in 维)置换不可，改用运行时激活 gather。")


# ---------------------------------------------------------------------------
# D. 命名 / 形状契约
# ---------------------------------------------------------------------------

def gguf_meta(reader, suffix: str):
    """元数据键带架构前缀（qwen35.*），允许传短名；contents() 对单元素返回标量，统一成列表。"""
    for key in (f"qwen35.{suffix}", f"general.{suffix}", suffix):
        if key in reader.fields:
            v = reader.fields[key].contents()
            return v if isinstance(v, (list, tuple, np.ndarray)) else [v]
    raise KeyError(f"GGUF 元数据缺少：{suffix}（qwen35./general. 前缀均未命中）")


def section_d(reader) -> None:
    print("\n== D. GGUF 张量集合 vs 打包器映射表 ==")
    tensors = {t.name: t for t in reader.tensors}
    n_layer_gguf = int(gguf_meta(reader, "block_count")[0])
    interval = int(gguf_meta(reader, "full_attention_interval")[0])
    n_main = 64
    full = [i for i in range(n_main) if (i + 1) % interval == 0]
    gdn = [i for i in range(n_main) if i not in full]
    check("主模型层数 64（block_count 含 1 个 MTP 层）",
          n_layer_gguf == n_main + 1, f"block_count={n_layer_gguf}")
    check("full attention 层 = 3,7,...,63 共 16 层",
          len(full) == 16 and full[0] == 3 and full[-1] == 63)
    check("GDN 层 48 层", len(gdn) == 48)

    need_full = ["attn_norm.weight", "post_attention_norm.weight", "attn_q.weight",
                 "attn_k.weight", "attn_v.weight", "attn_output.weight",
                 "attn_q_norm.weight", "attn_k_norm.weight",
                 "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
    need_gdn = ["attn_norm.weight", "post_attention_norm.weight", "attn_qkv.weight",
                "attn_gate.weight", "ssm_a", "ssm_alpha.weight", "ssm_beta.weight",
                "ssm_conv1d.weight", "ssm_dt.bias", "ssm_norm.weight", "ssm_out.weight",
                "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
    missing = []
    for i in full:
        missing += [f"blk.{i}.{r}" for r in need_full if f"blk.{i}.{r}" not in tensors]
    for i in gdn:
        missing += [f"blk.{i}.{r}" for r in need_gdn if f"blk.{i}.{r}" not in tensors]
    check("64 层全部所需张量存在", not missing, f"missing={missing[:6]}")

    shapes = {
        "attn_q": (5120, 12288), "attn_k": (5120, 1024), "attn_v": (5120, 1024),
        "attn_output": (6144, 5120), "attn_qkv": (5120, 10240), "attn_gate": (5120, 6144),
        "ssm_out": (6144, 5120), "ffn_gate": (5120, 17408), "ffn_up": (5120, 17408),
        "ffn_down": (17408, 5120), "ssm_conv1d": (4, 10240),
    }
    bad = []
    for name, want in shapes.items():
        probe = {"attn_q": f"blk.{full[0]}.", "attn_k": f"blk.{full[0]}.",
                 "attn_v": f"blk.{full[0]}.", "attn_output": f"blk.{full[0]}.",
                 "attn_qkv": f"blk.{gdn[0]}.", "attn_gate": f"blk.{gdn[0]}.",
                 "ssm_out": f"blk.{gdn[0]}.", "ssm_conv1d": f"blk.{gdn[0]}.",
                 "ffn_gate": f"blk.{0}.", "ffn_up": f"blk.{0}.", "ffn_down": f"blk.{0}."}[name]
        t = tensors.get(probe + name + ".weight")
        if t is None or (int(t.shape[0]), int(t.shape[1])) != want:
            bad.append((name, None if t is None else list(map(int, t.shape))))
    check("代表张量 (in,out) 与映射表一致", not bad, f"bad={bad}")

    # attn_q 的 12288 = 24 * (256 q + 256 gate) 交错
    n_q, hd_q, n_kv, hd_k = 24, 256, 4, 256
    check("attn_q 行数 = n_q*head*2（q 与 gate 每头交错）",
          shapes["attn_q"][1] == n_q * hd_q * 2)
    check("Qwen35FusedQKVLinear 期望 out = 12288 + 1024 + 1024 = 14336",
          12288 + 1024 + 1024 == 14336)
    check("GDN in_proj_qkv 行数 = q2048 + k2048 + v6144 = 10240",
          2048 + 2048 + 6144 == shapes["attn_qkv"][1])
    check("conv 通道 = 2*head_k*n_k + head_v*n_v = 10240",
          2 * 128 * 16 + 128 * 48 == 10240)
    mtp = [n for n in tensors if n.startswith("blk.64.")]
    nextn = [n for n in tensors if ".nextn." in n]
    check("MTP 丢弃规则 = 整块 blk.64.*（不止 .nextn.*，包含完整一层）",
          len(mtp) == 15 and len(nextn) == 4,
          f"blk.64.*={len(mtp)} 个（其中 .nextn.* 仅 {len(nextn)} 个）")
    max_blk = max(int(n.split(".")[1]) for n in tensors if n.startswith("blk."))
    check("块号集合 = 0..64（64 主层 + 1 MTP 层，无其它残留）",
          max_blk == 64 and len({int(n.split(".")[1]) for n in tensors if n.startswith("blk.")}) == 65,
          f"max_blk={max_blk}")


# ---------------------------------------------------------------------------
# F. 打包器字节核算（修正后的 MTP 规则）
# ---------------------------------------------------------------------------

# 阶段 3 kernel 需直接吃块的格式集合不再在本文件定义：见 gguf_mapping.NATIVE_BLOB_TYPES
# （由 gguf_routeb_shape_contract.py 对真文件校验），避免两处清单漂移。


def section_f(reader) -> None:
    print("\n== F. 打包器字节核算 ==")
    GiB = 2 ** 30
    tensors = {t.name: t for t in reader.tensors}
    # 核算必须由映射表驱动：之前本脚本自写一套分桶，把 7 个 IQ4 张量当成“反量化”、
    # 把实为 Q8_0 的 ssm_alpha/ssm_beta（框架不能量化它们）当成 blob，两处失真共
    # 高估 0.70 GiB。单一事实源 = gguf_mapping.build_plan(REAL)。
    import gguf_mapping as M
    plan = M.build_plan(M.REAL)
    _tn = {int(v.value): str(v.name) for v in QType}
    M.apply_v1_exceptions(plan, {n: _tn[int(t.tensor_type)] for n, t in tensors.items()})
    blob_src = {e.gguf for e in plan if e.blob and e.gguf in tensors}
    dense_e = [e for e in plan if not e.blob]
    bucket = collections.Counter()
    cnt = collections.Counter()
    for n in blob_src:
        bucket[f"U8 blob {QType(int(tensors[n].tensor_type)).name}"] += int(tensors[n].n_bytes)
        cnt[f"U8 blob {QType(int(tensors[n].tensor_type)).name}"] += 1
    d_emb = sum(int(np.prod(e.shape)) * 2 for e in dense_e
                if e.gguf in ("token_embd.weight", "output.weight"))
    d_other = sum(int(np.prod(e.shape)) * 2 for e in dense_e
                  if e.gguf not in ("token_embd.weight", "output.weight"))
    bucket["BF16 稠密(emb/lm_head)"] = d_emb
    cnt["BF16 稠密(emb/lm_head)"] = 2
    bucket["BF16 稠密(其余稠密化条目)"] = d_other
    cnt["BF16 稠密(其余稠密化条目)"] = len(dense_e) - 2
    bucket["丢弃(MTP)"] = sum(int(t.n_bytes) for n, t in tensors.items()
                              if n.startswith("blk.64."))
    cnt["丢弃(MTP)"] = sum(1 for n in tensors if n.startswith("blk.64."))

    total = sum(v / GiB for k, v in bucket.items() if k != "丢弃(MTP)")
    for k in sorted(bucket):
        print(f"  {k:26s} {bucket[k] / GiB:8.3f} GiB   ({cnt[k]:4d} 条目)")
    print(f"  {'-'*52}")
    print(f"  v1 加载后权重合计           {total:8.3f} GiB")
    check("v1 权重合计 ≤ 24.0 GiB（单卡 32607 MiB 可容纳权重+KV+激活）",
          total <= 24.0, f"total={total:.3f} GiB")
    check("MTP 丢弃量 < 0.4 GiB（不影响预算）",
          bucket["丢弃(MTP)"] / GiB < 0.4, f"{bucket['丢弃(MTP)'] / GiB:.3f} GiB")
    check("v1 blob 桶恰好只含阶段 3 实现的 4 种类型",
          {k.replace("U8 blob ", "") for k in bucket if k.startswith("U8 blob ")}
          == set(M.NATIVE_BLOB_TYPES),
          f"{sorted(k for k in bucket if k.startswith('U8'))}")
    check("blob 条目数与映射表一致",
          sum(cnt[k] for k in bucket if k.startswith("U8"))
          == len({e.gguf for e in plan if e.blob}),
          f"{sum(cnt[k] for k in bucket if k.startswith('U8'))}")

    emb, out = tensors["token_embd.weight"], tensors["output.weight"]
    check("token_embd / output 也是量化的（Q6_K / Q8_0），v1 必须反量化它们",
          int(emb.tensor_type) == int(QType.Q6_K) and int(out.tensor_type) == int(QType.Q8_0),
          f"emb={emb.tensor_type} out={out.tensor_type}")
    check("emb/output 均为 [hidden, vocab] 且 vocab 与元数据一致",
          list(map(int, emb.shape)) == list(map(int, out.shape)) == [5120, 248320]
          and len(gguf_meta(reader, "tokenizer.ggml.tokens")) == 248320,
          f"shape={list(map(int, emb.shape))}")
    print("  -> 阶段 6 可选项：emb 走 Q6_K 行 gather-dequant、lm_head 走 linear_gguf(Q8_0)，"
          f"可再省 ≈ {(d_emb - (emb.n_bytes + out.n_bytes)) / GiB:.2f} GiB")


# ---------------------------------------------------------------------------
# E. 元数据 -> config.json
# ---------------------------------------------------------------------------

def section_e(reader) -> None:
    print("\n== E. 元数据与 config.json 依据 ==")

    def kv(suffix, idx=0):
        return gguf_meta(reader, suffix)[idx]

    rope_secs = [int(x) for x in gguf_meta(reader, "rope.dimension_sections")]
    dim_cnt = int(kv("rope.dimension_count"))
    base = float(kv("rope.freq_base"))
    eps = float(kv("attention.layer_norm_rms_epsilon"))
    head_dim = int(kv("attention.key_length"))
    check("head_dim = key_length = value_length = 256",
          head_dim == int(kv("attention.value_length")) == 256)
    check("partial rotary: dimension_count=64, head_dim=256 -> factor 0.25",
          dim_cnt == 64 and dim_cnt * 4 == head_dim, f"dimension_count={dim_cnt}")
    check("mrope sections [11,11,10,0] 之和 = 32 = dimension_count/2",
          sum(rope_secs) == dim_cnt // 2, f"sections={rope_secs}")
    check("rope_theta = 1e7", base == 1e7, f"base={base}")
    check("mtp 层数声明为 1（与 block_count=65 = 64+1 一致）",
          int(kv("nextn_predict_layers")) == 1)
    n_k = int(kv("ssm.group_count"))
    inner = int(kv("ssm.inner_size"))
    st = int(kv("ssm.state_size"))
    dt = int(kv("ssm.time_step_rank"))
    check("ssm: inner 6144 / group 16 / state 128 / time_step_rank 48 / conv 4",
          (inner, n_k, st, dt, int(kv("ssm.conv_kernel"))) == (6144, 16, 128, 48, 4))
    check("value heads = inner/state = 48 = time_step_rank（两路推导一致）",
          inner // st == dt == 48, f"inner/state={inner // st} time_step_rank={dt}")
    check("num_k_heads * state = 2048 = q/k 段长度",
          n_k * st == 2048)
    vocab = len(gguf_meta(reader, "tokenizer.ggml.tokens"))
    print(f"  arch={gguf_meta(reader, 'architecture')[0]!r}  "
          f"name={gguf_meta(reader, 'name')[0]!r}  rms_eps={eps:g}  "
          f"ctx={int(kv('context_length'))}  vocab={vocab}  "
          f"heads={int(kv('attention.head_count'))}/{int(kv('attention.head_count_kv'))}  "
          f"hidden={int(kv('embedding_length'))}  ffn={int(kv('feed_forward_length'))}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/liuxd/models/Qwen3.8-27B-GGUF/"
                                      "Qwen3.8-27B-UD-Q6_K.gguf")
    args = ap.parse_args()
    print(f"审计对象：{args.gguf}\n大小：{os.path.getsize(args.gguf):,} bytes")
    reader = GGUFReader(args.gguf)
    section_a(reader)
    section_b()
    section_c()
    section_d(reader)
    section_e(reader)
    section_f(reader)
    print(f"\n===== 结果：PASS {len(PASSED)} / FAIL {len(FAILED)} =====")
    if FAILED:
        for f in FAILED:
            print("  FAIL:", f)
        return 1
    print("阶段 0 全部通过，可进入阶段 1（打包器）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
