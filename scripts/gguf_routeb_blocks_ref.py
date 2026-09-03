#!/usr/bin/env python3
"""
InfiniLM 路线 B —— 阶段 3.1 验收：ggml_blocks.h 的 block 解码位精正确认

四方交叉，任何两方不一致都会炸出来：

  A. numpy reference（本文件）：照 llama.cpp `ggml/src/ggml-quants.c` 的
     `dequantize_row_q8_0/q4_K/q5_K/q6_K` 标量语义逐行翻过来，含
     `get_scale_min_k4` 的 6-bit 解包与**浮点结合顺序**（先 d*scale 再碰 quant）。
  B. gguf-py 的 numpy 实现（`gguf.quants.Q8_0/Q4_K/Q5_K/Q6_K.dequantize_blocks`）：
     它是阶段 4「单 block 级 max|Δ| == 0」的基准。它解包 scale 用的是
     reshape/split 另一条路径，与 A 相互独立 —— 两边逐位相同才说明 6-bit
     打包的解读没读歪。
  C. 被测对象 `InfiniCore/src/infiniop/ops/linear_gguf/ggml_blocks.h`，经
     `scripts/gguf_routeb_blocks_probe.cpp` 编出的 host driver 跑真数据。
  D. 同一个头经 `scripts/gguf_routeb_blocks_probe.cu` 编出的 CUDA driver：
     证明这个头确实设备无关（GPU 上编得过、跑得动、与 host 逐位相同），
     顺带验证 bf16 舍入 `float_to_bf16()` 与 torch 的 `.to(bfloat16)` 一致。

样本来自真实打包产物里 `*.weight_bytes` 的 block 字节，再加一批手造边界 block
（次正规 half、int8 scale = -128、scale/min 全 63、全 FF），因为 K-quant 的
scale 解包最容易在极值上翻车。随机 block 的 half 域被限制为有限值，好让判据
能要求 100% 逐位相同，而不是退化成"近似"。

用法：
  /usr/bin/python3 scripts/gguf_routeb_blocks_ref.py \
      [--model-path /home/liuxd/models/Qwen3.8-27B-GGUF-native-mini8] \
      [--blocks 20000] [--no-cuda] [--keep]
退出码 0 = 全部 PASS。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import struct
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LLAMA_CPP = os.environ.get("LLAMA_CPP_DIR", "/home/liuxd/llama.cpp")
_INFINICORE = os.environ.get("INFINICORE_DIR", "/home/liuxd/InfiniCore")
sys.path.insert(0, os.path.join(_LLAMA_CPP, "gguf-py"))

import gguf.quants as gq                                                # noqa: E402
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as Q  # noqa: E402

HEADER_DIR = os.path.join(_INFINICORE, "src", "infiniop", "ops", "linear_gguf")
PROBE_CPP = os.path.join(_HERE, "gguf_routeb_blocks_probe.cpp")
PROBE_CU = os.path.join(_HERE, "gguf_routeb_blocks_probe.cu")

TYPES = (8, 12, 13, 14)                    # 与 pack_report.json 的 blob_type_ids 一致
QK_K, QK8_0 = 256, 32
TYPE_SIZE = {t: GGML_QUANT_SIZES[Q(t)][1] for t in TYPES}
BLOCK_SIZE = {t: GGML_QUANT_SIZES[Q(t)][0] for t in TYPES}

_PASS = 0
_FAIL = 0
_SKIP = 0


def check(name, ok, detail=""):
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print("  PASS  %s" % name)
    else:
        _FAIL += 1
        print("  FAIL  %s%s" % (name, ("\n        %s" % detail) if detail else ""))
    return ok


def skip(name, why):
    global _SKIP
    _SKIP += 1
    print("  SKIP  %s（%s）" % (name, why))


# ------------------------------------------------------- A. numpy 参考实现
def _u16_le(col0, col1):
    return col0.astype(np.uint32) | (col1.astype(np.uint32) << np.uint32(8))


def half_to_float(bits):
    """IEEE binary16 -> float32，等价于 ggml FP16_TO_FP32 / __half2float。"""
    bits = np.asarray(bits, np.uint32)
    sign = (bits >> np.uint32(15)) << np.uint32(31)
    exp = (bits >> np.uint32(10)) & np.uint32(0x1F)
    mant = bits & np.uint32(0x3FF)
    out = np.zeros(bits.shape, np.uint32)
    zneg = (exp == 0) & (mant == 0)          # ±0：符号位必须留住，否则 -0.0 被写成正零
    out[zneg] = sign[zneg]
    norm = (exp != 0) & (exp != 31)
    out[norm] = sign[norm] | ((exp[norm] + np.uint32(112)) << np.uint32(23)) | (
        mant[norm] << np.uint32(13))
    special = exp == 31
    out[special] = sign[special] | np.uint32(0x7F800000) | (mant[special] << np.uint32(13))
    sub = (exp == 0) & (mant != 0)
    if sub.any():
        m = mant[sub].astype(np.int64)
        e = np.full(m.shape, -14, np.int64)
        for _ in range(11):
            need = (m & 0x400) == 0
            if not need.any():
                break
            m[need] <<= 1
            e[need] -= 1
        out[sub] = (sign[sub].astype(np.int64) | ((e + 127) << 23)
                    | ((m & 0x3FF) << 13)).astype(np.uint32)
    return out.view(np.float32)


def float_to_bf16_bits(f):
    """binary32 -> bf16 位模式，round-to-nearest-even，与头里那份同语义。"""
    b = np.asarray(f, np.float32).view(np.uint32).astype(np.int64)
    exp = (b >> 23) & 0xFF
    nan = (exp == 0xFF) & ((b & 0x7FFFFF) != 0)
    bias = 0x7FFF + ((b >> 16) & 1)
    out = ((b + bias) >> 16).astype(np.uint32)
    out[nan] = ((b[nan] >> 16) | 0x0040).astype(np.uint32)
    return out.astype(np.uint16)


def get_scale_min_k4(scales):
    """q4_K / q5_K：12 字节 -> 8 组 (scale, min)，照抄 ggml-quants.c 的分支。"""
    nb = scales.shape[0]
    d = np.empty((nb, 8), np.uint8)
    m = np.empty((nb, 8), np.uint8)
    for j in range(8):
        if j < 4:
            d[:, j] = scales[:, j] & 63
            m[:, j] = scales[:, j + 4] & 63
        else:
            d[:, j] = (scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4)
            m[:, j] = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    return d, m


def ref_q8_0(blk):
    nb = blk.shape[0]
    d = half_to_float(_u16_le(blk[:, 0], blk[:, 1])).reshape(nb, 1)
    q = blk[:, 2:34].view(np.int8).astype(np.float32)
    return q * d                                        # C: qs[j] * d


def ref_q4_K(blk):
    nb = blk.shape[0]
    d = half_to_float(_u16_le(blk[:, 0], blk[:, 1]))
    dmin = half_to_float(_u16_le(blk[:, 2], blk[:, 3]))
    sc, m = get_scale_min_k4(blk[:, 4:16])
    d_eff = (d[:, None] * sc.astype(np.float32)).reshape(nb, 8, 1)
    m_eff = (dmin[:, None] * m.astype(np.float32)).reshape(nb, 8, 1)
    qs = blk[:, 16:144].reshape(nb, 4, 32)
    q = np.stack([qs & 0xF, qs >> 4], axis=2).reshape(nb, 8, 32).astype(np.float32)
    return (d_eff * q - m_eff).reshape(nb, QK_K)


def ref_q5_K(blk):
    nb = blk.shape[0]
    d = half_to_float(_u16_le(blk[:, 0], blk[:, 1]))
    dmin = half_to_float(_u16_le(blk[:, 2], blk[:, 3]))
    sc, m = get_scale_min_k4(blk[:, 4:16])
    d_eff = (d[:, None] * sc.astype(np.float32)).reshape(nb, 8, 1)
    m_eff = (dmin[:, None] * m.astype(np.float32)).reshape(nb, 8, 1)
    qs = blk[:, 48:176].reshape(nb, 4, 32)
    qh = blk[:, 16:48][:, None, :]
    lo_shift = (2 * np.arange(4)).reshape(4, 1)         # u1 = 1 << 2g
    hi_shift = lo_shift + 1                             # u2 = 2 << 2g
    lo = (qs & 0xF) | (((qh >> lo_shift) & 1) << 4).astype(np.uint8)
    hi = (qs >> 4) | (((qh >> hi_shift) & 1) << 4).astype(np.uint8)
    q = np.stack([lo, hi], axis=2).reshape(nb, 8, 32).astype(np.float32)
    return (d_eff * q - m_eff).reshape(nb, QK_K)


def ref_q6_K(blk):
    nb = blk.shape[0]
    d = half_to_float(_u16_le(blk[:, 208], blk[:, 209]))
    sc = blk[:, 192:208].view(np.int8).astype(np.float32)
    d_eff = d[:, None] * sc                             # (nb,16) 先 d*sc，同 C 结合顺序
    out = np.empty((nb, QK_K), np.float32)
    l = np.arange(32)
    isidx = l // 16
    for c in (0, 1):
        ql = blk[:, 64 * c:64 * c + 64]
        qh = blk[:, 128 + 32 * c:128 + 32 * c + 32]
        base = 128 * c
        q1 = ((ql[:, 0:32] & 0xF) | (((qh >> 0) & 3) << 4)).astype(np.int32) - 32
        q2 = ((ql[:, 32:64] & 0xF) | (((qh >> 2) & 3) << 4)).astype(np.int32) - 32
        q3 = ((ql[:, 0:32] >> 4) | (((qh >> 4) & 3) << 4)).astype(np.int32) - 32
        q4 = ((ql[:, 32:64] >> 4) | ((qh >> 6) << 4)).astype(np.int32) - 32
        for part, (q, off) in enumerate(((q1, 0), (q2, 32), (q3, 64), (q4, 96))):
            # C 里每处理一个 128 元素段就 `sc += 8`，所以段 1 的 scale 下标整体偏移 8
            s = d_eff[:, 8 * c + isidx + 2 * part]
            out[:, base + off:base + off + 32] = s * q.astype(np.float32)
    return out


REF = {8: ref_q8_0, 12: ref_q4_K, 13: ref_q5_K, 14: ref_q6_K}


def gguf_py_dequant(t, blk):
    return getattr(gq, Q(t).name).dequantize_blocks(np.ascontiguousarray(blk))


def check_half_decode():
    """参考实现自己的回归护袋：全部 65536 个 half 位模式与 numpy 硬件转换逐位相同。

    次正规 / 0 / inf 都在这 65536 个里，负数那一半特别重要（曾经把符号位
    当成 bit16 丢掉了，只会让带负 d 的 Q6_K block 整批错）。
    NaN 只要求“也是 NaN”，不比 payload。
    """
    h = np.arange(65536, dtype=np.uint16)
    truth = h.view(np.float16).astype(np.float32)
    mine = half_to_float(h.astype(np.uint32))
    finite = np.isfinite(truth)
    ok = np.array_equal(mine[finite].view(np.uint32), truth[finite].view(np.uint32))
    n_nan = int(np.isnan(truth).sum())
    ok_nan = bool(np.array_equal(np.isnan(mine), np.isnan(truth))
                  and np.array_equal(np.isinf(mine) & (mine > 0), np.isinf(truth) & (truth > 0)))
    neq = np.flatnonzero(mine.view(np.uint32) != truth.view(np.uint32))
    check("numpy 参考的 half_to_float：有限值逐位相同（%d 个）+ NaN 仍为 NaN（%d 个）"
          % (int(finite.sum()), n_nan), ok and ok_nan,
          "不同 %d 个，首个 0x%04X：%s vs %s" % (neq.size, int(h[neq[0]]) if neq.size else 0,
                                                 float(mine[neq[0]]) if neq.size else 0,
                                                 float(truth[neq[0]]) if neq.size else 0))


# ------------------------------------------------- 差异度量（要求逐位相同）
def bitwise_diff(a, b):
    """返回 (非有限值个数, 逐位不同的元素数, max|Δ|, 首个差异描述)。"""
    fa, fb = np.asarray(a, np.float32), np.asarray(b, np.float32)
    bad = ~np.isfinite(fa) | ~np.isfinite(fb)
    n_bad = int(bad.sum())
    ok_mask = ~bad
    ua = fa[ok_mask].view(np.uint32)
    ub = fb[ok_mask].view(np.uint32)
    neq = ua != ub
    n_diff = int(neq.sum())
    maxabs = float(np.abs(fa[ok_mask] - fb[ok_mask]).max()) if ok_mask.any() else 0.0
    first = ""
    if n_diff:
        i = int(np.flatnonzero(neq)[0])
        first = ("第 %d 个非有限值以外的元素 a=%s(0x%08X) b=%s(0x%08X)"
                 % (i, float(ua[i]), ua[i], float(ub[i]), ub[i]))
    elif n_bad:
        i = int(np.flatnonzero(bad)[0])
        first = "非有限值 a=%s b=%s @flat %d" % (fa.reshape(-1)[i], fb.reshape(-1)[i], i)
    return n_bad, n_diff, maxabs, first


# ---------------------------------------------------------- 产物字节取样
class Artifact:
    """打包产物的 blob 张量字节入口。

    两个代表产物的类型表键形态不同：mini8 表键 = 张量名（带前缀 + .weight_bytes）；
    全量表键 = `layers.0...in_proj_q.weight`（不带前缀、不带 .weight_bytes，而
    key_prefix 又是 None ⇒ 与 index 张量名零交集）。所以既不能拿表键直接当张量名，
    也不能只剔一个前缀：先只剔尾缀归一，再要求“全等或唯一后缀匹配”，
    匹配不唯一 / 找不到都是打包回归，直接报错而不是猜。
    """

    def __init__(self, path):
        self.path = path
        cfg = json.load(open(os.path.join(path, "config.json")))
        qc = cfg["quantization_config"]
        table = qc["ggml_types"]
        self.prefix = qc.get("key_prefix") or ""
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.shards = {}
        for name in sorted(set(idx.values())):
            p = os.path.join(path, name)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            self.shards[p] = (n + 8, hdr)

        def norm(k):
            # 只剔尾缀；前缀差异交给下面的唯一后缀匹配，不硬编码模型字段名
            if k.endswith(".weight_bytes"):
                k = k[: -len(".weight_bytes")]
            elif k.endswith(".weight"):
                k = k[: -len(".weight")]
            return k

        self.norm = norm
        self.table_norm = {}
        for k, v in table.items():
            if not isinstance(v, int):
                continue
            n = norm(k)
            if n in self.table_norm and self.table_norm[n] != v:
                raise RuntimeError("归一化后 %s 撞键且 ggml type 不同（%d vs %d）"
                                   % (n, self.table_norm[n], v))
            self.table_norm[n] = v
        self.n_table_blob = len(self.table_norm)

        def lookup(tn):
            """张量名（已归一）-> 表键；要求全等或唯一后缀命中。"""
            if tn in self.table_norm:
                return tn, "exact"
            cands = [k for k in self.table_norm if tn.endswith("." + k)]
            if len(cands) == 1:
                return cands[0], "suffix"
            if len(cands) > 1:
                raise RuntimeError("张量 %s 在表里后缀命中 %d 个键，歧义：%s"
                                   % (tn, len(cands), sorted(cands)[:5]))
            raise RuntimeError("张量 %s 在类型表里找不到对应条目" % tn)

        self.blobs = {}
        self.match_form = collections.Counter()
        self.matched_table_keys = set()
        for tname in sorted(idx):
            if not tname.endswith(".weight_bytes"):
                continue
            key, form = lookup(norm(tname))
            self.match_form[form] += 1
            self.matched_table_keys.add(key)
            t = self.table_norm[key]
            if t not in TYPES:
                raise RuntimeError("%s 的 ggml type %d 不在路线 B 支持的 %s 里"
                                   % (tname, t, list(TYPES)))
            shard = os.path.join(self.path, idx[tname])
            base, hdr = self.shards[shard]
            e = hdr[tname]
            if e["dtype"] != "U8" or len(e["shape"]) != 2:
                raise RuntimeError("%s 应为 U8 [rows, row_bytes]，实为 %s %s"
                                   % (tname, e["dtype"], e["shape"]))
            self.blobs[tname] = (t, shard, base + e["data_offsets"][0],
                                 int(e["shape"][1]), int(e["shape"][0]))
        # 表里说自己是 blob、但产物里没有对应 weight_bytes 张量的条目（应为 0）
        self.orphan_table_keys = sorted(set(self.table_norm) - self.matched_table_keys)

    def type_names(self, t):
        return sorted(n for n, v in self.blobs.items() if v[0] == t)

    def sample(self, t, want, rng):
        """从该类型的真实张量里按整行取 block，返回 (n, type_size) uint8。"""
        ts = TYPE_SIZE[t]
        names = self.type_names(t)
        handles = {}
        out, touched = [], set()
        per_name = max(1, int(np.ceil(want / max(1, len(names)))))
        try:
            for name in names:
                _t, shard, base, row_bytes, nrows = self.blobs[name]
                bpr = row_bytes // ts
                if bpr * ts != row_bytes:
                    raise RuntimeError("%s 的 row_bytes=%d 不是 block_size %d 的整数倍"
                                       % (name, row_bytes, ts))
                rows_needed = int(np.ceil(per_name / bpr))
                rows = np.sort(rng.choice(nrows, size=min(rows_needed, nrows), replace=False))
                if shard not in handles:
                    handles[shard] = open(shard, "rb")
                fh = handles[shard]
                buf = np.empty((rows.size, row_bytes), np.uint8)
                for i, r in enumerate(rows):
                    fh.seek(base + int(r) * row_bytes)
                    buf[i] = np.frombuffer(fh.read(row_bytes), np.uint8)
                flat = buf.reshape(-1, ts)
                out.append(flat)
                touched.add(name)
                if sum(o.shape[0] for o in out) >= want:
                    break
        finally:
            for fh in handles.values():
                fh.close()
        if not out:
            return np.zeros((0, ts), np.uint8), touched
        blocks = np.concatenate(out, axis=0)[:want]
        return blocks, touched


def edge_blocks(t, rng, n_random=2048):
    """手造边界 block：全 0、全 FF、次正规 d、scale 极值，再加有限值随机块。"""
    ts = TYPE_SIZE[t]
    rows = [np.zeros(ts, np.uint8), np.full(ts, 0xFF, np.uint8),
            np.full(ts, 0x00, np.uint8), np.full(ts, 0x01, np.uint8)]
    b = np.full(ts, 0xFF, np.uint8)
    b[:] = 0
    if t == 8:                       # d = 最小次正规 half，qs 极值
        b[0:2] = [0x01, 0x00]
        b[2:] = 0x80                 # int8 -128
        rows.append(b.copy())
        b[2:] = 0x7F                 # int8 +127
        rows.append(b.copy())
    elif t in (12, 13):              # d / dmin 次正规，6-bit scale/min 全 63
        b[0:2] = [0x01, 0x00]
        b[2:4] = [0xFF, 0x00]        # dmin = 1023 * 2^-24
        b[4:16] = 0xFF
        rows.append(b.copy())
        b[0:2] = [0xFE, 0x7B]        # d = 65534（最大有限 half）
        b[2:4] = [0x00, 0x00]
        rows.append(b.copy())
    else:                            # Q6_K：int8 scale = -128 / +127
        b[192:208] = 0x80
        b[208:210] = [0x01, 0x00]
        rows.append(b.copy())
        b[192:208] = 0x7F
        b[208:210] = [0xFE, 0x7B]
        rows.append(b.copy())
    # 有限值随机块：随机字节 + 把 half 域换成非 inf/nan 的随机值
    for _ in range(n_random):
        r = rng.integers(0, 256, ts, dtype=np.uint8)
        for off in _half_offsets(t):
            h = int(rng.integers(0, 0x7BFF + 1))        # exp != 0x1F
            r[off], r[off + 1] = h & 0xFF, (h >> 8) & 0xFF
        rows.append(r)
    return np.stack(rows)


def _half_offsets(t):
    if t == 8:
        return (0,)
    if t in (12, 13):
        return (0, 2)
    return (208,)


def half_sweep_blocks(t, rng):
    """让每个 half 字段各自遍历全 65536 个位模式，其余字节随机。

    真实数据不一定会把次正规、负零、inf 这些 d 值送到解码路径上，全域扫描才能
    钉住头里那份 half_to_float()（包括上面 numpy 参考刚犯过的符号位错误）。
    返回 (blocks, 每个字段的扫描块起始行) 。
    """
    ts, offs = TYPE_SIZE[t], _half_offsets(t)
    per = 65536
    blocks = rng.integers(0, 256, (per * len(offs), ts), dtype=np.uint8)
    pats = np.arange(per, dtype=np.uint16)
    for i, off in enumerate(offs):
        sl = slice(i * per, (i + 1) * per)
        # 其他 half 字段固定为 1.0，避免 NaN/inf 乘上本字段后把结果全糊成 NaN
        for o2 in offs:
            if o2 != off:
                blocks[sl, o2] = 0x00
                blocks[sl, o2 + 1] = 0x3C
        blocks[sl, off] = (pats & 0xFF).astype(np.uint8)
        blocks[sl, off + 1] = (pats >> np.uint16(8)).astype(np.uint8)
    return blocks, [(off, i * per) for i, off in enumerate(offs)]


# ------------------------------------------------------------ probe 编译/调用
def build_probe(src, out, compiler, extra=()):
    cmd = [compiler, "-O2", "-std=c++17", "-I", HEADER_DIR, src, "-o", out] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("编译失败：%s\n%s" % (" ".join(cmd), (p.stderr or p.stdout)[-4000:]))
    return out


def run_probe(binary, t, blocks, workdir, tag):
    ts, bs = TYPE_SIZE[t], BLOCK_SIZE[t]
    inbin = os.path.join(workdir, "%s_t%d.in" % (tag, t))
    f32bin = os.path.join(workdir, "%s_t%d.f32" % (tag, t))
    bf16bin = os.path.join(workdir, "%s_t%d.bf16" % (tag, t))
    np.ascontiguousarray(blocks).tofile(inbin)
    p = subprocess.run([binary, str(t), str(blocks.shape[0]), inbin, f32bin, bf16bin],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("%s 失败（type=%d, rc=%d）：%s"
                           % (os.path.basename(binary), t, p.returncode,
                              (p.stderr or p.stdout).strip()[-2000:]))
    f32 = np.fromfile(f32bin, np.float32).reshape(-1, bs)
    bf16 = np.fromfile(bf16bin, np.uint16).reshape(-1, bs)
    m = re.search(r"elems=(\d+)", p.stdout)
    return f32, bf16, (int(m.group(1)) if m else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/liuxd/models/Qwen3.8-27B-GGUF-native-mini8")
    ap.add_argument("--blocks", type=int, default=20000, help="每种类型取多少真实 block")
    ap.add_argument("--workdir", default="/home/liuxd/tmp_routeb/blocks31")
    ap.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    ap.add_argument("--nvcc", default=os.environ.get("CUDACXX", "nvcc"))
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.workdir, exist_ok=True)
    print("产物：%s\n头文件：%s\n临时目录：%s\n每类型真实 block 目标：%d"
          % (args.model_path, os.path.join(HEADER_DIR, "ggml_blocks.h"), args.workdir,
             args.blocks))

    print("\n[0] 参考实现自检")
    check_half_decode()

    art = Artifact(args.model_path)
    n_blob_total = len(art.blobs)
    print("产物 blob 张量 %d 个（key_prefix=%r），按类型：%s"
          % (n_blob_total, art.prefix, {t: len(art.type_names(t)) for t in TYPES}))
    check("类型表 blob 条目与产物 weight_bytes 张量双向对平（表 %d / 张量 %d，孤儿 %d，"
          "匹配形态 %s）"
          % (art.n_table_blob, n_blob_total, len(art.orphan_table_keys), dict(art.match_form)),
          art.n_table_blob == n_blob_total and not art.orphan_table_keys,
          "孤儿键：%s" % art.orphan_table_keys[:5])

    print("\n[1] 编译 probe driver")
    host_bin = build_probe(PROBE_CPP, os.path.join(args.workdir, "blocks_probe_host"), args.cxx)
    print("  host driver ok：%s" % host_bin)
    dev_bin = None
    if args.no_cuda:
        skip("cuda driver 编译", "--no-cuda")
    else:
        try:
            dev_bin = build_probe(PROBE_CU, os.path.join(args.workdir, "blocks_probe_cuda"),
                                 args.nvcc, extra=["-x", "cu"])
            print("  cuda driver ok：%s" % dev_bin)
        except Exception as e:
            print("  ! %s" % e)
            dev_bin = None

    print("\n[2] 逐类型对拍（真实 block + 边界 block）")
    for t in TYPES:
        name = Q(t).name
        want = args.blocks
        blocks, touched = art.sample(t, want, rng)
        if not check("%s 采到 %d 个真实 block（目标 %d，覆盖 %d 个张量）"
                     % (name, blocks.shape[0], want, len(touched)),
                     blocks.shape[0] >= min(want, 100)):
            continue

        ref = REF[t](np.ascontiguousarray(blocks))
        py = gguf_py_dequant(t, blocks)
        n_bad, n_diff, maxabs, first = bitwise_diff(ref, py)
        check("%s numpy 参考 vs gguf-py（%d block 逐位相同）"
              % (name, blocks.shape[0]),
              n_diff == 0 and n_bad == 0,
              "差异 %d/%d 元素，非有限 %d，max|Δ|=%.3g，首个：%s"
              % (n_diff, ref.size, n_bad, maxabs, first))

        try:
            h_f32, h_bf16, elems = run_probe(host_bin, t, blocks, args.workdir, "host")
        except Exception as e:
            check("%s host probe 运行" % name, False, str(e))
            continue
        check("%s 头的 block_elems 与 GGML_QUANT_SIZES 一致（%d == %d）"
              % (name, elems, BLOCK_SIZE[t]), elems == BLOCK_SIZE[t])
        n_bad, n_diff, maxabs, first = bitwise_diff(h_f32, ref)
        check("%s 头(host) fp32 vs numpy 参考（%d 元素逐位相同）"
              % (name, h_f32.size), n_diff == 0 and n_bad == 0,
              "差异 %d，首个：%s" % (n_diff, first))

        want_bf16 = float_to_bf16_bits(ref)
        same_own = np.array_equal(h_bf16, want_bf16)
        check("%s 头(host) bf16 vs numpy RNE 舍入" % name, same_own,
              "首个差异 %s" % (np.flatnonzero(h_bf16 != want_bf16)[:5],))
        try:
            import torch
            tv = torch.from_numpy(np.ascontiguousarray(ref)).to(torch.bfloat16) \
                .view(torch.uint16).numpy()
            check("%s 头(host) bf16 vs torch .to(bfloat16)" % name, np.array_equal(h_bf16, tv))
        except Exception as e:
            skip("%s bf16 vs torch" % name, str(e).splitlines()[0][:80])

        if dev_bin is not None:
            try:
                d_f32, d_bf16, _ = run_probe(dev_bin, t, blocks, args.workdir, "cuda")
            except Exception as e:
                check("%s cuda probe 运行" % name, False, str(e))
            else:
                check("%s 头(cuda) fp32 vs 头(host) 逐位相同" % name,
                      np.array_equal(d_f32.view(np.uint32), h_f32.view(np.uint32)))
                check("%s 头(cuda) bf16 vs 头(host) 逐位相同" % name,
                      np.array_equal(d_bf16, h_bf16))

        eb = edge_blocks(t, rng)
        eref = REF[t](np.ascontiguousarray(eb))
        epy = gguf_py_dequant(t, eb)
        _, n_diff_e, maxabs_e, first_e = bitwise_diff(eref, epy)
        n_bad_e = int((~np.isfinite(eref) | ~np.isfinite(epy)).sum())
        h_e, _, _ = run_probe(host_bin, t, eb, args.workdir, "host_edge")
        _, n_diff_h, maxabs_h, first_h = bitwise_diff(h_e, eref)
        check("%s 边界块（%d 个）numpy vs gguf-py 逐位相同" % (name, eb.shape[0]),
              n_diff_e == 0, "差异 %d，非有限 %d，max|Δ|=%.3g，首个：%s"
              % (n_diff_e, n_bad_e, maxabs_e, first_e))
        check("%s 边界块（%d 个）头(host) vs numpy 逐位相同" % (name, eb.shape[0]),
              n_diff_h == 0, "差异 %d，max|Δ|=%.3g，首个：%s" % (n_diff_h, maxabs_h, first_h))

    print("\n[3] half 字段全域扫描（每个字段 65536 个位模式）")
    for t in TYPES:
        name = Q(t).name
        sb, _marks = half_sweep_blocks(t, rng)
        with np.errstate(all="ignore"):      # 扫描里故意喂 inf/nan half，告警与判据无关
            sref = REF[t](np.ascontiguousarray(sb))
        sh, _, _ = run_probe(host_bin, t, sb, args.workdir, "host_sweep")
        n_bad, n_diff, maxabs, first = bitwise_diff(sh, sref)
        check("%s 头(host) vs numpy 参考：half 全域扫描 %d block 逐位相同"
              % (name, sb.shape[0]), n_diff == 0,
              "差异 %d/%d 元素，非有限 %d（inf/nan 乘出的正常现象），max|Δ|=%.3g，首个：%s"
              % (n_diff, sh.size, n_bad, maxabs, first))

    print("\n[4] 不支持的类型必须被头拒绝")
    inbin = os.path.join(args.workdir, "reject.in")
    np.zeros(TYPE_SIZE[8], np.uint8).tofile(inbin)
    p = subprocess.run([host_bin, "10", "1", inbin,
                        os.path.join(args.workdir, "reject.f32"),
                        os.path.join(args.workdir, "reject.bf16")],
                       capture_output=True, text=True)
    check("头对 ggml type 10（TQ1_0，非本头范围）返回拒绝",
          p.returncode == 3 and "no decoder" in (p.stderr + p.stdout),
          "rc=%d stderr=%s" % (p.returncode, (p.stderr or p.stdout).strip()[-200:]))

    print("\n== 结果：%d PASS / %d FAIL / %d SKIP ==" % (_PASS, _FAIL, _SKIP))
    print("临时目录：%s" % args.workdir)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
