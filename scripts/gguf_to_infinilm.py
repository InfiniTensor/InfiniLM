#!/usr/bin/env python3
"""
InfiniLM 路线 B —— 阶段 1 打包器：GGUF -> InfiniLM 原生量化产物（执行方案 §5）

    Qwen3.8-27B-UD-Q6_K.gguf  ->  models/Qwen3.8-27B-GGUF-native/
                                     config.json + model-0000N-of-0000M.safetensors + index

铁律（阶段 0 的教训写在这里，别再用第二套定义）：
  * 键名 / shape / 哪些走 blob / 哪些稠密化，全部来自 `gguf_mapping.build_plan(REAL)`
    + `apply_v1_exceptions()`。本文件**不得**出现第二张表。
  * 反量化一律调 `gguf.quants.dequantize`，禁止自己实现解码。
  * 置换只沿 dim0 整行/整元素搬，块内字节绝不动（§2.7 已证明字节级可行）。
  * 取向：gguf-py 的 `tensor.data` 已经是 [out, in]（量化张量是 [out, row_bytes]），
    与 InfiniLM 参数同序 ⇒ 全程不转置数据。

用法：
    source /home/liuxd/InfiniLM/scripts/gguf_routeb_env.sh
    python3 scripts/gguf_to_infinilm.py [--dry-run] [--layers 4] [--verify all]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from math import prod

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(
    0, os.path.join(os.environ.get("LLAMA_CPP_DIR", "/home/liuxd/llama.cpp"), "gguf-py")
)

import gguf_mapping as M  # noqa: E402
import gguf_transforms as X  # noqa: E402
import numpy as np  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402
from gguf.constants import GGMLQuantizationType as Q  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

DEFAULT_GGUF = "/home/liuxd/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf"
DEFAULT_OUT = "/home/liuxd/models/Qwen3.8-27B-GGUF-native"
DEFAULT_TOKENIZER = "/home/liuxd/models/Qwen3.8-27B-BF16"

TYPE_NAME = {int(v.value): str(v.name) for v in Q}
TYPE_ID = {str(v.name): int(v.value) for v in Q}
UNQUANTIZED = ("F32", "F16", "BF16")

# 分词器配置文件：词表本身从 GGUF 导出，这些附属文件优先从 --tokenizer-dir 复制。
TOKENIZER_FILES = (
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
)

_GiB = 2**30


def log(msg: str) -> None:
    print(msg, flush=True)


def blk_sizes(type_name: str) -> tuple[int, int]:
    bs, ts = GGML_QUANT_SIZES[Q[type_name]]
    return int(bs), int(ts)


# safetensors 报的是 GGML 风 dtype 名（BF16/U8），torch 报的是 bfloat16/uint8，
# 不归一就会把 947 个键全判成不符（实测踩过）。
_DTYPE_ALIAS = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "U8": "uint8"}


def norm_dtype(s) -> str:
    s = str(s)
    return _DTYPE_ALIAS.get(s.upper() if s.isupper() else s, s.lower())


# ---------------------------------------------------------------------------
# 源 -> 目标：单一实现
# ---------------------------------------------------------------------------


def dense_float32(src: np.ndarray, type_name: str, chunk_rows: int) -> np.ndarray:
    """源张量的若干行 -> float32 [rows, in]。未量化类型只是换 dtype。"""
    if type_name in UNQUANTIZED:
        return np.asarray(src, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError("量化源张量应是 [out, row_bytes]，实测 %s" % (src.shape,))
    rows = src.shape[0]
    if rows == 0:
        return np.zeros((0,), dtype=np.float32)
    q = Q[type_name]
    first = np.asarray(dequantize(src[:chunk_rows], q), dtype=np.float32)
    if rows <= chunk_rows:
        return first
    # 预分配而不是 parts+concatenate：lm_head（248320×5120）峰值从 ~10 GB 降到 ~5 GB
    out = np.empty((rows,) + first.shape[1:], dtype=np.float32)
    out[:chunk_rows] = first
    for i in range(chunk_rows, rows, chunk_rows):
        out[i : i + chunk_rows] = np.asarray(
            dequantize(src[i : i + chunk_rows], q), dtype=np.float32
        )
    return out


def make_blob(e, t, dims, opt):
    """逐字节路径：U8 [out, row_bytes]，只在需要时做整行置换。"""
    bs, ts = blk_sizes(opt.types[t.name])
    n_out, n_in = int(e.shape[0]), int(e.shape[1])
    rb = M.row_bytes(n_in, bs, ts)
    if int(t.data.shape[-1]) != rb:
        raise ValueError(
            "%s: 源行字节 %d != 映射表期望 %d" % (e.gguf, int(t.data.shape[-1]), rb)
        )
    arr = t.data
    if e.slices:
        s, ep = e.slices[0]
        arr = arr[s:ep]
    if int(arr.shape[0]) != n_out:
        raise ValueError(
            "%s: 取段后 %d 行 != 映射表 %d" % (e.gguf, arr.shape[0], n_out)
        )
    if M.needs_vperm(e):
        arr = X.apply_vperm(arr, e, dims, opt.vperm)
    return torch_from(arr, np.uint8)


def entry_float32(e, t, dims, opt) -> np.ndarray:
    """稠密化条目的 float32 值。单一实现：写盘（make_dense）与自检（比 BF16 位）共用。"""
    tn = opt.types[t.name]
    src = t.data
    if e.slices:
        s, ep = e.slices[0]
        src = src[s:ep]
    if tn in UNQUANTIZED:
        arr = np.asarray(src, dtype=np.float32)
    else:
        arr = dense_float32(src, tn, opt.chunk_rows)
    for tr in e.transforms:
        if tr == M.T_ALOG:
            arr = X.alog_from_ssm_a(arr)
        elif tr in M.VPERM_TRANSFORMS:
            arr = X.apply_vperm(arr, e, dims, opt.vperm)
        elif tr in (M.T_DENSE, M.T_NONE):
            continue
        else:
            raise ValueError("%s: 未知 transform %r" % (e.infinilm, tr))
    want = tuple(int(x) for x in e.shape)
    if tuple(arr.shape) != want:
        if arr.size != prod(want):
            raise ValueError(
                "%s: 变换后 shape %s != 映射表 %s" % (e.infinilm, arr.shape, want)
            )
        arr = arr.reshape(want)  # §2.11 第 5 条：conv1d 补中间维
    return arr


def make_dense(e, t, dims, opt):
    """稠密化路径：反量化 / 换 dtype -> float32 -> BF16（cast 交给 torch，不自实现）。"""
    return torch_from(entry_float32(e, t, dims, opt), "bf16")


def torch_from(arr: np.ndarray, dtype):
    import torch

    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t.to(torch.bfloat16) if dtype == "bf16" else t


def _is_baked_plus1_norm(name: str) -> bool:
    """llama.cpp 转换时已对 norm.weight baked +1 的那些参数（conversion/qwen.py:394，
    linear_attn.norm 除外）。与 modeling_utils 的 `_remap_qwen3_5` 加载期 +1 集合一一对应：
    input/post_attention_layernorm、self_attn.q/k_norm、最终 model.norm 都以 'norm.weight' 结尾。"""
    return name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight")


def build(e, t, dims, opt, dense_all: bool):
    """一条映射条目 -> 一个待写盘的张量 + 名称。dense_all 用于 --emit-dense-ref。"""
    e2 = e
    if dense_all and e.blob:
        e2 = _as_dense(e)
    tens = make_blob(e2, t, dims, opt) if e2.blob else make_dense(e2, t, dims, opt)
    # dense-ref 的 ssm_out 列序必须从 GGUF 的 tiled 换成 grouped，否则与 blob 路径（运行时 gather）语义不同，
    # §8.3 的逐层 cos_sim 对拍就失去意义。稠密 BF16 可以随便换列（不像 blob 跨块要重量化），所以这里直接 permute。
    # vperm=none 时 blob 路径不做运行时 gather，denseref 也必须保持 GGUF 原生列序，二者才同构。
    if dense_all and e.act_vperm and opt.vperm != "none":
        n_k, r, hd = dims.lin_k_heads, dims.v_per_k, dims.lin_v_dim
        out_dim, in_dim = int(e.shape[0]), int(e.shape[1])
        if in_dim != n_k * r * hd:
            raise ValueError(
                "%s: in_dim %d != num_k_heads*num_v_per_k*head_dim = %d，无法按头分块置换列"
                % (e.infinilm, in_dim, n_k * r * hd)
            )
        # [out, in] 解释为 [out, r, n_k, hd]（tiled 序）-> 对调 1,2 轴 -> [out, n_k, r, hd]（grouped 序）-> flatten
        tens = (
            tens.view(out_dim, r, n_k, hd)
            .transpose(1, 2)
            .contiguous()
            .view(out_dim, in_dim)
        )
    # dense-ref 版删掉了 quantization_config（见主写盘处），框架按普通 HF 模型加载；
    # python 侧 `_remap_qwen3_5`（modeling_utils L808）对**非 gguf** 模型会把 norm 权重 +1
    # （HF 存 delta、C++ 用完整权重的约定）。而 dense-ref 的 norm 值是从 GGUF 原样搬来的
    # **已 baked +1 的完整权重**，再 +1 就变成 2+w（实测使块输入翻倍、级联污染 §8.3）。
    # 故 dense-ref 预存 (w-1)，让加载期 +1 恰好还原成 w，与 blob 路径（gguf=True 不 +1）同构。
    # 集合与 modeling_utils:799 `norm_weight_suffixes` 一致：linear_attn.norm 除外。
    if dense_all and _is_baked_plus1_norm(e.infinilm):
        tens = tens - 1
    name = M.ckpt_name(e2)
    return name, tens


_DENSE_CACHE: dict = {}


def _as_dense(e):
    """blob 条目的“同样内容但稠密化”视图（只给 dense-ref 用，不改原表）。"""
    key = (e.infinilm, e.vperm)
    v = _DENSE_CACHE.get(key)
    if v is None:
        tr = tuple(x for x in e.transforms if x != M.T_NONE) + (M.T_DENSE,)
        v = M.Entry(
            e.infinilm,
            e.gguf,
            e.shape,
            False,
            tr,
            e.types,
            e.slices,
            e.vperm,
            "dense-ref " + (e.note or ""),
        )
        _DENSE_CACHE[key] = v
    return v


# ---------------------------------------------------------------------------
# 维度：从 GGUF 元数据推导，并与映射表的 REAL 对账
# ---------------------------------------------------------------------------


def _dec(x) -> float:
    """float32 元数据归回十进制字面量（1e-6 而非 9.999999974752427e-07），
    让 config.json 与 HF 原始 config 逐字符一致。7 位有效数字对 float32 无损。"""
    return float("%.7g" % float(x))


def dims_from_gguf(reader) -> M.Dims:
    """元数据键名沿用 llama.cpp 标准写法，与审计脚本 E 节实测同一批键。"""
    g = lambda suffix, idx=0: X.gguf_meta(reader, suffix)[idx]  # noqa: E731
    n_layers = int(g("block_count")) - int(g("nextn_predict_layers"))
    inner = int(g("ssm.inner_size"))
    state = int(g("ssm.state_size"))
    head_dim = int(g("attention.key_length"))
    dim_cnt = int(g("rope.dimension_count"))
    sec = [int(x) for x in X.gguf_meta(reader, "rope.dimension_sections")]
    vocab = len(X.gguf_meta(reader, "tokenizer.ggml.tokens"))
    return M.Dims(
        hidden=int(g("embedding_length")),
        n_q_heads=int(g("attention.head_count")),
        n_kv_heads=int(g("attention.head_count_kv")),
        head_dim=head_dim,
        ffn=int(g("feed_forward_length")),
        lin_k_heads=int(g("ssm.group_count")),
        lin_v_heads=inner // state,
        lin_k_dim=state,
        lin_v_dim=state,
        conv_kernel=int(g("ssm.conv_kernel")),
        vocab=vocab,
        n_layers=n_layers,
        interval=int(g("full_attention_interval")),
        mrope_section=tuple(sec[:3]),  # 丢掉尾 0：§2.11 第 4 条
        rope_theta=_dec(g("rope.freq_base")),
        partial_rotary_factor=_dec(dim_cnt / head_dim),
        rms_norm_eps=_dec(g("attention.layer_norm_rms_epsilon")),
        max_position_embeddings=int(g("context_length")),
    )


def check_dims(d: M.Dims) -> None:
    """元数据推导必须与映射表钉死的 REAL 一致，否则说明换了模型还硬套表。

    float 字段用相对容差：GGUF 存的是 float32，1e-6 读回来是
    9.999999974e-07，按 == 比会误报（实测本文件的 rms_norm_eps 就撞在这上面）。
    """
    diff = []
    for f in _DIM_FIELDS:
        got, want = getattr(d, f.name), getattr(M.REAL, f.name)
        if isinstance(want, float) or isinstance(got, float):
            if not np.isclose(float(got), float(want), rtol=1e-6, atol=1e-12):
                diff.append("%s: %r != %r" % (f.name, got, want))
        elif got != want:
            diff.append("%s: %r != %r" % (f.name, got, want))
    if diff:
        raise SystemExit(
            "GGUF 元数据推导出的维度与 gguf_mapping.REAL 不符：%s\n"
            "=> 先按新模型实测重做阶段 0，不要改打包器来迁就。" % diff
        )
    log(
        "  rms_norm_eps：GGUF float32 %r -> config 写 HF 十进制 %r"
        % (float(d.rms_norm_eps), M.REAL.rms_norm_eps)
    )


from dataclasses import fields as _dc_fields  # noqa: E402

_DIM_FIELDS = [f for f in _dc_fields(M.Dims) if f.name != "architectures"]


# ---------------------------------------------------------------------------
# 分片写出
# ---------------------------------------------------------------------------


class ShardWriter:
    def __init__(self, out_dir: str, max_bytes: int):
        self.dir, self.max = out_dir, max_bytes
        self.buf: dict[str, object] = {}
        self.buf_bytes = 0
        self.shards: list[str] = []
        self.weight_map: dict[str, str] = {}
        self.total = 0

    def add(self, name: str, tens) -> None:
        nbytes = int(tens.numel()) * int(tens.element_size())
        if self.buf and self.buf_bytes + nbytes > self.max:
            self.flush()
        self.buf[name] = tens
        self.buf_bytes += nbytes
        self.total += nbytes

    def flush(self) -> None:
        if not self.buf:
            return
        self.shards.append("__pending__")
        idx = len(self.shards)
        fname = "model-%05d.safetensors" % idx
        from safetensors.torch import save_file

        save_file(self.buf, os.path.join(self.dir, fname), metadata={"format": "pt"})
        for k in self.buf:
            self.weight_map[k] = fname
        log(
            "  写出 %s（%.2f GiB，%d 个张量）"
            % (fname, self.buf_bytes / _GiB, len(self.buf))
        )
        self.shards[-1] = fname
        self.buf, self.buf_bytes = {}, 0

    def finish(self) -> None:
        self.flush()
        n = len(self.shards)
        renamed = {}
        for i, f in enumerate(self.shards, 1):
            new = "model-%05d-of-%05d.safetensors" % (i, n)
            if f != new:
                os.rename(os.path.join(self.dir, f), os.path.join(self.dir, new))
            renamed[f] = new
        self.weight_map = {k: renamed[v] for k, v in self.weight_map.items()}
        with open(os.path.join(self.dir, "model.safetensors.index.json"), "w") as fp:
            json.dump(
                {"metadata": {"total_size": self.total}, "weight_map": self.weight_map},
                fp,
                indent=1,
                sort_keys=True,
            )
        log("  分片 %d 个，合计 %.3f GiB" % (n, self.total / _GiB))


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def rows_hash(a) -> str:
    """把 [rows, cols] 字节阵的**行多重集**压成一个摘要（排序后逐行喂 hash）。

    用途：置换过的 blob 不能直接与源逐字节比（那等于拿置换代码自证），
    但可以无条件断言“产物行集 == 源行集”（置换只是整行搬，不允许改字节）。
    """
    import hashlib

    a = np.ascontiguousarray(a)
    v = a.view(np.void(a.shape[1] * a.dtype.itemsize)).ravel()
    h = hashlib.sha256()
    for x in np.sort(v):
        h.update(x.tobytes())
    return h.hexdigest()[:16]


def dense_bits_check(e, t, dims, opt, prod_t) -> bool:
    """BF16 条目的位级校验：逐行块算期望值并与产物对应行块比，峰值内存有界。

    上一版直接 `bf16_bits(整块 float32)`，在 lm_head（12.7 亿元素）上把进程 OOM kill 掉了。
    V 头置换 / A_log 是跨行或逐元素语义，不能切块，但这类条目都很小，走全量路径。
    """
    import torch

    if M.needs_vperm(e):
        return bool(
            np.array_equal(
                prod_t.view(torch.uint16).numpy(),
                X.bf16_bits(entry_float32(e, t, dims, opt)),
            )
        )
    src = np.asarray(t.data)
    if e.slices:
        s, ep = e.slices[0]
        src = src[s:ep]
    tn = opt.types[t.name]
    tail = tuple(int(x) for x in e.shape[1:])
    per_row = prod(tail) if tail else 1
    rows = max(1, int(_BIG_ELEMS // per_row))
    n = int(e.shape[0])
    if src.shape[0] != n:
        return False
    for i in range(0, n, rows):
        blk = src[i : i + rows]
        arr = (
            np.asarray(blk, dtype=np.float32)
            if tn in UNQUANTIZED
            else dense_float32(blk, tn, opt.chunk_rows)
        )
        exp = X.bf16_bits(arr.reshape((blk.shape[0],) + tail))
        got = prod_t[i : i + rows].view(torch.uint16).numpy()
        if not np.array_equal(got, exp):
            return False
    return True


_BIG_ELEMS = 64 * 1024 * 1024  # 切块阈值：一次最多算 64M 元素（float32 峰值 256 MB）


def verify(out_dir: str, plan, tensors, dims, opt, sample) -> int:
    """重读产物：全量比键/shape/dtype，分类抽样比字节。返回 FAIL 数。"""
    import torch
    from safetensors import safe_open

    log("\n== 自检：重读产物 ==")
    bs_files = sorted(
        f
        for f in os.listdir(out_dir)
        if f.endswith(".safetensors") and not f.startswith(".")
    )
    with open(os.path.join(out_dir, "model.safetensors.index.json")) as fp:
        index = json.load(fp)
    got: dict[str, tuple] = {}
    handles = {}
    for f in bs_files:
        h = safe_open(os.path.join(out_dir, f), framework="pt")
        handles[f] = h
        for k in h.keys():
            t = h.get_slice(k)
            got[k] = (tuple(int(x) for x in t.get_shape()), norm_dtype(t.get_dtype()))
    fails = 0
    want = {}
    for e in plan:
        name = M.ckpt_name(e)
        if e.blob:
            bs, ts = blk_sizes(opt.types[e.gguf])
            shape, dt = (int(e.shape[0]), M.row_bytes(int(e.shape[1]), bs, ts)), "uint8"
        else:
            shape, dt = tuple(int(x) for x in e.shape), "bfloat16"
        want[name] = (shape, dt, e)
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    for label, keys in (("缺键", missing), ("多键", extra)):
        if keys:
            fails += 1
            log("  FAIL %s %d 个：%s" % (label, len(keys), keys[:6]))
        else:
            log("  PASS 无%s" % label)
    bad = [k for k in set(want) & set(got) if want[k][:2] != got[k]]
    if bad:
        fails += 1
        log(
            "  FAIL shape/dtype 不符 %d 个：%s"
            % (len(bad), [(k, want[k][:2], got[k]) for k in sorted(bad)[:4]])
        )
    else:
        log("  PASS 全部 %d 个键的 shape+dtype 与映射表一致" % len(want))

    # config.json 的类型表必须与产物张量名**双向逐字相等**：阶段 2 的 C++ 就是拿
    # 这些名字查表决定 blob / 稠密（方案 §6.0 纠正 2），两边对不上会在运行期变成
    # “查不到 key”，那比 shape 错更难查。
    with open(os.path.join(out_dir, "config.json")) as fp:
        cfg = json.load(fp)
    qcfg = cfg.get("quantization_config") or {}
    table = qcfg.get("ggml_types") or {}
    if qcfg.get("quant_method") != "gguf":
        fails += 1
        log(
            "  FAIL config.json 顶层 quantization_config.quant_method != 'gguf'（或在 text_config 里）"
        )
    elif qcfg.get("key_prefix") != M.PREFIX:
        fails += 1
        log("  FAIL config.json 缺 key_prefix=%r（阶段 2 C++ 用它裁表 key）" % M.PREFIX)
    else:
        log("  PASS quantization_config 在顶层，key_prefix=%r" % M.PREFIX)
    for label, keys in (
        ("类型表缺键", sorted(set(got) - set(table))),
        ("类型表多键", sorted(set(table) - set(got))),
    ):
        if keys:
            fails += 1
            log("  FAIL %s %d 个：%s" % (label, len(keys), keys[:6]))
        else:
            log("  PASS 无%s（%d 个 key 与张量名逐字相等）" % (label, len(table)))

    # 分类抽样（按名排序取首个，可复现）：三种字节路径必须有各自的代表，
    # 纯随机抽 3 个会全部落在“未置换 memcpy”上，那样根本测不到置换与切片。
    def sel(pred):
        return sorted(k for k, (_, _, e) in want.items() if pred(e))

    cats = [
        (
            "blob 未置换",
            sel(lambda e: e.blob and not e.slices and not M.needs_vperm(e)),
        ),
        ("blob V 置换", sel(lambda e: e.blob and not e.slices and M.needs_vperm(e))),
        ("blob 融合切片", sel(lambda e: e.blob and e.slices)),
        (
            "bf16 反量化",
            sel(lambda e: not e.blob and not e.slices and not M.needs_vperm(e)),
        ),
        ("bf16 置换+alog", sel(lambda e: not e.blob and M.needs_vperm(e))),
        ("bf16 融合切片", sel(lambda e: not e.blob and e.slices)),
    ]
    picks = [c[1][0] for c in cats if c[1]]
    if sample == "all":
        picks = sorted(k for k, (_, _, e) in want.items() if e.blob)
    log(
        "  抽样 %d 个：%s"
        % (
            len(picks),
            "全部 blob"
            if sample == "all"
            else " ".join("%s=%s" % (c, len(v)) for c, v in cats),
        )
    )
    for k in picks:
        shape, dt, e = want[k]
        prod_t = handles[index["weight_map"][k]].get_tensor(k)
        src = np.asarray(tensors[e.gguf].data)
        if e.slices:
            s, ep = e.slices[0]
            src = src[s:ep]
        ref = build(e, tensors[e.gguf], dims, opt, False)[1]
        checks = []
        if tuple(int(x) for x in prod_t.shape) != tuple(int(x) for x in ref.shape):
            checks.append(("shape", False))
        elif e.blob:
            p = prod_t.numpy()
            checks.append(("与重建一致", bool(torch.equal(prod_t, ref))))
            if M.needs_vperm(e):
                checks.append(("行集与源相同", rows_hash(p) == rows_hash(src)))
            else:
                checks.append(("与 GGUF 源逐字节", np.array_equal(p, src)))
        else:
            # BF16：拿 numpy 的 RNE 位模式比，相当于独立验一次 torch 的 cast + 读写往返
            checks.append(
                (
                    "BF16 位与 numpy RNE 一致",
                    dense_bits_check(e, tensors[e.gguf], dims, opt, prod_t),
                )
            )
        ok = all(v for _, v in checks)
        fails += 0 if ok else 1
        log(
            "  %s %-52s %-16s %s"
            % (
                "PASS" if ok else "FAIL",
                k,
                str(tuple(int(x) for x in prod_t.shape)),
                "，".join("%s=%s" % (n, "Y" if v else "N") for n, v in checks),
            )
        )
    return fails


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER)
    ap.add_argument(
        "--dense-iq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="v1 把 5 个 IQ4_NL/IQ4_XS 稠密化（阶段 6 上了码本 kernel 后 --no-dense-iq）",
    )
    ap.add_argument(
        "--dense-embed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="v1 恒为 True；--no-dense-embed 需要阶段 6 的 embedding kernel",
    )
    ap.add_argument(
        "--vperm",
        choices=("inv", "fwd", "none"),
        default="inv",
        help="V 头 tiled->grouped 方向；阶段 4 A/B 用（§2.7）",
    )
    ap.add_argument(
        "--emit-dense-ref",
        metavar="PATH",
        default=None,
        help="额外产出一份全反量化 BF16 版（阶段 4 自洽基准，不部署）",
    )
    ap.add_argument("--max-shard-gib", type=float, default=4.0)
    ap.add_argument(
        "--chunk-rows", type=int, default=8192, help="反量化分块行数，限制峰值内存"
    )
    ap.add_argument(
        "--layers",
        type=int,
        default=None,
        help="只打前 N 层，并同步把 config 的 num_hidden_layers 改成 N"
        "（产物可直接被框架构造 + 加载，阶段 2/3 用小模型验收用）",
    )
    ap.add_argument("--verify", choices=("off", "sample", "all"), default="sample")
    ap.add_argument(
        "--skip-pack",
        action="store_true",
        help="不重写 23 GiB 权重，只做分词器导出 + 自检（迭代自检逻辑用）",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="全量校验取向/shape/字节数，不写盘（稠密化条目也只算 shape）",
    )
    a = ap.parse_args()

    if not a.dense_embed:
        raise SystemExit(
            "--no-dense-embed 需要阶段 6 的 embedding / lm_head 原生 kernel，"
            "v1 没有它们就只能稠密化（§2.4）"
        )

    t0 = time.time()
    log("读取 GGUF 元数据：%s" % a.gguf)
    reader = GGUFReader(a.gguf)
    tensors = {t.name: t for t in reader.tensors}
    dims = dims_from_gguf(reader)
    check_dims(dims)
    log(
        "  维度与映射表 REAL 一致：%d 层，hidden=%d，vocab=%d"
        % (dims.n_layers, dims.hidden, dims.vocab)
    )

    # --layers 必须在 check_dims **之后**覆盖：维度照旧逐项校 REAL（防止换模型后硬套本表），
    # 但 config 的 num_hidden_layers / layer_types 要跟着改，否则截断产物与 config
    # 不自洽，框架构造 64 层却只拿到 N 层权重（旧版本里这条表现为“不可加载”）。
    if a.layers is not None:
        if not 0 < a.layers < dims.n_layers:
            raise SystemExit(
                "--layers 必须在 (0, %d) 之间，实际 %d" % (dims.n_layers, a.layers)
            )
        log(
            "  --layers %d：num_hidden_layers %d -> %d，产物可加载"
            % (a.layers, dims.n_layers, a.layers)
        )
        dims.n_layers = a.layers

    opt = type("Opt", (), {})()
    opt.vperm, opt.chunk_rows = a.vperm, a.chunk_rows
    opt.types = {n: TYPE_NAME[int(t.tensor_type)] for n, t in tensors.items()}
    plan = M.build_plan(dims)
    n_exc = M.apply_v1_exceptions(plan, opt.types, enabled=a.dense_iq)
    log("  映射条目 %d，v1 稠密化例外命中 %d 个 IQ4" % (len(plan), n_exc))
    blob = [e for e in plan if e.blob]
    log(
        "  blob %d 个 / 稠密化 %d 个 / 丢弃 MTP 前缀 %s"
        % (len(blob), len(plan) - len(blob), M.DROP_PREFIXES)
    )

    if a.dry_run:
        log("\n== dry-run：逐条目校验取向与字节数（不写盘、不反量化）==")
        blob_bytes = dense_bytes = 0
        for e in plan:
            t = tensors[e.gguf]
            n_out = (e.slices[0][1] - e.slices[0][0]) if e.slices else int(e.shape[0])
            if e.blob:
                bs, ts = blk_sizes(opt.types[e.gguf])
                rb = M.row_bytes(int(e.shape[1]), bs, ts)
                if int(t.data.shape[-1]) != rb:
                    raise ValueError(
                        "%s: 源行字节 %d != 期望 %d"
                        % (e.gguf, int(t.data.shape[-1]), rb)
                    )
                if int(t.data.shape[0]) < n_out:
                    raise ValueError(
                        "%s: 源 %d 行 < 条目需 %d 行" % (e.gguf, t.data.shape[0], n_out)
                    )
                blob_bytes += n_out * rb
            else:
                n = prod(tuple(int(x) for x in e.shape))
                if not e.slices and prod(int(x) for x in t.shape) != n:
                    raise ValueError(
                        "%s: 源元素数 %s != 条目 shape %s"
                        % (e.gguf, t.shape, tuple(e.shape))
                    )
                dense_bytes += n * 2
        log(
            "  PASS %d 个条目取向/字节数自洽：blob %.3f GiB + 稠密化 BF16 %.3f GiB"
            " = 产物应占 %.3f GiB"
            % (
                len(plan),
                blob_bytes / _GiB,
                dense_bytes / _GiB,
                (blob_bytes + dense_bytes) / _GiB,
            )
        )
        return 0

    os.makedirs(a.out, exist_ok=True)
    w = ShardWriter(a.out, int(a.max_shard_gib * _GiB))
    ggml_types = {}
    for e in plan:
        ggml_types[M.type_table_key(M.ckpt_name(e))] = (
            TYPE_ID[opt.types[e.gguf]] if e.blob else "dense_bf16"
        )
    if a.skip_pack:
        with open(os.path.join(a.out, "model.safetensors.index.json")) as fp:
            w.total = json.load(fp)["metadata"]["total_size"]
        log("\n== --skip-pack：沿用已有权重，仅重写 config.json / 分词器与自检 ==")
    else:
        log("\n== 写出 %s ==" % a.out)
        for i, e in enumerate(plan):
            t = tensors[e.gguf]
            name, tens = build(e, t, dims, opt, False)
            w.add(name, tens)
            if (i + 1) % 100 == 0:
                log("  ... %d/%d 条目（%.1f s）" % (i + 1, len(plan), time.time() - t0))
        w.finish()

    # config.json 两条路都要写：activation_vperm 这类语义元数据只能在这里刷新，
    # 留在 else 里会让 --skip-pack 沿用旧 config（为了几个键重打包 7.2 GiB 不值）。
    # 规则由映射表派生（M.activation_vperm_rules），C++ 照单执行，不在两边各抄一份。
    rules = M.activation_vperm_rules(dims, plan)
    if a.vperm == "none":
        # --vperm none = 全链路不做任何 V 头置换：in_proj 不重排、out_proj 不 gather、
        # denseref 不列置换。config 必须同步清空规则，否则 C++ 照旧 gather。
        rules = []
    cfg = M.make_root_config(dims, ggml_types, rules)
    with open(os.path.join(a.out, "config.json"), "w") as fp:
        json.dump(cfg, fp, indent=1, sort_keys=True)
    log(
        "  config.json：%d 个 ggml_types 键（quantization_config 在顶层）+ 激活 V 头置换规则 %d 条：%s"
        % (
            len(ggml_types),
            len(rules),
            " ".join(
                "%s=%dx%dx%d"
                % (r["suffix"], r["num_k_heads"], r["num_v_per_k"], r["head_dim"])
                for r in rules
            )
            or "无",
        )
    )

    fails = export_tokenizer(reader, a.out, a.tokenizer_dir, dims)

    if not a.skip_pack:
        with open(os.path.join(a.out, "pack_report.json"), "w") as fp:
            json.dump(
                {
                    "gguf": os.path.abspath(a.gguf),
                    # 张量 data 区之和 != 文件大小（后者含元数据与对齐填充），
                    # 两者都记下来，免得日后拿这个数去对 stat 产生误会
                    "gguf_file_bytes": os.path.getsize(os.path.abspath(a.gguf)),
                    "gguf_tensor_data_bytes": sum(
                        int(t.n_bytes) for t in reader.tensors
                    ),
                    "n_gguf_tensors": len(tensors),
                    "v1_dense_iq": bool(a.dense_iq),
                    "vperm": a.vperm,
                    "n_entries": len(plan),
                    "n_blob": len(blob),
                    "n_v1_exceptions": n_exc,
                    "blob_type_ids": sorted({TYPE_ID[opt.types[e.gguf]] for e in blob}),
                    "out_bytes": w.total,
                    "shards": w.shards,
                    "seconds": round(time.time() - t0, 1),
                },
                fp,
                indent=1,
                sort_keys=True,
            )

    if a.verify != "off":
        fails += verify(a.out, plan, tensors, dims, opt, a.verify)

    if a.emit_dense_ref:
        log("\n== 额外产出稠密基准版 %s ==" % a.emit_dense_ref)
        os.makedirs(a.emit_dense_ref, exist_ok=True)
        wr = ShardWriter(a.emit_dense_ref, int(a.max_shard_gib * _GiB))
        for e in plan:
            name, tens = build(e, tensors[e.gguf], dims, opt, True)
            wr.add(name, tens)
        wr.finish()
        ref_cfg = M.make_root_config(
            dims, {M.type_table_key(k): "dense_bf16" for k in ggml_types}, rules
        )
        # 稠密基准版不写 quantization_config：框架默认 NoneQuantization，C++ 里没有人
        # 执行置换。它的 ssm_out 列序在打包期已置换为 grouped（见 build() 里的 act_vperm 分支），
        # 与 blob 路径（运行时 gather）语义相同，可做逐层 cos_sim 对拍（§8.3）。
        del ref_cfg["quantization_config"]
        with open(os.path.join(a.emit_dense_ref, "config.json"), "w") as fp:
            json.dump(ref_cfg, fp, indent=1, sort_keys=True)
        export_tokenizer(reader, a.emit_dense_ref, a.tokenizer_dir, dims)

    log(
        "\n===== 完成：%.1f s，产物 %.3f GiB，自检 FAIL %d 处 ====="
        % (time.time() - t0, w.total / _GiB, fails)
    )
    return 1 if fails else 0


def export_tokenizer(reader, out_dir: str, tokenizer_dir: str, dims) -> int:
    """产物自带完整分词器。为什么不是简单 copy：

    实测 `--tokenizer-dir`（models/Qwen3.8-27B-BF16）只有 vocab.json，**没有**
    merges.txt / tokenizer.json，`AutoTokenizer.from_pretrained` 直接报
    "`vocab` and `merges` must be both be from memory or both filenames"。
    GGUF 内嵌完整 byte-level BPE（实测 248320 tokens / 247587 merges，
    tokenizer.ggml.model=gpt2, pre=qwen35），词表与 embedding 行数同源，故以 GGUF 为准
    写 vocab.json + merges.txt，其余配置文件从 tokenizer_dir 复制。
    """
    tokens = [str(t) for t in X.gguf_meta(reader, "tokenizer.ggml.tokens")]
    merges = [str(m) for m in X.gguf_meta(reader, "tokenizer.ggml.merges")]
    model = str(X.gguf_meta(reader, "tokenizer.ggml.model")[0])
    if len(tokens) != dims.vocab:
        raise SystemExit(
            "GGUF 词表 %d != config vocab_size %d，词表与 embedding 不同源"
            % (len(tokens), dims.vocab)
        )
    if model != "gpt2":
        log(
            "  警告：tokenizer.ggml.model=%r 非 gpt2，vocab.json/merges.txt 写法需复核"
            % model
        )
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as fp:
        json.dump({t: i for i, t in enumerate(tokens)}, fp, ensure_ascii=False)
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as fp:
        fp.write("#version: 0.2\n" + "\n".join(merges) + "\n")
    log(
        "  词表来自 GGUF：vocab %d / merges %d（model=%s）"
        % (len(tokens), len(merges), model)
    )

    have = (
        os.listdir(tokenizer_dir)
        if tokenizer_dir and os.path.isdir(tokenizer_dir)
        else []
    )
    if not have:
        log("  警告：分词器配置目录不存在：%s（只写了词表）" % tokenizer_dir)
    copied = []
    for f in TOKENIZER_FILES:
        dst = os.path.join(out_dir, f)
        if f in have and not os.path.exists(dst):
            shutil.copy2(os.path.join(tokenizer_dir, f), dst)
            copied.append(f)
    log("  附属配置复制 %d 个：%s" % (len(copied), " ".join(sorted(copied))))
    if "tokenizer_config.json" not in copied + have:
        raise SystemExit(
            "产物缺 tokenizer_config.json：既没从 %s 复制到，也没导出兜底"
            % tokenizer_dir
        )
    return check_tokenizer(out_dir, dims)


def check_tokenizer(out_dir: str, dims) -> int:
    """真装一次 AutoTokenizer 并做编解码往返（阶段 5 的前置条件，现在就能测）。"""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        log("  SKIP 分词器自检：本环境无 transformers")
        return 0
    try:
        tk = AutoTokenizer.from_pretrained(out_dir)
        n, cls = len(tk), type(tk).__name__
        s = "你好，世界 hello world 27B"
        ids = tk.encode(s)
        ok = n == dims.vocab and tk.decode(ids) == s
        log(
            "  %s 分词器 %s vocab=%d 往返=%s"
            % ("PASS" if ok else "FAIL", cls, n, tk.decode(ids) == s)
        )
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001
        log("  FAIL 分词器加载：%s: %s" % (type(exc).__name__, str(exc)[:200]))
        return 1


if __name__ == "__main__":
    sys.exit(main())
