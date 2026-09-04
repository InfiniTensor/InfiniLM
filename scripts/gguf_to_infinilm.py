#!/usr/bin/env python3
"""Convert a GGUF model into an InfiniLM packed-weight checkpoint.

All names, shapes, packed/dense choices, and transformations come from
``gguf_mapping``. Dequantization uses ``gguf.quants.dequantize``. Packed data
may only be moved as complete rows; bytes within quantization blocks remain
unchanged. GGUF tensors already match InfiniLM's [out, in] orientation.

Example:
    python3 scripts/gguf_to_infinilm.py --gguf MODEL.gguf --out OUT_DIR
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
if os.environ.get("LLAMA_CPP_DIR"):
    sys.path.insert(0, os.path.join(os.environ["LLAMA_CPP_DIR"], "gguf-py"))

import gguf_mapping as M  # noqa: E402
import gguf_transforms as X  # noqa: E402
import numpy as np  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402
from gguf.constants import GGMLQuantizationType as Q  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

TYPE_NAME = {int(v.value): str(v.name) for v in Q}
TYPE_ID = {str(v.name): int(v.value) for v in Q}
UNQUANTIZED = ("F32", "F16", "BF16")

# Tokenizer vocabulary comes from GGUF; copy auxiliary files when available.
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


# Normalize safetensors and torch dtype names before comparison.
_DTYPE_ALIAS = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "U8": "uint8"}


def norm_dtype(s) -> str:
    s = str(s)
    return _DTYPE_ALIAS.get(s.upper() if s.isupper() else s, s.lower())


# ---------------------------------------------------------------------------
# Source-to-target conversion
# ---------------------------------------------------------------------------


def dense_float32(src: np.ndarray, type_name: str, chunk_rows: int) -> np.ndarray:
    """Convert source rows to float32 [rows, in]."""
    if type_name in UNQUANTIZED:
        return np.asarray(src, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(
            "quantized source must have shape [out, row_bytes], got %s" % (src.shape,)
        )
    rows = src.shape[0]
    if rows == 0:
        return np.zeros((0,), dtype=np.float32)
    q = Q[type_name]
    first = np.asarray(dequantize(src[:chunk_rows], q), dtype=np.float32)
    if rows <= chunk_rows:
        return first
    # Preallocate to bound peak memory for large tensors such as lm_head.
    out = np.empty((rows,) + first.shape[1:], dtype=np.float32)
    out[:chunk_rows] = first
    for i in range(chunk_rows, rows, chunk_rows):
        out[i : i + chunk_rows] = np.asarray(
            dequantize(src[i : i + chunk_rows], q), dtype=np.float32
        )
    return out


def make_blob(e, t, dims, opt):
    """Build a U8 [out, row_bytes] tensor, permuting only complete rows."""
    bs, ts = blk_sizes(opt.types[t.name])
    n_out, n_in = int(e.shape[0]), int(e.shape[1])
    rb = M.row_bytes(n_in, bs, ts)
    if int(t.data.shape[-1]) != rb:
        raise ValueError(
            "%s: source row bytes %d != expected %d"
            % (e.gguf, int(t.data.shape[-1]), rb)
        )
    arr = t.data
    if e.slices:
        s, ep = e.slices[0]
        arr = arr[s:ep]
    if int(arr.shape[0]) != n_out:
        raise ValueError(
            "%s: sliced rows %d != mapping rows %d" % (e.gguf, arr.shape[0], n_out)
        )
    if M.needs_vperm(e):
        arr = X.apply_vperm(arr, e, dims, opt.vperm)
    return torch_from(arr, np.uint8)


def entry_float32(e, t, dims, opt) -> np.ndarray:
    """Return float32 values shared by dense output and verification."""
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
            raise ValueError("%s: unknown transform %r" % (e.infinilm, tr))
    want = tuple(int(x) for x in e.shape)
    if tuple(arr.shape) != want:
        if arr.size != prod(want):
            raise ValueError(
                "%s: transformed shape %s != mapped shape %s"
                % (e.infinilm, arr.shape, want)
            )
        arr = arr.reshape(want)  # Restore a squeezed singleton convolution dimension.
    return arr


def make_dense(e, t, dims, opt):
    """Dequantize or cast through float32, then let torch produce BF16."""
    return torch_from(entry_float32(e, t, dims, opt), "bf16")


def torch_from(arr: np.ndarray, dtype):
    import torch

    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t.to(torch.bfloat16) if dtype == "bf16" else t


def _is_baked_plus1_norm(name: str) -> bool:
    """Return whether llama.cpp baked a +1 normalization offset into GGUF."""
    return name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight")


def build(e, t, dims, opt, dense_all: bool):
    """Build one output tensor and name; dense_all creates the dense reference."""
    e2 = e
    if dense_all and e.blob:
        e2 = _as_dense(e)
    tens = make_blob(e2, t, dims, opt) if e2.blob else make_dense(e2, t, dims, opt)
    # Dense-reference output can permute columns directly. Match the packed path,
    # which performs the equivalent activation permutation at runtime.
    if dense_all and e.act_vperm and opt.vperm != "none":
        n_k, r, hd = dims.lin_k_heads, dims.v_per_k, dims.lin_v_dim
        out_dim, in_dim = int(e.shape[0]), int(e.shape[1])
        if in_dim != n_k * r * hd:
            raise ValueError(
                "%s: input dimension %d != num_k_heads*num_v_per_k*head_dim %d; cannot permute complete heads"
                % (e.infinilm, in_dim, n_k * r * hd)
            )
        # [out, r, n_k, hd] tiled -> [out, n_k, r, hd] grouped -> flatten.
        tens = (
            tens.view(out_dim, r, n_k, hd)
            .transpose(1, 2)
            .contiguous()
            .view(out_dim, in_dim)
        )
    # The dense reference loads through the non-GGUF remap, which adds +1 to
    # selected norm weights. Store w-1 so loading reconstructs the baked GGUF w.
    if dense_all and _is_baked_plus1_norm(e.infinilm):
        tens = tens - 1
    name = M.ckpt_name(e2)
    return name, tens


_DENSE_CACHE: dict = {}


def _as_dense(e):
    """Return a dense-reference view of a packed entry without changing the plan."""
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
# Derive dimensions from GGUF metadata and validate the target profile.
# ---------------------------------------------------------------------------


def _dec(x) -> float:
    """Render float32 metadata with a stable seven-significant-digit decimal."""
    return float("%.7g" % float(x))


def dims_from_gguf(reader) -> M.Dims:
    """Derive dimensions from standard llama.cpp GGUF metadata keys."""
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
        mrope_section=tuple(sec[:3]),  # InfiniLM consumes three MRoPE sections.
        rope_theta=_dec(g("rope.freq_base")),
        partial_rotary_factor=_dec(dim_cnt / head_dim),
        rms_norm_eps=_dec(g("attention.layer_norm_rms_epsilon")),
        max_position_embeddings=int(g("context_length")),
    )


def check_dims(d: M.Dims) -> None:
    """Reject inputs whose metadata does not match the target model profile."""
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
            "GGUF metadata does not match gguf_mapping.REAL: %s\n"
            "Create and validate a model-specific mapping before conversion." % diff
        )
    log(
        "  rms_norm_eps: GGUF float32 %r -> config decimal %r"
        % (float(d.rms_norm_eps), M.REAL.rms_norm_eps)
    )


from dataclasses import fields as _dc_fields  # noqa: E402

_DIM_FIELDS = [f for f in _dc_fields(M.Dims) if f.name != "architectures"]


# ---------------------------------------------------------------------------
# Sharded output
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
            "  wrote %s (%.2f GiB, %d tensors)"
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
        log("  %d shards, %.3f GiB total" % (n, self.total / _GiB))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def rows_hash(a) -> str:
    """Hash the multiset of rows in a [rows, cols] byte array.

    A row permutation cannot be compared positionally, but it must preserve the
    complete multiset of packed rows.
    """
    import hashlib

    a = np.ascontiguousarray(a)
    v = a.view(np.void(a.shape[1] * a.dtype.itemsize)).ravel()
    h = hashlib.sha256()
    for x in np.sort(v):
        h.update(x.tobytes())
    return h.hexdigest()[:16]


def dense_bits_check(e, t, dims, opt, prod_t) -> bool:
    """Verify dense BF16 entries bitwise in bounded row chunks."""
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


_BIG_ELEMS = 64 * 1024 * 1024  # At most 64M float32 elements per verification chunk.


def verify(out_dir: str, plan, tensors, dims, opt, sample) -> int:
    """Reload output, validate metadata, and sample bytes. Return failure count."""
    import torch
    from safetensors import safe_open

    log("\n== Verification: reload output ==")
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
    for label, keys in (("missing keys", missing), ("extra keys", extra)):
        if keys:
            fails += 1
            log("  FAIL %s (%d): %s" % (label, len(keys), keys[:6]))
        else:
            log("  PASS no %s" % label)
    bad = [k for k in set(want) & set(got) if want[k][:2] != got[k]]
    if bad:
        fails += 1
        log(
            "  FAIL shape/dtype mismatch (%d): %s"
            % (len(bad), [(k, want[k][:2], got[k]) for k in sorted(bad)[:4]])
        )
    else:
        log("  PASS shape and dtype for all %d keys" % len(want))

    # Type-table keys must exactly match output tensor names in both directions.
    with open(os.path.join(out_dir, "config.json")) as fp:
        cfg = json.load(fp)
    qcfg = cfg.get("quantization_config") or {}
    table = qcfg.get("ggml_types") or {}
    if qcfg.get("quant_method") != "gguf":
        fails += 1
        log("  FAIL top-level quantization_config.quant_method is not 'gguf'")
    elif qcfg.get("key_prefix") != M.PREFIX:
        fails += 1
        log("  FAIL config.json is missing key_prefix=%r" % M.PREFIX)
    else:
        log("  PASS top-level quantization_config with key_prefix=%r" % M.PREFIX)
    for label, keys in (
        ("type-table missing keys", sorted(set(got) - set(table))),
        ("type-table extra keys", sorted(set(table) - set(got))),
    ):
        if keys:
            fails += 1
            log("  FAIL %s (%d): %s" % (label, len(keys), keys[:6]))
        else:
            log("  PASS no %s (%d exact tensor-name matches)" % (label, len(table)))

    # Deterministically sample every conversion category rather than relying on
    # random samples that may miss permutations and slices.
    def sel(pred):
        return sorted(k for k, (_, _, e) in want.items() if pred(e))

    cats = [
        (
            "packed unchanged",
            sel(lambda e: e.blob and not e.slices and not M.needs_vperm(e)),
        ),
        (
            "packed V permutation",
            sel(lambda e: e.blob and not e.slices and M.needs_vperm(e)),
        ),
        ("packed fused slice", sel(lambda e: e.blob and e.slices)),
        (
            "BF16 dequantization",
            sel(lambda e: not e.blob and not e.slices and not M.needs_vperm(e)),
        ),
        ("BF16 permutation and A_log", sel(lambda e: not e.blob and M.needs_vperm(e))),
        ("BF16 fused slice", sel(lambda e: not e.blob and e.slices)),
    ]
    picks = [c[1][0] for c in cats if c[1]]
    if sample == "all":
        picks = sorted(k for k, (_, _, e) in want.items() if e.blob)
    log(
        "  sampled %d entries: %s"
        % (
            len(picks),
            "all packed entries"
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
            checks.append(("matches reconstruction", bool(torch.equal(prod_t, ref))))
            if M.needs_vperm(e):
                checks.append(
                    ("same source-row multiset", rows_hash(p) == rows_hash(src))
                )
            else:
                checks.append(("byte-identical to GGUF source", np.array_equal(p, src)))
        else:
            # Independently compare BF16 bits against NumPy RNE conversion.
            checks.append(
                (
                    "BF16 bits match NumPy RNE",
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
                ", ".join("%s=%s" % (n, "Y" if v else "N") for n, v in checks),
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
    ap.add_argument("--gguf", required=True, help="source GGUF file")
    ap.add_argument("--out", required=True, help="output checkpoint directory")
    ap.add_argument(
        "--tokenizer-dir",
        default="",
        help="optional directory with auxiliary tokenizer files",
    )
    ap.add_argument(
        "--dense-iq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="convert IQ4_NL/IQ4_XS tensors to dense BF16",
    )
    ap.add_argument(
        "--dense-embed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="convert embedding and output head to dense BF16",
    )
    ap.add_argument(
        "--vperm",
        choices=("inv", "fwd", "none"),
        default="inv",
        help="value-head permutation direction",
    )
    ap.add_argument(
        "--emit-dense-ref",
        metavar="PATH",
        default=None,
        help="also emit a fully dequantized BF16 reference checkpoint",
    )
    ap.add_argument("--max-shard-gib", type=float, default=4.0)
    ap.add_argument(
        "--chunk-rows", type=int, default=8192, help="rows per dequantization chunk"
    )
    ap.add_argument(
        "--layers",
        type=int,
        default=None,
        help="convert only the first N layers and update num_hidden_layers",
    )
    ap.add_argument("--verify", choices=("off", "sample", "all"), default="sample")
    ap.add_argument(
        "--skip-pack",
        action="store_true",
        help="reuse existing weights while refreshing config, tokenizer, and verification",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate orientation, shapes, and packed byte sizes without writing output",
    )
    a = ap.parse_args()

    if not a.dense_embed:
        raise SystemExit(
            "--no-dense-embed requires native packed embedding and lm_head kernels"
        )

    t0 = time.time()
    log("Reading GGUF metadata: %s" % a.gguf)
    reader = GGUFReader(a.gguf)
    tensors = {t.name: t for t in reader.tensors}
    dims = dims_from_gguf(reader)
    check_dims(dims)
    log(
        "  dimensions match mapping profile: %d layers, hidden=%d, vocab=%d"
        % (dims.n_layers, dims.hidden, dims.vocab)
    )

    # Validate the full model profile before applying the optional layer limit,
    # then keep the emitted config consistent with the truncated checkpoint.
    if a.layers is not None:
        if not 0 < a.layers < dims.n_layers:
            raise SystemExit(
                "--layers must be in (0, %d), got %d" % (dims.n_layers, a.layers)
            )
        log(
            "  --layers %d: num_hidden_layers %d -> %d"
            % (a.layers, dims.n_layers, a.layers)
        )
        dims.n_layers = a.layers

    opt = type("Opt", (), {})()
    opt.vperm, opt.chunk_rows = a.vperm, a.chunk_rows
    opt.types = {n: TYPE_NAME[int(t.tensor_type)] for n, t in tensors.items()}
    plan = M.build_plan(dims)
    n_exc = M.apply_v1_exceptions(plan, opt.types, enabled=a.dense_iq)
    log("  mapping entries %d, dense IQ4 fallbacks %d" % (len(plan), n_exc))
    blob = [e for e in plan if e.blob]
    log(
        "  packed %d / dense %d / excluded MTP prefixes %s"
        % (len(blob), len(plan) - len(blob), M.DROP_PREFIXES)
    )

    if a.dry_run:
        log("\n== Dry run: validate orientation and byte sizes ==")
        blob_bytes = dense_bytes = 0
        for e in plan:
            t = tensors[e.gguf]
            n_out = (e.slices[0][1] - e.slices[0][0]) if e.slices else int(e.shape[0])
            if e.blob:
                bs, ts = blk_sizes(opt.types[e.gguf])
                rb = M.row_bytes(int(e.shape[1]), bs, ts)
                if int(t.data.shape[-1]) != rb:
                    raise ValueError(
                        "%s: source row bytes %d != expected %d"
                        % (e.gguf, int(t.data.shape[-1]), rb)
                    )
                if int(t.data.shape[0]) < n_out:
                    raise ValueError(
                        "%s: source has %d rows, entry requires %d"
                        % (e.gguf, t.data.shape[0], n_out)
                    )
                blob_bytes += n_out * rb
            else:
                n = prod(tuple(int(x) for x in e.shape))
                if not e.slices and prod(int(x) for x in t.shape) != n:
                    raise ValueError(
                        "%s: source element count %s != entry shape %s"
                        % (e.gguf, t.shape, tuple(e.shape))
                    )
                dense_bytes += n * 2
        log(
            "  PASS %d entries: packed %.3f GiB + dense BF16 %.3f GiB"
            " = expected output %.3f GiB"
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
        log("\n== --skip-pack: refresh config, tokenizer, and verification ==")
    else:
        log("\n== Writing %s ==" % a.out)
        for i, e in enumerate(plan):
            t = tensors[e.gguf]
            name, tens = build(e, t, dims, opt, False)
            w.add(name, tens)
            if (i + 1) % 100 == 0:
                log(
                    "  ... %d/%d entries (%.1f s)"
                    % (i + 1, len(plan), time.time() - t0)
                )
        w.finish()

    # Refresh semantic metadata even with --skip-pack. Derive rules from the
    # mapping so Python and C++ do not maintain duplicate definitions.
    rules = M.activation_vperm_rules(dims, plan)
    if a.vperm == "none":
        # Keep conversion, runtime, and dense-reference paths aligned.
        rules = []
    cfg = M.make_root_config(dims, ggml_types, rules)
    with open(os.path.join(a.out, "config.json"), "w") as fp:
        json.dump(cfg, fp, indent=1, sort_keys=True)
    log(
        "  config.json: %d ggml_types keys and %d activation V-head rules: %s"
        % (
            len(ggml_types),
            len(rules),
            " ".join(
                "%s=%dx%dx%d"
                % (r["suffix"], r["num_k_heads"], r["num_v_per_k"], r["head_dim"])
                for r in rules
            )
            or "none",
        )
    )

    fails = export_tokenizer(reader, a.out, a.tokenizer_dir, dims)

    if not a.skip_pack:
        with open(os.path.join(a.out, "pack_report.json"), "w") as fp:
            json.dump(
                {
                    "gguf": os.path.abspath(a.gguf),
                    # Tensor payload excludes file metadata and alignment padding.
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
        log("\n== Writing dense reference %s ==" % a.emit_dense_ref)
        os.makedirs(a.emit_dense_ref, exist_ok=True)
        wr = ShardWriter(a.emit_dense_ref, int(a.max_shard_gib * _GiB))
        for e in plan:
            name, tens = build(e, tensors[e.gguf], dims, opt, True)
            wr.add(name, tens)
        wr.finish()
        ref_cfg = M.make_root_config(
            dims, {M.type_table_key(k): "dense_bf16" for k in ggml_types}, rules
        )
        # The dense reference omits quantization_config and stores grouped
        # columns directly, matching the packed path's runtime semantics.
        del ref_cfg["quantization_config"]
        with open(os.path.join(a.emit_dense_ref, "config.json"), "w") as fp:
            json.dump(ref_cfg, fp, indent=1, sort_keys=True)
        export_tokenizer(reader, a.emit_dense_ref, a.tokenizer_dir, dims)

    log(
        "\n===== Complete: %.1f s, output %.3f GiB, verification failures %d ====="
        % (time.time() - t0, w.total / _GiB, fails)
    )
    return 1 if fails else 0


def export_tokenizer(reader, out_dir: str, tokenizer_dir: str, dims) -> int:
    """Export GGUF BPE vocabulary and copy optional tokenizer configuration."""
    tokens = [str(t) for t in X.gguf_meta(reader, "tokenizer.ggml.tokens")]
    merges = [str(m) for m in X.gguf_meta(reader, "tokenizer.ggml.merges")]
    model = str(X.gguf_meta(reader, "tokenizer.ggml.model")[0])
    if len(tokens) != dims.vocab:
        raise SystemExit(
            "GGUF vocabulary size %d != config vocab_size %d"
            % (len(tokens), dims.vocab)
        )
    if model != "gpt2":
        log(
            "  WARNING tokenizer.ggml.model=%r is not gpt2; verify vocab/merges export"
            % model
        )
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as fp:
        json.dump({t: i for i, t in enumerate(tokens)}, fp, ensure_ascii=False)
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as fp:
        fp.write("#version: 0.2\n" + "\n".join(merges) + "\n")
    log(
        "  tokenizer from GGUF: vocab %d / merges %d (model=%s)"
        % (len(tokens), len(merges), model)
    )

    have = (
        os.listdir(tokenizer_dir)
        if tokenizer_dir and os.path.isdir(tokenizer_dir)
        else []
    )
    if not have:
        log("  WARNING tokenizer config directory not found: %s" % tokenizer_dir)
    copied = []
    for f in TOKENIZER_FILES:
        dst = os.path.join(out_dir, f)
        if f in have and not os.path.exists(dst):
            shutil.copy2(os.path.join(tokenizer_dir, f), dst)
            copied.append(f)
    log(
        "  copied %d auxiliary tokenizer files: %s"
        % (len(copied), " ".join(sorted(copied)))
    )
    if "tokenizer_config.json" not in copied + have:
        raise SystemExit(
            "output is missing tokenizer_config.json; it was not available in %s"
            % tokenizer_dir
        )
    return check_tokenizer(out_dir, dims)


def check_tokenizer(out_dir: str, dims) -> int:
    """Load AutoTokenizer and verify an encode/decode round trip."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        log("  SKIP tokenizer verification: transformers is unavailable")
        return 0
    try:
        tk = AutoTokenizer.from_pretrained(out_dir)
        n, cls = len(tk), type(tk).__name__
        s = "Hello, world 27B"
        ids = tk.encode(s)
        ok = n == dims.vocab and tk.decode(ids) == s
        log(
            "  %s tokenizer %s vocab=%d round_trip=%s"
            % ("PASS" if ok else "FAIL", cls, n, tk.decode(ids) == s)
        )
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001
        log("  FAIL tokenizer load: %s: %s" % (type(exc).__name__, str(exc)[:200]))
        return 1


if __name__ == "__main__":
    sys.exit(main())
