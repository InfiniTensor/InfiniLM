#!/usr/bin/env python3
"""Pure NumPy transforms shared by the GGUF-to-InfiniLM converter.

Value-head ordering follows llama.cpp's Qwen conversion:
    HF / InfiniLM = grouped [key][value]
    GGUF          = tiled   [value][key]

Therefore ``reorder_v`` converts grouped to tiled order and
``reorder_v_inverse`` converts tiled to grouped order.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Value-head permutation
# ---------------------------------------------------------------------------


def reorder_v(t: np.ndarray, n_k: int, n_v_per_k: int, hd: int) -> np.ndarray:
    """Convert grouped to tiled order along dimension 0.

    Trailing dimensions are preserved, including 1-D scalars per head,
    2-D weight rows, and 3-D convolution weights.
    """
    rest = t.shape[1:]
    return (
        t.reshape((n_k, n_v_per_k, hd) + rest)
        .transpose((1, 0, 2) + tuple(range(3, 3 + len(rest))))
        .reshape((n_k * n_v_per_k * hd,) + rest)
    )


def reorder_v_inverse(t: np.ndarray, n_k: int, n_v_per_k: int, hd: int) -> np.ndarray:
    """Convert tiled to grouped order along dimension 0."""
    rest = t.shape[1:]
    return (
        t.reshape((n_v_per_k, n_k, hd) + rest)
        .transpose((1, 0, 2) + tuple(range(3, 3 + len(rest))))
        .reshape((n_k * n_v_per_k * hd,) + rest)
    )


_VPERM = {"inv": reorder_v_inverse, "fwd": reorder_v, "none": None}


def vperm_head_dim(e, dims) -> int:
    """Return the number of elements per value head for a mapping entry.

    Derive the value from shape rather than tensor names so the converter does
    not need a second name table.
    """
    n_heads = dims.lin_v_heads
    rows = int(e.shape[0]) if e.vperm == "all" else dims.value_dim
    if rows % n_heads:
        raise ValueError(
            "%s: scope rows %d are not divisible by %d value heads"
            % (e.infinilm, rows, n_heads)
        )
    return rows // n_heads


def apply_vperm(arr: np.ndarray, e, dims, direction: str = "inv") -> np.ndarray:
    """Permute value heads along dimension 0 within an all or v_tail scope."""
    fn = _VPERM[direction]
    if fn is None:
        return arr
    n_k, hd = dims.lin_k_heads, vperm_head_dim(e, dims)
    v_per_k = dims.lin_v_heads // n_k
    if int(dims.lin_v_heads) % n_k:
        raise ValueError(
            "lin_v_heads %d is not divisible by lin_k_heads %d"
            % (dims.lin_v_heads, n_k)
        )
    if e.vperm == "v_tail":
        n_v = n_k * v_per_k * hd
        if arr.shape[0] < n_v:
            raise ValueError(
                "%s: dimension 0 size %d is smaller than value segment %d"
                % (e.infinilm, arr.shape[0], n_v)
            )
        out = np.asarray(arr, dtype=arr.dtype)
        return np.concatenate([out[:-n_v], fn(out[-n_v:], n_k, v_per_k, hd)], axis=0)
    return fn(np.asarray(arr, dtype=arr.dtype), n_k, v_per_k, hd)


# ---------------------------------------------------------------------------
# Other transforms
# ---------------------------------------------------------------------------


def alog_from_ssm_a(a: np.ndarray) -> np.ndarray:
    """Recover the HF ``A_log`` convention from GGUF ``-exp(A_log)`` values."""
    a = np.asarray(a, dtype=np.float32)
    if not np.all(a < 0):
        raise ValueError(
            "ssm_a contains a non-negative value (min=%g); cannot compute log(-x). "
            "Check the A_log convention in conversion/qwen.py." % float(a.min())
        )
    return np.log(-a)


def gguf_meta(reader, suffix: str):
    """Read metadata with architecture/general prefixes and return a list."""
    for key in ("qwen35.%s" % suffix, "general.%s" % suffix, suffix):
        if key in reader.fields:
            v = reader.fields[key].contents()
            return v if isinstance(v, (list, tuple, np.ndarray)) else [v]
    raise KeyError(
        "missing GGUF metadata %s (no qwen35/general/unprefixed match)" % suffix
    )


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """Return round-to-nearest-even bfloat16 bit patterns as uint16.

    Keep arithmetic in uint32 to avoid doubling memory for large tensors. An
    overflow beyond bit 31 cannot affect the retained bfloat16 bits.
    """
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u + bias) >> np.uint32(16)).astype(np.uint16)
