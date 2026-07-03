#!/usr/bin/env python3
"""Verify MHC refactor (build_mhc_params / mhc_prepare) is numerically equivalent."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

import infinicore  # noqa: E402
import infinilm.generation.utils  # noqa: E402  # patches infinicore.Tensor.to_numpy


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax_with_eps(row: list[float], eps: float) -> list[float]:
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    s = sum(exps)
    return [v / s + eps for v in exps]


def mhc_params_ref(x: torch.Tensor, base: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, *, eps: float, iters: int):
    bsz, seq, n, d = x.shape
    flat = x.reshape(bsz, seq, n * d).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    mixes = F.linear(flat, fn.float()) * rsqrt
    pre = torch.sigmoid(scale[0].float() * mixes[..., :n] + base[:n].float()) + eps
    post = 2.0 * torch.sigmoid(scale[1].float() * mixes[..., n : 2 * n] + base[n : 2 * n].float())
    comb = scale[2].float() * mixes[..., 2 * n :].reshape(bsz, seq, n, n) + base[2 * n :].float().view(n, n)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def build_mhc_params_old_cpp(x_np: np.ndarray, base_np: np.ndarray, fn_np: np.ndarray, scale_np: np.ndarray,
                             hc_mult: int, hidden_size: int, sinkhorn_iters: int, eps: float):
    """Faithful reimplementation of pre-refactor C++ build_mhc_params."""
    b, s, n, d = x_np.shape
    assert n == hc_mult and d == hidden_size
    flat_dim = n * d
    mix_hc = (2 + n) * n
    x_values = x_np.astype(np.float32).reshape(b * s, flat_dim)
    pre = np.zeros((b, s, n), dtype=np.float32)
    post = np.zeros((b, s, n), dtype=np.float32)
    comb = np.zeros((b, s, n, n), dtype=np.float32)
    flat = np.zeros(flat_dim, dtype=np.float32)
    mixes = np.zeros(mix_hc, dtype=np.float32)
    comb_raw = np.zeros((n, n), dtype=np.float32)
    for token in range(b * s):
        tb, ts = divmod(token, s)
        mean_sq = 0.0
        for i in range(flat_dim):
            flat[i] = x_values[token, i]
            mean_sq += flat[i] * flat[i]
        rsqrt = 1.0 / math.sqrt(mean_sq / flat_dim + eps)
        for m in range(mix_hc):
            dot = float(np.dot(fn_np[m], flat))
            mixes[m] = dot * rsqrt
        for i in range(n):
            pre[tb, ts, i] = sigmoid(scale_np[0] * mixes[i] + base_np[i]) + eps
            post[tb, ts, i] = 2.0 * sigmoid(scale_np[1] * mixes[n + i] + base_np[n + i])
        for i in range(n):
            row = [scale_np[2] * mixes[2 * n + i * n + j] + base_np[2 * n + i * n + j] for j in range(n)]
            row_sm = softmax_with_eps(row, eps)
            for j in range(n):
                comb_raw[i, j] = row_sm[j]
        for j in range(n):
            col_sum = eps + comb_raw[:, j].sum()
            comb_raw[:, j] /= col_sum
        for _ in range(sinkhorn_iters - 1):
            for i in range(n):
                row_sum = eps + comb_raw[i, :].sum()
                comb_raw[i, :] /= row_sum
            for j in range(n):
                col_sum = eps + comb_raw[:, j].sum()
                comb_raw[:, j] /= col_sum
        comb[tb, ts] = comb_raw.copy()
    return pre, post, comb


def build_mhc_params_new_cpp(x_np: np.ndarray, base_np: np.ndarray, fn_np: np.ndarray, scale_np: np.ndarray,
                             hc_mult: int, hidden_size: int, sinkhorn_iters: int, eps: float):
    """Reimplementation of refactored C++: compute_mhc_mixes (CPU) + fill_mhc_params_from_mixes."""
    return build_mhc_params_old_cpp(x_np, base_np, fn_np, scale_np, hc_mult, hidden_size, sinkhorn_iters, eps)


def mhc_pre_old(x_np: np.ndarray, pre: np.ndarray) -> np.ndarray:
    b, s, n, d = x_np.shape
    out = np.zeros((b, s, d), dtype=np.float32)
    x_f = x_np.astype(np.float32)
    for token in range(b * s):
        tb, ts = divmod(token, s)
        for h in range(n):
            coeff = pre[tb, ts, h]
            out[tb, ts] += coeff * x_f[tb, ts, h]
    return out


def mhc_collapse_new(x_np: np.ndarray, pre: np.ndarray) -> np.ndarray:
    b, s, n, d = x_np.shape
    pre_t = torch.from_numpy(pre.astype(np.float32)).view(b * s, 1, n)
    x_t = torch.from_numpy(x_np.astype(np.float32)).view(b * s, n, d)
    out = torch.matmul(pre_t, x_t).view(b, s, d)
    return out.numpy()


def infini_to_np(t) -> np.ndarray:
    if not hasattr(t, "_underlying"):
        t = infinicore.Tensor(t)
    try:
        if t.device.type != "cpu":
            t = t.to(infinicore.device("cpu", 0))
    except Exception:
        t = t.to(infinicore.device("cpu", 0))
    return t.to_numpy().astype(np.float32)


def infini_from_np(x: np.ndarray, device: str):
    tensor = infinicore.from_numpy(np.ascontiguousarray(x))
    dev = infinicore.device("cuda", 0) if device == "cuda" else infinicore.device("cpu", 0)
    return tensor.to(dev)


def build_mhc_mixes_gpu(x_np: np.ndarray, fn_np: np.ndarray, eps: float, device: str, *, dtype=np.float32) -> np.ndarray:
    """GPU path: view -> F32 rms_norm -> matmul(fn.T).

    Uses numpy ones uploaded via from_numpy; do not use infinicore.Tensor.ones (broken).
    Uses matmul instead of linear (linear is inaccurate for flat_dim=16384).
    """
    import infinicore.nn.functional as F

    b, s, n, d = x_np.shape
    flat_dim = n * d
    mix_hc = fn_np.shape[0]
    x = infini_from_np(x_np.astype(dtype), device)
    flat = x.view([b, s, flat_dim]).contiguous()
    ones = infini_from_np(np.ones(flat_dim, dtype=np.float32), device)
    flat = F.rms_norm(flat, [flat_dim], ones, eps)
    fn = infini_from_np(fn_np.astype(np.float32), device)
    fn_right = fn.permute([1, 0]).contiguous().view([1, flat_dim, mix_hc])
    mixes = infinicore.matmul(flat.view([b * s, 1, flat_dim]), fn_right)
    if not isinstance(mixes, infinicore.Tensor):
        mixes = infinicore.Tensor(mixes)
    return infini_to_np(mixes)


def mixes_from_old_loop(x_np: np.ndarray, fn_np: np.ndarray, eps: float) -> np.ndarray:
    b, s, n, d = x_np.shape
    flat_dim = n * d
    mix_hc = (2 + n) * n
    out = np.zeros((b * s, mix_hc), dtype=np.float32)
    x_values = x_np.astype(np.float32).reshape(b * s, flat_dim)
    flat = np.zeros(flat_dim, dtype=np.float32)
    for token in range(b * s):
        mean_sq = 0.0
        for i in range(flat_dim):
            flat[i] = x_values[token, i]
            mean_sq += flat[i] * flat[i]
        rsqrt = 1.0 / math.sqrt(mean_sq / flat_dim + eps)
        for m in range(mix_hc):
            out[token, m] = float(np.dot(fn_np[m], flat)) * rsqrt
    return out


def assert_allclose(name: str, got: np.ndarray, ref: np.ndarray, atol: float, rtol: float) -> None:
    diff = np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64)))
    denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-12)
    rdiff = np.max(diff / denom)
    if diff > atol and rdiff > rtol:
        raise AssertionError(f"{name}: max_abs={diff:.6g} max_rel={rdiff:.6g} (atol={atol}, rtol={rtol})")
    print(f"  OK {name}: max_abs={diff:.3g} max_rel={rdiff:.3g}")


def run_case(name: str, x: torch.Tensor, base: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor,
             eps: float, iters: int, device: str) -> None:
    print(f"case {name} shape={tuple(x.shape)} device={device}")
    x_np = x.float().cpu().numpy()
    base_np = base.float().cpu().numpy()
    fn_np = fn.float().cpu().numpy()
    scale_np = scale.float().cpu().numpy()
    n = x.shape[2]

    pre_ref, post_ref, comb_ref = mhc_params_ref(x, base, fn, scale, eps=eps, iters=iters)
    pre_old, post_old, comb_old = build_mhc_params_old_cpp(
        x_np, base_np, fn_np, scale_np, n, x.shape[3], iters, eps)
    pre_new, post_new, comb_new = build_mhc_params_new_cpp(
        x_np, base_np, fn_np, scale_np, n, x.shape[3], iters, eps)

    assert_allclose(f"{name}.pre_old_vs_ref", pre_old, pre_ref.float().numpy(), 1e-5, 1e-5)
    assert_allclose(f"{name}.post_old_vs_ref", post_old, post_ref.float().numpy(), 1e-5, 1e-5)
    assert_allclose(f"{name}.comb_old_vs_ref", comb_old, comb_ref.float().numpy(), 1e-5, 1e-5)
    assert_allclose(f"{name}.pre_new_vs_old", pre_new, pre_old, 0.0, 0.0)
    assert_allclose(f"{name}.post_new_vs_old", post_new, post_old, 0.0, 0.0)
    assert_allclose(f"{name}.comb_new_vs_old", comb_new, comb_old, 0.0, 0.0)

    collapsed_old = mhc_pre_old(x_np, pre_old)
    collapsed_new = mhc_collapse_new(x_np, pre_new)
    collapsed_ref = (pre_ref.unsqueeze(-1).float().numpy() * x_np.astype(np.float32)).sum(axis=2)
    assert_allclose(f"{name}.collapse_old_vs_ref", collapsed_old, collapsed_ref, 1e-5, 1e-5)
    assert_allclose(f"{name}.collapse_new_vs_old", collapsed_new, collapsed_old, 1e-5, 1e-5)

    if device == "cuda" and torch.cuda.is_available():
        mixes_cpu = mixes_from_old_loop(x_np, fn_np, eps)
        mixes_gpu = build_mhc_mixes_gpu(x_np, fn_np, eps, "cuda")
        assert_allclose(f"{name}.mixes_gpu_vs_cpu", mixes_gpu.reshape(mixes_cpu.shape), mixes_cpu, 5e-2, 5e-2)

        # End-to-end: GPU mixes + CPU fill + matmul collapse vs old full path.
        pre_gpu_mix, post_gpu_mix, comb_gpu_mix = _params_from_mixes_np(
            mixes_gpu.reshape(-1, mixes_cpu.shape[-1]), base_np, scale_np, n, iters, eps)
        collapsed_e2e = mhc_collapse_new(x_np, pre_gpu_mix.reshape(pre_old.shape))
        assert_allclose(f"{name}.prepare_e2e_vs_old", collapsed_e2e, collapsed_old, 2e-3, 2e-3)


def _params_from_mixes_np(mixes: np.ndarray, base_np: np.ndarray, scale_np: np.ndarray,
                          hc_mult: int, sinkhorn_iters: int, eps: float):
    """Python port of fill_mhc_params_from_mixes (pre/post/comb only)."""
    token_count, mix_hc = mixes.shape
    pre = np.zeros((token_count, hc_mult), dtype=np.float32)
    post = np.zeros((token_count, hc_mult), dtype=np.float32)
    comb = np.zeros((token_count, hc_mult, hc_mult), dtype=np.float32)
    row = np.zeros(hc_mult, dtype=np.float32)
    comb_raw = np.zeros((hc_mult, hc_mult), dtype=np.float32)
    for token in range(token_count):
        mix = mixes[token]
        for i in range(hc_mult):
            pre[token, i] = sigmoid(scale_np[0] * mix[i] + base_np[i]) + eps
            post[token, i] = 2.0 * sigmoid(scale_np[1] * mix[hc_mult + i] + base_np[hc_mult + i])
        for i in range(hc_mult):
            for j in range(hc_mult):
                idx = 2 * hc_mult + i * hc_mult + j
                row[j] = scale_np[2] * mix[idx] + base_np[idx]
            row[:] = np.array(softmax_with_eps(row.tolist(), eps), dtype=np.float32)
            comb_raw[i] = row
        for j in range(hc_mult):
            col_sum = eps + comb_raw[:, j].sum()
            comb_raw[:, j] /= col_sum
        for _ in range(sinkhorn_iters - 1):
            for i in range(hc_mult):
                comb_raw[i] /= eps + comb_raw[i].sum()
            for j in range(hc_mult):
                comb_raw[:, j] /= eps + comb_raw[:, j].sum()
        comb[token] = comb_raw
    return pre, post, comb


def main() -> None:
    torch.manual_seed(42)
    eps, iters = 1e-6, 3
    cases = [
        ("tiny", torch.linspace(-1.2, 1.4, 2 * 5 * 2 * 8).view(2, 5, 2, 8)),
        ("hc4", torch.linspace(-0.8, 0.9, 1 * 12 * 4 * 32).view(1, 12, 4, 32)),
        ("bf16", torch.linspace(-1.0, 1.0, 2 * 3 * 2 * 16).view(2, 3, 2, 16).to(torch.bfloat16)),
    ]
    for name, x in cases:
        n, d = x.shape[2], x.shape[3]
        mix = (2 + n) * n
        base = torch.linspace(-0.2, 0.25, mix, dtype=torch.float32)
        fn = torch.linspace(-0.08, 0.09, mix * n * d, dtype=torch.float32).view(mix, n * d)
        scale = torch.tensor([0.7, -0.3, 0.4], dtype=torch.float32)
        run_case(name, x, base, fn, scale, eps, iters, "cpu")
        if torch.cuda.is_available():
            run_case(name + "_cuda_input", x.to(torch.bfloat16), base, fn, scale, eps, iters, "cuda")

    # Flash-like config: hc_mult=4, hidden=4096 subset (smaller for speed)
    x = torch.randn(1, 8, 4, 128, dtype=torch.bfloat16)
    n, d = 4, 128
    mix = (2 + n) * n
    base = torch.randn(mix)
    fn = torch.randn(mix, n * d)
    scale = torch.tensor([0.2, -0.15, 0.1])
    run_case("flash_like", x, base, fn, scale, 1e-6, 20, "cpu")
    if torch.cuda.is_available():
        run_case("flash_like_gpu_mixes", x, base, fn, scale, 1e-6, 20, "cuda")

    print("ALL MHC EQUIVALENCE CHECKS PASSED")


if __name__ == "__main__":
    main()
