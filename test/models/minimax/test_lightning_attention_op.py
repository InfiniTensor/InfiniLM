#!/usr/bin/env python3
"""
Direct unit test for the InfiniCore `lightning_attention` op (indexed pool mode).

Validates the op output and the in-place final state against an independent
numpy reference of the MiniMax-01 recurrence:
    S  = ratio[h] * S + k_t^T v_t       (ratio[h] = exp(-slope[h]))
    o_t = q_t @ S
"""
import ctypes
import sys

import numpy as np
import torch

import infinicore


CTYPE = {np.float32: ctypes.c_float, np.int32: ctypes.c_int32}


def t2raw(t: torch.Tensor, dev):
    # Python Tensor wrapper (the package-level op wrapper expects wrappers).
    t = t.contiguous()
    return infinicore.from_blob(t.data_ptr(), list(t.shape),
                                dtype={torch.float32: infinicore.float32,
                                       torch.int32: infinicore.int32}[t.dtype],
                                device=dev)


def read(t, dtype=np.float32):
    ctype = CTYPE[dtype]
    buf = (ctype * int(t.numel())).from_address(t.data_ptr())
    shape = [int(t.size(i)) for i in range(int(t.ndim))]
    return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()


def reference(q, k, v, slope, pool, init_idx, final_idx):
    B, T, H, D = q.shape
    out = np.zeros_like(q)
    pool = pool.copy()
    ratio = np.exp(-slope)  # [H]
    for b in range(B):
        S = pool[init_idx[b]].copy()  # [H, D, D]
        for t in range(T):
            for h in range(H):
                kh = k[b, t, h]      # [D]
                vh = v[b, t, h]      # [D]
                qh = q[b, t, h]      # [D]
                S[h] = ratio[h] * S[h] + np.outer(kh, vh)
                out[b, t, h] = qh @ S[h]
        pool[final_idx[b]] = S
    return out, pool


def run_case(B, T, H, D, pool_size, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, T, H, D)).astype(np.float32)
    k = rng.standard_normal((B, T, H, D)).astype(np.float32)
    v = rng.standard_normal((B, T, H, D)).astype(np.float32)
    slope = rng.uniform(0.0, 1.0, (H,)).astype(np.float32)
    pool = rng.standard_normal((pool_size, H, D, D)).astype(np.float32)
    init_idx = np.array([b % pool_size for b in range(B)], dtype=np.int32)
    final_idx = np.array([(b + 1) % pool_size for b in range(B)], dtype=np.int32)

    dev = infinicore.device("cpu")
    q_t = t2raw(torch.from_numpy(q), dev)
    k_t = t2raw(torch.from_numpy(k), dev)
    v_t = t2raw(torch.from_numpy(v), dev)
    s_t = t2raw(torch.from_numpy(slope), dev)
    p_t = t2raw(torch.from_numpy(pool), dev)
    i_t = t2raw(torch.from_numpy(init_idx), dev)
    f_t = t2raw(torch.from_numpy(final_idx), dev)

    out_w = infinicore.lightning_attention(q_t, k_t, v_t, s_t, p_t, i_t, f_t)

    out = read(out_w._underlying)
    pool_after = read(p_t._underlying)
    ref_out, ref_pool = reference(q, k, v, slope, pool, init_idx, final_idx)

    out_err = float(np.abs(out - ref_out).max())
    pool_err = float(np.abs(pool_after - ref_pool).max())
    print(f"  B={B} T={T} H={H} D={D} pool={pool_size}: out_err={out_err:.3e} pool_err={pool_err:.3e}")
    assert out_err < 1e-4, f"out mismatch {out_err}"
    assert pool_err < 1e-4, f"state mismatch {pool_err}"
    return True


def main():
    print("[1/2] decode case (T=1, batched requests, in-place state write)")
    run_case(B=3, T=1, H=4, D=8, pool_size=4, seed=1)
    print("[2/2] prefill case (T=6, per-request state evolution)")
    run_case(B=2, T=6, H=4, D=8, pool_size=4, seed=2)
    print("PASS: lightning_attention op matches reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())


