#!/usr/bin/env python3
"""
InfiniLM 路线 B —— 阶段 3.2 + 3.3 验收：linear_gguf 两条 NVIDIA 路径的数值正确性

被测对象（两条路径都由算子本体所在的头文件提供，probe 直接 include）：
  * `InfiniCore/src/infiniop/ops/linear_gguf/nvidia/linear_gguf_gemv.cuh`
    —— M <= kMaxDecodeM 的 decode 路径（一 warp 一行、寄存器内解码 + fp32 累加）；
  * `InfiniCore/src/infiniop/ops/linear_gguf/nvidia/linear_gguf_dequant.cuh`
    —— M > kMaxDecodeM 的 prefill 路径（64 行权重解码到 BF16 scratch + cublasGemmEx）。
`scripts/gguf_routeb_gemv_probe.cu` 里的路由谓词与算子 `calculate` 用的是同一个
`kMaxDecodeM`，所以每条用例走的真是发布路径上那条 kernel；probe 还会在 stdout 报
`path=gemv|prefill`，脚本据此**断言路由本身**（见下面的“路径”判据）。

判据不新造：
  * 主判据 = 方案 §1.2 第 2 条「GEMM 输出与稠密 BF16 权重 @ x 的 cos_sim > 0.999」。
    这里的“稠密权重”用的是 3.1 已证与 gguf-py / 头逐位相同的 numpy 参考（`REF`），
    所以这条判据同时就把「解码正确」与「GEMV / prefill 正确」两件事串在了一起。
    prefill 路径把权重先舍到 BF16 再乘，与这条基准口径一致。
  * 权重字节全部取自真实打包产物的 `*.weight_bytes` **整行**（不是随机 block），
    因为 kernel 依赖“一行 = 整数个 block”这个契约，随机 block 拼不出来。每种类型
    取首/中/尾三个张量（跨层），避开“两份产物挑到同一层同一张量”的假独立性。
  * 行数默认 200（不是 64 的整数倍），这样 prefill 的 tile 循环会走到
    “最后一片不满”的分支。
  * 附带两条拒绝（必须报错、不许静默出结果）：未知 type、K 不是 block 元素数整数倍。
    （原来那条「M=9 超过 decode 上限必须被拒」在 3.3 之后不再成立，M=9 现在既是
    prefill 的下边界、又是一条正例，见 --ms 默认值。）

累加顺序与 numpy 不同（gemv：块内顺序求和 -> 沿 block 累加 -> warp shuffle 归约；
prefill：cublas 分块），所以这里**不要求逐位相同**，而是把逐位相同率当作观测量报出来，
cos_sim 当判据。

用法：
  /usr/bin/python3 scripts/gguf_routeb_gemv_check.py \
      [--model-path /home/liuxd/models/Qwen3.8-27B-GGUF-native-mini8] \
      [--rows 200] [--ms 1,8,9,16,32,64,256,1024] [--keep]
退出码 0 = 全部 PASS。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import gguf_routeb_blocks_ref as bref                             # noqa: E402
from gguf_routeb_blocks_ref import (Artifact, BLOCK_SIZE, REF,    # noqa: E402
                                    TYPE_SIZE, TYPES, check, skip)

GEMV_DIR = os.path.join(bref.HEADER_DIR, "nvidia")
PROBE_SRC = os.path.join(_HERE, "gguf_routeb_gemv_probe.cu")
MAX_M = 8                    # kMaxDecodeM：M <= 8 走 gemv，M > 8 走 prefill
PREFILL_MS = "9,16,32,64,256,1024"      # 9 = prefill 下边界（§1.2 第 3 条含 16/32/64/256/1024）
PATH_RE = re.compile(r"path=(\w+)")
T_NAME = {8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}


def reset_counters():
    bref._PASS = bref._FAIL = bref._SKIP = 0


# --------------------------------------------------------------- bf16 位模式
def bf16_to_f32(bits):
    return (np.asarray(bits, np.uint16).astype(np.uint32)
            << np.uint32(16)).view(np.float32)


def f32_to_bf16(x):
    return bref.float_to_bf16_bits(x)


def cos_sim(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float(np.array_equal(a, b))
    return float(a @ b / (na * nb))


# ------------------------------------------------------------- 真实权重整行
def pick_rows(art, t, want_rows, rng, which=0):
    """从类型 t 的某个真实张量里取连续若干行字节（which 指定取哪个）。"""
    names = art.type_names(t)
    if not names:
        return None
    ts = TYPE_SIZE[t]
    name = names[min(which, len(names) - 1)]
    _t, shard, base, row_bytes, nrows = art.blobs[name]
    blocks_per_row = row_bytes // ts
    if blocks_per_row * ts != row_bytes or blocks_per_row < 1:
        raise RuntimeError("%s 的 row_bytes=%d 不是 block_size %d 的整数倍"
                           % (name, row_bytes, ts))
    rows = min(want_rows, nrows)
    r0 = int(rng.integers(0, nrows - rows + 1))
    with open(shard, "rb") as fh:
        fh.seek(base + r0 * row_bytes)
        buf = np.frombuffer(fh.read(rows * row_bytes), np.uint8)
    W = buf.reshape(rows, row_bytes).copy()
    return name, W, blocks_per_row * BLOCK_SIZE[t], r0


def dense_weights(t, W, rows, K):
    """numpy 参考反量化：W[rows, row_bytes] -> float32 [rows, K]。"""
    ts, bs = TYPE_SIZE[t], BLOCK_SIZE[t]
    blocks = W.reshape(-1, ts)
    dec = REF[t](blocks)                        # (n_blocks, bs) float32，3.1 已证逐位正确
    return dec.reshape(rows, K)


# ------------------------------------------------------------------ 驱动调用
# probe 一个可执行文件覆盖两条路径（名字沿用 3.2），具体走哪条由它内部的
# m > kMaxDecodeM 谓词决定，并由 stdout 的 path= 字段报回来。
def run_gemv(binary, t, A_bf16, W, K, workdir, tag):
    m, _ = A_bf16.shape
    n, row_bytes = W.shape
    abin = os.path.join(workdir, "%s_m%d_t%d.a" % (tag, m, t))
    wbin = os.path.join(workdir, "%s_m%d_t%d.w" % (tag, m, t))
    cbin = os.path.join(workdir, "%s_m%d_t%d.c" % (tag, m, t))
    A_bf16.astype(np.uint16).tofile(abin)
    np.ascontiguousarray(W).tofile(wbin)
    cmd = [binary, str(t), str(m), str(n), str(K), str(row_bytes), abin, wbin, cbin]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p, cbin


def check_type(binary, art, t, rows, Ms, rng, workdir, which_list):
    """对同一类型的多个张量（刻意跨层）各跑一轮。

    只取排序后第一个张量会在两份产物上挑到同一个张量（字节完全相同），那
    时候选产物就只是“两种键形态”而不是两份独立权重证据，所以这里固定取
    首/中/尾三个（不足则去重）。注意 mini8 是完整模型的前若干层，layer 0/1
    的张量在两份产物里字节相同，取样点落在这些层时仍然撞——这是数据的性质，
    不是取样能修的（Q4_K 尤其：全模型只有 4 个张量且都在 layer 1）。
    """
    names = art.type_names(t)
    picks = sorted({min(w, len(names) - 1) for w in which_list})
    done = set()
    for wi in picks:
        picked = pick_rows(art, t, rows, rng, wi)
        if picked is None:
            skip("%d 张量 #%d" % (t, wi), "产物不含该类型")
            continue
        name, W, K, r0 = picked
        if name in done:
            continue
        done.add(name)
        check_type_one(binary, art, t, name, W, K, r0, Ms, rng, workdir)


def check_type_one(binary, art, t, name, W, K, r0, Ms, rng, workdir):
    n = W.shape[0]
    Wf32 = dense_weights(t, W, n, K)
    # 稠密 BF16 权重 @ x 这条基准：先把反量化结果舍到 bf16 再算，同 §1.2 第 2 条口径
    Wdense = bf16_to_f32(f32_to_bf16(Wf32))
    print("  %s：%s（起始行 %d），%d 行 x K=%d，row_bytes=%d"
          % (T_NAME.get(t, t), name, r0, n, K, W.shape[1]))
    for m in Ms:
        A = (rng.standard_normal((m, K)) * 0.5).astype(np.float32)
        Abits = f32_to_bf16(A)
        Af = bf16_to_f32(Abits)                          # kernel 看到的就是这份值
        p, cbin = run_gemv(binary, t, Abits, W, K, workdir, "gemv")
        if not check("%s M=%d：kernel 退出码 0" % (T_NAME.get(t, t), m), p.returncode == 0,
                     "rc=%d %s" % (p.returncode, (p.stderr or p.stdout).strip()[-600:])):
            continue
        # 路由判据：probe 报的 path 必须等于算子在该 M 上会选的路径。数值过了但
        # 路走错了同样不可接受（那意味着门测的不是发布路径）。
        want_path = "gemv" if m <= MAX_M else "prefill"
        pm = PATH_RE.search(p.stdout or "")
        got_path = pm.group(1) if pm else "?"
        check("%s M=%d：走 %s 路径（与算子 calculate 的谓词一致）" % (T_NAME.get(t, t), m, want_path),
              got_path == want_path, "probe 报 path=%s" % got_path)
        got = bf16_to_f32(np.fromfile(cbin, np.uint16).reshape(m, n))
        assert got.shape == (m, n)
        ref = (Af @ Wdense.T).astype(np.float32)         # §1.2 第 2 条口径的基准
        ref_exact = (Af @ Wf32.T).astype(np.float32)     # 不先把权重舍到 bf16
        c = cos_sim(got, ref)
        check("%s M=%d：cos_sim(kernel, 稠密 BF16 权重 @ x) > 0.999" % (T_NAME.get(t, t), m),
              c > 0.999, "cos_sim=%.8f" % c)
        # 观测量（不作判据）：bf16 位相同率、最大绝对/相对偏差、vs 未舍入基准的 cos_sim
        same = float(np.mean(f32_to_bf16(got) == f32_to_bf16(ref)))
        dg = got.astype(np.float64) - ref.astype(np.float64)
        absd = float(np.max(np.abs(dg)))
        # 相对偏差只在“有意义的元素”上算（|ref| >= 最大幅值的 1%），否则会被近零
        # 元素除出几十倍的假大数，那种数字没有判读价值。
        sig = np.abs(ref.astype(np.float64)) >= 0.01 * float(np.max(np.abs(ref)))
        rel = float(np.max(np.abs(dg[sig]) / np.abs(ref.astype(np.float64)[sig]))) if sig.any() else 0.0
        print("        观测：cos_sim(kernel, 稠密 BF16 权重)=%.10f"
              "  cos_sim(kernel, 未舍入基准)=%.10f  bf16 逐位相同率=%.4f"
              "  max|Δ|=%.3e  max 相对偏差(|ref|≥最大幅值1%% 的子集)=%.3e  %s"
              % (c, cos_sim(got, ref_exact), same, absd, rel,
                 p.stdout.strip().split("ok")[-1].strip()))


def check_rejections(binary, art, workdir):
    """必须报错的输入：不许静默出结果。

    3.3 之前这里还有一条「M=9 超过 decode 上限被拒」，现在 prefill 接管了 M>8，
    该用例已反转成 --ms 里的正例（prefill 下边界）。
    """
    name_ok = None
    for t in TYPES:
        picked = pick_rows(art, t, 4, np.random.default_rng(7))
        if picked:
            name_ok, W, K = t, picked[1], picked[2]
            break
    A = np.zeros((1, K), np.float32)
    Abits = f32_to_bf16(A)

    p, _ = run_gemv(binary, 10, Abits, W, K, workdir, "rej")
    check("未知 ggml type 10 被拒（rc=3，不启动 kernel）", p.returncode == 3,
          "rc=%d %s" % (p.returncode, p.stderr.strip()[-300:]))

    bad_k = K + (BLOCK_SIZE[name_ok] - 1)                # 不再是整数个 block
    A_bad = f32_to_bf16(np.zeros((1, bad_k), np.float32))
    p, _ = run_gemv(binary, name_ok, A_bad, W, bad_k, workdir, "rej")
    check("K 不是 block 元素数整数倍被拒（rc=3）", p.returncode == 3,
          "rc=%d %s" % (p.returncode, p.stderr.strip()[-300:]))

    # 同一条约束在 prefill 路径上也必须成立（两条路径各自有谓词，不能只查 gemv）
    A_bad_p = f32_to_bf16(np.zeros((MAX_M + 1, bad_k), np.float32))
    p, _ = run_gemv(binary, name_ok, A_bad_p, W, bad_k, workdir, "rej")
    check("prefill 路径同样拒掉不整除的 K（rc=3）", p.returncode == 3,
          "rc=%d %s" % (p.returncode, p.stderr.strip()[-300:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/liuxd/models/Qwen3.8-27B-GGUF-native-mini8")
    ap.add_argument("--rows", type=int, default=200, help="每个张量取多少行权重（不是 64 的整数倍才能盖住 tile 余数）")
    ap.add_argument("--ms", default="1,8," + PREFILL_MS,
                    help="逗号分隔；<=8 走 gemv，>8 走 prefill（两条路径同一份门）")
    ap.add_argument("--workdir", default="/home/liuxd/tmp_routeb/gemv32")
    ap.add_argument("--nvcc", default=os.environ.get("CUDACXX", "nvcc"))
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--no-reject", action="store_true")
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args()

    reset_counters()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.workdir, exist_ok=True)
    Ms = [int(x) for x in args.ms.split(",") if x.strip()]
    print("产物：%s\n被测：\n  %s\n  %s\n  %s\n临时目录：%s\n每种类型权重行数：%d，M 取 %s"
          % (args.model_path,
             os.path.join(GEMV_DIR, "linear_gguf_gemv.cuh"),
             os.path.join(GEMV_DIR, "linear_gguf_dequant.cuh"),
             PROBE_SRC, args.workdir, args.rows, Ms))

    binary = os.path.join(args.workdir, "gemv_probe")
    print("\n[1] 编译两条路径的驱动（prefill 需要 -lcublas）")
    if args.skip_build:
        skip("编译", "--skip-build")
    else:
        try:
            bref.build_probe(PROBE_SRC, binary, args.nvcc,
                             extra=["-I", GEMV_DIR, "-lcublas"])
            check("nvcc 编译 %s 通过（含两个 kernel 头 + cublas）"
                  % os.path.basename(PROBE_SRC), True)
        except Exception as exc:                           # noqa: BLE001
            check("nvcc 编译 %s 通过" % os.path.basename(PROBE_SRC), False, str(exc)[-2000:])
            return 1

    art = Artifact(args.model_path)
    print("\n[2] 真实权重对 numpy 稠密基准（gemv + prefill，判据：cos_sim > 0.999）")
    print("产物 blob 张量 %d 个（key_prefix=%r），按类型：%s"
          % (len(art.blobs), art.prefix, {t: len(art.type_names(t)) for t in TYPES}))
    for t in TYPES:
        n_avail = len(art.type_names(t))
        # 张量本来就少（Q4_K 全模型只有 4 个，且都在 layer 1）时全取，否则首/中/尾
        which = list(range(n_avail)) if n_avail <= 6 else [0, n_avail // 2, n_avail - 1]
        which = which or [0]
        check_type(binary, art, t, args.rows, Ms, rng, args.workdir, which)

    if args.no_reject:
        skip("非法输入用例", "--no-reject")
    else:
        print("\n[3] 非法输入必须被拒（不许静默出结果）")
        check_rejections(binary, art, args.workdir)

    print("\n== 结果：%d PASS / %d FAIL / %d SKIP ==" % (bref._PASS, bref._FAIL, bref._SKIP))
    print("临时目录：%s" % args.workdir)
    return 0 if bref._FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
