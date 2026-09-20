"""Group graph-node kernels into replays by time gaps, report median
per-replay time+count per (kernel, grid), with analytic bytes/GBs for the
known Qwen3-0.6B GEMV call sites.

Usage: python3 step_breakdown5.py /root/prof_w1_06b_n.sqlite
"""
import sqlite3
import sys
from collections import Counter, defaultdict

db = sqlite3.connect(sys.argv[1])
rows = db.execute(
    "SELECT k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ "
    "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "JOIN StringIds s ON k.demangledName = s.id "
    "WHERE k.graphNodeId IS NOT NULL ORDER BY k.start"
).fetchall()
print(f"graph kernels: {len(rows)}")

GAP = 100_000  # 100us: inter-step gap is ~0.5ms+, intra-step gaps are ~us
replays, cur = [], []
for r in rows:
    if cur and r[0] - cur[-1][1] > GAP:
        replays.append(cur)
        cur = []
    cur.append(r)
if cur:
    replays.append(cur)

sizes = Counter(len(r) for r in replays)
print("replay sizes (kernels: count):", sizes.most_common(6))
mode_size = sizes.most_common(1)[0][0]
steady = [r for r in replays if len(r) == mode_size]
print(f"mode replay size: {mode_size}, steady replays: {len(steady)}")


def short(name):
    n = name
    if "internal::gemvx::kernel" in n:
        return "gemvx"
    if n.startswith("void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_"):
        return "wmma_" + n.split("gemm_bf16_")[1].split(">")[0]
    if "flashAttentionDecodeHd128SplitKvCta" in n:
        return "paged_splitkv_cta"
    if "flashAttentionDecodeHd128SplitKvCombine" in n:
        return "paged_splitkv_combine"
    if "DeviceReduceSingleTileKernel" in n:
        return "cub_reduce_single"
    if "DeviceReduceKernel" in n:
        return "cub_reduce"
    return n.split("<")[0].split("(")[0].replace("void ", "")[:56]


agg = defaultdict(lambda: [[], []])
walls = []
for rep in steady:
    walls.append(rep[-1][1] - rep[0][0])
    per = defaultdict(lambda: [0, 0])
    for start, end, name, gx, gy, gz in rep:
        key = f"{short(name)} g({gx},{gy},{gz})"
        per[key][0] += end - start
        per[key][1] += 1
    for k, (t, c) in per.items():
        agg[k][0].append(t)
        agg[k][1].append(c)

walls.sort()
med_wall = walls[len(walls) // 2]
items = []
for k, (ts, cs) in agg.items():
    ts.sort()
    cs.sort()
    items.append((ts[len(ts) // 2], cs[len(cs) // 2], k))
items.sort(reverse=True)
tot = sum(t for t, _, _ in items)
print(f"\nmedian step wall: {med_wall/1e3:.1f} us, GPU-busy: {tot/1e3:.1f} us, idle: {(med_wall-tot)/1e3:.1f} us ({100.0*(med_wall-tot)/med_wall:.0f}%)")
print(f"{'kernel @ grid':<56} {'us/step':>8} {'n/step':>6} {'us/call':>8} {'%':>5}")
for t, c, k in items[:22]:
    print(f"{k:<56} {t/1e3:>8.1f} {c:>6} {t/c/1e3:>8.2f} {100.0*t/tot:>4.1f}")
