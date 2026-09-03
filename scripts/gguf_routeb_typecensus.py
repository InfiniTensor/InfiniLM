import collections
import sys

sys.path.insert(0, "/home/liuxd/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGMLQuantizationType as QType  # noqa: E402

GGUF = "/home/liuxd/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf"
r = GGUFReader(GGUF)
T = {t.name: t for t in r.tensors}


def tn(name):
    return QType(int(T[name].tensor_type)).name


per = collections.defaultdict(collections.Counter)
for i in range(64):
    full = (i + 1) % 4 == 0
    names = (
        [
            f"blk.{i}.attn_q.weight",
            f"blk.{i}.attn_k.weight",
            f"blk.{i}.attn_v.weight",
            f"blk.{i}.attn_output.weight",
        ]
        if full
        else [
            f"blk.{i}.attn_qkv.weight",
            f"blk.{i}.attn_gate.weight",
            f"blk.{i}.ssm_out.weight",
        ]
    )
    names += [
        f"blk.{i}.ffn_gate.weight",
        f"blk.{i}.ffn_up.weight",
        f"blk.{i}.ffn_down.weight",
    ]
    for n in names:
        per[n.split(".")[2]][tn(n)] += 1

print("=== 按张量角色的类型分布（64 层）===")
for k, v in sorted(per.items()):
    print(f"  {k:14s}", dict(v))

print("=== full-attn 层内 q/k/v 类型是否一致（决定融合 blob 能否共用一块 buffer）===")
bad = [
    (i, [tn(f"blk.{i}.attn_{x}.weight") for x in ("q", "k", "v")])
    for i in range(3, 64, 4)
]
bad = [b for b in bad if len(set(b[1])) != 1]
print(f"  不一致层数 = {len(bad)}  样例 = {bad[:6]}")

print("=== ffn gate/up 类型是否一致（决定 GateUp 融合 blob 能否共用一块 buffer）===")
bad2 = [(i, [tn(f"blk.{i}.ffn_{x}.weight") for x in ("gate", "up")]) for i in range(64)]
bad2 = [b for b in bad2 if len(set(b[1])) != 1]
print(f"  不一致层数 = {len(bad2)}  样例 = {bad2[:6]}")

print("=== GDN 层 attn_qkv / attn_gate / ssm_out 抽样类型 ===")
for i in (0, 1, 2, 4, 62):
    print(
        "  ",
        i,
        {
            s: tn(f"blk.{i}.{s}.weight")
            for s in ("attn_qkv", "attn_gate", "ssm_out", "ffn_gate", "ffn_down")
        },
    )

print("=== 每个 Linear 的 (角色 -> 类型) 逐层矩阵，看同一角色跨层是否稳定 ===")
for role in (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
):
    c = collections.Counter()
    for i in range(64):
        n = f"blk.{i}.{role}.weight"
        if n in T:
            c[tn(n)] += 1
    print(f"  {role:12s}", dict(c))
