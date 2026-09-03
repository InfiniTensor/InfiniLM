#!/usr/bin/env python3
"""
InfiniLM 路线 B —— 阶段 0.3 shape 契约回归（执行方案 §4.2 第 3 条）

三方对账，任何一处对不上都在此暴露，而不是留到阶段 5 被 strict=False 静默丢权重：

  1. 框架侧：CPU 构造 mini qwen3_5 InferEngine，导出 C++ 真实参数键 + shape，
     与 gguf_mapping.build_plan(MINI) 做双向 diff（缺键 / 多键 / shape 不符即 FAIL）。
  2. GGUF 侧：build_plan(REAL) 的每条 gguf 名必须在真文件中存在，shape 必须与
     ne 反序一致（含 conv1d 的 squeeze），blob 条目的行字节必须能被块大小整除；
     共用同一源张量的条目（attn_qkv -> q|k|v）其 slices 必须无重叠地精确覆盖全行。
  3. 反向无遗漏：真文件中未被丢弃、又未被任何条目消费的张量 = 0。
  4. 阶段 3 作用域：统计 blob 实际用到的 GGML 类型集合，作为 kernel 必须覆盖的清单。

用法：
  source scripts/gguf_routeb_env.sh
  python3 scripts/gguf_routeb_shape_contract.py [--skip-min] [--engine-device cpu]
退出码 0 表示全部 PASS。
"""

from __future__ import annotations

import argparse
import collections
from math import prod
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get("LLAMA_CPP_DIR", "/home/liuxd/llama.cpp"), "gguf-py"))

import gguf_mapping as M                                                   # noqa: E402
from gguf import GGUFReader                                                # noqa: E402
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as Q     # noqa: E402

DEFAULT_GGUF = "/home/liuxd/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf"
TYPE_NAME = {int(v.value): str(v.name) for v in Q}

_PASS = 0
_FAIL = 0


def check(name, ok, detail=""):
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print("  PASS  %s" % name)
    else:
        _FAIL += 1
        print("  FAIL  %s%s" % (name, ("\n        %s" % detail) if detail else ""))
    return ok


def dims_from_text_config(tc):
    """config.json 的 text_config 段 -> Dims。打包器写出 config.json 后也用它自检。"""
    return M.Dims(
        hidden=tc["hidden_size"], n_q_heads=tc["num_attention_heads"],
        n_kv_heads=tc["num_key_value_heads"], head_dim=tc["head_dim"],
        ffn=tc["intermediate_size"], lin_k_heads=tc["linear_num_key_heads"],
        lin_v_heads=tc["linear_num_value_heads"], lin_k_dim=tc["linear_key_head_dim"],
        lin_v_dim=tc["linear_value_head_dim"], conv_kernel=tc["linear_conv_kernel_dim"],
        vocab=tc["vocab_size"], n_layers=tc["num_hidden_layers"],
        interval=tc["full_attention_interval"],
    )


def framework_side(engine_device):
    print("\n== 1. 框架侧：mini InferEngine vs build_plan(MINI) ==")
    import infinicore
    from infinilm.cache import StaticKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from gguf_routeb_probe_params import CFG

    check("探针 CFG 与 MINI 维度一致", dims_from_text_config(CFG["text_config"]) == M.MINI,
          "cfg=%s\n MINI=%s" % (dims_from_text_config(CFG["text_config"]), M.MINI))

    # 不能用 /tmp：开发机上只读，写不进去。缓存在 HOME 下，无需清理权限。
    tmp = os.path.join(os.environ.get("XDG_CACHE_HOME")
                       or os.path.expanduser("~/.cache"), "gguf_routeb_mini_cfg")
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump(CFG, f)
    eng = InferEngine(model_path=tmp, device=infinicore.device(engine_device, 0),
                      distributed_config=DistConfig(1),
                      cache_config=StaticKVCacheConfig(max_batch_size=1, max_cache_len=16))
    sd = eng.state_dict()[0]
    actual = {k: tuple(int(x) for x in sd[k].shape) for k in eng.state_dict_keyname()}
    print("  -> 引擎导出 %d 个参数（device=%s）" % (len(actual), engine_device))

    plan = M.build_plan(M.MINI)
    want = {}
    for e in plan:
        assert e.infinilm not in want, "映射表内重复键：%s" % e.infinilm
        want[e.infinilm] = M.compress(e.shape)
    check("映射表无重复键（%d 条）" % len(plan), len(want) == len(plan), "%d vs %d" % (len(want), len(plan)))

    missing = sorted(set(actual) - set(want))
    extra = sorted(set(want) - set(actual))
    check("无缺键（框架要但映射表未提供 -> 会保持随机初始化）", not missing, str(missing[:12]))
    check("无多键（映射表提供但框架无此参数 -> strict=False 下静默丢）", not extra, str(extra[:12]))

    bad = [(k, want[k], M.compress(actual[k])) for k in sorted(set(actual) & set(want))
           if M.compress(actual[k]) != want[k]]
    check("逐键 shape 全等（压缩长度为 1 的维后）", not bad, str(bad[:8]))


def dense_iq_bf16(plan, tensors, gguf_types, prod):
    """v1 被稠密化的那 5 个 IQ4 张量若改回 blob，可省下的显存字节数。"""
    return sum(prod(e.shape) * 2 - int(tensors[e.gguf].n_bytes) for e in plan
               if not e.blob and gguf_types.get(e.gguf) in M.V1_IQUANT_DENSE
               and e.gguf in tensors)


def gguf_side(path):
    print("\n== 2. GGUF 侧：build_plan(REAL) vs 真文件 ==")
    reader = GGUFReader(path)
    tensors = {t.name: t for t in reader.tensors}
    gguf_types = {n: TYPE_NAME.get(int(t.tensor_type), str(t.tensor_type))
                  for n, t in tensors.items()}
    plan = M.build_plan(M.REAL)
    n_exc = M.apply_v1_exceptions(plan, gguf_types)     # v1 稠密化 IQ4（阶段 6 取消）
    check("映射条目数 = %d" % len(plan), len(plan) == 947, str(len(plan)))
    check("v1 稠密化例外命中 5 个 IQ4 张量", n_exc == 5, str(n_exc))

    bad_name, bad_shape, bad_type, bad_rows = [], [], [], []
    ok_blob = 0
    type_hist = collections.defaultdict(collections.Counter)
    for e in plan:
        t = tensors.get(e.gguf)
        if t is None:
            bad_name.append(e.gguf)
            continue
        ne = tuple(int(x) for x in t.shape)          # GGML ne 序 = [in, out]
        hf = tuple(reversed(ne))                     # HF/InfiniLM 序 = [out, in]
        tn = TYPE_NAME.get(int(t.tensor_type), str(t.tensor_type))
        suffix = e.gguf.split(".", 2)[2] if e.gguf.startswith("blk.") else e.gguf
        type_hist[suffix][tn] += 1

        allowed = M.NATIVE_BLOB_TYPES if e.blob else M.DENSE_SRC_TYPES
        if tn not in allowed:
            bad_type.append("%s: %s 不在 %s" % (e.gguf, tn, allowed))
        # 共用源张量的条目只占一个行段，比对该段长度而非全量
        exp = M.compress(e.shape)
        got = M.compress(hf)
        if e.slices:
            s, ep = e.slices[0]
            got = (ep - s,) + got[1:]
        if exp != got:
            bad_shape.append("%s: 表 %s vs GGUF %s%s" % (e.gguf, exp, got,
                           "" if not e.slices else "（按行段 %s）" % (e.slices[0],)))
            continue
        if e.blob:
            blk, ts = (int(x) for x in GGML_QUANT_SIZES[Q[tn]])
            n_in = hf[-1]
            if n_in % blk:
                bad_rows.append("%s: in=%d 不能被块大小 %d 整除" % (e.gguf, n_in, blk))
            else:
                row_bytes = n_in // blk * ts
                if row_bytes * hf[0] != int(t.n_bytes):
                    bad_rows.append("%s: %d 行 x %d B != n_bytes %d"
                                    % (e.gguf, hf[0], row_bytes, t.n_bytes))
                else:
                    ok_blob += 1

    check("每条目的 GGUF 源张量都存在", not bad_name, str(sorted(set(bad_name))[:10]))
    check("源类型均在可实现集合内", not bad_type, str(bad_type[:6]))
    check("shape 与 ne 反序全等（含 conv1d squeeze）", not bad_shape, str(bad_shape[:8]))
    check("blob 条目行字节可整除且与 n_bytes 相符（%d 条）" % ok_blob, not bad_rows, str(bad_rows[:6]))

    print("\n== 3. 切片覆盖 + 反向无遗漏 ==")
    shared = collections.defaultdict(list)
    for e in plan:
        shared[e.gguf].append(e)
    cov = []
    for name, es in shared.items():
        if name not in tensors:
            continue
        n_out = int(tuple(reversed(tensors[name].shape))[0])
        segs = sorted((s, ep) for e in es for s, ep in e.slices)
        if len(es) == 1 and not segs:
            continue
        if not segs:
            cov.append("%s: %d 个条目共用但无 slices 声明" % (name, len(es)))
        elif segs[0][0] != 0 or segs[-1][1] != n_out or \
                any(segs[i][1] != segs[i + 1][0] for i in range(len(segs) - 1)):
            cov.append("%s: 切片 %s 未无重叠覆盖 [0,%d)" % (name, segs, n_out))
    check("共用源张量的切片精确覆盖全行", not cov, str(cov[:6]))

    used = {e.gguf for e in plan}
    dropped = {n for n in tensors if n.startswith(M.DROP_PREFIXES)}
    orphan = sorted(set(tensors) - used - dropped)
    check("无既未消费又未丢弃的张量", not orphan, str(orphan[:10]))
    print("  -> 消费 %d 个 / 丢弃 %d 个（MTP blk.%d.*）/ 文件共 %d 个"
          % (len(set(tensors) & used), len(dropped), M.MTP_BLOCK, len(tensors)))

    print("\n== 4. 阶段 3 kernel 作用域 ==")
    all_types = collections.Counter()
    for hist in type_hist.values():
        all_types.update(hist)
    print("  按条目统计：" + ", ".join("%s x%d" % (tn, c) for tn, c in all_types.most_common()))
    blob_types = {e.gguf: TYPE_NAME.get(int(tensors[e.gguf].tensor_type))
                  for e in plan if e.blob and e.gguf in tensors}
    seen = collections.Counter(blob_types.values())
    print("  blob 条目源类型：" + ", ".join("%s x%d" % (tn, c) for tn, c in seen.most_common()))
    check("阶段 3 v1 需实现的类型集合 = %s" % sorted(seen),
          set(seen) == set(M.NATIVE_BLOB_TYPES),
          "缺 %s / 多 %s" % (set(M.NATIVE_BLOB_TYPES) - set(seen), set(seen) - set(M.NATIVE_BLOB_TYPES)))
    check("IQ4_* 已被 v1 稠密化例外排除", not ({"IQ4_NL", "IQ4_XS"} & set(seen)), str(sorted(seen)))
    giB = 2 ** 30
    total = sum(int(t.n_bytes) for t in reader.tensors)
    blob_src = {e.gguf for e in plan if e.blob and e.gguf in tensors}
    dense_src = {e.gguf for e in plan if not e.blob and e.gguf in tensors} - blob_src
    b = sum(int(tensors[n].n_bytes) for n in blob_src)
    d_src = sum(int(tensors[n].n_bytes) for n in dense_src)
    drop = sum(int(t.n_bytes) for n, t in tensors.items() if n in dropped)
    print("  -> 文件 %.3f GiB = 逐字节 blob %.3f（%d 个） + 稠密化源 %.3f（%d 个）"
          " + MTP 丢弃 %.3f" % (total / giB, b / giB, len(blob_src),
                                 d_src / giB, len(dense_src), drop / giB))
    # 稠密化条目的显存 = InfiniLM 元素数 x 2B（按行段拆分的条目只算自己那段）
    dense_bf16 = sum(prod(e.shape) * 2 for e in plan if not e.blob)
    budget = (b + dense_bf16) / giB
    print("  -> v1 显存预算：blob %.3f + 稠密化 BF16 %.3f = %.3f GiB"
          % (b / giB, dense_bf16 / giB, budget))
    check("v1 权重预算 <= 24.0 GiB（单卡 5090 32.6 GiB 留 KV 余量）",
          budget <= 24.0, "%.3f GiB" % budget)
    # 阶段 6 复利：IQ4 上原生 kernel 后再省；emb/lm_head 上 kernel 再省 2.51 GiB
    st6 = budget - dense_iq_bf16(plan, tensors, gguf_types, prod) / giB
    emb_out_blob = int(tensors["token_embd.weight"].n_bytes) + int(tensors["output.weight"].n_bytes)
    emb_out_bf16 = sum(prod(e.shape) * 2 for e in plan
                       if not e.blob and e.gguf in ("token_embd.weight", "output.weight"))
    st6b = st6 - (emb_out_bf16 - emb_out_blob) / giB
    print("  -> 阶段 6：IQ4 原生 kernel %.3f GiB；再 emb/lm_head 原生 %.3f GiB"
          % (st6, st6b))
    check("阶段 6 预算单调下降", st6b < st6 < budget, "%.3f / %.3f / %.3f" % (st6b, st6, budget))
    check("阶段 6 目标态 <= 20.5 GiB（相对路线 A 的 26.6 GiB 权重）", st6b <= 20.5,
          "%.3f GiB" % st6b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--skip-min", action="store_true", help="跳过需要 infinilm 的框架侧检查")
    ap.add_argument("--engine-device", default="cpu")
    a = ap.parse_args()

    if not a.skip_min:
        framework_side(a.engine_device)
    gguf_side(a.gguf)

    print("\n== 结果：%d PASS / %d FAIL ==" % (_PASS, _FAIL))
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
