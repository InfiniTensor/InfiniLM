#!/usr/bin/env python3
"""
InfiniLM 路线 B —— 阶段 2 验收（执行方案 §6.3 判据 1–3）

拿 mini8 产物（8 层 / 121 条目 / blob 61 + 稠密 60）在**新写的
GGUFBlockQuantization** 上走一遍「构造 -> 键对账 -> 加载 -> 首次 forward」。
每条判据都能独立失败，不是「能加载」的同义反复：

  1. 构造：C++ 侧每个 Linear 都用自己的 checkpoint stem 查类型表。stem 拼错 /
     融合组没登记 / 表外 ggml type -> resolve() 抛错，构造直接失败。
     所以「构造通过」= 所有被查询的 stem 都恰好命中 1 个候选。
  2. 键双向 diff：引擎 state_dict_keyname() 与产物 index 的张量名必须完全相等。
  3. 逐键 shape 对账：blob 必须是 [out, row_bytes]，(block_size, type_size) 直接从
     gguf-py 的 GGML_QUANT_SIZES 取（**独立于 gguf.cpp 里那份常量表**）——两侧谁算
     窄了/算宽了都会在下层的 load_no_sync 里炸，这里先炸出来，报错更好读。
  4. 加载：load_model_state_dict_by_file 末尾的 check_parameters 对缺键/多键直接
     raise，等于框架替我们做 strict=False 的兜底审查（判据 1）。
  5. 首次 forward：blob Linear 必须真的进了 linear_gguf 并返回（判据 3：没有静默
     回落稠密 GEMM）。阶段 2 时这里期望的是抛「阶段 3 实现」占位，3.2 落地后
     期望反过来：日志里出现带 M/N/K/ggml_type/row_bytes 的契约行，且 row_bytes
     用 gguf-py 的 (block_size, type_size) 独立重算相等。整模端到端（generate）
     由 scripts/gguf_routeb_stage3_check.py 覆盖：forward_raw 的 python 签名不暴
     露 mamba_*_state_indices，GDN 模型走完 in_proj 后会在下游 conv1d 里因可选
     入参为空抛 bad_optional_access —— 上游 API 缺口，与 GGUF 无关。

用法：
  source /home/liuxd/InfiniLM/scripts/gguf_routeb_env.sh
  /usr/bin/python3 scripts/gguf_routeb_stage2_check.py [--device cuda:0] [--no-forward]
退出码 0 = 全部 PASS。
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import json
import os
import re
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(
    0, os.path.join(os.environ.get("LLAMA_CPP_DIR", "/home/liuxd/llama.cpp"), "gguf-py")
)

DEFAULT_MODEL = "/home/liuxd/models/Qwen3.8-27B-GGUF-native-mini8"
BLOB_SUFFIX = "weight_bytes"
# 与 csrc/layers/quantization/gguf.cpp 里那条诊断日志的格式对应
BLOB_RE = re.compile(
    r"linear_gguf: 首个 blob 前向 (\S+) — M=(\d+) N=(\d+) K=(\d+) "
    r"ggml_type=(\d+) row_bytes=(\d+)"
)
MAX_DECODE_M = 8  # kMaxDecodeM：<=8 走 gemv，>8 走 prefill（阶段 3.3 起不再是上限）
PROMPT_TOKENS = 3  # 下面 forward_raw 喂的 token 数，用来核对契约行的 M

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-forward", action="store_true", help="跳过首次 forward 判据")
    a = ap.parse_args()

    import infinicore
    from gguf.constants import GGML_QUANT_SIZES
    from infinilm.cache import StaticKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file
    from safetensors import safe_open

    # ---------------------------------------------------------------- 0. config
    print("\n== 0. 产物 config.json ==")
    with open(os.path.join(a.model_path, "config.json")) as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config") or {}
    check(
        "quantization_config 在顶层且 quant_method=gguf",
        qc.get("quant_method") == "gguf",
        "qc keys=%s" % sorted(qc),
    )
    table = qc.get("ggml_types") or {}
    check("类型表非空（%d 条）" % len(table), bool(table))
    bs_ts = {int(t): (int(v[0]), int(v[1])) for t, v in GGML_QUANT_SIZES.items()}
    ids = sorted({v for v in table.values() if isinstance(v, int)})
    check(
        "表内 type id 都能从 gguf-py 查出 (block_size, type_size)：%s" % ids,
        all(i in bs_ts for i in ids),
        str([i for i in ids if i not in bs_ts]),
    )

    with open(os.path.join(a.model_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    n_blob = sum(1 for v in table.values() if isinstance(v, int))
    print(
        "  -> 类型表 %d 条：blob %d / 稠密 %d；产物 index %d 个张量；key_prefix='%s'"
        % (
            len(table),
            n_blob,
            len(table) - n_blob,
            len(weight_map),
            qc.get("key_prefix"),
        )
    )
    # 溯源：表键有两种历史形态。新规则（§6.0 纠正 2）= 张量名原文（与产物 index 同名）；
    # 旧规则 = 去前缀的相对名且 blob 归一成 .weight（与 index 不同名）。两者 C++ 都能
    # 命中（裁前缀时 key_prefix 缺失就取 ""，探键时 weight_bytes / weight 都探），
    # 但必须知道眼下这份产物是哪一种，不然对不上时会查错方向。
    n_ident = len(set(table) & set(weight_map))
    print(
        "  -> 表键形态：%d/%d 条与产物张量名同名（新规则），其余 %d 条为相对名或前缀外键"
        % (n_ident, len(table), len(table) - n_ident)
    )

    # ------------------------------------------------------------- 1. 构造引擎
    print("\n== 1. 用 GGUFBlockQuantization 构造引擎（device=%s）==" % a.device)
    # infinicore.device("cuda:0", 0) 会报 “index should not be provided”，带冒号就不能再传 index
    dev_spec = (
        infinicore.device(a.device)
        if ":" in a.device
        else infinicore.device(a.device, 0)
    )
    try:
        eng = InferEngine(
            model_path=a.model_path,
            device=dev_spec,
            distributed_config=DistConfig(1),
            cache_config=StaticKVCacheConfig(max_batch_size=1, max_cache_len=16),
        )
        ok, err = True, ""
    except Exception as e:  # noqa: BLE001
        ok, err = False, "%s: %s" % (type(e).__name__, str(e)[:1200])
    check(
        "构造通过（= 被查询的 stem 全部恰好命中 1 个候选，且无 TP/bias 违规）", ok, err
    )
    if not ok:
        print("\n构造都没过，后面全部跳过\n" + traceback.format_exc())
        return 1
    check(
        "引擎确实走 GGUF 方案",
        (eng.hf_config.get("quantization_config") or {}).get("quant_method") == "gguf",
    )

    # --------------------------------------------------------- 2. 键双向 diff
    print("\n== 2. 引擎参数键 vs 产物张量名 ==")
    keys = list(eng.state_dict_keyname())
    extra = sorted(set(keys) - set(weight_map))
    missing = sorted(set(weight_map) - set(keys))
    check(
        "产物有、引擎不要（多键 -> strict=False 下静默丢权重）",
        not extra,
        str(extra[:12]),
    )
    check("引擎要、产物没有（缺键 -> 保持随机初始化）", not missing, str(missing[:12]))
    check(
        "键数一致（引擎 %d / 产物 %d）" % (len(keys), len(weight_map)),
        len(keys) == len(weight_map),
    )

    # ---------------------------------------------- 3. 逐键 dtype / shape 对账
    print("\n== 3. 逐键 shape 对账（blob 行字节独立重算）==")
    sd_keys = set(keys)
    meta = {}
    for fn in sorted(set(weight_map.values())):
        with safe_open(os.path.join(a.model_path, fn), framework="pt") as f:
            for k in f.keys():
                if k in sd_keys:
                    meta[k] = (
                        f.get_slice(k).get_dtype(),
                        list(f.get_slice(k).get_shape()),
                    )
    eng_sd = eng.state_dict()[0]

    # 照抄 C++ GGUFBlockQuantization::resolve() 的查表语义：表键 = 产物名裁掉
    # 已声明的 key_prefix（未声明则为 ""，即保留原样），探 stem+"weight_bytes" 与
    # stem+"weight" 两个候选。引擎侧的绝对键 = 模型参数路径，可能与表键不同形，
    # 所以这里按候选集查而不是 table[k] 直查（命中数 != 1 算 FAIL，不让脚本 KeyError）。
    MOD_PREFIX = "model.language_model."
    W_BLOB = "." + BLOB_SUFFIX

    def table_hits(k):
        cands = {k, k[: -len(W_BLOB)] + ".weight" if k.endswith(W_BLOB) else k}
        for base in list(cands):
            if base.startswith(MOD_PREFIX):
                cands.add(base[len(MOD_PREFIX) :])
        declared = qc.get("key_prefix") or ""
        for base in list(cands):
            if declared and base.startswith(declared):
                cands.add(base[len(declared) :])
        return sorted(c for c in cands if c in table)

    bad_shape, bad_dtype, n_blob_eng, n_table_form = [], [], 0, collections.Counter()
    for k in sorted(sd_keys & set(meta)):
        e_shape = [int(x) for x in eng_sd[k].shape]
        if e_shape != list(meta[k][1]):
            bad_shape.append("%s: 引擎 %s vs 产物 %s" % (k, e_shape, meta[k][1]))
        if k.endswith("." + BLOB_SUFFIX):
            n_blob_eng += 1
            if "U8" not in str(eng_sd[k].dtype).upper() or meta[k][0] != "U8":
                bad_dtype.append(
                    "%s: 引擎 %s / 产物 %s" % (k, eng_sd[k].dtype, meta[k][0])
                )
            hits = table_hits(k)
            if len(hits) != 1:
                bad_shape.append(
                    "%s: 类型表命中 %d 个候选 %s（C++ 会抛或静默走稠密）"
                    % (k, len(hits), hits[:4])
                )
                continue
            n_table_form["与张量名同名" if hits[0] == k else "相对名/归一后缀"] += 1
            _bs, ts = bs_ts[int(table[hits[0]])]
            if e_shape and ts and e_shape[-1] % ts:
                bad_shape.append(
                    "%s: row_bytes=%d 不是 type_size %d 的整数倍" % (k, e_shape[-1], ts)
                )
    check(
        "%d 个 blob 键在引擎侧与产物侧都是 U8" % n_blob_eng,
        not bad_dtype,
        str(bad_dtype[:6]),
    )
    check(
        "全部 %d 键 shape 逐字相等（blob 为 [out, row_bytes]）" % len(sd_keys),
        not bad_shape,
        str(bad_shape[:8]),
    )
    print(
        "  -> %d 个 blob 命中的表键形态：%s"
        % (
            n_blob_eng,
            ", ".join("%s x%d" % kv for kv in n_table_form.most_common()) or "无",
        )
    )

    # ----------------------------------------------------------------- 4. 加载
    print("\n== 4. 加载（末尾 check_parameters 会对缺/多键抛错 = 判据 1）==")
    try:
        load_model_state_dict_by_file(eng, a.model_path, dtype=eng.dtype)
        ok, err = True, ""
    except Exception as e:  # noqa: BLE001
        ok, err = False, "%s: %s" % (type(e).__name__, str(e)[:1200])
    check("%d 个条目全部装载完毕" % len(weight_map), ok, err)

    # ------------------------------------------------------- 5. 首次 forward
    if a.no_forward:
        print("\n== 5. 跳过（--no-forward）==")
    else:
        print(
            "\n== 5. 首个 blob Linear 必须进 linear_gguf 并返回（判据 3：不静默回落稠密）=="
        )
        import torch

        def to_dev(t):
            return infinicore.from_torch(
                t.cuda(0) if a.device.startswith("cuda") else t
            )

        ids = to_dev(torch.tensor([[114, 5, 7]], dtype=torch.int32))
        # qwen3_5 是 mrope（position_id_axes=3），position_ids 的轴序在 C++ 侧
        # 只要求最后一维是 seq，这里按 [axes, seq] / [seq] 两种形状各试一次，
        # 目的是越过入参校验走到第一个 Linear —— 判据只看那里抛的是什么。
        cands = [
            to_dev(torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=torch.int32)),
            to_dev(torch.tensor([[0, 1, 2]], dtype=torch.int32)),
        ]

        # RankWorker 会把工作线程里的异常换个文案再抛一次（python 侧只看到
        # “RankWorker is closing”），真实抛出只落在 spdlog 里。实测 spdlog 走的是
        # **stdout**（把 2 单独分流到文件后 “linear_gguf” 那条 [error] 仍留在
        # stdout），所以 fd 1、2 都得用 memfd 接住（沙箱里 /tmp 只读）。
        def open_cap():
            try:
                return os.memfd_create("stage2_log")
            except AttributeError:
                return os.open(
                    os.path.join(_HERE, ".stage2_log.tmp"),
                    os.O_RDWR | os.O_CREAT | os.O_TRUNC,
                    0o600,
                )

        libc = ctypes.CDLL(None)
        caps = {fd: open_cap() for fd in (1, 2)}
        saved = {fd: os.dup(fd) for fd in caps}
        msgs = []
        try:
            # 先把手头的正常输出推完再换管道：否则 step 4 的 PASS 还躺在 python
            # 的块缓冲里，换完才被 flush，会打到 memfd 里而不在日志文件中。
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, mem in caps.items():
                os.dup2(mem, fd)
            for pos in cands:
                try:
                    eng.forward_raw(input_ids=ids, position_ids=pos)
                    msgs.append("<没抛异常：blob 被当成稠密权重跑了！>")
                    break
                except Exception as e:  # noqa: BLE001
                    msgs.append(
                        "%s: %s"
                        % (type(e).__name__, str(e).strip().splitlines()[0][:200])
                    )
        finally:
            libc.fflush(None)  # C++ 侧重定向到文件时是块缓冲，不冲读不到
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, mem in caps.items():
                os.fsync(mem)
                os.dup2(saved[fd], fd)
                os.close(saved[fd])
        captured = ""
        for mem in caps.values():
            os.lseek(mem, 0, os.SEEK_SET)
            captured += os.read(mem, 1 << 20).decode("utf-8", "replace")
            os.close(mem)
        line = next((ln for ln in captured.splitlines() if "linear_gguf" in ln), "")
        m = BLOB_RE.search(line)
        if not m:
            check(
                "首个 blob Linear 进入 linear_gguf 并返回（未回落稠密）",
                False,
                "python: %s\n        日志尾部: %s"
                % (" | ".join(msgs), captured[-600:]),
            )
        else:
            M, N, K, tid, row_bytes = [int(m.group(i)) for i in range(2, 7)]
            check("首个 blob Linear 进入 linear_gguf 并返回（未回落稠密）", True)
            # 只留 linear_gguf 之后的部分：spdlog 前缀占掉大半行，按整行截断会把张量名切掉
            print("        %s" % line[line.find("linear_gguf") :].strip())
            bs, ts = bs_ts[tid]
            # 阶段 3.3 前这里评的是“M <= 8”（当时的 decode 护栏）；现在 M 的唯一
            # 契约是“等于本次喂进去的 token 数”，大了小了都算错。
            check(
                "契约行自洽：M=%d 等于 prompt token 数 %d 且 row_bytes=%d == (K/%d)*%d"
                % (M, PROMPT_TOKENS, row_bytes, bs, ts),
                M == PROMPT_TOKENS and row_bytes == (K // bs) * ts,
                "type=%d (block_size, type_size)=(%d,%d)" % (tid, bs, ts),
            )

    print("\n== 结果：%d PASS / %d FAIL ==" % (_PASS, _FAIL))
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
