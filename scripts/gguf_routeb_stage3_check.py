#!/usr/bin/env python3
"""
InfiniLM 路线 B —— 阶段 3 端到端验收（执行方案 §7.1 判据 4/5）

拿 mini8 产物在**量化形态**下真跑一遍 generate，逐条判据独立可失败：

  1. PagedKVCacheConfig + attention_backend="paged-attn" 的引擎能构造并加载 121 条目。
     （必须 paged：Qwen3NextCausalConv1D::forward 取 mamba_metadata 的三个
      optional<Tensor>.value()，而 forward_raw 的 python 签名不暴露
      mamba_*_state_indices —— 上游 API 缺口，与 GGUF 无关，见 §7.2 备注。）
  2. **prefill 正例**：prompt 长度 12（> kMaxDecodeM=8）的 generate 必须跑完。阶段 3.3
     之前这里是必抛「超过 decode kernel 的上限」，现在反过来：抛就算 FAIL。
  3. 日志里出现「首个 blob 前向 …」契约行，且 **M 等于 prompt 长度**（证明整个
     批量一次进了 kernel、没被拆开也没回落），row_bytes 用 gguf-py 的
     (block_size, type_size) 独立重算相等。
  4. 贪心（top_k=1 / temperature=0）两次同 prompt 结果逐字相同 —— 说明 kernel
     没有 NaN/不确定行为（数值对不对是阶段 4 的比对，这里不比数值）。
  5. token id 落在词表内。
  6. **decode 回归**：prompt 长度 4（<= kMaxDecodeM）仍走 gemv、仍跑完 —— 撤护栏不
     许把已经能用的短 prompt 路径弄坏。
  7. （--count-blob-calls）把自己在 gdb 下重跑一遍，用断点命中次数证明
     「每一步、每个 blob 模块」都进了 kernel：命中数 == 步数 × blob 条目数。
     这条与路径无关（gemv/prefill 都过同一个 infiniopLinearGguf），少了就是有
     blob 静默回落稠密，多了就是有别的稠密 Linear 被误开。

用法：
  source /home/liuxd/InfiniLM/scripts/gguf_routeb_env.sh
  /usr/bin/python3 scripts/gguf_routeb_stage3_check.py [--new-tokens 8] [--count-blob-calls]
退出码 0 = 全部 PASS。
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(
    os.environ.get("LLAMA_CPP_DIR", "/home/liuxd/llama.cpp"), "gguf-py"))

DEFAULT_MODEL = "/home/liuxd/models/Qwen3.8-27B-GGUF-native-mini8"
BLOB_RE = re.compile(
    r"linear_gguf: 首个 blob 前向 (\S+) — M=(\d+) N=(\d+) K=(\d+) "
    r"ggml_type=(\d+) row_bytes=(\d+)")
MAX_DECODE_M = 8          # kMaxDecodeM：<=8 走 gemv，>8 走 prefill（两条路径同一个谓词）
PREFILL_M = 12            # > MAX_DECODE_M：阶段 3.3 的 prefill 正例（旧行为是必抛）
DECODE_M = 4              # <= MAX_DECODE_M：decode 回归用例

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


# --------------------------------------------------------------- C++ 日志捕获
def _open_cap():
    try:
        return os.memfd_create("stage3_log")
    except AttributeError:
        path = os.path.join(tempfile.gettempdir(), ".stage3_log.tmp")
        try:
            return os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        except OSError:
            return os.open(os.path.join(_HERE, ".stage3_log.tmp"),
                           os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)


class capture:
    """把 fd 1/2 换到内存文件，用于读 spdlog 的输出。

    RankWorker 会把工作线程里的异常换个文案再抛一次（python 侧只看到
    “RankWorker …”），真实抛出点只落在 spdlog 里；实测 spdlog 走 stdout，
    所以 1、2 两个 fd 都得接。
    """

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        libc = ctypes.CDLL(None)
        self._libc = libc
        self.captured = ""
        self._caps = {fd: _open_cap() for fd in (1, 2)}
        self._saved = {fd: os.dup(fd) for fd in self._caps}
        for fd, mem in self._caps.items():
            os.dup2(mem, fd)
        return self

    def __exit__(self, *exc):
        self._libc.fflush(None)      # C++ 侧块缓冲，不冲就读不到
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, mem in self._caps.items():
            try:
                os.fsync(mem)
            except OSError:
                pass
            os.lseek(mem, 0, os.SEEK_SET)
            self.captured += os.read(mem, 1 << 22).decode("utf-8", "replace")
            os.dup2(self._saved[fd], fd)
            os.close(self._saved[fd])
            os.close(mem)
        return False


# ------------------------------------------------------------------ gdb 计数
def count_blob_calls(inner_argv):
    """在 gdb 下重跑本脚本（inner_argv 已带 --route-b-inner），读断点命中次数。"""
    script = os.path.join(tempfile.gettempdir(), "stage3_count.gdb")
    try:
        with open(script, "w") as f:
            f.write("set pagination off\nset confirm off\n"
                    "set breakpoint pending on\n"
                    "break infiniopLinearGguf\ncommands\nsilent\ncontinue\nend\n"
                    "run\nprintf \"\\n===BPSTAT===\\n\"\ninfo breakpoints\n")
    except OSError:
        script = os.path.join(_HERE, ".stage3_count.gdb")
        with open(script, "w") as f:
            f.write("set pagination off\nset confirm off\n"
                    "set breakpoint pending on\n"
                    "break infiniopLinearGguf\ncommands\nsilent\ncontinue\nend\n"
                    "run\nprintf \"\\n===BPSTAT===\\n\"\ninfo breakpoints\n")
    cmd = ["gdb", "-q", "-batch", "-x", script, "--args", sys.executable] + inner_argv
    print("  -> %s" % " ".join(cmd[:8]))
    p = subprocess.run(cmd, capture_output=True, text=True)
    tail = p.stdout + p.stderr
    m = re.search(r"breakpoint already hit (\d+) times", tail)
    return int(m.group(1)) if m else None, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--new-tokens", type=int, default=8)
    ap.add_argument("--num-blocks", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--count-blob-calls", action="store_true")
    ap.add_argument("--route-b-inner", action="store_true",
                    help="内部用：gdb 子进程模式，只做前 6 条判据")
    a, _unknown = ap.parse_known_args()

    import infinicore
    from infinilm.cache import PagedKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import GenerationConfig, InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file
    from gguf.constants import GGML_QUANT_SIZES
    import json

    with open(os.path.join(a.model_path, "config.json")) as f:
        cfg = json.load(f)
    table = (cfg.get("quantization_config") or {}).get("ggml_types") or {}
    n_blob = sum(1 for v in table.values() if isinstance(v, int))
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    vocab = int(text_cfg.get("vocab_size") or 0)

    def build():
        return InferEngine(
            model_path=a.model_path,
            device=infinicore.device("cuda:0"),
            distributed_config=DistConfig(1),
            cache_config=PagedKVCacheConfig(a.num_blocks, a.block_size,
                                            max_batch_size=1),
            attention_backend="paged-attn",
        )

    # ------------------------------------------------------------- 1. 构造加载
    print("\n== 1. paged 引擎构造 + 加载（blob %d / 稠密 %d）==" % (n_blob,
                                                                    len(table) - n_blob))
    try:
        eng = build()
        ok, err = True, ""
    except Exception as e:                                        # noqa: BLE001
        ok, err = False, "%s: %s" % (type(e).__name__, str(e)[:1200])
    check("构造通过（PagedKVCacheConfig + paged-attn）", ok, err)
    if not ok:
        return 1
    check("has_mamba_cache 且 enable_paged_attn（GDN 模型只能走这条路）",
          eng.has_mamba_cache and eng.enable_paged_attn)
    try:
        load_model_state_dict_by_file(eng, a.model_path, dtype=eng.dtype)
        ok, err = True, ""
    except Exception as e:                                        # noqa: BLE001
        ok, err = False, "%s: %s" % (type(e).__name__, str(e)[:1200])
    check("权重装载完毕", ok, err)
    if not ok:
        return 1

    def do_generate(tokens):
        ids = infinicore.from_list([tokens], dtype=infinicore.int64)
        out = eng.generate(ids, GenerationConfig(
            max_new_tokens=a.new_tokens, temperature=0.0, top_k=1, top_p=1.0,
            eos_token_id=None, stop_on_eos=False))
        return [int(x.to_numpy().reshape(-1)[0]) for x in out]

    # 成功的 generate 次数；每完成一次 = 1 次 prefill + (new_tokens-1) 次 decode
    # = new_tokens 个前向步，每步每个 blob 各进 kernel 一次（判据 7 的期望值）。
    done_generates = 0

    def one_generate(tokens):
        nonlocal done_generates
        toks = do_generate(tokens)
        done_generates += 1
        return toks

    # ------------------------------------- 2/3/4/5. prefill 正例（prompt > decode 上限）
    print("\n== 2-5. prefill：prompt=%d token（> kMaxDecodeM=%d）==" % (PREFILL_M, MAX_DECODE_M))
    pre_prompt = list(range(100, 100 + PREFILL_M))
    with capture() as cap:
        try:
            toks1 = one_generate(pre_prompt)
            perr = ""
        except BaseException as e:                                # noqa: BLE001
            toks1, perr = None, "%s: %s" % (
                type(e).__name__, str(e).strip().splitlines()[:1])
    log1 = cap.captured
    check("prefill generate 走完 %d 步（M=%d 不再抛）" % (a.new_tokens, PREFILL_M),
          toks1 is not None, perr + "\n        日志尾部: " + log1[-500:])
    if toks1 is None:
        print("\n== 结果：%d PASS / %d FAIL ==" % (_PASS, _FAIL))
        return 1
    print("        tokens=%s" % toks1)
    check("token id 落在词表 [0,%d) 内" % vocab,
          not vocab or all(0 <= t < vocab for t in toks1))

    toks2 = one_generate(pre_prompt)
    check("贪心两次结果逐字相同", toks1 == toks2, "%s vs %s" % (toks1, toks2))

    m = BLOB_RE.search(log1)
    check("日志出现 blob 前向契约行（= blob 没被当稠密权重跑）", bool(m),
          "捕获 %d 字节，未见 linear_gguf 行" % len(log1))
    if m:
        key, M, N, K, tid, row_bytes = m.group(1), *[int(m.group(i)) for i in
                                                     range(2, 7)]
        print("        %s — M=%d N=%d K=%d ggml_type=%d row_bytes=%d"
              % (key, M, N, K, tid, row_bytes))
        bs, ts = GGML_QUANT_SIZES[tid]
        # 契约行是进 kernel 的第一个 blob，而第一个 blob 就在 prompt 的 prefill 里。
        # M 必须等于 prompt 长度：小了就是上层把 prompt 拆碎了/没走 prefill。
        check("契约行 M=%d 等于 prompt 长度 %d（整批进 kernel）" % (M, PREFILL_M),
              M == PREFILL_M, "M=%d" % M)
        check("该批只能由 prefill 路径处理（M=%d > kMaxDecodeM=%d）" % (M, MAX_DECODE_M),
              M > MAX_DECODE_M, "M=%d" % M)
        check("契约行 row_bytes == (K/%d)*%d 自洽" % (bs, ts),
              row_bytes == (K // int(bs)) * int(ts),
              "row_bytes=%d 期望=%d" % (row_bytes, (K // int(bs)) * int(ts)))

    # ------------------------------------------- 6. decode 回归（短 prompt 仍可用）
    print("\n== 6. decode 回归：prompt=%d token（<= %d，仍走 gemv）=="
          % (DECODE_M, MAX_DECODE_M))
    dec_prompt = list(range(300, 300 + DECODE_M))
    try:
        toks3 = one_generate(dec_prompt)
        err3 = ""
    except BaseException as e:                                    # noqa: BLE001
        toks3, err3 = None, "%s: %s" % (type(e).__name__, str(e).strip()[:200])
    check("短 prompt 用例走完 %d 步（撤护栏未弄坏 gemv 路径）" % a.new_tokens,
          toks3 is not None, err3)
    if toks3 is not None:
        print("        tokens=%s" % toks3)

    # --------------------------------------------------- 7. 断点命中数（可选）
    if a.count_blob_calls and not a.route_b_inner:
        print("\n== 7. gdb 断点计数：每步 × 每个 blob ==")
        inner = [os.path.abspath(sys.argv[0])] + \
            [x for x in sys.argv[1:] if x != "--count-blob-calls"] + \
            ["--route-b-inner"]
        n, tail = count_blob_calls(inner)
        steps = re.search(r"INNER_STEPS=(\d+)", tail)
        steps = int(steps.group(1)) if steps else None
        expect = steps * n_blob if steps else None
        check("infiniopLinearGguf 命中 %s 次 == 步数 %s × blob %d = %s"
              % (n, steps, n_blob, expect), n is not None and n == expect,
              "实际 %s / 期望 %s\n        子进程输出尾部: %s"
              % (n, expect, tail[-600:]))
    elif a.route_b_inner:
        # 子进程里：把实际完成的前向步数报给外层。每次 generate = 1 次 prefill +
        # (max_new_tokens-1) 次 decode = max_new_tokens 步；本脚本一共跑 3 次。
        print("INNER_STEPS=%d" % (done_generates * a.new_tokens))

    print("\n== 结果：%d PASS / %d FAIL ==" % (_PASS, _FAIL))
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
