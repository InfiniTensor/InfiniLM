"""End-to-end benchmark: fused FFN vs non-fused FFN on the csrc engine.

The csrc decoder layer for FM9G re-reads `INFINILM_USE_FUSED_FFN` on every
`forward` call, so this script can flip `os.environ` between back-to-back
`generate()` calls and interleave NF/F rounds to cancel thermal drift. The
measurement window is `time.perf_counter` around the model call; warmup,
correctness verification, and a markdown report are produced as well.

Usage:
    python bench_fused_ffn_csrc.py --model /path/to/FM9G-7B \\
        [--device iluvatar] [--tp 1] [--warmup 3] [--rounds 5] \\
        [--max-new-tokens 64] [--output report.md] [--skip-verify] \\
        [--no-chat-template]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

# Meaningful Chinese natural-language prompts.
PROMPTS: List[str] = [
    # short prompts — exercise small ntok path
    "你好",
    "今天天气怎么样？",
    "讲个笑话。",
    "1+1等于几？",
    "你叫什么名字？",
    # medium / longer creative prompts
    "如果猫能写诗，它们会写些什么？",
    "描述一个没有重力的世界。",
    "如果地球停止自转，会发生什么？",
    "假设你是一只会飞的鲸鱼，描述你的日常生活。",
    "如果人类可以与植物沟通，世界会变成什么样？",
    "描述一个由糖果构成的城市。",
    "如果时间旅行成为可能，你最想去哪个时代？",
    "想象一下，如果地球上只有蓝色，其他颜色都消失了。",
    "如果动物能上网，它们会浏览什么网站？",
    "描述一个没有声音的世界。",
    "如果人类可以在水下呼吸，城市会如何变化？",
    "想象一下，如果天空是绿色的，云是紫色的。",
    "如果你能与任何历史人物共进晚餐，你会选择谁？",
    "描述一个没有夜晚的星球。",
    "如果地球上只有一种语言，世界会如何运作？",
    "想象一下，如果所有的书都变成了音乐。",
    "如果你可以变成任何一种动物，你会选择什么？",
    "描述一个由机器人统治的未来世界。",
    "如果你能与任何虚构角色成为朋友，你会选择谁？",
    "想象一下，如果每个人都能读懂他人的思想。",
]

FUSED_ENV = "INFINILM_USE_FUSED_FFN"


def set_fused(mode: bool) -> None:
    os.environ[FUSED_ENV] = "1" if mode else "0"


# ── Statistics ─────────────────────────────────────────────────────────────

def summarize(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"mean": 0.0, "trimmed_mean": 0.0, "median": 0.0,
                "min": 0.0, "max": 0.0, "stdev": 0.0, "p99": 0.0, "n": 0}
    arr = sorted(samples)
    n = len(arr)
    trimmed = arr[1:-1] if n >= 4 else arr
    p99_idx = max(0, min(n - 1, int(round((n - 1) * 0.99))))
    return {"mean": mean(arr), "trimmed_mean": mean(trimmed),
            "median": median(arr), "min": arr[0], "max": arr[-1],
            "stdev": stdev(arr) if n > 1 else 0.0, "p99": arr[p99_idx], "n": n}


def speedup_pct(nf: float, f: float) -> float:
    return (nf - f) / nf * 100.0 if nf > 0 else 0.0


def speedup_ratio(nf: float, f: float) -> float:
    return nf / f if f > 0 else 0.0


# ── csrc engine wrapper ────────────────────────────────────────────────────

class CsrcModel:
    """Thin wrapper over infinilm.InferEngine. Mirrors examples/bench.py."""

    # Map convenient hardware names (matches python/infinilm/base_config.py:280+)
    # to the infinicore.device("cuda"/"mlu"/... ) string the SDK expects.
    _DEVICE_ALIAS = {
        "iluvatar": "cuda",
        "nvidia":   "cuda",
        "metax":    "cuda",
        "hygon":    "cuda",
        "kunlun":   "cuda",
        "ali":      "cuda",
        "qy":       "cuda",
        "moore":    "musa",
        "cambricon": "mlu",
        "ascend":   "npu",
        "cpu":      "cpu",
    }

    def __init__(self, model_path: str, device_name: str, tp: int,
                 kv_cache_dtype: Optional[str] = None) -> None:
        import infinicore
        from infinilm.distributed import DistConfig
        from infinilm.infer_engine import InferEngine
        from infinilm.modeling_utils import load_model_state_dict_by_file
        from infinilm.processors import AutoInfinilmProcessor

        canon = self._DEVICE_ALIAS.get(device_name, device_name)
        device = infinicore.device(canon, 0)
        kwargs: Dict[str, Any] = {
            "device": device,
            "distributed_config": DistConfig(tp),
            "cache_config": None,
        }
        if kv_cache_dtype is not None:
            kwargs["kv_cache_dtype"] = kv_cache_dtype

        engine = InferEngine(model_path, **kwargs)
        load_model_state_dict_by_file(engine, model_path, dtype=engine.dtype)

        processor = AutoInfinilmProcessor.from_pretrained(model_path)
        self.engine = engine
        self.processor = processor
        self.tokenizer = processor.get_tokenizer()
        self.infinicore = infinicore

    def encode_prompts(self, prompts: List[str],
                       use_chat_template: bool = True) -> List[List[int]]:
        encoded: List[List[int]] = []
        for p in prompts:
            if use_chat_template:
                text = self.processor.apply_chat_template(
                    conversation=[{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                text = p
            ids = self.tokenizer.encode(text)
            encoded.append(list(ids))
        return encoded

    def reset_cache(self, batch_size: int, total_seq_len: int) -> None:
        from infinilm.cache import StaticKVCacheConfig
        self.engine.reset_cache(StaticKVCacheConfig(
            max_batch_size=batch_size, max_cache_len=total_seq_len))

    def generate_batch(self, input_ids_list: List[List[int]],
                       max_new_tokens: int, temperature: float = 1.0,
                       top_k: int = 1, top_p: float = 1.0) -> Tuple[List[List[int]], float]:
        from infinilm.infer_engine import GenerationConfig
        input_ids_infini = self.infinicore.from_list(input_ids_list)

        t0 = time.perf_counter()
        output_ids = self.engine.generate(
            input_ids_infini,
            GenerationConfig(
                max_new_tokens=max_new_tokens,
                eos_token_id=[],
                top_k=top_k, top_p=top_p, temperature=temperature,
                stop_on_eos=False,
            ),
            _measure_and_log_time=False,
        )
        t1 = time.perf_counter()

        out: List[List[int]] = []
        for tensor in output_ids:
            arr = tensor.to_numpy()
            out.append(arr[0].tolist() if arr.ndim > 1 else arr.tolist())
        return out, (t1 - t0) * 1000.0  # ms


# ── Correctness verification ──────────────────────────────────────────────

def verify_correctness(model: CsrcModel, input_ids_list: List[List[int]],
                       max_new_tokens: int) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("CORRECTNESS  (greedy NF vs F — output tokens must agree)")
    print("=" * 60)

    total_seq_len = max(len(ids) for ids in input_ids_list) + max_new_tokens

    nf_out: List[List[int]] = []
    for ids in input_ids_list:
        model.reset_cache(1, len(ids) + max_new_tokens)
        set_fused(False)
        out, _ = model.generate_batch([ids], max_new_tokens,
                                       temperature=1.0, top_k=1, top_p=1.0)
        nf_out.append(out[0])

    f_out: List[List[int]] = []
    for ids in input_ids_list:
        model.reset_cache(1, len(ids) + max_new_tokens)
        set_fused(True)
        out, _ = model.generate_batch([ids], max_new_tokens,
                                       temperature=1.0, top_k=1, top_p=1.0)
        f_out.append(out[0])

    matches = 0
    mismatches: List[Tuple[int, int]] = []
    for i, (nf, f) in enumerate(zip(nf_out, f_out)):
        n = min(len(nf), len(f))
        equal_prefix = 0
        for j in range(n):
            if nf[j] == f[j]:
                equal_prefix += 1
            else:
                break
        if equal_prefix == max_new_tokens and len(nf) == len(f):
            matches += 1
        else:
            mismatches.append((i, equal_prefix))

    passed = matches == len(input_ids_list)
    print(f"  total prompts            : {len(input_ids_list)}")
    print(f"  byte-identical outputs   : {matches}")
    print(f"  divergent outputs        : {len(mismatches)}")
    if mismatches[:3]:
        print(f"  first divergences (idx, common_prefix_tokens): {mismatches[:3]}")
    print(f"  status                   : {'PASS' if passed else 'WARN (some divergence — kernels may differ in low bits)'}")

    return {
        "total": len(input_ids_list),
        "matches": matches,
        "mismatches": len(mismatches),
        "first_divergences": mismatches[:3],
        "passed": passed,
    }


# ── Interleaved measurement ───────────────────────────────────────────────

def benchmark_prompt(model: CsrcModel, input_ids: List[int], prompt_idx: int,
                     prompt_text: str, max_new_tokens: int, warmup: int,
                     rounds: int) -> Dict[str, Any]:
    print(f"\n[prompt #{prompt_idx}] input_tokens={len(input_ids)}  out={max_new_tokens}")
    print(f"  text: {prompt_text[:40]}...")

    total_seq_len = len(input_ids) + max_new_tokens
    model.reset_cache(1, total_seq_len)

    for _ in range(warmup):
        set_fused(False)
        model.generate_batch([input_ids], max_new_tokens)
        model.reset_cache(1, total_seq_len)
    for _ in range(warmup):
        set_fused(True)
        model.generate_batch([input_ids], max_new_tokens)
        model.reset_cache(1, total_seq_len)

    nf_samples: List[float] = []
    f_samples: List[float] = []
    for _ in range(rounds):
        set_fused(False)
        _, t_nf = model.generate_batch([input_ids], max_new_tokens)
        nf_samples.append(t_nf)
        model.reset_cache(1, total_seq_len)

        set_fused(True)
        _, t_f = model.generate_batch([input_ids], max_new_tokens)
        f_samples.append(t_f)
        model.reset_cache(1, total_seq_len)

    nf_stat = summarize(nf_samples)
    f_stat = summarize(f_samples)
    sp = speedup_pct(nf_stat["mean"], f_stat["mean"])
    ratio = speedup_ratio(nf_stat["mean"], f_stat["mean"])

    print(f"  {'metric':<22} {'non-fused':>12}  {'fused':>12}")
    print(f"  {'':->50}")
    for k, label in [("mean", "mean    (ms)"), ("trimmed_mean", "trimmed (ms)"),
                     ("median", "median  (ms)"), ("stdev", "stdev   (ms)"),
                     ("min", "min     (ms)"), ("p99", "p99     (ms)")]:
        print(f"  {label:<22} {nf_stat[k]:>12.3f}  {f_stat[k]:>12.3f}")
    print(f"  → e2e speedup: {sp:+.2f}%   ({ratio:.3f}×)")

    return {"nf_samples": nf_samples, "f_samples": f_samples,
            "nf": nf_stat, "f": f_stat, "speedup_pct": sp, "ratio": ratio,
            "input_tokens": len(input_ids), "prompt": prompt_text}


# ── Markdown report ───────────────────────────────────────────────────────

def save_markdown_report(args, verify_data: Optional[Dict[str, Any]],
                         per_prompt: List[Dict[str, Any]]) -> None:
    L: List[str] = []

    def W(s: str = "") -> None:
        L.append(s)

    W("# Fused FFN End-to-End Benchmark (csrc engine, FM9G)")
    W()
    W(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    W(f"- **Model:** `{args.model}`")
    W(f"- **Device:** {args.device} (tp={args.tp})")
    W(f"- **Prompts:** {len(per_prompt)} meaningful Chinese natural-language prompts")
    W(f"- **Warmup:** {args.warmup} rounds per mode")
    W(f"- **Rounds:** {args.rounds} interleaved NF/F pairs per prompt")
    W(f"- **max_new_tokens:** {args.max_new_tokens}")
    W()
    W("> The measurement window is `time.perf_counter` around the `engine.generate(...)` "
      "call. NF and F runs are interleaved (NF, F, NF, F, ...) so that drift in clock or "
      "thermal state affects both modes equally and cancels in the ratio. The fused/non-fused "
      "toggle is set via `INFINILM_USE_FUSED_FFN`, which the csrc decoder layer re-reads on "
      "every forward — there is no model reload between measurements.")
    W()

    if verify_data:
        W("## Correctness")
        W()
        W(f"- prompts checked: {verify_data['total']}")
        W(f"- byte-identical outputs: {verify_data['matches']}")
        W(f"- divergent outputs: {verify_data['mismatches']}")
        if verify_data.get("first_divergences"):
            W(f"- first divergences (idx, common prefix tokens): "
              f"{verify_data['first_divergences']}")
        W(f"- status: **{'PASS' if verify_data['passed'] else 'WARN'}** "
          "(greedy decoding — token streams must match if kernels are bit-exact)")
        W()

    W("## Per-Prompt Results")
    W()
    W("| # | input_tok | NF mean (ms) | F mean (ms) | NF p99 | F p99 | Speedup | Ratio |")
    W("|---|-----------|--------------|-------------|--------|-------|---------|-------|")
    for i, r in enumerate(per_prompt):
        W(f"| {i} | {r['input_tokens']} | {r['nf']['mean']:.2f} | {r['f']['mean']:.2f} | "
          f"{r['nf']['p99']:.2f} | {r['f']['p99']:.2f} | {r['speedup_pct']:+.2f}% | "
          f"{r['ratio']:.3f}× |")
    W()

    overall_nf = mean([r["nf"]["mean"] for r in per_prompt]) if per_prompt else 0.0
    overall_f = mean([r["f"]["mean"] for r in per_prompt]) if per_prompt else 0.0
    overall_sp = speedup_pct(overall_nf, overall_f)
    overall_ratio = speedup_ratio(overall_nf, overall_f)

    W("## Aggregate")
    W()
    W(f"- Mean NF latency across prompts: **{overall_nf:.2f} ms**")
    W(f"- Mean F  latency across prompts: **{overall_f:.2f} ms**")
    W(f"- Overall end-to-end speedup: **{overall_sp:+.2f}%**  ({overall_ratio:.3f}×)")
    W()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--device", default="iluvatar",
                        help="Device name passed to infinicore.device (default: iluvatar)")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=10,
                        help="How many prompts from the pool to benchmark (default: 10)")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Tokenize the raw prompt without wrapping in the "
                             "model's chat template. Use this to measure the "
                             "very-short-input fused-FFN regime (e.g. \"你好\" "
                             "→ 2 tokens instead of ~11 with the chat wrap).")
    parser.add_argument("--output", default="bench_fused_ffn_csrc_report.md")
    parser.add_argument("--samples-json", default=None,
                        help="If set, write raw per-round samples here")
    parser.add_argument("--kv-cache-dtype", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("FUSED FFN END-TO-END BENCHMARK  (csrc engine, FM9G)")
    print("=" * 60)
    print(f"Model         : {args.model}")
    print(f"Device        : {args.device} (tp={args.tp})")
    print(f"Prompts       : {args.num_prompts} (warmup={args.warmup}, rounds={args.rounds})")
    print(f"Max new toks  : {args.max_new_tokens}")
    print()

    model = CsrcModel(args.model, args.device, args.tp,
                      kv_cache_dtype=args.kv_cache_dtype)

    prompts = PROMPTS[:args.num_prompts]
    input_ids_list = model.encode_prompts(
        prompts, use_chat_template=not args.no_chat_template)

    verify_data = None
    if not args.skip_verify:
        verify_data = verify_correctness(model, input_ids_list[:4],
                                         max_new_tokens=min(16, args.max_new_tokens))

    per_prompt: List[Dict[str, Any]] = []
    for i, (ids, text) in enumerate(zip(input_ids_list, prompts)):
        r = benchmark_prompt(model, ids, i, text,
                             max_new_tokens=args.max_new_tokens,
                             warmup=args.warmup, rounds=args.rounds)
        per_prompt.append(r)

    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    print(f"{'#':<3} {'in_tok':>7} {'NF mean':>11} {'F mean':>11} {'speedup':>9} {'ratio':>7}")
    print("-" * 76)
    for i, r in enumerate(per_prompt):
        print(f"{i:<3} {r['input_tokens']:>7} {r['nf']['mean']:>10.2f}  "
              f"{r['f']['mean']:>10.2f}  {r['speedup_pct']:>+7.2f}%  "
              f"{r['ratio']:>6.3f}×")
    overall_nf = mean([r["nf"]["mean"] for r in per_prompt])
    overall_f = mean([r["f"]["mean"] for r in per_prompt])
    print("-" * 76)
    print(f"AVG: NF={overall_nf:.2f} ms  F={overall_f:.2f} ms  "
          f"speedup={speedup_pct(overall_nf, overall_f):+.2f}%  "
          f"ratio={speedup_ratio(overall_nf, overall_f):.3f}×")

    save_markdown_report(args, verify_data, per_prompt)
    print(f"\nMarkdown report → {args.output}")

    if args.samples_json:
        with open(args.samples_json, "w", encoding="utf-8") as f:
            json.dump({
                "args": vars(args),
                "verify": verify_data,
                "per_prompt": [{
                    "prompt": r["prompt"],
                    "input_tokens": r["input_tokens"],
                    "nf_samples": r["nf_samples"],
                    "f_samples": r["f_samples"],
                } for r in per_prompt],
            }, f, ensure_ascii=False, indent=2)
        print(f"Raw samples    → {args.samples_json}")


if __name__ == "__main__":
    main()
