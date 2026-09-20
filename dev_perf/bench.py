"""Offline dual-engine perf baseline (project #2 gap analysis).

One script, two interpreters:

  InfiniLM (any venv with torch; imports the built package from this checkout):
    python dev_perf/bench.py --engine infinilm --model Qwen/Qwen3-1.7B

  vLLM (isolated venv):
    /path/to/vllm-venv/bin/python dev_perf/bench.py --engine vllm --model Qwen/Qwen3-1.7B

Fairness contract: greedy decoding (temperature=0, top_k=1), ignore_eos=True,
identical prompts and max_tokens from workload.py. vLLM runs with its v1
defaults (CUDA graphs + prefix caching on); InfiniLM runs with
enable_graph=False + prefix caching on. Both facts are recorded in the output.
"""

import argparse
import json
import os
import sys
import time

from workload import (
    SHORT_PROMPTS,
    WORKLOADS,
    MemSampler,
    concurrent_prefill_workload,
    ctx_sweep_workloads,
    decode_stall_workload,
    resolve_model_path,
)

# Checkout whose built infinilm package we benchmark. Defaults to this repo's
# own python/ dir (where `xmake install _infinilm` lands its .so); override
# with INF_MAIN_PYTHON to benchmark a different checkout's build.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_CHECKOUT_PYTHON = os.environ.get("INF_MAIN_PYTHON", os.path.join(_REPO_ROOT, "python"))
# INFINI_ROOT selects the InfiniCore install (e.g. ~/.infini-fa for the
# ATen+FA build, required by the flash-attn backend); defaults to ~/.infini.
_INFINI_CORE_LIB = os.path.join(
    os.environ.get("INFINI_ROOT", os.path.expanduser("~/.infini")), "lib"
)
def _torch_lib_dir():
    """site-packages/torch/lib, needed on LD_LIBRARY_PATH for ATen-enabled
    InfiniCore builds (they link libtorch)."""
    for p in sys.path:
        d = os.path.join(p, "torch", "lib")
        if os.path.isdir(d):
            return d
    return None


LIB_DIRS = [
    _INFINI_CORE_LIB,
    os.path.join(MAIN_CHECKOUT_PYTHON, "infinilm", "lib"),
] + ([_torch_lib_dir()] if _torch_lib_dir() else [])


def ensure_ld_library_path():
    """infinilm/infinicore extensions find their .so deps via LD_LIBRARY_PATH.

    Re-exec the interpreter with the env set if needed (the dynamic linker
    reads LD_LIBRARY_PATH only at process start).
    """
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if all(d in cur.split(":") for d in LIB_DIRS):
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(LIB_DIRS + ([cur] if cur else []))
    os.execv(sys.executable, [sys.executable] + sys.argv)


def build_engine(args):
    if args.engine == "infinilm":
        ensure_ld_library_path()
        sys.path.insert(0, MAIN_CHECKOUT_PYTHON)
        from infinilm.llm.llm import LLM
        from infinilm.llm.sampling_params import SamplingParams

        speculative_method = (
            None if args.speculative_method == "none" else args.speculative_method
        )
        llm = LLM(
            model_path=args.model,
            device="cuda",
            dtype="bfloat16",
            cache_type="paged",
            attn_backend=args.attn_backend,
            max_batch_size=64,
            num_blocks=args.num_blocks,
            enable_prefix_caching=True,
            enable_graph=args.enable_graph,
            draft_model_path=args.draft_model,
            speculative_method=speculative_method,
            num_draft_tokens=args.num_draft_tokens,
        )

        def generate(prompts, max_tokens):
            return llm.generate(
                prompts=prompts,
                sampling_params=SamplingParams(
                    temperature=0.0, top_k=1, max_tokens=max_tokens, ignore_eos=True
                ),
                use_tqdm=False,
            )

        def close():
            llm.close()

        # w6_decode_stall drives the engine directly (mid-stream add_request)
        generate.llm = llm

        def spec_stats():
            """投机采样累计计数快照；未启用投机时返回 None。"""
            return llm.engine.model_runner.get_speculative_stats()

        generate.spec_stats = spec_stats

        engine_notes = {
            "engine": "infinilm",
            "cuda_graph": bool(args.enable_graph),
            "prefix_caching": True,
            "attn_backend": args.attn_backend,
            "num_blocks": args.num_blocks,
            "speculative_method": speculative_method,
            "num_draft_tokens": (
                args.num_draft_tokens if speculative_method else None
            ),
        }

    elif args.engine == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=4096,
            seed=0,
        )

        def generate(prompts, max_tokens):
            return llm.generate(
                prompts,
                SamplingParams(
                    temperature=0.0, top_k=1, max_tokens=max_tokens, ignore_eos=True
                ),
                use_tqdm=False,
            )

        def close():
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception:
                pass

        engine_notes = {"engine": "vllm", "cuda_graph": True, "prefix_caching": True, "gpu_memory_utilization": args.gpu_mem_util}

    else:
        raise ValueError(f"unknown engine: {args.engine}")

    return generate, close, engine_notes


def run_decode_stall(llm, params, dump_outputs=False):
    """w6 driver: steady decode stream + one long prompt injected mid-flight.

    n_decode short-prompt requests generate 512 tokens each; after
    INJECT_AFTER_STEPS engine steps one long prompt is added. Per-step wall
    times are recorded so the injected prefill shows up as an ITL spike for
    the decode stream (plain FCFS) or as a few mildly-longer mixed steps
    (chunked prefill). Returns a bench record dict.
    """
    from infinilm.llm.request import InferenceRequest
    from infinilm.llm.sampling_params import SamplingParams

    n_decode, inject_prompt = params
    decode_max_tokens = 512
    inject_after_steps = 64
    inject_max_tokens = 32

    engine = llm.engine

    def add(prompt, max_tokens, tag):
        req = InferenceRequest(
            request_id=f"w6-{tag}-{time.time_ns()}",
            prompt=prompt,
            prompt_token_ids=engine.tokenize(prompt),
            sampling_params=SamplingParams(
                temperature=0.0, top_k=1, max_tokens=max_tokens, ignore_eos=True
            ),
            eos_token_ids=engine.eos_token_ids,
        )
        engine.add_request(req)
        return req

    decode_reqs = [
        add(f"w6问题 {i + 1}：{SHORT_PROMPTS[i % len(SHORT_PROMPTS)]}",
            decode_max_tokens, f"d{i}")
        for i in range(n_decode)
    ]
    injected = None
    t_inject = None
    inject_ttft = None
    step_times = []
    inject_step_idx = None

    t_start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        did_work, _ = engine.step()
        dt = time.perf_counter() - t0
        if did_work:
            step_times.append(dt)
        if injected is None and len(step_times) >= inject_after_steps:
            injected = add(inject_prompt, inject_max_tokens, "p")
            t_inject = time.perf_counter()
            inject_step_idx = len(step_times)
        if (
            injected is not None
            and inject_ttft is None
            and injected.get_num_generated_tokens() >= 1
        ):
            inject_ttft = time.perf_counter() - t_inject
        if all(r.is_finished() for r in decode_reqs) and (
            injected is None or injected.is_finished()
        ):
            break
    e2e = time.perf_counter() - t_start

    pre = step_times[:inject_step_idx]
    post = step_times[inject_step_idx:]
    post_sorted = sorted(post)
    p90 = post_sorted[int(len(post_sorted) * 0.9)] if post_sorted else 0.0

    all_reqs = decode_reqs + ([injected] if injected else [])
    out_token_counts = [r.get_num_generated_tokens() for r in all_reqs]
    total_out = sum(out_token_counts)
    rec = {
        "workload": "w6_decode_stall",
        "num_requests": len(all_reqs),
        "prompt_tokens_total": sum(len(r.prompt_token_ids) for r in all_reqs),
        "output_tokens_total": total_out,
        "e2e_seconds": round(e2e, 3),
        "output_tokens_per_sec": round(total_out / e2e, 2),
        "ms_per_output_token": round(1000.0 * e2e / max(total_out, 1), 3),
        "min_output_tokens": min(out_token_counts),
        "max_output_tokens": max(out_token_counts),
        # decode-stream smoothness: mean step time before injection vs the
        # worst/p90 step after the long prompt lands
        "decode_step_ms_pre_inject": round(1000.0 * sum(pre) / max(len(pre), 1), 3),
        "step_ms_post_inject_max": round(1000.0 * max(post), 2) if post else None,
        "step_ms_post_inject_p90": round(1000.0 * p90, 2),
        "inject_prompt_tokens": len(injected.prompt_token_ids) if injected else 0,
        "inject_ttft_ms": round(1000.0 * inject_ttft, 1) if inject_ttft else None,
    }
    if dump_outputs:
        rec["output_token_ids"] = [list(r.generated_token_ids) for r in all_reqs]
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["infinilm", "vllm"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    )
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        help="infinilm only: enable graph compiling (CUDA-graph-like capture)",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=512,
        help="infinilm only: paged KV cache blocks (x256 tokens); lower it for a low-VRAM run",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.85,
        help="vllm only: gpu_memory_utilization; lower it to stay under VRAM watchdogs",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated workload names to run (default: all)",
    )
    parser.add_argument(
        "--concurrent-prefill-n",
        type=int,
        default=8,
        help="number of concurrent long prompts in w5_concurrent_prefill",
    )
    parser.add_argument(
        "--decode-stall-n",
        type=int,
        default=8,
        help="number of streaming decode requests in w6_decode_stall",
    )
    parser.add_argument(
        "--attn-backend",
        default="paged-attn",
        choices=["default", "static-attn", "paged-attn", "flash-attn", "flashinfer", "hybrid"],
        help="infinilm only: attention backend",
    )
    parser.add_argument(
        "--speculative-method",
        default="none",
        choices=["none", "eagle", "prompt_lookup"],
        help="infinilm only: speculative decoding method (prompt_lookup needs no draft model)",
    )
    parser.add_argument(
        "--draft-model",
        default=None,
        help="infinilm only: Eagle/MTP draft model directory (speculative-method=eagle)",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=4,
        help="infinilm only: draft tokens verified per speculative step",
    )
    parser.add_argument(
        "--ctx-sweep",
        action="store_true",
        help="replace the workload list with a decode-vs-context-length sweep "
        "(single request, 128 decode tokens, ~0.2k~5k ctx)",
    )
    parser.add_argument(
        "--dump-outputs",
        action="store_true",
        help="record per-request output token ids in the JSON (for cross-engine correctness diffing)",
    )
    args = parser.parse_args()

    # WSL2: vLLM disables pinned memory by default, which makes its V2 model
    # runner's UvaBuffer raise "UVA is not available". Pinned memory works on
    # WSL2 kernels >= 4.19.121 (verified locally), so opt back in.
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

    args.model = resolve_model_path(args.model)
    print(f"[bench] engine={args.engine} model={args.model}", flush=True)

    sampler = MemSampler()
    sampler.start()

    t_load0 = time.perf_counter()
    generate, close, engine_notes = build_engine(args)
    load_seconds = time.perf_counter() - t_load0
    mem_after_load = sampler.latest
    print(f"[bench] load took {load_seconds:.1f}s, mem={mem_after_load} MiB", flush=True)

    def _spec_snapshot():
        fn = getattr(generate, "spec_stats", None)
        return fn() if fn is not None else None

    def _spec_rec(before, after):
        """本 workload 区间的投机采样增量统计；未启用投机时返回 None。"""
        if after is None:
            return None
        before = before or {}
        drafted = after["drafted_tokens"] - before.get("drafted_tokens", 0)
        accepted = after["accepted_tokens"] - before.get("accepted_tokens", 0)
        steps = after["spec_steps"] - before.get("spec_steps", 0)
        emitted = after["emitted_tokens"] - before.get("emitted_tokens", 0)
        alloc_fail = after["verify_alloc_failures"] - before.get(
            "verify_alloc_failures", 0
        )
        return {
            "spec_drafted_tokens": drafted,
            "spec_accepted_tokens": accepted,
            "spec_accept_rate": round(accepted / drafted, 4) if drafted else None,
            "spec_avg_tokens_per_step": round(emitted / steps, 3) if steps else None,
            "spec_verify_alloc_failures": alloc_fail,
        }

    workloads = (
        ctx_sweep_workloads()
        if args.ctx_sweep
        else [
            concurrent_prefill_workload(args.concurrent_prefill_n)
            if name == "w5_concurrent_prefill"
            else decode_stall_workload(args.decode_stall_n)
            if name == "w6_decode_stall"
            else (name, prompts, max_tokens)
            for name, prompts, max_tokens in WORKLOADS
        ]
    )

    # Warmup (untimed): runs the first workload once to trigger lazy init /
    # graph capture.
    w1_name, w1_prompts, w1_mt = workloads[0]
    generate(w1_prompts, w1_mt)

    only = set(args.only.split(",")) if args.only else None
    results = []
    for name, prompts, max_tokens in workloads:
        if only and name not in only:
            continue
        if name == "w6_decode_stall":
            llm = getattr(generate, "llm", None)
            if llm is None:
                print(
                    f"[bench] {name}: skipped (engine exposes no raw llm handle)",
                    flush=True,
                )
                continue
            spec_before = _spec_snapshot()
            rec = run_decode_stall(llm, prompts, dump_outputs=args.dump_outputs)
            rec.update(_spec_rec(spec_before, _spec_snapshot()) or {})
            results.append(rec)
            print(
                f"[bench] {name}: e2e={rec['e2e_seconds']:.2f}s "
                f"out={rec['output_tokens_total']} tok "
                f"({rec['output_tokens_per_sec']} tok/s) "
                f"step pre={rec['decode_step_ms_pre_inject']}ms "
                f"post_max={rec['step_ms_post_inject_max']}ms "
                f"post_p90={rec['step_ms_post_inject_p90']}ms "
                f"inject_ttft={rec['inject_ttft_ms']}ms",
                flush=True,
            )
            continue
        spec_before = _spec_snapshot()
        t0 = time.perf_counter()
        outputs = generate(prompts, max_tokens)
        e2e = time.perf_counter() - t0

        prompt_tokens = sum(len(o.prompt_token_ids or []) for o in outputs)
        out_token_counts = [len(o.outputs[0].token_ids) for o in outputs]
        total_out = sum(out_token_counts)
        rec = {
            "workload": name,
            "num_requests": len(prompts),
            "prompt_tokens_total": prompt_tokens,
            "output_tokens_total": total_out,
            "e2e_seconds": round(e2e, 3),
            "output_tokens_per_sec": round(total_out / e2e, 2),
            "ms_per_output_token": round(1000.0 * e2e / max(total_out, 1), 3),
            "min_output_tokens": min(out_token_counts),
            "max_output_tokens": max(out_token_counts),
        }
        rec.update(_spec_rec(spec_before, _spec_snapshot()) or {})
        if args.dump_outputs:
            rec["output_token_ids"] = [list(o.outputs[0].token_ids) for o in outputs]
        results.append(rec)
        spec_note = ""
        if rec.get("spec_avg_tokens_per_step") is not None:
            spec_note = (
                f" spec_accept={rec['spec_accept_rate']}"
                f" avg_tok/step={rec['spec_avg_tokens_per_step']}"
            )
        print(
            f"[bench] {name}: e2e={e2e:.2f}s out={total_out} tok "
            f"({rec['output_tokens_per_sec']} tok/s, "
            f"{rec['ms_per_output_token']} ms/tok){spec_note}",
            flush=True,
        )

    sampler.stop()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        **engine_notes,
        "load_seconds": round(load_seconds, 2),
        "mem_after_load_mib": mem_after_load,
        "mem_peak_mib": sampler.peak,
        "workloads": results,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.basename(args.model.rstrip("/"))
    out_path = os.path.join(
        args.out_dir, f"{args.engine}_{tag}_{time.strftime('%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[bench] wrote {out_path}", flush=True)

    # Shut down only after results are persisted; engine teardown must not
    # be allowed to take the report down with it.
    try:
        close()
    except Exception as e:
        print(f"[bench] close() failed (results already saved): {e}", flush=True)


if __name__ == "__main__":
    main()
