"""
Balanced single-request benchmark harness for InfiniLM.

Targets a fixed workload (prompt≈256 tokens, decode≈256 tokens) and produces one JSON
artifact with TTFT + decode inter-token latency, aligned with the existing InfiniLM
timing path in `InferEngine.generate(_measure_and_log_time=True)`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

from transformers import AutoTokenizer

_ex_dir = os.path.dirname(os.path.abspath(__file__))
if _ex_dir not in sys.path:
    sys.path.insert(0, _ex_dir)

from flash_attn_preload import maybe_load_flash_attn_global

maybe_load_flash_attn_global()

import infinicore

from infinilm.cache import PagedKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import GenerationConfig, InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file


_DEFAULT_BLOCK_SIZE = 256


@dataclass
class BenchRow:
    engine: str
    model_path: str
    device: str
    tp: int
    attn_backend: str
    enable_paged_attn: bool
    paged_kv_block_size: int
    prompt_tokens_target: int
    prompt_tokens_actual: int
    max_new_tokens: int
    warmup_steps: int
    runs: int
    seed: int
    load_weights_s: float
    run_wall_s: float
    ttft_ms: float
    ttft_cpu_prep_ms: float
    ttft_gpu_forward_ms: float
    ttft_gpu_sampling_ms: float
    ttft_gpu_d2h_ms: float
    ttft_unaccounted_ms: float
    avg_decode_itl_ms: float
    total_ms: float


def _read_default_prompt() -> str:
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "bench_prompt.md"), "r") as f:
        return f.read()


def _repeat_prompt_tokens(input_ids: list[int], target_len: int) -> list[int]:
    if target_len <= 0:
        return []
    if not input_ids:
        raise ValueError("Tokenized prompt is empty.")
    repeat_times = (target_len + len(input_ids) - 1) // len(input_ids)
    return (input_ids * repeat_times)[:target_len]


def _device_from_args(args) -> str:
    if args.cpu:
        return "cpu"
    if args.nvidia or args.qy or args.metax or args.iluvatar or args.ali or args.hygon:
        return "cuda"
    if args.moore:
        return "musa"
    if args.cambricon:
        return "mlu"
    raise SystemExit(
        "Pick a device flag: --cpu | --nvidia | --qy | --metax | --moore | --iluvatar | --cambricon | --ali | --hygon"
    )


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--nvidia", action="store_true")
    ap.add_argument("--qy", action="store_true")
    ap.add_argument("--metax", action="store_true")
    ap.add_argument("--moore", action="store_true")
    ap.add_argument("--iluvatar", action="store_true")
    ap.add_argument("--cambricon", action="store_true")
    ap.add_argument("--ali", action="store_true")
    ap.add_argument("--hygon", action="store_true")

    ap.add_argument("--model", "--model-path", dest="model_path", type=str, required=True)
    ap.add_argument("--tp", "--tensor-parallel-size", dest="tp", type=int, default=1)

    ap.add_argument("--attn", type=str, default="flash-attn", choices=["default", "paged-attn", "flash-attn"])
    ap.add_argument("--enable-paged-attn", action="store_true", default=False)
    ap.add_argument("--paged-kv-block-size", type=int, default=_DEFAULT_BLOCK_SIZE)

    ap.add_argument("--prompt", type=str, default=None, help="Raw user prompt content (pre chat-template).")
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=256)

    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument(
        "--timing-discard-runs",
        type=int,
        default=1,
        help="Full generate() passes with timing enabled, discarded before recorded runs "
        "(reduces first-pass inflation vs cold CUDA / autotune). Use 0 to disable.",
    )
    ap.add_argument(
        "--bench-print",
        action="store_true",
        help="Print InfiniLM per-generate timing to stdout (default: suppressed).",
    )

    ap.add_argument("--json-out", type=str, required=True)
    ap.add_argument("--print-json", action="store_true", help="Also print the JSON row to stdout.")
    ap.add_argument(
        "--print-generated",
        action="store_true",
        help="Decode and print model output tokens from the last measured run (validation).",
    )
    return ap.parse_args()


def main() -> None:
    args = get_args()
    if not args.bench_print:
        os.environ["INFINILM_SUPPRESS_BENCH_PRINTS"] = "1"
    # Enable C++ worker step timing so TTFT breakdown fields are meaningful.
    os.environ.setdefault("INFINILM_PROFILE_STEP_TIMING", "1")

    model_path = os.path.expanduser(args.model_path)
    device_str = _device_from_args(args)
    infini_device = infinicore.device(device_str, 0)

    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)

    enable_paged_attn = bool(args.enable_paged_attn)
    attn_backend = args.attn
    if enable_paged_attn and attn_backend == "default":
        attn_backend = "paged-attn"

    tok0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    prompt = _read_default_prompt() if args.prompt is None else args.prompt
    prompt_text = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    enc = tokenizer(prompt_text, truncation=True, max_length=8192)
    base_ids = enc["input_ids"]
    prompt_ids = _repeat_prompt_tokens(base_ids, args.prompt_tokens)
    prompt_tokens_actual = len(prompt_ids)
    _ = time.perf_counter() - tok0

    block_size = int(args.paged_kv_block_size)
    cache_config = None
    if enable_paged_attn:
        max_total_tokens = prompt_tokens_actual + int(args.max_new_tokens)
        num_blocks = ((max_total_tokens + block_size - 1) // block_size) * 1
        cache_config = PagedKVCacheConfig(num_blocks=num_blocks, block_size=block_size)

    t_load0 = time.perf_counter()
    engine = InferEngine(
        model_path,
        device=infini_device,
        distributed_config=DistConfig(args.tp),
        cache_config=cache_config,
        enable_graph_compiling=False,
        attention_backend=attn_backend,
    )
    load_model_state_dict_by_file(engine, model_path, dtype=engine.config.dtype)
    load_s = time.perf_counter() - t_load0

    # Warmup: run short decode to populate kernels and cache paths.
    if args.warmup_steps > 0:
        warm_ids = [prompt_ids[: min(64, len(prompt_ids))]]
        for _ in range(int(args.warmup_steps)):
            if cache_config is not None:
                engine.reset_cache(cache_config)
            engine.generate(
                infinicore.from_list(warm_ids),
                GenerationConfig(
                    max_new_tokens=max(1, int(args.max_new_tokens)),
                    eos_token_id=[],
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    stop_on_eos=False,
                ),
                _measure_and_log_time=False,
            )

    # Same path as recorded runs (timing + breakdown); discarded to avoid cold-start inflation.
    for _ in range(max(0, int(args.timing_discard_runs))):
        if cache_config is not None:
            engine.reset_cache(cache_config)
        engine.generate(
            infinicore.from_list([prompt_ids]),
            GenerationConfig(
                max_new_tokens=int(args.max_new_tokens),
                eos_token_id=[],
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                stop_on_eos=False,
            ),
            _measure_and_log_time=True,
            _return_time_measurements=True,
            _return_step_breakdown=True,
        )

    # Measured runs
    ttft_ms_list: list[float] = []
    avg_itl_ms_list: list[float] = []
    total_ms_list: list[float] = []
    run_wall_s_list: list[float] = []
    ttft_cpu_prep_ms_list: list[float] = []
    ttft_gpu_forward_ms_list: list[float] = []
    ttft_gpu_sampling_ms_list: list[float] = []
    ttft_gpu_d2h_ms_list: list[float] = []
    ttft_unaccounted_ms_list: list[float] = []

    last_output_ids: list | None = None
    for _ in range(int(args.runs)):
        if cache_config is not None:
            engine.reset_cache(cache_config)
        input_ids_infini = infinicore.from_list([prompt_ids])
        t_run0 = time.perf_counter()
        gen_out, time_measurements, breakdown = engine.generate(
            input_ids_infini,
            GenerationConfig(
                max_new_tokens=int(args.max_new_tokens),
                eos_token_id=[],
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                stop_on_eos=False,
            ),
            _measure_and_log_time=True,
            _return_time_measurements=True,
            _return_step_breakdown=True,
        )
        last_output_ids = gen_out
        run_wall_s = time.perf_counter() - t_run0

        if not time_measurements:
            raise RuntimeError("No time measurements were recorded.")

        ttft_ms = float(time_measurements[0] * 1000.0)
        cpu_prep_ms = float(breakdown["cpu_prep_s"][0] * 1000.0) if breakdown["cpu_prep_s"] else 0.0
        gpu_fwd_ms = float(breakdown["gpu_forward_ms"][0]) if breakdown["gpu_forward_ms"] else 0.0
        gpu_samp_ms = float(breakdown["gpu_sampling_ms"][0]) if breakdown["gpu_sampling_ms"] else 0.0
        gpu_d2h_ms = float(breakdown["gpu_d2h_ms"][0]) if breakdown["gpu_d2h_ms"] else 0.0
        unaccounted_ms = max(0.0, ttft_ms - (cpu_prep_ms + gpu_fwd_ms + gpu_samp_ms + gpu_d2h_ms))
        if len(time_measurements) > 1:
            avg_itl_ms = float(np.mean(time_measurements[1:]) * 1000.0)
        else:
            avg_itl_ms = 0.0
        total_ms = float(np.sum(time_measurements) * 1000.0)

        ttft_ms_list.append(ttft_ms)
        ttft_cpu_prep_ms_list.append(cpu_prep_ms)
        ttft_gpu_forward_ms_list.append(gpu_fwd_ms)
        ttft_gpu_sampling_ms_list.append(gpu_samp_ms)
        ttft_gpu_d2h_ms_list.append(gpu_d2h_ms)
        ttft_unaccounted_ms_list.append(unaccounted_ms)
        avg_itl_ms_list.append(avg_itl_ms)
        total_ms_list.append(total_ms)
        run_wall_s_list.append(run_wall_s)

    row = BenchRow(
        engine="infinilm",
        model_path=model_path,
        device=device_str,
        tp=int(args.tp),
        attn_backend=attn_backend,
        enable_paged_attn=enable_paged_attn,
        paged_kv_block_size=block_size,
        prompt_tokens_target=int(args.prompt_tokens),
        prompt_tokens_actual=int(prompt_tokens_actual),
        max_new_tokens=int(args.max_new_tokens),
        warmup_steps=int(args.warmup_steps),
        runs=int(args.runs),
        seed=int(args.seed),
        load_weights_s=float(load_s),
        run_wall_s=float(np.mean(run_wall_s_list)),
        ttft_ms=float(np.mean(ttft_ms_list)),
        ttft_cpu_prep_ms=float(np.mean(ttft_cpu_prep_ms_list)),
        ttft_gpu_forward_ms=float(np.mean(ttft_gpu_forward_ms_list)),
        ttft_gpu_sampling_ms=float(np.mean(ttft_gpu_sampling_ms_list)),
        ttft_gpu_d2h_ms=float(np.mean(ttft_gpu_d2h_ms_list)),
        ttft_unaccounted_ms=float(np.mean(ttft_unaccounted_ms_list)),
        avg_decode_itl_ms=float(np.mean(avg_itl_ms_list)),
        total_ms=float(np.mean(total_ms_list)),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(asdict(row), f, indent=2)

    if args.print_generated and last_output_ids is not None:
        gen_ids: list[int] = []
        for t in last_output_ids:
            arr = t.to_numpy()
            gen_ids.append(int(arr.reshape(-1)[0]))
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        print("\n--- generated (new tokens only) ---", flush=True)
        print(gen_text, flush=True)
        print("--- token ids:", gen_ids, "---\n", flush=True)

    if args.print_json:
        print(json.dumps(asdict(row), indent=2))


if __name__ == "__main__":
    main()

