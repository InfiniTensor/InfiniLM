import argparse
import os
import time

import torch
from transformers import AutoTokenizer

from infllmv2_loader import preload_infllmv2_if_available

preload_infllmv2_if_available()

import infinicore
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.cache import StaticKVCacheConfig
from infinilm.modeling_utils import load_model_state_dict_by_file


def _sync_infini_device():
    # Best-effort sync for timing stability.
    try:
        infinicore.sync_stream()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--prompt", default="How are you")
    ap.add_argument("--seq_len", type=int, default=None, help="If set, pad/truncate tokenized prompt to this length")
    ap.add_argument("--out", default="/tmp/torchprof_prefill_infinilm.txt", help="Write launch summary and table to this file")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--active", type=int, default=3)
    args = ap.parse_args()

    # Tokenize on CPU.
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    enc = tok(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    if args.seq_len is not None:
        if input_ids.numel() >= args.seq_len:
            input_ids = input_ids[: args.seq_len]
        else:
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
            input_ids = torch.cat([input_ids, torch.full((args.seq_len - input_ids.numel(),), pad_id, dtype=input_ids.dtype)])
    seqlen = int(input_ids.numel())

    # Build InfiniLM engine.
    inf_cuda_index = int(os.environ.get("INFINILM_CUDA_INDEX", "0"))
    inf_dev = infinicore.device("cuda", inf_cuda_index)
    eng = InferEngine(
        model_path=args.model_path,
        device=inf_dev,
        distributed_config=DistConfig(1),
        cache_config=StaticKVCacheConfig(max_batch_size=1, max_cache_len=max(2048, seqlen + 8)),
        enable_graph_compiling=False,
        attention_backend="default",
    )
    load_model_state_dict_by_file(eng, args.model_path, dtype=infinicore.bfloat16)
    _sync_infini_device()

    # Prepare all inputs ONCE to avoid profiling HtoD setup work.
    # Workaround used elsewhere: int32 ids.
    ids_i32 = input_ids.to(torch.int32).view(1, seqlen).contiguous()
    pos = torch.arange(seqlen, dtype=torch.int64).view(1, seqlen).contiguous()
    input_offsets = torch.tensor([0, seqlen], dtype=torch.int32).contiguous()
    past = torch.tensor([0], dtype=torch.int32).contiguous()
    total = torch.tensor([seqlen], dtype=torch.int32).contiguous()

    ids_inf = infinicore.from_torch(ids_i32)
    pos_inf = infinicore.from_torch(pos)
    input_offsets_inf = infinicore.from_torch(input_offsets)
    past_inf = infinicore.from_torch(past)
    total_inf = infinicore.from_torch(total)

    def run_once():
        _ = eng.forward_logits(
            ids_inf,
            position_ids=pos_inf,
            past_kv_lengths=past_inf,
            total_kv_lengths=total_inf,
            input_offsets=input_offsets_inf,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        )
        _sync_infini_device()

    # Warmup (not profiled).
    for _ in range(args.warmup):
        run_once()

    # Profile active runs.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
        with_modules=False,
    ) as prof:
        t0 = time.time()
        for _ in range(args.active):
            run_once()
        t1 = time.time()

    events = prof.key_averages()
    n_launch = sum(e.count for e in events if "LaunchKernel" in e.key or e.key == "cudaLaunchKernel")
    n_memcpy = sum(e.count for e in events if "Memcpy" in e.key or "memcpy" in e.key.lower())
    cpu_launch_us = sum(e.cpu_time_total for e in events if "LaunchKernel" in e.key or e.key == "cudaLaunchKernel")

    print(f"prefill_only seq_len={seqlen} warmup={args.warmup} active={args.active}")
    print(f"elapsed_s={t1 - t0:.3f}")
    print(f"[launch_summary] cudaLaunchKernel_count={n_launch} cudaMemcpy_count={n_memcpy} cpu_launch_time_us={cpu_launch_us:.0f} (over {args.active} runs, divide by {args.active} for per-prefill)")
    print()
    print(events.table(sort_by="self_cuda_time_total", row_limit=40))

    if args.out and args.out != "-":
        with open(args.out, "w") as f:
            f.write(f"prefill seq_len={seqlen} active={args.active}\n")
            f.write(f"cudaLaunchKernel_count={n_launch} cudaMemcpy_count={n_memcpy}\n\n")
            f.write(events.table(sort_by="self_cuda_time_total", row_limit=80))
        print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()

