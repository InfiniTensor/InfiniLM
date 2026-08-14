#!/usr/bin/env python3
"""Minimal Transformers TP benchmark runner for Qwen3_235B-A22B.

The scenario wrappers are intentionally directly executable.  When started as
``python wrapper.py`` this module replaces the process with a torchrun agent;
the original timeout therefore remains responsible for the agent, which in
turn terminates every worker on SIGTERM.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCRIPT_ROOT = str(Path(__file__).resolve().parents[1])
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from _gpu_guard import require_idle_gpu as _require_idle_gpu


MODEL_NAME = "Qwen3_235B"
DEFAULT_MODEL = "/data1/Qwen3_235B"
DEFAULT_PROMPT_FILE = "examples/bench_prompt.md"
FALLBACK_PROMPT = """High-performance language-model inference processes a prompt in a prefill phase
and then produces one token per request during decode. Tensor-parallel ranks must
exchange identical partial results, while the key/value cache keeps each request
lane isolated. The benchmark uses deterministic prompt tokens and greedy decoding
so that every reported run has an exact, auditable token count."""
MEASURED_INPUT_LENGTHS = 1
REPEATS_PER_INPUT_LENGTH = 3
MEASURED_ITERATIONS = REPEATS_PER_INPUT_LENGTH
MEASUREMENT_SEMANTICS = "one_fixed_shape_x_three_measurements"
SMOKE_OUTPUT_TOKENS = 64
HYGON_TP_PLAN = {
    "lm_head": "colwise_gather_output",
    "model.layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "model.layers.*.mlp.experts.down_proj": "rowwise",
    "model.layers.*.mlp.experts": "moe_tp_experts",
}
EXPECTED_QWEN3_235B_ARCHITECTURE = {
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "head_dim": 128,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "num_hidden_layers": 94,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 1536,
    "vocab_size": 151936,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    batch_size: int
    input_tokens: int
    output_tokens: int

    @property
    def input_lengths(self) -> tuple[int]:
        return (self.input_tokens,)

    @property
    def total_context_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _parse_args(scenario: Scenario) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transformers Qwen3_235B TP8 benchmark: "
            f"batch={scenario.batch_size}, input={scenario.input_tokens}, "
            f"output={scenario.output_tokens}, "
            f"total={scenario.total_context_tokens} tokens"
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--output-tokens", type=int, default=scenario.output_tokens)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "load the full model but run only batch=1, input=16 and output=64 "
            "to validate the TP/attention/cache path"
        ),
    )
    parser.add_argument(
        "--attention",
        choices=("eager",),
        default="eager",
        help="BW1100 correctness path; SDPA is unsupported for this model stack",
    )
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if args.smoke:
        args.output_tokens = SMOKE_OUTPUT_TOKENS
    if args.output_tokens < 2:
        parser.error("--output-tokens must be at least 2 to measure decode speed")
    if args.tp_size < 1:
        parser.error("--tp-size must be positive")
    if len(set(scenario.input_lengths)) != MEASURED_INPUT_LENGTHS:
        parser.error(
            f"scenario must define exactly {MEASURED_INPUT_LENGTHS} lengths"
        )
    return args


def _effective_scenario(scenario: Scenario, smoke: bool) -> Scenario:
    if not smoke:
        return scenario
    return Scenario(f"{scenario.name}_smoke", 1, 16, SMOKE_OUTPUT_TOKENS)


def _validate_qwen3_235b_architecture(model_config: Any) -> dict[str, int]:
    if getattr(model_config, "model_type", None) != "qwen3_moe":
        raise RuntimeError(
            "this benchmark requires model_type='qwen3_moe', got "
            f"{getattr(model_config, 'model_type', None)!r}"
        )
    actual: dict[str, int] = {}
    mismatches: list[str] = []
    for field, expected in EXPECTED_QWEN3_235B_ARCHITECTURE.items():
        value = getattr(model_config, field, None)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            mismatches.append(f"{field}={value!r} (expected {expected})")
            continue
        actual[field] = parsed
        if parsed != expected:
            mismatches.append(f"{field}={parsed} (expected {expected})")
    architectures = tuple(getattr(model_config, "architectures", None) or ())
    if "Qwen3MoeForCausalLM" not in architectures:
        mismatches.append(
            "architectures does not contain 'Qwen3MoeForCausalLM': "
            f"{architectures!r}"
        )
    if mismatches:
        raise RuntimeError(
            "checkpoint is not the expected Qwen3_235B-A22B architecture: "
            + "; ".join(mismatches)
        )
    return actual


def _build_qwen3_moe_tp_plan(
    model_config: Any,
    tp_size: int,
    scenario: Scenario,
    output_tokens: int,
) -> tuple[str | dict[str, str], dict[str, Any]]:
    """Use the correctness-first TP8 layout validated on BW1100.

    Attention stays replicated. Qwen3_235B has four KV heads, so attempting to
    tensor-parallelize attention over eight ranks produces an invalid local GQA
    layout in this Transformers/DTK stack. Only MoE experts and the LM head are
    sharded, matching the working Hygon container example.
    """
    if getattr(model_config, "model_type", None) != "qwen3_moe":
        raise RuntimeError(
            "this benchmark only supports model_type='qwen3_moe', got "
            f"{getattr(model_config, 'model_type', None)!r}"
        )

    global_query_heads = int(model_config.num_attention_heads)
    global_kv_heads = int(model_config.num_key_value_heads)
    head_dim = int(model_config.head_dim)
    num_hidden_layers = int(model_config.num_hidden_layers)
    if min(
        global_query_heads,
        global_kv_heads,
        head_dim,
        num_hidden_layers,
        tp_size,
    ) < 1:
        raise RuntimeError("TP and attention dimensions must all be positive")
    if global_query_heads % global_kv_heads:
        raise RuntimeError(
            f"global Q heads ({global_query_heads}) must be divisible by global "
            f"KV heads ({global_kv_heads})"
        )
    tp_plan = dict(HYGON_TP_PLAN)
    maximum_sequence_tokens = scenario.input_tokens + output_tokens
    dtype_bytes = 2  # BF16 K and V elements.
    kv_cache_bytes_per_rank = (
        scenario.batch_size
        * maximum_sequence_tokens
        * num_hidden_layers
        * 2
        * global_kv_heads
        * head_dim
        * dtype_bytes
    )
    plan_payload = tp_plan if isinstance(tp_plan, dict) else {"mode": tp_plan}
    metadata = {
        "tp_plan_mode": f"qwen3_moe_tp{tp_size}_experts_lm_head_only",
        "tp_plan_sha256": _stable_hash(plan_payload),
        "attention_strategy": "replicated_eager",
        "kv_projection_strategy": "replicated",
        "kv_cache_replication_factor_across_tp_ranks": tp_size,
        "global_query_heads": global_query_heads,
        "global_kv_heads": global_kv_heads,
        "head_dim": head_dim,
        "num_hidden_layers": num_hidden_layers,
        "local_query_heads": global_query_heads,
        "local_kv_heads": global_kv_heads,
        "local_gqa_groups": global_query_heads // global_kv_heads,
        "maximum_sequence_tokens": maximum_sequence_tokens,
        "estimated_dense_bf16_kv_cache_gib_per_rank": (
            kv_cache_bytes_per_rank / (1024**3)
        ),
        "kv_cache_estimate_excludes_allocator_and_cache_metadata": True,
    }
    return tp_plan, metadata


def _validate_and_set_local_gqa(
    model: Any, tp_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Verify that attention stayed fully replicated on every TP rank."""
    base_model_prefix = getattr(model, "base_model_prefix", None)
    base_model = getattr(model, base_model_prefix, None)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise RuntimeError(
            f"cannot locate {base_model_prefix!r}.layers on loaded model"
        )

    head_dim = int(tp_metadata["head_dim"])
    expected_query_heads = int(tp_metadata["local_query_heads"])
    expected_kv_heads = int(tp_metadata["local_kv_heads"])
    expected_groups = int(tp_metadata["local_gqa_groups"])
    expected_query_width = expected_query_heads * head_dim
    expected_kv_width = expected_kv_heads * head_dim
    observed_groups: set[int] = set()

    for layer_index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise RuntimeError(f"layer {layer_index} has no self_attn module")
        query_width = int(attention.q_proj.out_features)
        key_width = int(attention.k_proj.out_features)
        value_width = int(attention.v_proj.out_features)
        if query_width != expected_query_width:
            raise RuntimeError(
                f"layer {layer_index} local Q width={query_width}, expected "
                f"{expected_query_width} ({expected_query_heads} heads)"
            )
        if key_width != expected_kv_width or value_width != expected_kv_width:
            raise RuntimeError(
                f"layer {layer_index} local K/V widths={key_width}/{value_width}, "
                f"expected {expected_kv_width} ({expected_kv_heads} heads)"
            )
        observed_groups.add(int(attention.num_key_value_groups))

    if observed_groups != {expected_groups}:
        raise RuntimeError(
            "attention GQA metadata changed despite replicated attention: "
            f"observed={sorted(observed_groups)}, expected={expected_groups}"
        )

    expected_layers = int(tp_metadata["num_hidden_layers"])
    if len(layers) != expected_layers:
        raise RuntimeError(
            f"loaded model has {len(layers)} transformer layers, expected "
            f"{expected_layers}"
        )
    return {
        "validated_attention_layers": len(layers),
        "local_query_projection_width": expected_query_width,
        "local_kv_projection_width": expected_kv_width,
        "attention_replication_validated": True,
        "local_gqa_groups": expected_groups,
    }


def _launch_torchrun(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    target_path = (
        "/root/.local/bin:/opt/dtk/cuda/cuda/bin:/opt/dtk/bin:/opt/dtk/hip/bin"
    )
    target_library_path = ":".join(
        (
            "/usr/local/lib/python3.10/dist-packages/torch/lib",
            "/opt/dtk/dcc/gcvm/lib",
            "/opt/dtk/hip/lib",
            "/opt/dtk/llvm/lib",
            "/opt/dtk/lib",
            "/opt/dtk/lib64",
            "/opt/hyhal/lib",
            "/opt/hyhal/lib64",
            "/opt/dtk/dushmem/lib",
            "/opt/dtk/opencl/lib",
            "/opt/ucx/lib",
            "/opt/mpi/lib",
            "/opt/hwloc/lib",
        )
    )
    env["PATH"] = f"{target_path}:{env.get('PATH', '')}"
    inherited_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{target_library_path}:{inherited_library_path}"
        if inherited_library_path
        else target_library_path
    )
    inherited_python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"/usr/local:{inherited_python_path}"
        if inherited_python_path
        else "/usr/local"
    )
    visible_devices = ",".join(str(index) for index in range(args.tp_size))
    env.setdefault("HIP_VISIBLE_DEVICES", visible_devices)
    env.setdefault("CUDA_VISIBLE_DEVICES", visible_devices)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HSA_FORCE_FINE_GRAIN_PCIE", "1")
    env.setdefault("NCCL_DEBUG", "WARN")

    script = str(Path(sys.argv[0]).resolve())
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={args.tp_size}",
        "--max-restarts=0",
        "--monitor-interval=1",
        script,
        *sys.argv[1:],
    ]
    os.execvpe(sys.executable, command, env)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _install_hygon_grouped_mm_guard(torch: Any) -> bool:
    """Install the grouped-MM fallback validated in the BW1100 image."""
    if getattr(torch.version, "hip", None) is None:
        return False

    from transformers.integrations import moe as transformers_moe

    if getattr(transformers_moe, "_hygon_grouped_mm_guard_installed", False):
        return True

    def grouped_mm(input_tensor: Any, weight: Any, offs: Any) -> Any:
        ends = [int(value) for value in offs.detach().cpu().tolist()]
        output = input_tensor.new_empty(
            (input_tensor.shape[0], weight.shape[-1]), dtype=weight.dtype
        )
        start = 0
        for expert_index, end in enumerate(ends):
            if end > start:
                output[start:end] = input_tensor[start:end].to(weight.dtype).matmul(
                    weight[expert_index]
                )
            start = end
        return output

    transformers_moe._grouped_mm = grouped_mm
    transformers_moe._hygon_grouped_mm_guard_installed = True
    return True


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _emit(rank: int, tag: str, payload: dict[str, Any]) -> None:
    if rank != 0:
        return
    if tag == "PYTORCH_QWEN3_235B_CONFIG":
        print(
            "[Transformers] "
            f"model={payload['model']} batch={payload['batch_size']} "
            f"input={payload['input_lengths'][0]} "
            f"output={payload['output_tokens_per_request']} "
            f"tp={payload['tp_size']} attention={payload['attention_implementation']}",
            flush=True,
        )
        print(
            f" load weights over! {payload['model_load_seconds'] * 1000.0:.2f} ms ",
            flush=True,
        )
    elif tag == "PYTORCH_QWEN3_235B_COMPLETE":
        print(f"Transformers benchmark status: {payload['status']}", flush=True)


def _print_infinilm_style_metrics(
    rank: int, measurement: dict[str, Any], decoded_output: str
) -> None:
    if rank != 0:
        return
    print(
        f"\n Generation completed in {measurement['generation_seconds'] * 1000.0:.2f} ms",
        flush=True,
    )
    print(
        f" Batchsize={measurement['batch_size']}  "
        f"Per_Batch_Input_Len={measurement['input_tokens_per_request']}  "
        f"Per_Batch_New_Tokens={measurement['output_tokens_per_request']}",
        flush=True,
    )
    print(
        f"\n Prefill TTFT: {measurement['ttft_seconds'] * 1000.0:.2f} ms  "
        f"Throughput: {measurement['prefill_tokens_per_second']:.2f} tok/s",
        flush=True,
    )
    print(
        f"\n Decode  Avg ITL: {measurement['inter_token_latency_ms']:.2f} ms   "
        f"Throughput: {measurement['decode_tokens_per_second']:.2f} tok/s\n",
        flush=True,
    )
    print(decoded_output or "（未生成可显示文本）", flush=True)


def _validate_decoded_output(decoded_output: str) -> dict[str, Any]:
    text = decoded_output.strip()
    if not text:
        raise RuntimeError("generated output decoded to an empty string")
    if "\ufffd" in text:
        raise RuntimeError("generated output contains Unicode replacement characters")
    printable_characters = sum(
        character.isprintable() or character in "\n\t" for character in text
    )
    printable_ratio = printable_characters / len(text)
    cjk_characters = sum("\u3400" <= character <= "\u9fff" for character in text)
    url_fragments = text.lower().count("http")
    if printable_ratio < 0.95:
        raise RuntimeError(
            f"generated output printable ratio is too low: {printable_ratio:.3f}"
        )
    if cjk_characters < 8:
        raise RuntimeError(
            f"generated output is not a substantive Chinese response: CJK={cjk_characters}"
        )
    if url_fragments > 2:
        raise RuntimeError(
            f"generated output contains suspicious URL fragments: {url_fragments}"
        )
    return {
        "nonempty_decoded_text": True,
        "no_replacement_characters": True,
        "printable_ratio": printable_ratio,
        "cjk_character_count": cjk_characters,
        "url_fragment_count": url_fragments,
    }


def _max_across_ranks(value: float, torch: Any, dist: Any, device: Any) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _make_prompt_base(
    tokenizer: Any, prompt_file: str
) -> tuple[list[int], dict[str, Any]]:
    prompt_path = Path(prompt_file).resolve()
    prompt_file_exists = prompt_path.is_file()
    prompt_text = (
        prompt_path.read_text(encoding="utf-8")
        if prompt_file_exists
        else FALLBACK_PROMPT
    ).strip()
    if not prompt_text:
        raise RuntimeError(f"benchmark prompt file is empty: {prompt_path}")
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("model tokenizer does not define a chat template")
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Match InfiniLM's benchmark path: use tokenizer.encode defaults after the
    # model's chat template, then repeat this exact base sequence to each length.
    token_ids = list(tokenizer.encode(rendered_prompt))
    if not token_ids:
        raise RuntimeError("the fixed benchmark prompt tokenized to an empty list")
    return token_ids, {
        "prompt_source": (
            "file_chat_template" if prompt_file_exists else "embedded_fallback"
        ),
        "prompt_file": str(prompt_path),
        "prompt_file_exists": prompt_file_exists,
        "prompt_file_sha256": hashlib.sha256(
            prompt_text.encode("utf-8")
        ).hexdigest(),
        "rendered_prompt_sha256": hashlib.sha256(
            rendered_prompt.encode("utf-8")
        ).hexdigest(),
    }


def _repeat_prompt(token_ids: Sequence[int], target_length: int) -> list[int]:
    if target_length < 1:
        raise ValueError("target_length must be positive")
    base = list(token_ids)
    if target_length <= len(base):
        # Preserve the assistant-generation suffix instead of cutting it off.
        result = base[-target_length:]
    else:
        prefix_length = target_length - len(base)
        repeats = (prefix_length + len(base) - 1) // len(base)
        result = (base * repeats)[:prefix_length] + base
    if len(result) != target_length:
        raise RuntimeError(f"expected {target_length} prompt tokens, got {len(result)}")
    return result


def _materialize_logits(logits: Any) -> Any:
    # A replicated TP lm_head returns Tensor.  Keep this guard for TP plans that
    # leave the vocabulary output as a DTensor.
    if type(logits).__name__ == "DTensor" and hasattr(logits, "full_tensor"):
        return logits.full_tensor()
    return logits


def _forward(
    model: Any,
    logits_limit_argument: str,
    input_ids: Any,
    past_key_values: Any | None = None,
) -> tuple[Any, Any]:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": True,
        "return_dict": True,
        logits_limit_argument: 1,
    }
    outputs = model(**kwargs)
    if outputs.past_key_values is None:
        raise RuntimeError("model did not return past_key_values with use_cache=True")
    logits = _materialize_logits(outputs.logits)
    if logits.ndim != 3 or logits.shape[0] != input_ids.shape[0]:
        raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")
    if logits.shape[1] != 1:
        raise RuntimeError(
            f"expected one retained logits position, got shape {tuple(logits.shape)}"
        )
    return logits[:, -1, :], outputs.past_key_values


def _validate_output(
    generated: Any,
    last_logits: Any,
    batch_size: int,
    output_tokens: int,
    vocab_size: int,
    torch: Any,
    dist: Any,
) -> tuple[Any, dict[str, Any]]:
    expected_shape = (batch_size, output_tokens)
    if tuple(generated.shape) != expected_shape:
        raise RuntimeError(
            f"expected generated shape {expected_shape}, got {tuple(generated.shape)}"
        )

    rank_minimum = generated.clone()
    rank_maximum = generated.clone()
    dist.all_reduce(rank_minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(rank_maximum, op=dist.ReduceOp.MAX)
    rank_consensus = bool(torch.equal(rank_minimum, rank_maximum))

    valid_ids = bool(
        torch.logical_and(generated >= 0, generated < vocab_size).all().item()
    )
    finite_logits = bool(torch.isfinite(last_logits).all().item())
    checks = torch.tensor(
        [int(rank_consensus), int(valid_ids), int(finite_logits)],
        dtype=torch.int32,
        device=generated.device,
    )
    dist.all_reduce(checks, op=dist.ReduceOp.MIN)
    rank_consensus, valid_ids, finite_logits = [bool(value) for value in checks.tolist()]
    if not (rank_consensus and valid_ids and finite_logits):
        raise RuntimeError(
            "correctness validation failed: "
            f"rank_consensus={rank_consensus}, valid_ids={valid_ids}, "
            f"finite_logits={finite_logits}"
        )

    generated_cpu = generated.cpu()
    matrix = generated_cpu.tolist()
    return generated_cpu, {
        "exact_output_shape": True,
        "rank_consensus": rank_consensus,
        "valid_token_ids": valid_ids,
        "finite_last_logits": finite_logits,
        "output_token_ids_sha256": _stable_hash(matrix),
        "first_request_first_16_tokens": matrix[0][:16],
    }


def _run_iteration(
    model: Any,
    prompt_base: Sequence[int],
    batch_size: int,
    input_tokens: int,
    output_tokens: int,
    vocab_size: int,
    logits_limit_argument: str,
    torch: Any,
    dist: Any,
    device: Any,
) -> dict[str, Any]:
    prompt = _repeat_prompt(prompt_base, input_tokens)
    input_ids = (
        torch.tensor(prompt, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    prefill_start = time.perf_counter()
    logits, past_key_values = _forward(
        model, logits_limit_argument, input_ids, past_key_values=None
    )
    next_token = torch.argmax(logits, dim=-1)
    torch.cuda.synchronize(device)
    prefill_local_seconds = time.perf_counter() - prefill_start
    prefill_seconds = _max_across_ranks(
        prefill_local_seconds, torch, dist, device
    )

    generated_tokens = [next_token]
    decode_start = time.perf_counter()
    for _ in range(output_tokens - 1):
        logits, past_key_values = _forward(
            model,
            logits_limit_argument,
            next_token.unsqueeze(1),
            past_key_values=past_key_values,
        )
        next_token = torch.argmax(logits, dim=-1)
        generated_tokens.append(next_token)
    torch.cuda.synchronize(device)
    decode_local_seconds = time.perf_counter() - decode_start
    decode_seconds = _max_across_ranks(decode_local_seconds, torch, dist, device)

    peak_allocated_gib = _max_across_ranks(
        torch.cuda.max_memory_allocated(device) / (1024**3), torch, dist, device
    )
    peak_reserved_gib = _max_across_ranks(
        torch.cuda.max_memory_reserved(device) / (1024**3), torch, dist, device
    )
    generated = torch.stack(generated_tokens, dim=1)
    generated_cpu, correctness = _validate_output(
        generated,
        logits,
        batch_size,
        output_tokens,
        vocab_size,
        torch,
        dist,
    )

    total_prompt_tokens = batch_size * input_tokens
    decode_token_count = batch_size * (output_tokens - 1)
    generated_token_count = batch_size * output_tokens
    total_seconds = prefill_seconds + decode_seconds
    result = {
        "batch_size": batch_size,
        "input_tokens_per_request": input_tokens,
        "prompt_token_ids_sha256": _stable_hash(prompt),
        "output_tokens_per_request": output_tokens,
        "total_context_tokens_per_request": input_tokens + output_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": generated_token_count,
        "ttft_seconds": prefill_seconds,
        "prefill_tokens_per_second": total_prompt_tokens / prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": decode_token_count / decode_seconds,
        "decode_tokens_per_second_per_request": (
            (output_tokens - 1) / decode_seconds
        ),
        "inter_token_latency_ms": decode_seconds * 1000.0 / (output_tokens - 1),
        "generation_seconds": total_seconds,
        "generated_tokens_per_second": generated_token_count / total_seconds,
        "peak_memory_allocated_gib_max_rank": peak_allocated_gib,
        "peak_memory_reserved_gib_max_rank": peak_reserved_gib,
        "correctness": correctness,
        "first_request_output_token_ids": generated_cpu[0].tolist(),
    }

    del generated_cpu, generated, generated_tokens, logits, next_token
    del past_key_values, input_ids
    return result


def _median_summary(measurements: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "ttft_seconds",
        "prefill_tokens_per_second",
        "decode_seconds",
        "decode_tokens_per_second",
        "decode_tokens_per_second_per_request",
        "inter_token_latency_ms",
        "generation_seconds",
        "generated_tokens_per_second",
        "peak_memory_allocated_gib_max_rank",
        "peak_memory_reserved_gib_max_rank",
    )
    return {
        f"median_{field}": statistics.median(
            float(measurement[field]) for measurement in measurements
        )
        for field in fields
    }


def _per_length_medians(
    measurements: Sequence[dict[str, Any]], input_lengths: Sequence[int]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for input_tokens in input_lengths:
        records = [
            measurement
            for measurement in measurements
            if int(measurement["input_tokens_per_request"]) == input_tokens
        ]
        if len(records) != REPEATS_PER_INPUT_LENGTH:
            raise RuntimeError(
                f"input length {input_tokens}: recorded {len(records)} repeats; "
                f"expected {REPEATS_PER_INPUT_LENGTH}"
            )
        summaries.append(
            {
                "input_tokens_per_request": input_tokens,
                "measured_repeats": len(records),
                **_median_summary(records),
            }
        )
    return summaries


def _overall_median_of_per_length_medians(
    per_length_medians: Sequence[dict[str, Any]],
) -> dict[str, float]:
    metric_names = [
        name for name in per_length_medians[0] if name.startswith("median_")
    ]
    return {
        name: statistics.median(
            float(length_summary[name]) for length_summary in per_length_medians
        )
        for name in metric_names
    }


def _run_worker_impl(args: argparse.Namespace, scenario: Scenario) -> int:
    import inspect

    import torch
    import torch.distributed as dist
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    grouped_mm_fallback = _install_hygon_grouped_mm_guard(torch)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    if rank != 0:
        transformers.utils.logging.disable_progress_bar()
    if world_size != args.tp_size:
        raise RuntimeError(
            f"expected WORLD_SIZE={args.tp_size}, got WORLD_SIZE={world_size}"
        )
    if local_world_size != args.tp_size:
        raise RuntimeError(
            "this benchmark requires all TP ranks on one host: "
            f"LOCAL_WORLD_SIZE={local_world_size}, TP={args.tp_size}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda is unavailable")
    if torch.cuda.device_count() < local_world_size:
        raise RuntimeError(
            f"need {local_world_size} visible GPUs, found {torch.cuda.device_count()}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    # Each torchrun worker owns one device.  Avoid manual_seed_all(), which can
    # make every worker initialize a context on all eight visible GPUs.
    torch.random.default_generator.manual_seed(0)
    torch.cuda.manual_seed(0)

    model_config = AutoConfig.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    architecture_signature = _validate_qwen3_235b_architecture(model_config)
    quantization_config = getattr(model_config, "quantization_config", None)
    if quantization_config:
        raise RuntimeError(
            "the Transformers benchmark is BF16-only; refusing quantized "
            f"checkpoint {args.model!r} with quantization_config="
            f"{quantization_config!r}"
        )
    tp_plan, tp_metadata = _build_qwen3_moe_tp_plan(
        model_config,
        args.tp_size,
        scenario,
        args.output_tokens,
    )

    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=model_config,
        dtype=torch.bfloat16,
        attn_implementation=args.attention,
        tp_plan=tp_plan,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    tp_validation = _validate_and_set_local_gqa(model, tp_metadata)
    model.eval()
    torch.cuda.synchronize(device)
    if not dist.is_initialized():
        raise RuntimeError(
            "Transformers tp_plan='auto' did not initialize torch.distributed"
        )
    load_seconds = _max_across_ranks(
        time.perf_counter() - load_start, torch, dist, device
    )

    tp_plan = getattr(model, "_tp_plan", None)
    if not tp_plan:
        raise RuntimeError("model loaded without a non-empty Transformers TP plan")
    resolved_attention = getattr(model.config, "_attn_implementation", None)
    if resolved_attention != args.attention:
        raise RuntimeError(
            f"requested attention={args.attention!r}, loaded model resolved "
            f"attention={resolved_attention!r}"
        )
    forward_parameters = inspect.signature(model.forward).parameters
    if "logits_to_keep" in forward_parameters:
        logits_limit_argument = "logits_to_keep"
    elif "num_logits_to_keep" in forward_parameters:
        logits_limit_argument = "num_logits_to_keep"
    else:
        raise RuntimeError(
            "model.forward has no logits_to_keep argument; refusing to materialize "
            "full [batch, context, vocab] logits for this benchmark"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    prompt_base, prompt_metadata = _make_prompt_base(tokenizer, args.prompt_file)
    vocab_size = int(model.config.vocab_size)
    if any(token < 0 or token >= vocab_size for token in prompt_base):
        raise RuntimeError("fixed prompt contains a token outside model vocabulary")

    maximum_position_embeddings = int(
        getattr(model.config, "max_position_embeddings", 0) or 0
    )
    maximum_requested = scenario.input_tokens + args.output_tokens
    if maximum_position_embeddings and maximum_requested > maximum_position_embeddings:
        raise RuntimeError(
            f"requested sequence length {maximum_requested} exceeds "
            f"max_position_embeddings={maximum_position_embeddings}"
        )

    config = {
        "framework": "transformers",
        "scenario": scenario.name,
        "model_name": MODEL_NAME,
        "model": str(Path(args.model).absolute()),
        "model_realpath": str(Path(args.model).resolve()),
        "model_class": type(model).__name__,
        "dtype": "bfloat16",
        "checkpoint_quantized": False,
        "validated_qwen3_235b_architecture": architecture_signature,
        "attention_implementation": resolved_attention,
        "tp_plan": tp_metadata["tp_plan_mode"],
        "tp_plan_rules": tp_plan,
        "tp_plan_rule_count": len(tp_plan) if isinstance(tp_plan, dict) else None,
        "tp_size": args.tp_size,
        "smoke": args.smoke,
        "batch_size": scenario.batch_size,
        "input_lengths": list(scenario.input_lengths),
        "output_tokens_per_request": args.output_tokens,
        "total_context_tokens_per_request": maximum_requested,
        "measured_iterations": MEASURED_ITERATIONS,
        "measured_input_lengths": MEASURED_INPUT_LENGTHS,
        "repeats_per_input_length": REPEATS_PER_INPUT_LENGTH,
        "measurement_semantics": MEASUREMENT_SEMANTICS,
        "model_load_seconds": load_seconds,
        "fixed_prompt_base_tokens": len(prompt_base),
        "fixed_prompt_base_sha256": _stable_hash(prompt_base),
        **prompt_metadata,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "flash_attn_version": _package_version("flash-attn"),
        "hygon_transformers_grouped_mm_fallback": grouped_mm_fallback,
        "gpu_name": torch.cuda.get_device_name(device),
        **tp_metadata,
        **tp_validation,
    }
    _emit(rank, "PYTORCH_QWEN3_235B_CONFIG", config)

    measurements: list[dict[str, Any]] = []
    iteration = 0
    for length_index, input_tokens in enumerate(scenario.input_lengths, start=1):
        with torch.inference_mode():
            shape_warmup = _run_iteration(
                model,
                prompt_base,
                scenario.batch_size,
                input_tokens,
                args.output_tokens,
                vocab_size,
                logits_limit_argument,
                torch,
                dist,
                device,
            )
        shape_warmup_hash = shape_warmup["correctness"][
            "output_token_ids_sha256"
        ]
        shape_warmup_prompt_hash = shape_warmup["prompt_token_ids_sha256"]
        _emit(
            rank,
            "PYTORCH_QWEN3_235B_SHAPE_WARMUP",
            {
                "scenario": scenario.name,
                "length_index": length_index,
                "batch_size": scenario.batch_size,
                "input_tokens_per_request": input_tokens,
                "prompt_token_ids_sha256": shape_warmup_prompt_hash,
                "output_tokens_per_request": args.output_tokens,
                "output_token_ids_sha256": shape_warmup_hash,
                "correctness": shape_warmup["correctness"],
            },
        )
        del shape_warmup
        gc.collect()

        for repeat in range(1, REPEATS_PER_INPUT_LENGTH + 1):
            iteration += 1
            with torch.inference_mode():
                measurement = _run_iteration(
                    model,
                    prompt_base,
                    scenario.batch_size,
                    input_tokens,
                    args.output_tokens,
                    vocab_size,
                    logits_limit_argument,
                    torch,
                    dist,
                    device,
                )
            measured_hash = measurement["correctness"][
                "output_token_ids_sha256"
            ]
            measured_prompt_hash = measurement["prompt_token_ids_sha256"]
            if measured_prompt_hash != shape_warmup_prompt_hash:
                raise RuntimeError(
                    f"input length {input_tokens} repeat {repeat}: measured prompt "
                    f"hash {measured_prompt_hash} does not match exact-shape "
                    f"warmup prompt hash {shape_warmup_prompt_hash}"
                )
            # Hygon BF16 kernels can make numerically valid MoE routing choices
            # differ across independent runs. Keep the replay hash observable,
            # while treating per-run shape/range/finite/rank checks as correctness.
            measurement["correctness"]["output_matches_exact_shape_warmup"] = (
                measured_hash == shape_warmup_hash
            )
            measurement["exact_shape_warmup_output_sha256"] = shape_warmup_hash
            measurement = {
                "scenario": scenario.name,
                "iteration": iteration,
                "length_index": length_index,
                "repeat": repeat,
                **measurement,
            }
            output_token_ids = measurement.pop("first_request_output_token_ids")
            decoded_output = tokenizer.decode(
                output_token_ids, skip_special_tokens=True
            ).strip()
            measurement["correctness"]["decoded_output"] = (
                _validate_decoded_output(decoded_output)
            )
            measurements.append(measurement)
            _print_infinilm_style_metrics(rank, measurement, decoded_output)
            _emit(rank, "PYTORCH_QWEN3_235B_ITERATION", measurement)
            gc.collect()

    if len(measurements) != MEASURED_ITERATIONS:
        raise RuntimeError(
            f"expected {MEASURED_ITERATIONS} measurements, got {len(measurements)}"
        )
    per_length_medians = _per_length_medians(
        measurements, scenario.input_lengths
    )
    overall_medians = _overall_median_of_per_length_medians(per_length_medians)
    summary = {
        "scenario": scenario.name,
        "status": "PASS",
        "measured_iterations": len(measurements),
        "measured_input_lengths": MEASURED_INPUT_LENGTHS,
        "repeats_per_input_length": REPEATS_PER_INPUT_LENGTH,
        "measurement_semantics": MEASUREMENT_SEMANTICS,
        "input_lengths": list(scenario.input_lengths),
        "batch_size": scenario.batch_size,
        "output_tokens_per_request": args.output_tokens,
        "total_context_tokens_per_request": maximum_requested,
        "per_length_medians": per_length_medians,
        "overall_aggregate": {
        "aggregation_method": "median_of_three_fixed_shape_measurements",
        "measurement_count": len(measurements),
        "mixed_input_lengths": False,
            **overall_medians,
        },
        # Compatibility aliases for existing table consumers. Their scope is the
        # explicitly labeled overall aggregate above, not a single input length.
        **overall_medians,
        "output_token_ids_sha256": [
            item["correctness"]["output_token_ids_sha256"] for item in measurements
        ],
    }
    _emit(rank, "PYTORCH_QWEN3_235B_SUMMARY", summary)
    return len(measurements)


def _run_worker(args: argparse.Namespace, scenario: Scenario) -> None:
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    measured_iterations = 0
    caught: BaseException | None = None
    caught_traceback: Any = None
    teardown_errors: list[str] = []
    process_group_was_initialized = False
    try:
        measured_iterations = _run_worker_impl(args, scenario)
    except BaseException as error:
        caught = error
        caught_traceback = error.__traceback__
    finally:
        process_group_was_initialized = dist.is_initialized()
        if process_group_was_initialized:
            if caught is None:
                try:
                    dist.barrier()
                except BaseException as error:
                    caught = error
                    caught_traceback = error.__traceback__
                    teardown_errors.append(
                        f"barrier: {type(error).__name__}: {error}"
                    )
            try:
                dist.destroy_process_group()
            except BaseException as error:
                if caught is None:
                    caught = error
                    caught_traceback = error.__traceback__
                teardown_errors.append(
                    f"destroy_process_group: {type(error).__name__}: {error}"
                )

        teardown_complete = not dist.is_initialized() and not teardown_errors
        status = (
            "PASS"
            if caught is None
            and measured_iterations == MEASURED_ITERATIONS
            and teardown_complete
            else "ERROR"
        )
        completion: dict[str, Any] = {
            "scenario": scenario.name,
            "status": status,
            "exit_code": 0 if status == "PASS" else 1,
            "measured_iterations": measured_iterations,
            "measured_input_lengths": MEASURED_INPUT_LENGTHS,
            "repeats_per_input_length": REPEATS_PER_INPUT_LENGTH,
            "measurement_semantics": MEASUREMENT_SEMANTICS,
            "process_group_was_initialized": process_group_was_initialized,
            "distributed_teardown_complete": teardown_complete,
        }
        if caught is not None:
            completion["error"] = {
                "type": type(caught).__name__,
                "message": str(caught),
            }
        if teardown_errors:
            completion["teardown_errors"] = teardown_errors
        _emit(rank, "PYTORCH_QWEN3_235B_COMPLETE", completion)

    if caught is not None:
        raise caught.with_traceback(caught_traceback)


def main(scenario: Scenario) -> None:
    args = _parse_args(scenario)
    scenario = _effective_scenario(scenario, args.smoke)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if "LOCAL_RANK" not in os.environ and world_size == 1:
        _require_idle_gpu()
        _launch_torchrun(args)
        raise AssertionError("os.execvpe returned unexpectedly")
    _run_worker(args, scenario)
