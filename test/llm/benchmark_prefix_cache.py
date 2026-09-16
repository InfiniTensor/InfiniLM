#!/usr/bin/env python3
"""Reproducible control-plane and model prefix-cache benchmark."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import platform
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from types import ModuleType

BLOCK_SIZE = 256
OUTPUT_TOKENS = 32
REQUEST_COUNT = 128
WARMUP_REQUESTS = 16
SOURCE_TEXTS = (
    "You are a careful technical assistant. Follow the system instructions and "
    "answer with precise, verifiable details. ",
    "Paged key value caches store completed attention states in fixed sized blocks. "
    "Prefix reuse avoids repeated prefill computation when requests share input. ",
    "Explain how deterministic experiments separate control plane overhead from "
    "model execution and why complete traces are needed for reproducibility. ",
)


def _percentile(values, percentile):
    ordered = sorted(values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * percentile
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _distribution(values):
    return {
        "median_ns": statistics.median(values) if values else None,
        "p95_ns": _percentile(values, 0.95),
        "valid_samples": len(values),
    }


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_request_timing(request_id, prompt_tokens, started, token_ids, token_times):
    if not token_ids or len(token_ids) != len(token_times):
        raise RuntimeError(
            "The request produced no token IDs or mismatched timestamps."
        )
    return {
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "output_token_ids": list(token_ids),
        "ttft_seconds": token_times[0] - started,
        "delivery_intervals_seconds": [
            later - earlier for earlier, later in zip(token_times, token_times[1:])
        ],
        "latency_seconds": token_times[-1] - started,
    }


def summarize_requests(requests, wall_seconds):
    generated = sum(len(item["output_token_ids"]) for item in requests)
    if wall_seconds <= 0:
        raise ValueError("wall_seconds must be positive")
    return {
        "request_count": len(requests),
        "generated_tokens": generated,
        "wall_seconds": wall_seconds,
        "throughput_tokens_per_second": generated / wall_seconds,
        "ttft_median_seconds": statistics.median(
            item["ttft_seconds"] for item in requests
        ),
    }


class SchedulerAccounting:
    """Record cache work only for requests returned as scheduled prefill work."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.original = scheduler.schedule
        self.by_request = {}
        self.admission_counts = {}
        self.max_observed_block_ref_count = 0

    def install(self):
        def measured_schedule():
            output = self.original()
            if output is not None and output.is_prefill:
                for request in output.scheduled_requests:
                    self.admission_counts[request.request_id] = (
                        self.admission_counts.get(request.request_id, 0) + 1
                    )
                    self.by_request[request.request_id] = {
                        "local_cached_tokens": request.num_local_cached_tokens,
                        "prefill_tokens": len(request.slot_mapping),
                    }
                    manager = getattr(self.scheduler, "cache_manager", None)
                    for block_id in getattr(request, "block_table", ()):
                        self.max_observed_block_ref_count = max(
                            self.max_observed_block_ref_count,
                            manager.blocks[block_id].ref_count if manager else 0,
                        )
            return output

        self.scheduler.schedule = measured_schedule

    def restore(self):
        self.scheduler.schedule = self.original

    def require_exactly_once(self, request_ids):
        missing = [
            request_id
            for request_id in request_ids
            if self.admission_counts.get(request_id, 0) == 0
        ]
        duplicates = [
            request_id
            for request_id in request_ids
            if self.admission_counts.get(request_id, 0) > 1
        ]
        if missing or duplicates:
            raise RuntimeError(
                "Invalid admitted-prefill accounting: "
                f"missing={missing}, duplicate={duplicates}"
            )


def _expanded_tokens(token_sources, minimum=4096):
    tokens = [token for source in token_sources for token in source]
    if not tokens:
        raise ValueError("Tokenizer produced no ordinary token IDs")
    return (tokens * (minimum // len(tokens) + 2))[:minimum]


def _family(corpus, seed, family_index, length):
    # A family-local PRNG composes only real corpus IDs while avoiding periodic
    # rotations when the source text contains fewer than 128 distinct positions.
    rng = random.Random(f"prefix-family-{seed}-{family_index}")
    return [corpus[rng.randrange(len(corpus))] for _ in range(length)]


def build_trace(scenario, token_sources, seed, request_count=REQUEST_COUNT):
    rng = random.Random(seed)
    corpus = _expanded_tokens(token_sources)

    def tail(index):
        start = (2048 + index * 37) % len(corpus)
        return _family(corpus, seed, f"tail-{start}", 32)

    requests = []
    for index in range(request_count):
        phase = "measurement"
        group = None
        if scenario == "hot-cold":
            family_index = 0 if index % 4 < 3 else index + 1
            prefix_length = 1024
            group = "hot" if family_index == 0 else "cold"
        elif scenario == "hot-shift":
            family_index = 0 if index < 64 else 1
            prefix_length = 1024
            if index >= 64:
                group = "post_shift_first_16" if index < 80 else "post_shift_stable"
            else:
                group = "prefix_a"
        elif scenario == "no-reuse":
            family_index = index + rng.randrange(1, 1 << 30) * request_count
            prefix_length = 1024
        elif scenario == "over-capacity":
            family_index = index % 32
            prefix_length = 1024
        elif scenario == "mixed-length":
            prefix_length = (256, 1024, 2048)[index % 3]
            family_index = index % 4
            group = str(prefix_length)
        elif scenario == "shared":
            family_index = 0
            prefix_length = 1024
            group = f"batch-{index // 4}"
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        prompt = _family(corpus, seed, family_index, prefix_length) + tail(index)
        if index < WARMUP_REQUESTS:
            phase = "warmup"
        requests.append(
            {
                "request_id": f"request-{index:03d}",
                "phase": phase,
                "group": group,
                "prompt_token_ids": prompt,
            }
        )
    return requests


def _load_block_manager():
    """Load metadata modules without importing infinilm or native extensions."""
    source = Path(__file__).resolve().parents[2] / "python/infinilm/llm"
    saved = dict(sys.modules)
    try:
        infinilm = ModuleType("infinilm")
        llm = ModuleType("infinilm.llm")
        infinilm.__path__ = []
        llm.__path__ = []
        sys.modules["infinilm"] = infinilm
        sys.modules["infinilm.llm"] = llm
        for name in ("prefix_cache", "cache_manager"):
            fullname = f"infinilm.llm.{name}"
            spec = importlib.util.spec_from_file_location(
                fullname, source / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module
            spec.loader.exec_module(module)
        return sys.modules["infinilm.llm.cache_manager"].BlockManager
    finally:
        loaded = sys.modules.get("infinilm.llm.cache_manager")
        sys.modules.clear()
        sys.modules.update(saved)
        if loaded is not None:
            sys.modules["_benchmark_cache_manager"] = loaded


def _filled_manager(block_manager, num_blocks, pinned_ratio):
    manager = block_manager(num_blocks, BLOCK_SIZE)
    table, _ = manager.allocate_slots(num_blocks * BLOCK_SIZE)
    hashes = [index.to_bytes(16, "little") for index in range(1, num_blocks + 1)]
    manager.publish_computed_blocks(table, hashes, 0, num_blocks * BLOCK_SIZE)
    manager.free_blocks(table)
    pinned = []
    for block_hash in hashes[: int(num_blocks * pinned_ratio)]:
        blocks, _ = manager.get_computed_blocks([block_hash], BLOCK_SIZE)
        pinned.extend(blocks)
    return manager, hashes, pinned


def _policy_examples(block_manager):
    manager = block_manager(3, BLOCK_SIZE)
    hashes = []
    for value in (11, 22, 33):
        table, _ = manager.allocate_slots(BLOCK_SIZE)
        block_hash = bytes([value]) * 16
        manager.publish_computed_blocks(table, [block_hash], 0, BLOCK_SIZE)
        manager.free_blocks(table)
        hashes.append(block_hash)
    touched, _ = manager.get_computed_blocks([hashes[0]], BLOCK_SIZE)
    manager.free_blocks(touched)
    manager.allocate_slots(BLOCK_SIZE)
    survivor_hits = []
    for block_hash in hashes:
        pinned, hit = manager.get_computed_blocks([block_hash], BLOCK_SIZE)
        survivor_hits.append(hit)
        if pinned:
            manager.free_blocks(pinned)

    manager = block_manager(2, BLOCK_SIZE)
    table, _ = manager.allocate_slots(2 * BLOCK_SIZE)
    tail_hashes = [b"prefix".ljust(16, b"0"), b"tail".ljust(16, b"0")]
    manager.publish_computed_blocks(table, tail_hashes, 0, 2 * BLOCK_SIZE)
    manager.free_blocks(table)
    manager.allocate_slots(BLOCK_SIZE)
    _, prefix_hit = manager.get_computed_blocks(tail_hashes, 2 * BLOCK_SIZE)
    return {
        "recent_reuse": {"hit_tokens_after_pressure": survivor_hits},
        "tail_first": {"prefix_hit_tokens_after_pressure": prefix_hit},
    }


def run_metadata(args):
    block_manager = _load_block_manager()
    cases = []
    for pinned_ratio in (0.0, 0.5, 0.9):
        manager, _, _ = _filled_manager(block_manager, args.num_blocks, pinned_ratio)
        usable_times = []
        for _ in range(args.repeat):
            started = time.perf_counter_ns()
            usable = manager.get_total_usable_blocks()
            usable_times.append(time.perf_counter_ns() - started)
        reclaim = {}
        for count in (1, 8, 64):
            elapsed = []
            outcomes = []
            reclaimed_counts = []
            for _ in range(args.repeat):
                sample, _, _ = _filled_manager(
                    block_manager, args.num_blocks, pinned_ratio
                )
                free_before = sample.get_num_free_blocks()
                started = time.perf_counter_ns()
                outcome = sample.try_free_blocks(count)
                ended = time.perf_counter_ns()
                outcomes.append(outcome)
                elapsed.append(ended - started)
                reclaimed_counts.append(sample.get_num_free_blocks() - free_before)
            reclaim[str(count)] = {
                **_distribution(elapsed),
                "successful_samples": sum(outcomes),
                "requested_blocks": count,
                "actual_reclaimed_blocks": reclaimed_counts,
                "capacity_failure": any(actual < count for actual in reclaimed_counts),
            }
        cases.append(
            {
                "pinned_ratio": pinned_ratio,
                "usable_blocks": usable,
                "usable_query": _distribution(usable_times),
                "reclaim": reclaim,
            }
        )

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    memory_manager, _, _ = _filled_manager(block_manager, args.num_blocks, 0.0)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del memory_manager
    return {
        "schema_version": 1,
        "mode": "metadata",
        "status": "success",
        "python_version": platform.python_version(),
        "num_blocks": args.num_blocks,
        "block_size": BLOCK_SIZE,
        "repeat": args.repeat,
        "seed": args.seed,
        "harness_sha256": _file_sha256(__file__),
        "cache_manager_source_sha256": _file_sha256(
            Path(__file__).resolve().parents[2] / "python/infinilm/llm/cache_manager.py"
        ),
        "cache_manager_source": str(
            Path(__file__).resolve().parents[2] / "python/infinilm/llm/cache_manager.py"
        ),
        "timing_scope": "method call only; pool construction, hashing, and JSON excluded",
        "memory_scope": "Python tracemalloc peak increment; GPU memory excluded",
        "pool_peak_increment_bytes": peak - before_current,
        "pool_current_increment_bytes": current - before_current,
        "policy_examples": _policy_examples(block_manager),
        "cases": cases,
    }


def _git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_root_for(path):
    candidate = Path(path).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _dtype_name(value):
    if value is None:
        return None
    return getattr(value, "name", None) or str(value)


def _load_json_provenance(environment_name, description):
    value = os.environ.get(environment_name)
    if not value:
        raise RuntimeError(f"{environment_name} must point to {description}")
    path = Path(value).resolve()
    manifest = json.loads(path.read_text())
    return {
        "source": str(path),
        "sha256": _file_sha256(path),
        "manifest": manifest,
    }


def load_build_provenance():
    provenance = _load_json_provenance(
        "INFINILM_BUILD_PROVENANCE", "the native build provenance manifest"
    )
    manifest = provenance["manifest"]
    for project in ("infinicore", "infinilm"):
        if not isinstance(manifest.get(project), dict) or not manifest[project].get(
            "git_sha"
        ):
            raise RuntimeError(f"Build provenance is missing {project}.git_sha")
    return provenance


def _validate_native_artifact(build_provenance, binary_path):
    binary_path = Path(binary_path).resolve()
    actual_sha = _file_sha256(binary_path)
    matching = [
        artifact
        for artifact in build_provenance["manifest"].get("artifacts", [])
        if Path(artifact.get("path", "")).name == binary_path.name
    ]
    if len(matching) != 1 or matching[0].get("sha256") != actual_sha:
        raise RuntimeError(
            f"Build manifest does not uniquely match imported binary {binary_path}"
        )
    return {
        "path": str(binary_path),
        "sha256": actual_sha,
        "manifest_artifact": matching[0],
    }


def _provenance(engine, model_path, build_provenance):
    import infinicore
    import infinicore.lib._infinicore as core_native
    from infinilm.lib import _infinilm as lm_native

    model_engine = engine.engine.model_runner.model_engine
    if model_engine.dtype != infinicore.float16:
        raise RuntimeError(
            f"Effective model dtype is {_dtype_name(model_engine.dtype)}, not float16"
        )
    nested_cache = model_engine.get_kv_cache()
    cache_tensors = [tensor for layer in nested_cache for tensor in layer]
    if not cache_tensors:
        raise RuntimeError("Native model engine returned no KV cache tensors")
    invalid_cache_dtypes = {
        _dtype_name(tensor.dtype)
        for tensor in cache_tensors
        if tensor.dtype != infinicore.float16
    }
    if invalid_cache_dtypes:
        raise RuntimeError(
            f"Effective KV cache tensor dtypes are not float16: {invalid_cache_dtypes}"
        )
    effective_dtype = _dtype_name(model_engine.dtype)
    model_provenance = _load_json_provenance(
        "INFINILM_MODEL_PROVENANCE", "the model revision/overlay manifest"
    )
    overlays = model_provenance["manifest"]
    overlay = next(
        (item for item in overlays if item.get("experiment_path") == str(model_path)),
        None,
    )
    if overlay is None or not overlay.get("repo_id") or not overlay.get("revision"):
        raise RuntimeError(
            "INFINILM_MODEL_PROVENANCE must name a manifest containing this "
            "experiment_path, repo_id, and immutable revision"
        )
    try:
        gpu = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            .strip()
            .splitlines()
        )
    except (OSError, subprocess.CalledProcessError):
        gpu = []
    core_source = Path(infinicore.__file__).resolve()
    core_root = _git_root_for(core_source)
    core_binary = _validate_native_artifact(build_provenance, core_native.__file__)
    lm_binary = _validate_native_artifact(build_provenance, lm_native.__file__)
    try:
        cuda_toolkit = subprocess.check_output(["nvcc", "--version"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        cuda_toolkit = None
    return {
        "infinilm_sha": _git_sha(Path(__file__).resolve().parents[2]),
        "infinicore_sha": os.environ.get("INFINICORE_GIT_SHA")
        or (_git_sha(core_root) if core_root else None),
        "imported_infinicore_source": str(core_source),
        "infinicore_native_binary": core_binary,
        "infinilm_native_binary": lm_binary,
        "imported_llm_source": sys.modules["infinilm.llm.llm"].__file__,
        "harness_sha256": _file_sha256(__file__),
        "imported_llm_source_sha256": _file_sha256(
            sys.modules["infinilm.llm.llm"].__file__
        ),
        "model_path": str(model_path),
        "model_id": overlay.get("repo_id") if overlay else None,
        "model_revision": overlay.get("revision") if overlay else None,
        "fp16_overlay": overlay,
        "model_provenance": model_provenance,
        "build_provenance": build_provenance,
        "effective_model_dtype": effective_dtype,
        "effective_kv_cache_dtype": _dtype_name(cache_tensors[0].dtype),
        "effective_kv_cache_tensor_count": len(cache_tensors),
        "gpu_and_driver": gpu,
        "cuda_toolkit": cuda_toolkit,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "build_environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("INFINI", "CUDA", "LD_LIBRARY_PATH"))
        },
    }


async def _collect(engine, item, sampling_params):
    started = time.perf_counter()
    request = engine.add_request(
        messages=None,
        prompt_token_ids=item["prompt_token_ids"],
        sampling_params=sampling_params,
        request_id=item["request_id"],
    )
    times = []
    token_ids = []
    async for output in engine.stream_request(request):
        if output.token_id >= 0:
            times.append(time.perf_counter())
            token_ids.append(output.token_id)
    result = build_request_timing(
        item["request_id"], len(item["prompt_token_ids"]), started, token_ids, times
    )
    result.update({"phase": item["phase"], "group": item["group"]})
    return result


async def _run_model_async(args):
    from infinilm.llm.llm import AsyncLLMEngine
    from infinilm.llm.sampling_params import SamplingParams

    build_provenance = load_build_provenance()
    engine = None
    accounting = None
    trace = None
    trace_sha256 = None
    provenance = None
    results = []
    measured_started = None
    engine_started = False
    try:
        engine = AsyncLLMEngine(
            model_path=args.model,
            device="cuda",
            dtype="float16",
            cache_type="paged",
            enable_graph=False,
            attn_backend="paged-attn",
            tensor_parallel_size=1,
            block_size=BLOCK_SIZE,
            num_blocks=args.num_blocks,
            max_batch_size=4,
            max_tokens=OUTPUT_TOKENS,
            enable_prefix_caching=args.prefix_cache == "on",
        )
        provenance = _provenance(engine, Path(args.model), build_provenance)
        accounting = SchedulerAccounting(engine.engine.scheduler)
        accounting.install()
        tokenizer = engine.engine.tokenizer
        token_sources = []
        for text in SOURCE_TEXTS:
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                ids = tokenizer.encode(text)
            special_ids = set(getattr(tokenizer, "all_special_ids", ()))
            token_sources.append(
                [token_id for token_id in ids if token_id not in special_ids]
            )
        trace = build_trace(args.scenario, token_sources, args.seed)
        trace_bytes = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        trace_sha256 = hashlib.sha256(trace_bytes).hexdigest()
        _write_json(
            args.output,
            {
                "schema_version": 1,
                "mode": "model",
                "status": "running",
                "repeat": args.child_repeat,
                "config": _model_config(args),
                "trace_sha256": trace_sha256,
                "trace": trace,
                "requests": [],
                "provenance": provenance,
            },
        )
        sampling = SamplingParams(top_k=1, max_tokens=OUTPUT_TOKENS, ignore_eos=True)
        engine.start()
        engine_started = True
        for offset in range(0, len(trace), args.concurrency):
            if offset == WARMUP_REQUESTS:
                for result in results:
                    result.update(accounting.by_request[result["request_id"]])
                _write_json(
                    args.output,
                    {
                        "schema_version": 1,
                        "mode": "model",
                        "status": "running",
                        "repeat": args.child_repeat,
                        "config": _model_config(args),
                        "checkpoint": "warmup_complete",
                        "trace_sha256": trace_sha256,
                        "trace": trace,
                        "requests": results,
                        "provenance": provenance,
                    },
                )
                measured_started = time.perf_counter()
            batch = trace[offset : offset + args.concurrency]
            results.extend(
                await asyncio.gather(
                    *(_collect(engine, item, sampling) for item in batch)
                )
            )
        wall_seconds = time.perf_counter() - measured_started
        accounting.require_exactly_once([item["request_id"] for item in trace])
        # A separate normal-EOS request verifies that the smoke path does not rely on
        # ignore_eos semantics. Its output is excluded from performance accounting.
        eos_item = {
            "request_id": "correctness-eos",
            "prompt_token_ids": trace[-1]["prompt_token_ids"],
            "phase": "correctness",
            "group": None,
        }
        eos_result = await _collect(
            engine,
            eos_item,
            SamplingParams(top_k=1, max_tokens=OUTPUT_TOKENS, ignore_eos=False),
        )
        for result in results:
            result.update(accounting.by_request[result["request_id"]])
            if len(result["output_token_ids"]) != OUTPUT_TOKENS:
                raise RuntimeError(
                    f"{result['request_id']} delivered "
                    f"{len(result['output_token_ids'])} tokens; expected {OUTPUT_TOKENS}"
                )
        measured = [item for item in results if item["phase"] == "measurement"]
        aggregate = summarize_requests(measured, wall_seconds)
        aggregate["local_cached_tokens"] = sum(
            item["local_cached_tokens"] for item in measured
        )
        aggregate["prefill_tokens"] = sum(item["prefill_tokens"] for item in measured)
        aggregate["max_observed_block_ref_count"] = (
            accounting.max_observed_block_ref_count
        )
        if (
            args.scenario == "shared"
            and args.prefix_cache == "on"
            and accounting.max_observed_block_ref_count < 2
        ):
            raise RuntimeError(
                "Shared scenario did not observe concurrent cached-block owners"
            )
        groups = {}
        for group in sorted({item["group"] for item in measured if item["group"]}):
            selected = [item for item in measured if item["group"] == group]
            groups[group] = {
                "request_count": len(selected),
                "ttft_median_seconds": statistics.median(
                    item["ttft_seconds"] for item in selected
                ),
            }
        return {
            "schema_version": 1,
            "mode": "model",
            "status": "success",
            **_model_config(args),
            "trace_sha256": trace_sha256,
            "trace": trace,
            "requests": results,
            "correctness_eos_request": eos_result,
            "aggregate": aggregate,
            "groups": groups,
            "provenance": provenance,
        }
    except Exception as error:
        for result in results:
            if accounting and result["request_id"] in accounting.by_request:
                result.update(accounting.by_request[result["request_id"]])
        return {
            "schema_version": 1,
            "mode": "model",
            "status": "failure",
            **_model_config(args),
            "error_type": type(error).__name__,
            "error": str(error),
            "trace_sha256": trace_sha256,
            "trace": trace,
            "requests": results,
            "provenance": provenance,
            "build_provenance": build_provenance,
        }
    finally:
        if accounting is not None:
            accounting.restore()
        if engine_started:
            engine.stop()


def _model_config(args):
    return {
        "model": args.model,
        "prefix_cache": args.prefix_cache,
        "scenario": args.scenario,
        "concurrency": args.concurrency,
        "num_blocks": args.num_blocks,
        "block_size": BLOCK_SIZE,
        "seed": args.seed,
        "repeat": args.repeat,
        "repeat_timeout_seconds": args.repeat_timeout_seconds,
        "warmup_requests": WARMUP_REQUESTS,
        "attention_backend": "paged-attn",
        "engine_config": {
            "device": "cuda",
            "requested_dtype": "float16",
            "cache_type": "paged",
            "enable_graph": False,
            "tensor_parallel_size": 1,
            "max_batch_size": 4,
            "max_tokens": OUTPUT_TOKENS,
        },
    }


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def assemble_repeat_artifact(records, config, build_provenance=None):
    successful = sum(record["status"] == "success" for record in records)
    failed = len(records) - successful
    return {
        "schema_version": 1,
        "mode": "model",
        "status": "failure" if failed else "success",
        "config": config,
        "build_provenance": build_provenance,
        "successful_repeats": successful,
        "failed_repeats": failed,
        "repeats": records,
    }


def _read_child_payload(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _run_child(command, timeout_seconds):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return process.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return None, stdout, stderr, True


def _run_model_repeats(args):
    if args.child_repeat is not None:
        return asyncio.run(_run_model_async(args))
    try:
        build_provenance = load_build_provenance()
    except Exception as error:
        records = [
            {
                "repeat": index,
                "status": "failure",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            for index in range(args.repeat)
        ]
        return assemble_repeat_artifact(records, _model_config(args))
    records = []
    with tempfile.TemporaryDirectory(prefix="infinilm-prefix-benchmark-") as directory:
        for index in range(args.repeat):
            child_output = Path(directory) / f"repeat-{index}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                "model",
                "--output",
                str(child_output),
                "--seed",
                str(args.seed),
                "--repeat",
                "1",
                "--num-blocks",
                str(args.num_blocks),
                "--model",
                args.model,
                "--prefix-cache",
                args.prefix_cache,
                "--scenario",
                args.scenario,
                "--concurrency",
                str(args.concurrency),
                "--repeat-timeout-seconds",
                str(args.repeat_timeout_seconds),
                "--child-repeat",
                str(index),
            ]
            try:
                returncode, stdout, stderr, timed_out = _run_child(
                    command, args.repeat_timeout_seconds
                )
            except OSError as error:
                records.append(
                    {
                        "repeat": index,
                        "status": "failure",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                continue
            payload = _read_child_payload(child_output)
            if timed_out:
                record = {
                    "repeat": index,
                    "status": "timeout",
                    "timeout_seconds": args.repeat_timeout_seconds,
                    "stdout": stdout,
                    "stderr": stderr,
                }
                if payload is not None:
                    record["payload"] = payload
            elif returncode or payload is None or payload.get("status") != "success":
                record = {
                    "repeat": index,
                    "status": "failure",
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }
                if payload is not None:
                    record["payload"] = payload
            else:
                record = {"repeat": index, "status": "success", "payload": payload}
            records.append(record)
    return assemble_repeat_artifact(
        records, _model_config(args), build_provenance=build_provenance
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("metadata", "model"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--model")
    parser.add_argument("--prefix-cache", choices=("on", "off"))
    parser.add_argument(
        "--scenario",
        choices=(
            "hot-cold",
            "hot-shift",
            "no-reuse",
            "over-capacity",
            "mixed-length",
            "shared",
        ),
    )
    parser.add_argument("--concurrency", type=int, choices=(1, 4))
    parser.add_argument("--repeat-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--child-repeat", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeat <= 0 or args.num_blocks <= 0 or args.repeat_timeout_seconds <= 0:
        parser.error(
            "--repeat, --num-blocks, and --repeat-timeout-seconds must be positive"
        )
    if args.mode == "model" and not all(
        (args.model, args.prefix_cache, args.scenario, args.concurrency)
    ):
        parser.error(
            "model mode requires --model, --prefix-cache, --scenario, and --concurrency"
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = (
            run_metadata(args) if args.mode == "metadata" else _run_model_repeats(args)
        )
    except Exception as error:
        result = {
            "schema_version": 1,
            "mode": args.mode,
            "status": "failure",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_json(args.output, result)
        raise
    _write_json(args.output, result)
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
