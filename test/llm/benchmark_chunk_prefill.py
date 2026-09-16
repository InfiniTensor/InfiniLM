"""Short native long-prefill/active-decode experiment with recorded step boundaries."""

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

SOURCE_TEXTS = (
    "You are a careful technical assistant. Follow the system instructions and "
    "answer with precise, verifiable details. ",
    "Paged key value caches store completed attention states in fixed sized blocks. "
    "Prefix reuse avoids repeated prefill computation when requests share input. ",
    "Explain how deterministic experiments separate control plane overhead from "
    "model execution and why complete traces are needed for reproducibility. ",
)


def provenance(engine, model_path):
    import infinicore
    import infinilm.llm.llm as llm_source
    from infinicore.lib import _infinicore as core_native
    from infinilm.lib import _infinilm as lm_native

    tree = Path(llm_source.__file__).resolve().parents[3]
    model_engine = engine.engine.model_runner.model_engine
    assert model_engine.dtype == infinicore.float16
    ranks = model_engine.get_kv_cache()
    assert len(ranks) == engine.config.tensor_parallel_size
    assert {t.device.index for rank in ranks for t in rank} == set(range(len(ranks)))
    assert all(t.dtype == infinicore.float16 for rank in ranks for t in rank)
    record = dict(
        imported_llm_source=llm_source.__file__,
        infinilm_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=tree, text=True
        ).strip(),
        git_status=subprocess.check_output(
            ["git", "status", "--short"], cwd=tree, text=True
        ),
        model_path=str(model_path),
        model_config=json.loads((model_path / "config.json").read_text()),
        model_config_sha256=hashlib.sha256(
            (model_path / "config.json").read_bytes()
        ).hexdigest(),
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
        rank_devices=[[str(t.device) for t in rank] for rank in ranks],
        native_binaries={
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (lm_native.__file__, core_native.__file__)
        },
    )
    maps = Path("/proc/self/maps")
    if maps.exists():
        paths = sorted(
            {
                line.split()[-1]
                for line in maps.read_text().splitlines()
                if any(name in line for name in ("libinfiniccl.so", "libnccl.so"))
            }
        )
        record["communication_libraries"] = {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths
        }
    for name in ("INFINILM_BUILD_PROVENANCE", "INFINILM_MODEL_PROVENANCE"):
        path = os.getenv(name)
        if path:
            record[name] = dict(
                path=path,
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                manifest=json.loads(Path(path).read_text()),
            )
    return record


async def run(args):
    from infinilm.llm.llm import AsyncLLMEngine
    from infinilm.llm.sampling_params import SamplingParams

    config = dict(
        model_path=args.model,
        device="cuda",
        dtype="float16",
        cache_type="paged",
        enable_graph=args.graph,
        prefix_cache_policy=args.policy,
        attn_backend="paged-attn",
        tensor_parallel_size=args.tp,
        block_size=256,
        num_blocks=128,
        max_batch_size=2,
        max_tokens=128,
        enable_prefix_caching=False,
    )
    if args.chunk_size:
        config["prefill_chunk_size"] = args.chunk_size
    engine = AsyncLLMEngine(**config)
    try:
        evidence = provenance(engine, Path(config["model_path"]).resolve())
        tree = Path(evidence["imported_llm_source"]).resolve().parents[3]
        hashes = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                *[
                    tree / name
                    for name in (
                        "csrc/engine/infer_engine.cpp",
                        "csrc/engine/rank_worker.cpp",
                        "csrc/engine/rank_worker.hpp",
                        "csrc/models/infinilm_model.hpp",
                        "csrc/pybind11/engine/engine.hpp",
                        "csrc/layers/causal_lm_templates/text_causal_lm.hpp",
                    )
                ],
                *[
                    tree / "python/infinilm" / name
                    for name in (
                        "infer_engine.py",
                        "llm/model_runner/model_runner.py",
                        "llm/scheduler.py",
                        "llm/request.py",
                        "llm/llm.py",
                        "processors/basic_llm_processor.py",
                        "config/engine_config.py",
                        "base_config.py",
                    )
                ],
            ]
        }
        if args.legacy_chunk_output:
            # Experimental control: use the same binary and schedule, restoring
            # logits/sampling/token transfer for intermediate chunks only.
            forward = engine.engine.model_runner.model_engine.forward

            def legacy_forward(**kwargs):
                intermediate = kwargs.get("prefill_only", False)
                if intermediate:
                    kwargs["prefill_only"] = False
                result = forward(**kwargs)
                if intermediate:
                    result.to_numpy().tolist()
                return result

            engine.engine.model_runner.model_engine.forward = legacy_forward

        corpus = engine.engine.tokenizer.encode(
            " ".join(SOURCE_TEXTS), add_special_tokens=False
        )
        corpus = [x for x in corpus if x not in engine.engine.tokenizer.all_special_ids]
        prompts = {
            name: random.Random(seed).choices(corpus, k=length)
            for name, seed, length in [
                ("active", 51, 128),
                ("long", 52, args.long_tokens),
                ("late", 53, 128),
            ]
        }
        prompts["warmup"] = corpus[:128]
        steps = []
        original = engine.engine.model_runner.execute_model

        def execute(output):
            row = dict(
                start=time.perf_counter(),
                prefill=output.is_prefill,
                requests=[
                    dict(
                        id=r.request_id,
                        slots=len(r.slot_mapping),
                        cached=r.num_local_cached_tokens,
                        computed=r.num_computed_tokens,
                        generated=len(r.generated_token_ids),
                    )
                    for r in output.scheduled_requests
                ],
            )
            result = original(output)
            row["end"] = time.perf_counter()
            steps.append(row)
            return result

        engine.engine.model_runner.execute_model = execute
        results = {}
        active_ready = asyncio.Event()

        async def collect(name, count):
            started = time.perf_counter()
            request = engine.add_request(
                messages=None,
                prompt_token_ids=prompts[name],
                request_id=name,
                sampling_params=SamplingParams(
                    top_k=1, max_tokens=count, ignore_eos=True
                ),
            )
            times = []
            ids = []
            async for output in engine.stream_request(request):
                if output.token_id >= 0:
                    times.append(time.perf_counter())
                    ids.append(output.token_id)
                    if name == "active" and len(ids) == 8:
                        active_ready.set()
            assert len(ids) == count, (name, len(ids), count)
            gaps = [b - a for a, b in zip(times, times[1:])]
            results[name] = dict(
                start=started,
                times=times,
                token_ids=ids,
                ttft=times[0] - started,
                itl_median=statistics.median(gaps),
                itl_max=max(gaps),
                itl_p95=sorted(gaps)[int(0.95 * (len(gaps) - 1))],
                status=str(request.status),
            )

        engine.start()
        tasks = []
        try:
            # A separate warmup does not populate the test prefix cache.
            await asyncio.wait_for(collect("warmup", 8), 30)
            results.pop("warmup")
            prompts.pop("warmup")
            steps.clear()
            tasks.append(asyncio.create_task(collect("active", 128)))
            await asyncio.wait_for(active_ready.wait(), 30)
            tasks.append(asyncio.create_task(collect("long", 16)))
            await asyncio.sleep(0.02)
            tasks.append(asyncio.create_task(collect("late", 16)))
            await asyncio.wait_for(asyncio.gather(*tasks), 90)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if len(results) == 3:
                assert all(
                    b.ref_count == 0
                    for b in engine.engine.scheduler.cache_manager.blocks
                )
        return dict(
            status="success",
            chunk_size=args.chunk_size,
            legacy_chunk_output=args.legacy_chunk_output,
            config=config,
            provenance=evidence,
            source_hashes=hashes,
            prompts=prompts,
            results=results,
            steps=steps,
        )
    finally:
        if engine._running:
            engine.stop()
        else:
            # stop() skips close after a failed worker or before start().
            engine.engine.close()
        if engine._step_thread is not None:
            assert not engine._step_thread.is_alive()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--long-tokens", type=int, default=8192)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--policy", choices=("lru", "slru"), default="lru")
    parser.add_argument(
        "--legacy-chunk-output",
        action="store_true",
        help="benchmark control: compute and discard intermediate outputs",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = asyncio.run(run(args))
    except Exception as error:
        args.output.write_text(
            json.dumps(dict(status="failure", error=repr(error)), indent=2) + "\n"
        )
        raise
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {m: v for m, v in d.items() if m not in ("times", "token_ids")}
                for k, d in payload["results"].items()
            },
            indent=2,
        )
    )
