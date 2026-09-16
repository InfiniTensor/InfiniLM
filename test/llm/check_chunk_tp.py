"""Opt-in real-model chunk/KV check; run explicitly with --model and --output."""

import argparse
import ctypes
import hashlib
import json
import random
import time
from pathlib import Path


def host_fp16(tensor, core):
    """Export FP16 without relying on the native to_numpy dtype support."""
    import numpy as np

    assert tensor.dtype == core.float16
    core.set_device(tensor.device)
    core.sync_device()
    cpu = tensor.to(core.device("cpu", 0)).contiguous()
    raw = (ctypes.c_ubyte * (cpu.numel() * 2)).from_address(cpu.data_ptr())
    result = np.frombuffer(raw, dtype=np.float16).reshape(cpu.shape).copy()
    core.set_device(core.device("cuda", 0))
    return result


def run(args):
    import infinicore as core
    import numpy as np
    from infinilm.config.engine_config import EngineConfig
    from infinilm.llm.llm import LLMEngine
    from infinilm.llm.request import InferenceRequest
    from infinilm.llm.sampling_params import SamplingParams

    config = EngineConfig(
        model_path=args.model,
        device="cuda",
        dtype="float16",
        tensor_parallel_size=args.tp,
        cache_type="paged",
        enable_graph=args.graph,
        pipeline_parallel_size=args.pp,
        pipeline_parallel_stage=args.stage,
        master_port=args.port,
        prefix_cache_policy=args.policy,
        attn_backend="paged-attn",
        block_size=256,
        num_blocks=16,
        max_batch_size=2,
        prefill_chunk_size=args.chunk_size,
        enable_prefix_caching=not args.cache_off,
    )

    def local_caches(model):
        from infinilm.lib import _infinilm

        return [
            [core.Tensor(t) for t in rank if t]
            for rank in _infinilm.InferEngine.get_kv_cache(model)
        ]

    if args.stage:
        from infinilm.distributed.pipeline_transport import PipelineWorkerClient
        from infinilm.llm.model_runner.model_runner import ModelRunner

        runner = ModelRunner(config, initialize_processor=False)
        observed = [
            t for rank in local_caches(runner.model_engine) for t in (rank[0], rank[-1])
        ]
        checks = []
        forward = runner.model_engine.forward

        def checked_forward(**inputs):
            slots = inputs["slot_mapping"].to_numpy().reshape(-1).tolist()
            allowed = np.zeros(16 * 256, dtype=bool)
            allowed[slots] = True
            before = []
            for tensor in observed:
                raw = host_fp16(tensor, core)
                for slot in slots:
                    raw[:, slot // 256, :, slot % 256, :] = np.nan
                core.set_device(tensor.device)
                tensor.copy_(core.from_numpy(raw, device=tensor.device))
                core.sync_device()
                before.append(raw.transpose(1, 3, 0, 2, 4).reshape(16 * 256, -1))
            result = forward(**inputs)
            for index, tensor in enumerate(observed):
                after = (
                    host_fp16(tensor, core)
                    .transpose(1, 3, 0, 2, 4)
                    .reshape(16 * 256, -1)
                )
                assert np.isfinite(after[allowed]).all()
                assert np.array_equal(
                    before[index][~allowed], after[~allowed], equal_nan=True
                )
            checks.append(
                dict(prefill_only=inputs.get("prefill_only", False), slots=len(slots))
            )
            return result

        runner.model_engine.forward = checked_forward
        try:
            PipelineWorkerClient(
                runner, config.master_addr, config.master_port, args.stage
            ).serve_forever()
            return dict(status="worker_success", stage=args.stage, checks=checks)
        finally:
            runner.close()
    counter = ctypes.CDLL(None).graph_launch_count if args.graph else lambda: 0
    if args.graph:
        counter.restype = ctypes.c_ulonglong
    engine = LLMEngine(config)
    try:
        manager = engine.scheduler.cache_manager
        ranks = local_caches(engine.model_runner.model_engine)
        assert len(ranks) == args.tp
        assert {t.device.index for rank in ranks for t in rank} == set(range(args.tp))
        # Inspect the first attention layer on every TP rank. Other layers run normally.
        observed = [rank[i] for rank in ranks for i in (0, len(rank) - 1)]
        sentinel = np.float16(np.nan)
        for tensor in observed:
            assert (
                len(tensor.shape) == 5
                and tensor.shape[0] == 2
                and tensor.shape[3] == 256
            )
            core.set_device(tensor.device)
            tensor.copy_(
                core.from_numpy(
                    np.full(tensor.shape, sentinel, dtype=np.float16),
                    device=tensor.device,
                )
            )
            core.sync_device()
        core.set_device(core.device("cuda", 0))
        corpus = engine.tokenizer.encode(
            "The engine processes a long document while serving short requests. "
            "Pages store key and value vectors, and attention reuses computed prefixes. "
            * 8,
            add_special_tokens=False,
        )
        corpus = [t for t in corpus if t not in engine.tokenizer.all_special_ids]
        prompt = random.Random(82).choices(corpus, k=1027)
        steps = []
        publication_checks = []
        span_checks = []
        requests = {}
        original = engine.model_runner.execute_model

        def execute(output):
            row = dict(
                prefill=output.is_prefill,
                end=getattr(output, "prefill_end", None),
                requests=[
                    dict(
                        id=r.request_id,
                        cached=r.num_local_cached_tokens,
                        computed=r.num_computed_tokens,
                        generated=len(r.generated_token_ids),
                        slots=list(r.slot_mapping),
                    )
                    for r in output.scheduled_requests
                ],
            )
            before = []
            if output.is_prefill:
                # Poison every scheduled slot, including reused physical pages, so a
                # missing K/V write cannot pass merely because stale values are finite.
                for tensor in observed:
                    raw = host_fp16(tensor, core)
                    for request in output.scheduled_requests:
                        for slot in request.slot_mapping:
                            raw[:, slot // 256, :, slot % 256, :] = sentinel
                    core.set_device(tensor.device)
                    tensor.copy_(core.from_numpy(raw, device=tensor.device))
                    core.sync_device()
                    before.append(raw.transpose(1, 3, 0, 2, 4).reshape(16 * 256, -1))
                core.set_device(core.device("cuda", 0))
            before_launches = counter()
            started = time.perf_counter()
            result = original(output)
            row["elapsed_ms"] = (time.perf_counter() - started) * 1000
            row["graph_launches"] = counter() - before_launches
            if args.graph:
                assert row["graph_launches"] == (0 if output.is_prefill else args.tp), (
                    row
                )
            if output.is_prefill:
                allowed = np.zeros(16 * 256, dtype=bool)
                for r in output.scheduled_requests:
                    allowed[r.slot_mapping] = True
                for rank, tensor in enumerate(observed):
                    after = (
                        host_fp16(tensor, core)
                        .transpose(1, 3, 0, 2, 4)
                        .reshape(16 * 256, -1)
                    )
                    assert np.array_equal(
                        before[rank][~allowed], after[~allowed], equal_nan=True
                    ), (
                        rank,
                        "KV write outside scheduled slots",
                    )
                    assert np.isfinite(after[allowed]).all()
                    span_checks.append(
                        dict(
                            rank=rank,
                            request=row["requests"][0]["id"],
                            end=row["end"],
                            written_span=int(allowed.sum()),
                            outside_unchanged=True,
                        )
                    )
            steps.append(row)
            return result

        engine.model_runner.execute_model = execute

        def new(name, tokens=prompt, count=8):
            r = InferenceRequest(
                name,
                prompt_token_ids=tokens,
                sampling_params=SamplingParams(
                    top_k=1, max_tokens=count, ignore_eos=True
                ),
            )
            engine.add_request(r)
            requests[name] = r
            return r

        def state(idle=False):
            assert all(b.ref_count >= 0 for b in manager.blocks)
            if idle:
                assert all(b.ref_count == 0 for b in manager.blocks)
                assert manager.get_total_usable_blocks() == 16

        def tick():
            worked, pending = engine.step()
            assert worked and not pending
            state()

        def drain(active):
            for _ in range(100):
                if all(r.is_finished() for r in active):
                    break
                tick()
            assert all(
                r.is_finished() and len(r.generated_token_ids) == 8 for r in active
            )
            state(idle=True)

        first = new("first")
        # Both ranks must write exactly [0, chunk_size), preserving untouched future slots.
        tick()
        if args.chunk_size:
            assert (
                first.num_computed_tokens == args.chunk_size
                and not first.generated_token_ids
            )
            expected = np.zeros(16 * 256, dtype=bool)
            for i in range(args.chunk_size):
                expected[first.block_table[i // 256] * 256 + i % 256] = True
            for rank, tensor in enumerate(observed):
                values = (
                    host_fp16(tensor, core)
                    .transpose(1, 3, 0, 2, 4)
                    .reshape(16 * 256, -1)
                )
                changed = np.any(~np.isnan(values), axis=1)
                assert np.array_equal(changed, expected), (
                    rank,
                    np.flatnonzero(changed != expected).tolist(),
                )
                assert np.isfinite(values[expected]).all()
                publication_checks.append(
                    dict(
                        rank=rank,
                        device=str(tensor.device),
                        changed_slots=int(changed.sum()),
                        expected_slots=int(expected.sum()),
                        indexed_blocks=first.num_cache_indexed_blocks,
                    )
                )
            assert first.num_cache_indexed_blocks == (
                0 if args.cache_off else args.chunk_size // 256
            )
            # A new request may reuse only the fully computed first page. Both hold shared ownership.
            second = new("shared", prompt + corpus[:2])
            shared_dispatch = None
            for _ in range(5):
                tick()
                row = steps[-1]
                if row["requests"][0]["id"] == "shared":
                    shared_dispatch = row
                    break
            assert shared_dispatch is not None
            hit = shared_dispatch["requests"][0]["cached"]
            if not args.cache_off:
                assert hit > 0 and hit % 256 == 0
                assert all(
                    manager.blocks[b].ref_count == 2
                    for b in second.block_table[: hit // 256]
                )
            else:
                assert hit == 0
            first.mark_canceled()
            drain([second])
            assert not first.generated_token_ids
            state(idle=True)
            # Repeat now matches the completed prefix; the uncached suffix is shorter than a chunk.
            repeat = new("repeat", prompt + corpus[:2])
            drain([repeat])
            if not args.cache_off:
                assert any(
                    r["id"] == "repeat" and r["cached"] == 1024
                    for s in steps
                    if s["prefill"]
                    for r in s["requests"]
                )
            # Abort after forward: final cached suffix when caching is on;
            # an intermediate segment when caching is off. Both must release ownership.
            canceled = new("abort-forward", prompt + corpus[:2])
            run_forward = engine.model_runner.execute_model

            def abort_after_forward(output):
                result = run_forward(output)
                canceled.abort()
                return result

            engine.model_runner.execute_model = abort_after_forward
            tick()
            engine.model_runner.execute_model = run_forward
            assert canceled.is_finished() and not canceled.generated_token_ids
            state(idle=True)
        else:
            drain([first])
            drain([new("shared", prompt + corpus[:2])])
            drain([new("repeat", prompt + corpus[:2])])
        result = dict(
            status="success",
            tp=args.tp,
            pp=args.pp,
            policy=args.policy,
            graph=args.graph,
            chunk_size=args.chunk_size,
            cache_off=args.cache_off,
            rank_devices=[[str(t.device) for t in rank] for rank in ranks],
            checks=publication_checks,
            span_checks=span_checks,
            steps=steps,
            requests={
                name: dict(
                    prompt=list(r.prompt_token_ids),
                    tokens=list(r.generated_token_ids),
                    status=str(r.status),
                )
                for name, r in requests.items()
            },
            final_refs=[b.ref_count for b in manager.blocks],
            final_usable=manager.get_total_usable_blocks(),
        )
        import infinilm.llm.llm as source
        from infinilm.lib import _infinilm as native

        result["source"] = str(Path(source.__file__).resolve())
        result["native_sha256"] = hashlib.sha256(
            Path(native.__file__).read_bytes()
        ).hexdigest()
        result["same_prompt_greedy_equal"] = (
            result["requests"]["shared"]["tokens"]
            == result["requests"]["repeat"]["tokens"]
        )
        assert result["same_prompt_greedy_equal"], "shared and repeated tokens differ"
        result["script_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        return result
    finally:
        engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), default=2)
    parser.add_argument("--chunk-size", type=int, choices=(0, 300, 512), default=300)
    parser.add_argument("--cache-off", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--pp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--stage", type=int, choices=(0, 1), default=0)
    parser.add_argument("--port", type=int, default=29761)
    parser.add_argument("--policy", choices=("lru", "slru"), default="lru")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        args.output.write_text(
            json.dumps(dict(status="failure", error=repr(error)), indent=2) + "\n"
        )
        raise
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "tp", "chunk_size", "checks", "final_usable")
                if k in result
            },
            indent=2,
        )
    )
