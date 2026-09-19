"""GPU correctness: serial equivalence, checkpoint ownership, graphs and batching.

Set INFINILM_QWEN_MTP_TEST_MODEL to a tiny checkpoint with MTP weights;
INFINILM_QWEN_MTP_TEST_TP=1 or 2 selects the parallelism (prefix caching uses TP1).
"""

import os

import infinicore
import pytest
from infinilm.cache import PagedKVCacheConfig
from infinilm.llm.llm import LLM
from infinilm.llm.request import InferenceRequest
from infinilm.llm.sampling_params import SamplingParams


def create(**options):
    path = os.environ.get("INFINILM_QWEN_MTP_TEST_MODEL")
    if not path:
        pytest.skip("Set INFINILM_QWEN_MTP_TEST_MODEL to a tiny GPU checkpoint")
    config = dict(
        enable_mtp=True,
        device="cuda",
        dtype="bfloat16",
        tensor_parallel_size=int(os.environ.get("INFINILM_QWEN_MTP_TEST_TP", "1")),
        num_blocks=40,
        block_size=64,
        num_state_rows=13,
        max_batch_size=1,
        enable_prefix_caching=False,
        attn_backend="paged-attn",
        top_k=1,
        top_p=1.0,
        weight_load_mode="sync",
    )
    return LLM(path, **(config | options))


def request(name, prompt, count=17):
    return InferenceRequest(
        name,
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(max_tokens=count, ignore_eos=True, top_k=1),
    )


def drain(engine, reqs, *, admit=True, recapture=False):
    if admit:
        for req in reqs:
            engine.add_request(req)
    for step in range(100):
        if all(r.is_finished() for r in reqs):
            break
        assert engine.step()[0]
        if recapture and step == 1:
            engine.model_runner.model_engine.compile()
    assert all(r.is_finished() for r in reqs)
    assert not engine.scheduler.mamba_cache_manager.used_block_ids
    cache = engine.scheduler.cache_manager
    assert all(b.ref_count == 0 for b in cache.blocks)
    assert cache.get_total_usable_blocks() == cache.num_blocks
    return [list(r.generated_token_ids) for r in reqs]


@pytest.mark.parametrize("candidates,graph", [(1, True), (2, False), (4, False)])
def test_mtp_matches_ordinary_with_reused_and_recaptured_state(candidates, graph):
    llm = create(num_draft_tokens=candidates, enable_graph=graph)
    engine = llm.engine
    runner = engine.model_runner
    mtp = runner.speculative_runner
    raw = runner.model_engine

    prompt = [i % 63 + 1 for i in range(63)]

    def generate(recapture=False):
        return drain(engine, [request("test", prompt, 20)], recapture=recapture)[0]

    try:
        runner.speculative_runner = None
        engine.config.enable_mtp = False
        expected = generate()
        runner.speculative_runner = mtp
        engine.config.enable_mtp = True
        mtp.device_tokens = False
        assert generate() == expected
        mtp.device_tokens = True
        assert generate(recapture=graph) == expected
        raw.reset_cache(PagedKVCacheConfig(40, 64, 1, llm.config.num_state_rows))
        assert generate() == expected
        # Exercise every device-side acceptance length, then consume the
        # selected Conv/GDN checkpoint with ordinary Q=1 continuation.
        forward = raw.forward_raw
        for prefix in range(candidates + 1):
            verified = []

            def force_prefix(**inputs):
                if inputs.get("verify_draft"):
                    ids = expected[: candidates + 1].copy()
                    if prefix < candidates:
                        ids[prefix + 1] = (ids[prefix + 1] + 1) % 63
                    inputs["input_ids"] = infinicore.from_list(
                        [ids], dtype=infinicore.int64
                    ).to(inputs["input_ids"].device)
                result = forward(**inputs)
                if inputs.get("verify_draft"):
                    verified.append(result["accepted_draft_tokens"])
                return result

            req = request("test", prompt, 20)
            engine.add_request(req)
            engine.step()
            raw.forward_raw = force_prefix
            try:
                engine.step()
            finally:
                raw.forward_raw = forward
            assert verified == [prefix]
            runner.speculative_runner = None
            engine.config.enable_mtp = False
            assert drain(engine, [req], admit=False)[0] == expected
            runner.speculative_runner = mtp
            engine.config.enable_mtp = True
    finally:
        llm.close()


def test_bounded_prefix_hit_eviction_and_rebuild():
    llm = create(
        num_draft_tokens=1,
        enable_graph=True,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        mtp_prefix_cache_bytes=8 * 1024**2,
    )
    engine = llm.engine
    runner = engine.model_runner
    mtp = runner.speculative_runner
    prompt = [i % 59 + 1 for i in range(63)]
    try:
        runner.speculative_runner = None
        engine.config.enable_mtp = False
        expected = drain(engine, [request("ordinary", prompt)])[0]
        runner.speculative_runner = mtp
        engine.config.enable_mtp = True
        assert drain(engine, [request("cold", prompt)])[0] == expected
        cache = mtp.prefix_cache
        assert cache.entries and cache.used_bytes <= cache.budget_bytes
        cache.budget_bytes = cache.used_bytes
        hits = cache.hits
        calls = []
        forward = runner.model_engine.forward_raw

        def capture(**kw):
            calls.append(kw["input_ids"].shape[-1])
            return forward(**kw)

        runner.model_engine.forward_raw = capture
        assert drain(engine, [request("hit", prompt)])[0] == expected
        assert cache.hits == hits + 1 and max(calls) <= 2
        runner.model_engine.forward_raw = forward
        drain(engine, [request("different", [7, 2, 19, 8])])
        assert cache.evictions == 1 and cache.used_bytes <= cache.budget_bytes
        assert drain(engine, [request("evicted", prompt)])[0] == expected
        assert cache.evictions == 2
        generation = runner.model_engine.cache_generation
        runner.model_engine.reset_cache(PagedKVCacheConfig(40, 64, 1, 13))
        assert runner.model_engine.cache_generation > generation
        hits = cache.hits
        assert drain(engine, [request("reset", prompt)])[0] == expected
        assert cache.hits == hits
    finally:
        llm.close()
    assert not mtp.prefix_cache.entries


def test_batched_mtp_matches_serial_with_cancellation():
    llm = create(
        num_draft_tokens=2,
        max_batch_size=3,
        enable_graph=False,
    )
    engine = llm.engine
    runner = engine.model_runner
    mtp = runner.speculative_runner
    prompts = [[i % 59 + 1 for i in range(63)], [7, 2, 19, 8], [19, 9, 13, 5]]
    try:
        runner.speculative_runner = None
        engine.config.enable_mtp = False
        expected = [drain(engine, [request("serial", p)])[0] for p in prompts]
        runner.speculative_runner = mtp
        engine.config.enable_mtp = True
        reqs = [request(str(i), p) for i, p in enumerate(prompts)]
        assert drain(engine, reqs) == expected
        cancel = request("cancel", prompts[0])
        keep = request("keep", prompts[1])
        engine.add_request(cancel)
        engine.add_request(keep)
        engine.step()
        engine.step()
        cancel.mark_canceled()
        other = request("new", prompts[2])
        engine.add_request(other)
        actual = drain(engine, [cancel, keep, other], admit=False)
        assert actual[1:] == expected[1:]
    finally:
        llm.close()
