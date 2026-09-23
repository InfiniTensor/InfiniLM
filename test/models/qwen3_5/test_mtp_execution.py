"""GPU correctness: serial equivalence, checkpoint ownership, graphs and batching.

Set INFINILM_QWEN_MTP_TEST_MODEL to a tiny checkpoint with MTP weights;
INFINILM_QWEN_MTP_TEST_TP=1 or 2 selects the parallelism.
"""

import os

import infinicore
import pytest
import torch
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
            engine.model_runner.model_engine.process_weights_after_loading()
    assert all(r.is_finished() for r in reqs)
    assert not engine.scheduler.mamba_cache_manager.used_block_ids
    cache = engine.scheduler.cache_manager
    assert all(b.ref_count == 0 for b in cache.blocks)
    assert cache.get_total_usable_blocks() == cache.num_blocks
    return [list(r.generated_token_ids) for r in reqs]


@pytest.mark.parametrize("candidates", [1, 2, 4])
def test_mtp_matches_ordinary_with_reused_and_rebuilt_state(candidates):
    llm = create(num_draft_tokens=candidates)
    engine = llm.engine
    runner = engine.model_runner
    mtp = runner.speculative_runner
    raw = runner.model_engine

    prompt = [i % 63 + 1 for i in range(63)]

    def generate():
        return drain(engine, [request("test", prompt, 20)])[0]

    try:
        runner.speculative_runner = None
        engine.config.enable_mtp = False
        expected = generate()
        runner.speculative_runner = mtp
        engine.config.enable_mtp = True
        mtp.device_tokens = False
        assert generate() == expected
        mtp.device_tokens = True
        assert generate() == expected
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


def test_ordinary_graph_recapture_preserves_recurrent_state():
    llm = create(enable_mtp=False, enable_graph=True)
    prompt = [i % 63 + 1 for i in range(63)]
    try:
        expected = drain(llm.engine, [request("ordinary", prompt)])[0]
        actual = drain(llm.engine, [request("recapture", prompt)], recapture=True)[0]
        assert actual == expected
    finally:
        llm.close()


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


def test_packed_verification_preserves_request_and_causal_boundaries():
    llm = create(num_draft_tokens=2, max_batch_size=2, enable_graph=False)
    runner = llm.engine.model_runner.speculative_runner
    raw = llm.engine.model_runner.model_engine
    reqs = [
        request("left", [i % 59 + 1 for i in range(63)]),
        request("right", [i % 53 + 1 for i in range(127)]),
    ]
    # Different histories and query lengths cross distinct physical page boundaries.
    for row, (req, blocks) in enumerate(zip(reqs, ([0, 2], [1, 3, 4])), 1):
        req.block_table = blocks
        req.mamba_cache_index = row

    def logits_for(inputs):
        result = raw.forward_raw(**inputs, sample_all_positions=True)
        logits = result["logits"]
        cpu = torch.empty(logits.shape, dtype=torch.bfloat16)
        infinicore.from_torch(cpu).copy_(logits)
        infinicore.sync_device()
        return cpu[0]

    def verify(left, right):
        return logits_for(
            runner._pack(
                [
                    runner._inputs(reqs[0], left, 63, destinations=[3, 4, 5]),
                    runner._inputs(reqs[1], right, 127, destinations=[6, 7]),
                ]
            )
        )

    try:
        inputs = runner._pack(
            [runner._inputs(req, list(req.prompt_token_ids), 0) for req in reqs]
        )
        raw.forward_raw(**inputs)
        expected = verify([5, 7, 9], [11, 13])
        changed_future = verify([5, 17, 19], [11, 23])
        torch.testing.assert_close(
            changed_future[[0, 3]], expected[[0, 3]], rtol=0, atol=0
        )
        changed_request = verify([5, 7, 9], [29, 31])
        torch.testing.assert_close(changed_request[:3], expected[:3], rtol=0, atol=0)
        # Verification writes scratch rows, so the committed initial states can
        # also be consumed by ordinary Decode for an independent prefix check.
        ordinary = logits_for(
            runner._pack(
                [
                    runner._inputs(reqs[0], [5], 63),
                    runner._inputs(reqs[1], [11], 127),
                ]
            )
        )
        torch.testing.assert_close(ordinary, expected[[0, 3]], rtol=0, atol=0)
    finally:
        llm.close()
