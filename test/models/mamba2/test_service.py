"""Mamba-2 request ownership and service-level recurrence checks."""

import os

import pytest
from test_adaptation import forward


@pytest.fixture(scope="module")
def service():
    model = os.environ.get("INFINILM_MAMBA2_MODEL")
    if not model:
        pytest.skip("Set `INFINILM_MAMBA2_MODEL` for real-weight service tests.")
    from infinilm.config.engine_config import EngineConfig
    from infinilm.llm.llm import LLMEngine

    instance = LLMEngine(
        EngineConfig(
            model_path=model,
            tensor_parallel_size=int(os.environ.get("INFINILM_MAMBA2_TP", "1")),
            enable_graph=os.environ.get("INFINILM_MAMBA2_GRAPH") == "1",
            enable_prefix_caching=False,
            num_blocks=24,
            block_size=16,
            max_batch_size=8,
        )
    )
    yield instance
    instance.close()


def request(name, tokens, count=4):
    from infinilm.llm.request import InferenceRequest
    from infinilm.llm.sampling_params import SamplingParams

    return InferenceRequest(
        request_id=name,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=count, top_k=1, ignore_eos=True),
    )


def assert_released(service):
    manager = service.scheduler.mamba_cache_manager
    assert not manager.used_block_ids
    assert len(manager.free_block_ids) == manager.num_blocks - 1
    assert len(set(manager.free_block_ids)) == len(manager.free_block_ids)
    blocks = service.scheduler.cache_manager
    assert blocks.get_total_usable_blocks() == blocks.num_blocks
    assert all(block.ref_count == 0 for block in blocks.blocks)


def test_service_decode_matches_explicit_state_continuation(service):
    manager = service.scheduler.mamba_cache_manager
    requests = [request("a", [17, 83, 51]), request("b", [142, 73, 6, 89, 13])]
    reference_rows = [manager.allocate() for _ in requests]
    for item in requests:
        service.add_request(item)
    try:
        for step in range(4):
            sequences = [
                list(item.prompt_token_ids)
                if step == 0
                else [item.generated_token_ids[-1]]
                for item in requests
            ]
            past = [
                0 if step == 0 else item.get_total_length() - 1 for item in requests
            ]
            # Identical batch shapes isolate scheduler state ownership from
            # low-precision changes between full-prefix and single-token GEMMs.
            expected = (
                forward(
                    service.model_runner.model_engine,
                    sequences,
                    [0] * len(requests) if step == 0 else reference_rows,
                    reference_rows,
                    past,
                    sample_all_positions=False,
                )
                .argmax(-1)
                .tolist()
            )
            assert service.step()[0]
            assert [item.generated_token_ids[-1] for item in requests] == expected
        assert all(item.is_finished() for item in requests)
    finally:
        for row in reference_rows:
            manager.free(row)
        for item in requests:
            if not item.is_finished():
                item.mark_canceled()
        service.step()
    assert_released(service)


def test_state_pool_exhaustion_and_deferred_admission(service):
    manager = service.scheduler.mamba_cache_manager
    items = [
        request(f"pool-{i}", [17 + i, 83, 51], count=5)
        for i in range(manager.num_blocks + 1)
    ]
    for item in items:
        service.add_request(item)
    assert service.step()[0]
    assert len(manager.used_block_ids) == manager.num_blocks - 1
    assert sum(item.mamba_cache_index is None for item in items) == 2
    # Cancel one admitted request and one waiting request, then admit the survivor.
    items[0].mark_canceled()
    items[-1].mark_canceled()
    for _ in range(20):
        service.step()
        assert (
            len(manager.used_block_ids) + len(manager.free_block_ids)
            == manager.num_blocks - 1
        )
        if all(item.is_finished() for item in items):
            break
    assert all(item.is_finished() for item in items)
    assert len(items[-2].generated_token_ids) == 5
    service.step()
    assert_released(service)


def test_eos_configuration_and_early_finish_release_state(service):
    from infinilm.llm.request import FinishReason

    configured_eos = service.model_runner.eos_token_id
    if isinstance(configured_eos, int):
        assert service.eos_token_ids == [configured_eos]
    manager = service.scheduler.mamba_cache_manager
    row = manager.allocate()
    try:
        token = (
            forward(
                service.model_runner.model_engine,
                [[17, 83, 51]],
                [0],
                [row],
                sample_all_positions=False,
            )
            .argmax(-1)
            .item()
        )
    finally:
        manager.free(row)
    item = request("early-eos", [17, 83, 51], count=8)
    item.eos_token_ids = [token]
    item.sampling_params.ignore_eos = False
    service.add_request(item)
    assert service.step()[0]
    assert item.finish_reason == FinishReason.EOS_TOKEN
    assert len(item.generated_token_ids) == 1
    service.step()
    assert_released(service)


def test_repeated_finish_cancel_and_slot_reuse(service):
    first_token = None
    for iteration in range(12):
        item = request(f"cycle-{iteration}", [17, 83, 51], count=3)
        service.add_request(item)
        mode = iteration % 4
        if mode == 0:
            item.mark_canceled()
        else:
            assert service.step()[0]
            if first_token is None:
                first_token = item.generated_token_ids[0]
            assert item.generated_token_ids[0] == first_token
            if mode == 1:
                item.mark_canceled()
            else:
                assert service.step()[0]
                if mode == 2:
                    item.mark_canceled()
                else:
                    assert service.step()[0]
        service.step()
        assert item.is_finished()
        assert_released(service)
