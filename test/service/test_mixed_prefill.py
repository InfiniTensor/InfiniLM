"""Mixed prefill must preserve request-local KV, recurrent and image metadata."""

import numpy as np
import pytest
import torch

from infinilm.llm.request import InferenceRequest, RequestStatus
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler
from infinilm.processors.basic_llm_processor import BasicLLMProcessor
from infinilm.processors.qwen3_5_processor import Qwen35Processor


def request(name, tokens, **kw):
    return InferenceRequest(
        name,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=8),
        **kw,
    )


def ready_scheduler(budget=32, mamba=False):
    scheduler = Scheduler(
        max_batch_size=4,
        num_blocks=32,
        block_size=4,
        max_num_batched_tokens=budget,
        has_mamba_cache=mamba,
        enable_prefix_caching=False,
        enable_mixed_prefill=True,
    )
    first = request("running", [10, 11, 12, 13])
    scheduler.add_request(first)
    initial = scheduler.schedule()
    assert initial.request_is_prefill(first)
    first.append_generated_token_id(99)
    scheduler.complete_requests([first])
    return scheduler, first


def test_mixed_slots_and_token_budget():
    scheduler, first = ready_scheduler()
    old_blocks = list(first.block_table)
    second = request("new", [20, 21, 22])
    scheduler.add_request(second)
    batch = scheduler.schedule()
    assert batch.scheduled_requests == [second, first]
    assert batch.request_is_prefill(second)
    assert not batch.request_is_prefill(first)
    assert first.block_table[: len(old_blocks)] == old_blocks
    assert len(first.slot_mapping) == 1
    assert set(second.block_table).isdisjoint(first.block_table)
    inputs = BasicLLMProcessor.__new__(BasicLLMProcessor).build_model_inputs(batch)
    assert inputs["input_ids"].to_numpy().tolist() == [[20, 21, 22, 99]]
    assert inputs["input_offsets"].to_numpy().tolist() == [0, 3, 4]
    assert inputs["past_kv_lengths"].to_numpy().tolist() == [0, 4]
    assert inputs["total_kv_lengths"].to_numpy().tolist() == [3, 5]
    for req in batch.scheduled_requests:
        req.status = RequestStatus.CANCELED
    scheduler.complete_requests(batch.scheduled_requests)
    assert scheduler.get_cache_stats()["usable_blocks"] == 32
    assert all(block.ref_count == 0 for block in scheduler.cache_manager.blocks)


def test_full_prefill_budget_defers_decode_without_consuming_slot():
    scheduler, first = ready_scheduler(budget=4)
    before = list(first.block_table), list(first.slot_mapping)
    second = request("new", [20, 21, 22, 23])
    scheduler.add_request(second)
    batch = scheduler.schedule()
    assert batch.scheduled_requests == [second]
    assert scheduler.running_queue.sync_q.qsize() == 1
    assert (first.block_table, first.slot_mapping) == before


def test_mixed_qwen35_state_and_image_isolation():
    scheduler, first = ready_scheduler(mamba=True)
    pixels = torch.randn(16, 12)
    media = {
        "pixel_values": pixels,
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "image_bound": torch.tensor([[1, 5]]),
    }
    # An already-running multimodal request must not reinsert its old image.
    first.processed_inputs = media
    first.mrope_position_delta = -2
    second = request(
        "new-image",
        [20, 21, 21, 21, 21, 22],
        processed_inputs=media,
        has_multimodal_inputs=True,
    )
    scheduler.add_request(second)
    batch = scheduler.schedule()
    processor = Qwen35Processor.__new__(Qwen35Processor)
    processor.pixel_values_dtype = torch.bfloat16
    processor.spatial_merge_size = 2
    inputs = processor.build_model_inputs(batch)
    assert inputs["mamba_init_state_indices"].to_numpy().tolist() == [
        0,
        first.mamba_cache_index,
    ]
    assert inputs["mamba_final_state_indices"].to_numpy().tolist() == [
        second.mamba_cache_index,
        first.mamba_cache_index,
    ]
    assert inputs["image_req_ids"] == [0]
    assert len(inputs["pixel_values"]) == 1
    positions = inputs["position_ids"].to_numpy()
    assert positions.shape == (3, 7)
    np.testing.assert_array_equal(positions[:, -1], [2, 2, 2])
    assert first.mrope_position_delta == -2
    for req in batch.scheduled_requests:
        req.status = RequestStatus.FINISHED
    scheduler.complete_requests(batch.scheduled_requests)
    assert scheduler.get_cache_stats()["usable_blocks"] == 32
    assert all(block.ref_count == 0 for block in scheduler.cache_manager.blocks)
    assert scheduler.get_cache_stats()["num_used_mamba_cache_blocks"] == 0


@pytest.mark.parametrize(
    "partial_first, continuation_limit", [(False, 0), (True, 0), (True, 2)]
)
def test_chunked_prefill_emits_only_after_the_complete_prompt(
    partial_first, continuation_limit
):
    from types import SimpleNamespace
    from infinilm.llm.llm import LLMEngine
    from infinilm.llm.model_runner.model_runner import ModelRunnerOutput

    scheduler = Scheduler(
        max_batch_size=4,
        num_blocks=128,
        block_size=4,
        max_num_batched_tokens=16,
        has_mamba_cache=True,
        num_mamba_cache_blocks=9,
        enable_prefix_caching=False,
        enable_mixed_prefill=True,
        prefill_chunk_size=3,
        partial_prefill_first=partial_first,
        max_prefill_continuations=continuation_limit,
    )
    requests = [request(str(i), [10 + i] * 8) for i in range(8)]
    for req in requests:
        req.sampling_params.max_tokens = 2
        scheduler.add_request(req)
    seen_partial = []
    processor = Qwen35Processor.__new__(Qwen35Processor)
    processor.spatial_merge_size = 2
    processor.pixel_values_dtype = torch.bfloat16

    def execute(batch):
        inputs = processor.build_model_inputs(batch)
        initial = inputs["mamba_init_state_indices"].to_numpy().tolist()
        for i, req in enumerate(batch.scheduled_requests):
            assert initial[i] == (
                0 if req.num_local_cached_tokens == 0 else req.mamba_cache_index
            )
            if not batch.request_finishes_prefill(req):
                assert req.get_num_generated_tokens() == 0
                seen_partial.append(req.request_id)
        return ModelRunnerOutput(
            req_ids=[r.request_id for r in batch.scheduled_requests],
            sampled_token_ids=[42] * batch.num_requests,
        )

    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = scheduler
    engine.model_runner = SimpleNamespace(execute_model=execute)
    engine.tokenizer = SimpleNamespace(decode=lambda tokens: "x" * len(tokens))
    engine.eos_token_ids = []
    for _ in range(40):
        did_work, pending = engine.step()
        assert not pending
        if all(req.is_finished() for req in requests):
            break
        assert did_work
    assert len(seen_partial) >= 8
    assert all(list(req.generated_token_ids) == [42, 42] for req in requests)
    assert not scheduler.partial_prefill_requests
    assert scheduler.get_cache_stats()["usable_blocks"] == 128
    assert scheduler.get_cache_stats()["num_used_mamba_cache_blocks"] == 0


def test_chunk_boundary_never_splits_visual_span():
    scheduler = Scheduler(enable_mixed_prefill=True, prefill_chunk_size=8)
    scheduler._chunking_active = True
    req = request(
        "image",
        list(range(24)),
        mm_token_index_mappings=[
            {"start_index": 4, "end_index": 13, "identifier": "img"}
        ],
    )
    assert scheduler._prefill_token_count(req, 0) == 14
    assert scheduler._prefill_token_count(req, 14) == 8


def test_chunk_policy_preserves_one_pass_prefill_below_high_occupancy():
    scheduler = Scheduler(
        max_batch_size=32,
        num_blocks=128,
        block_size=4,
        max_num_batched_tokens=256,
        enable_prefix_caching=False,
        enable_mixed_prefill=True,
        prefill_chunk_size=3,
        partial_prefill_first=True,
    )
    requests = [request(str(i), [10 + i] * 8) for i in range(16)]
    for req in requests:
        scheduler.add_request(req)
    batch = scheduler.schedule()
    assert batch.num_requests == 16
    assert all(batch.request_token_count(req) == 8 for req in requests)
    assert all(batch.request_finishes_prefill(req) for req in requests)


@pytest.mark.parametrize("output_length, should_chunk", [(256, True), (1024, False)])
def test_prefill_pressure_does_not_depend_on_arrival_group_size(
    output_length, should_chunk
):
    scheduler = Scheduler(
        max_batch_size=32,
        num_blocks=1024,
        block_size=64,
        max_num_batched_tokens=16386,
        enable_prefix_caching=False,
        enable_mixed_prefill=True,
        prefill_chunk_size=1024,
    )
    requests = [request(str(i), [100 + i] * 1045) for i in range(15)]
    for req in requests:
        req.sampling_params.max_tokens = output_length
        scheduler.add_request(req)
    batch = scheduler.schedule()
    assert batch.num_requests == 15
    assert all(
        batch.request_token_count(req) == (1024 if should_chunk else 1045)
        for req in requests
    )
    for req in requests:
        req.status = RequestStatus.CANCELED
    scheduler.complete_requests(requests)
    assert not scheduler._prefill_heavy_requests
    assert scheduler.get_cache_stats()["usable_blocks"] == 1024


@pytest.mark.parametrize("mamba", [False, True])
def test_chunk_admission_uses_scheduled_tokens(mamba):
    scheduler = Scheduler(
        max_batch_size=32,
        num_blocks=1024,
        block_size=64,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        has_mamba_cache=mamba,
        num_mamba_cache_blocks=33,
        enable_mixed_prefill=True,
        prefill_chunk_size=1024,
    )
    requests = [request(str(i), [100 + i] * 1349) for i in range(32)]
    for req in requests:
        req.sampling_params.max_tokens = 256
        scheduler.add_request(req)
    batch = scheduler.schedule()
    assert batch.num_requests == 8
    assert (
        sum(batch.request_token_count(req) for req in batch.scheduled_requests) == 8192
    )
    assert all(
        not batch.request_finishes_prefill(req) for req in batch.scheduled_requests
    )
    # Disjoint cache reservations still cover each complete prompt.
    blocks = [block for req in batch.scheduled_requests for block in req.block_table]
    assert len(blocks) == len(set(blocks))
    for req in requests:
        req.status = RequestStatus.CANCELED
    scheduler.complete_requests(requests)
    assert not scheduler._prefill_heavy_requests
    assert scheduler.get_cache_stats()["usable_blocks"] == 1024


def test_arrivals_during_batch_construction_revise_all_prefill_chunks(monkeypatch):
    scheduler = Scheduler(
        max_batch_size=32,
        num_blocks=1024,
        block_size=64,
        max_num_batched_tokens=16386,
        enable_prefix_caching=False,
        enable_mixed_prefill=True,
        prefill_chunk_size=1024,
    )
    requests = [request(str(i), [100 + i] * 1045) for i in range(15)]
    for req in requests:
        req.sampling_params.max_tokens = 256
    scheduler.add_request(requests[0])
    allocate = scheduler.cache_manager.allocate_slots
    arrived = False

    def allocate_with_arrivals(*args, **kwargs):
        nonlocal arrived
        result = allocate(*args, **kwargs)
        if not arrived:
            arrived = True
            for req in requests[1:]:
                scheduler.add_request(req)
        return result

    monkeypatch.setattr(
        scheduler.cache_manager, "allocate_slots", allocate_with_arrivals
    )
    batch = scheduler.schedule()
    assert batch.num_requests == 15
    assert all(batch.request_token_count(req) == 1024 for req in requests)
    assert all(len(req.slot_mapping) == 1024 for req in requests)
    inputs = BasicLLMProcessor.__new__(BasicLLMProcessor).build_model_inputs(batch)
    assert inputs["input_offsets"].to_numpy().tolist() == [i * 1024 for i in range(16)]
    for req in requests:
        req.status = RequestStatus.CANCELED
    scheduler.complete_requests(requests)
    assert scheduler.get_cache_stats()["usable_blocks"] == 1024


@pytest.mark.parametrize(
    "temperature,top_k,expected", [(0, 0, 8), (1, 1, 8), (0.7, 20, 24)]
)
def test_chunking_checks_each_request_sampling(temperature, top_k, expected):
    scheduler = Scheduler(enable_mixed_prefill=True, prefill_chunk_size=8)
    scheduler._chunking_active = True
    req = request("sampling", list(range(24)))
    req.sampling_params.temperature = temperature
    req.sampling_params.top_k = top_k
    scheduler.add_request(req)
    assert scheduler._prefill_token_count(req, 0) == expected
    if expected == 24:
        assert req.request_id not in scheduler._prefill_heavy_requests


@pytest.mark.parametrize(
    "model_type,changes,device_name,enabled,override",
    [
        ("qwen3", {}, "ILUVATAR", True, None),
        ("qwen3_5", {}, "ILUVATAR", True, None),
        ("qwen3", {}, "NVIDIA", False, None),
        ("qwen3_5", {"dtype": "float16"}, "ILUVATAR", False, None),
        ("qwen3", {"tensor_parallel_size": 2}, "ILUVATAR", False, None),
        ("qwen3_5", {"pipeline_parallel_size": 2}, "ILUVATAR", False, None),
        ("qwen3", {"draft_model_path": "draft"}, "ILUVATAR", False, None),
        ("qwen3", {"enable_prefix_caching": True}, "ILUVATAR", False, None),
        ("qwen3_5", {"top_k": 20}, "ILUVATAR", False, None),
        ("other", {}, "ILUVATAR", False, None),
        ("qwen3_5", {}, "ILUVATAR", False, "0"),
    ],
)
def test_engine_selects_chunking_only_for_supported_runtime(
    monkeypatch, model_type, changes, device_name, enabled, override
):
    from importlib import import_module
    from types import SimpleNamespace
    from infinicore.lib import _infinicore
    from infinilm.config.engine_config import EngineConfig

    module = import_module("infinilm.llm.llm")
    for key in (
        "INFINILM_EXPERIMENTAL_MIXED_PREFILL",
        "INFINILM_EXPERIMENTAL_PREFILL_CHUNK_SIZE",
        "INFINILM_EXPERIMENTAL_PREFILL_ORDER",
        "INFINILM_EXPERIMENTAL_PREFILL_CONTINUATIONS",
    ):
        monkeypatch.delenv(key, raising=False)
    if override is not None:
        monkeypatch.setenv("INFINILM_EXPERIMENTAL_MIXED_PREFILL", override)
    monkeypatch.setenv("INFINILM_MAX_NUM_BATCHED_TOKENS", "8192")
    runner = SimpleNamespace(
        device=SimpleNamespace(
            _underlying=SimpleNamespace(
                type=getattr(_infinicore.Device.Type, device_name)
            )
        ),
        dtype=None,
        processor=SimpleNamespace(get_tokenizer=lambda: None),
        model_engine=SimpleNamespace(hf_config={"max_position_embeddings": 16384}),
        model_type=model_type,
        eos_token_id=[],
    )
    monkeypatch.setattr(module, "ModelRunner", lambda config: runner)
    monkeypatch.setattr(
        module, "read_hf_config", lambda path: {"model_type": model_type}
    )
    monkeypatch.setattr(module, "model_uses_mamba_cache", lambda config: False)
    options = dict(
        model_path="test",
        dtype="bfloat16",
        max_batch_size=32,
        enable_prefix_caching=False,
    )
    options.update(changes)
    engine = module.LLMEngine(EngineConfig(**options))
    assert engine.scheduler.enable_mixed_prefill == enabled
    assert engine.scheduler.prefill_chunk_size == (1024 if enabled else 0)
    if enabled:
        assert engine.scheduler.partial_prefill_first == (model_type == "qwen3_5")
        assert engine.scheduler.max_prefill_continuations == (
            8 if model_type == "qwen3_5" else 0
        )


@pytest.mark.parametrize("concurrency,expected", [(16, 2), (32, 22)])
def test_visual_work_budget_preserves_high_occupancy_throughput(concurrency, expected):
    scheduler = Scheduler(
        max_batch_size=32,
        num_blocks=1024,
        block_size=64,
        max_num_batched_tokens=8192,
        has_mamba_cache=True,
        num_mamba_cache_blocks=33,
        enable_prefix_caching=False,
        enable_mixed_prefill=True,
        prefill_chunk_size=1024,
        max_num_batched_visual_tokens=2400,
    )
    requests = [
        request(
            str(i),
            [100 + i] * 357,
            has_multimodal_inputs=True,
            processed_inputs={
                "image_grid_thw": torch.tensor([[1, 30, 40]]),
                "image_bound": torch.tensor([[1, 301]]),
            },
        )
        for i in range(concurrency)
    ]
    for req in requests:
        scheduler.add_request(req)
    batch = scheduler.schedule()
    assert batch.num_requests == expected
    assert all(batch.request_finishes_prefill(req) for req in batch.scheduled_requests)
    assert all(req.mamba_cache_index is None for req in requests[expected:])
    for req in requests:
        req.status = RequestStatus.CANCELED
    scheduler.complete_requests(requests)
    assert not scheduler._visual_prefill_spans
    assert scheduler.get_cache_stats()["num_used_mamba_cache_blocks"] == 0
    assert scheduler.get_cache_stats()["usable_blocks"] == 1024


def test_visual_work_charges_only_images_in_the_current_chunk():
    scheduler = Scheduler(max_num_batched_visual_tokens=2400)
    req = request(
        "two-images",
        list(range(32)),
        processed_inputs={
            "image_grid_thw": [torch.tensor([1, 20, 60]), torch.tensor([1, 40, 40])],
            "image_bound": torch.tensor([[[1, 9], [17, 25]]]),
        },
    )
    scheduler.add_request(req)
    assert scheduler._visual_token_count(req, 0, 12) == 1200
    assert scheduler._visual_token_count(req, 12, 5) == 0
    assert scheduler._visual_token_count(req, 17, 8) == 1600
    assert scheduler._visual_token_count(req, 25, 7) == 0
    assert scheduler._visual_token_count(req, 0, 32) == 2800


def test_visual_work_allows_one_oversized_request_to_progress():
    scheduler = Scheduler(
        max_batch_size=32,
        num_blocks=32,
        block_size=64,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        max_num_batched_visual_tokens=2400,
    )
    req = request(
        "large-image",
        list(range(320)),
        processed_inputs={
            "image_grid_thw": torch.tensor([[1, 80, 120]]),
            "image_bound": torch.tensor([[1, 301]]),
        },
    )
    scheduler.add_request(req)
    batch = scheduler.schedule()
    assert batch.scheduled_requests == [req]


def test_visual_work_admission_adapts_to_new_arrivals(monkeypatch):
    scheduler = Scheduler(
        max_batch_size=32,
        num_blocks=1024,
        block_size=64,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        max_num_batched_visual_tokens=2400,
    )
    requests = [
        request(
            str(i),
            [100 + i] * 357,
            processed_inputs={
                "image_grid_thw": torch.tensor([[1, 30, 40]]),
                "image_bound": torch.tensor([[1, 301]]),
            },
        )
        for i in range(32)
    ]
    scheduler.add_request(requests[0])
    allocate = scheduler.cache_manager.allocate_slots
    arrived = False

    def allocate_with_arrivals(*args, **kwargs):
        nonlocal arrived
        result = allocate(*args, **kwargs)
        if not arrived:
            arrived = True
            for req in requests[1:]:
                scheduler.add_request(req)
        return result

    monkeypatch.setattr(
        scheduler.cache_manager, "allocate_slots", allocate_with_arrivals
    )
    batch = scheduler.schedule()
    assert batch.num_requests == 22
