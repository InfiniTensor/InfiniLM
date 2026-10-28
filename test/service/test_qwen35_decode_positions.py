"""Decode position packing must preserve MRoPE and per-request state metadata."""

import numpy as np
import pytest

from infinilm.llm.request import InferenceRequest
from infinilm.llm.scheduler import SchedulerOutput
from infinilm.llm.static_scheduler import StaticSchedulerOutput
from infinilm.processors.qwen3_5_processor import Qwen35Processor


@pytest.mark.parametrize(
    "count,static", [(1, False), (3, False), (16, False), (32, False), (1, True)]
)
def test_decode_positions_match_legacy_across_reordering_and_continuation(
    monkeypatch, count, static
):
    processor = Qwen35Processor.__new__(Qwen35Processor)
    requests = []
    for i in range(count):
        req = InferenceRequest(str(i), prompt_token_ids=[100 + i] * (335 + i))
        req.mrope_position_delta = [0, -287, 9][i % 3]
        req.mamba_cache_index = count - i
        req.processed_inputs = {"pixel_values": object()}
        requests.append(req)
    for step in range(3):
        if step:
            requests.reverse()
        for i, req in enumerate(requests):
            req.append_generated_token_id(200 + step * count + i)
            req.num_local_cached_tokens = req.get_total_length() - 1
            req.block_table = list(range((req.get_total_length() + 63) // 64))
            req.slot_mapping = [10000 + i * 64 + step]
        scheduler_output = (StaticSchedulerOutput if static else SchedulerOutput)(
            requests, is_prefill=False
        )
        before = [
            (
                req.request_id,
                req.mrope_position_delta,
                req.mamba_cache_index,
                req.num_local_cached_tokens,
                list(req.block_table),
                list(req.slot_mapping),
            )
            for req in requests
        ]
        monkeypatch.setenv("INFINILM_DISABLE_BATCHED_DECODE_POSITIONS", "1")
        legacy = processor.build_model_inputs(scheduler_output)
        monkeypatch.delenv("INFINILM_DISABLE_BATCHED_DECODE_POSITIONS")
        packed = processor.build_model_inputs(scheduler_output)
        for key in legacy:
            left, right = legacy[key], packed[key]
            if hasattr(left, "to_numpy"):
                assert left.dtype == right.dtype
                np.testing.assert_array_equal(
                    left.to_numpy(), right.to_numpy(), err_msg=key
                )
            else:
                assert left == right
        expected = [
            req.get_total_length() - 1 + req.mrope_position_delta for req in requests
        ]
        np.testing.assert_array_equal(packed["position_ids"].to_numpy(), [expected] * 3)
        assert before == [
            (
                req.request_id,
                req.mrope_position_delta,
                req.mamba_cache_index,
                req.num_local_cached_tokens,
                list(req.block_table),
                list(req.slot_mapping),
            )
            for req in requests
        ]
