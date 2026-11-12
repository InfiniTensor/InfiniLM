"""Processor row-based metadata for v1 mixed scheduling."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from infinilm.llm.request import InferenceRequest, RequestStatus
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import ScheduledRow, Scheduler, SchedulerOutput
from infinilm.processors.basic_llm_processor import BasicLLMProcessor


def _req(rid: str, prompt_len: int, *, chunk_size: int = 4096) -> InferenceRequest:
    req = InferenceRequest(
        request_id=rid,
        prompt_token_ids=list(range(1, prompt_len + 1)),
        sampling_params=SamplingParams(max_tokens=1),
    )
    req.chunk_size = chunk_size
    return req


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def to_numpy(self):
        import numpy as np

        return np.asarray(self._values)


class ProcessorV1RowsTest(unittest.TestCase):
    def setUp(self):
        self.processor = BasicLLMProcessor.__new__(BasicLLMProcessor)

    def test_mixed_row_metadata_decode_cu_seqlens(self):
        """Decode row cu_seqlens uses total KV length, not 1."""
        sched = Scheduler(
            max_batch_size=4,
            num_blocks=512,
            block_size=256,
            enable_prefix_cache=False,
        )
        decode = _req("d", 4096, chunk_size=4096)
        decode.is_prefill = False
        decode.generated_token_ids = [99]
        decode.block_table, decode.slot_mapping, _ = (
            sched.cache_manager.allocate_blocks(decode.get_input_tokens(), [])
        )
        # Extend allocation for the generated token slot used at decode.
        decode.block_table, new_slot = sched.cache_manager.append_slot(
            decode.block_table, decode.get_total_length()
        )
        decode.slot_mapping = [new_slot]
        decode.num_blocks = len(decode.block_table)
        decode.status = RequestStatus.RUNNING
        decode.num_local_cached_tokens = decode.get_total_length() - 1
        decode.num_computed_tokens = decode.num_local_cached_tokens

        prefill = _req("p", 4096, chunk_size=4096)
        prefill.block_table, prefill.slot_mapping, cached = (
            sched.cache_manager.allocate_blocks(prefill.get_input_tokens(), [])
        )
        prefill.num_blocks = len(prefill.block_table)
        prefill.status = RequestStatus.RUNNING
        prefill.num_local_cached_tokens = cached
        prefill.num_computed_tokens = cached
        prefill.chunk_prefill_offset = 0

        rows = [
            ScheduledRow(decode, 1, False, False),
            ScheduledRow(prefill, 4096, True, False),
        ]
        out = SchedulerOutput(
            scheduled_requests=[decode, prefill],
            is_prefill=False,
            rows=rows,
        )

        # Avoid requiring a real infinicore extension: stub from_list.
        import sys
        import types

        fake_ic = types.ModuleType("infinicore")
        fake_ic.int32 = "int32"
        fake_ic.int64 = "int64"

        def _from_list(values, dtype=None):
            return _FakeTensor(values)

        fake_ic.from_list = _from_list
        sys.modules["infinicore"] = fake_ic

        model_input = self.processor._build_model_input_from_rows(
            out, temperature=1.0, top_p=1.0, top_k=1
        )
        cu = model_input["cu_seqlens"].to_numpy().tolist()
        self.assertEqual(model_input["scheduling_mode"], "MIXED")
        self.assertEqual(len(model_input["is_final_prefill_chunk"]), 2)
        self.assertTrue(model_input["is_final_prefill_chunk"][0])
        self.assertFalse(model_input["is_final_prefill_chunk"][1])
        self.assertEqual(cu[-1], 4096 + 4097)
        past = model_input["past_kv_lengths"].to_numpy().tolist()
        total = model_input["total_kv_lengths"].to_numpy().tolist()
        self.assertEqual(past[0], 4096)
        self.assertEqual(total[0], 4097)


if __name__ == "__main__":
    unittest.main()
