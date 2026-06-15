"""Regression tests for concurrent batch metadata and per-request token routing."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from infinilm.llm.llm import LLMEngine
from infinilm.llm.request import InferenceRequest, RequestStatus
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import ScheduledRow, SchedulerOutput, Scheduler
from infinilm.processors.basic_llm_processor import BasicLLMProcessor


def _req(
    request_id: str,
    prompt_len: int,
    *,
    chunk_size: int = 4096,
    max_tokens: int = 8,
) -> InferenceRequest:
    req = InferenceRequest(
        request_id=request_id,
        prompt_token_ids=list(range(1, prompt_len + 1)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )
    req.chunk_size = chunk_size
    return req


def _prime_decode(scheduler: Scheduler, req: InferenceRequest, last_token: int) -> None:
    req.is_prefill = False
    req.generated_token_ids = [last_token]
    req.block_table, req.slot_mapping, req.num_cached_tokens = (
        scheduler.cache_manager.allocate_blocks(req.get_input_tokens(), req.block_table)
    )
    req.num_blocks = len(req.block_table)
    req.status = RequestStatus.RUNNING
    req.num_cached_tokens = req.get_total_length() - 1


class ProcessorConcurrentBatchTest(unittest.TestCase):
    def setUp(self):
        self.processor = BasicLLMProcessor.__new__(BasicLLMProcessor)
        self.processor.num_blocks = 512
        self.scheduler = Scheduler(
            max_batch_size=4,
            num_blocks=512,
            block_size=256,
            enable_prefix_cache=False,
        )

    def test_block_tables_padded_to_num_blocks(self):
        short = _req("short", 128)
        long = _req("long", 4096)
        _prime_decode(self.scheduler, short, 1001)
        _prime_decode(self.scheduler, long, 2002)

        rows = [
            ScheduledRow(short, 1, False, False),
            ScheduledRow(long, 1, False, False),
        ]
        out = SchedulerOutput(
            scheduled_requests=[short, long],
            is_prefill=False,
            rows=rows,
        )
        model_input = self.processor._build_model_input_from_rows(
            out, temperature=1.0, top_p=1.0, top_k=1
        )
        block_tables = model_input["block_tables"].to_numpy().tolist()
        self.assertEqual(len(block_tables), 2)
        self.assertEqual(len(block_tables[0]), 512)
        self.assertEqual(len(block_tables[1]), 512)
        self.assertNotEqual(block_tables[0][:3], block_tables[1][:3])

    def test_concurrent_decode_rows_have_distinct_positions(self):
        a = _req("req-a", 512)
        b = _req("req-b", 2048)
        _prime_decode(self.scheduler, a, 42)
        _prime_decode(self.scheduler, b, 99)

        rows = [
            ScheduledRow(a, 1, False, False),
            ScheduledRow(b, 1, False, False),
        ]
        out = SchedulerOutput(
            scheduled_requests=[a, b],
            is_prefill=False,
            rows=rows,
        )
        model_input = self.processor._build_model_input_from_rows(
            out, temperature=1.0, top_p=1.0, top_k=1
        )
        position_ids = model_input["position_ids"].to_numpy().tolist()
        total_kv = model_input["total_kv_lengths"].to_numpy().tolist()
        self.assertEqual(position_ids, [512, 2048])
        self.assertEqual(total_kv, [513, 2049])
        self.assertEqual(model_input["scheduling_mode"], "DECODE")


class LLMTokenRoutingTest(unittest.TestCase):
    def test_update_requests_from_rows_maps_tokens_by_row_order(self):
        os.environ["INFINI_V1_SCHEDULER"] = "1"
        engine = LLMEngine.__new__(LLMEngine)
        engine.cache_type = "paged"
        engine.scheduler = mock.Mock()
        engine.scheduler.complete_requests = mock.Mock()
        engine.scheduler.cache_manager = mock.Mock()
        engine.tokenizer = mock.Mock()
        engine.tokenizer.decode.side_effect = lambda tokens: f"t{tokens[-1]}"

        req_a = _req("req-a", 16)
        req_b = _req("req-b", 32)
        req_a.is_prefill = False
        req_b.is_prefill = False
        req_a.generated_token_ids = [11]
        req_b.generated_token_ids = [22]

        rows = [
            ScheduledRow(req_a, 1, False, False),
            ScheduledRow(req_b, 1, False, False),
        ]
        scheduler_output = SchedulerOutput(
            scheduled_requests=[req_a, req_b],
            is_prefill=False,
            rows=rows,
        )

        with mock.patch.object(
            LLMEngine, "_check_request_finished", return_value=False
        ):
            LLMEngine._update_requests_from_rows(
                engine, scheduler_output, [101, 202]
            )

        self.assertEqual(req_a.generated_token_ids, [11, 101])
        self.assertEqual(req_b.generated_token_ids, [22, 202])
        self.assertEqual(req_a.generated_text, "t101")
        self.assertEqual(req_b.generated_text, "t202")
        os.environ.pop("INFINI_V1_SCHEDULER", None)

    def test_rows_needing_sampled_tokens_skips_mid_chunk_prefill(self):
        engine = LLMEngine.__new__(LLMEngine)
        mid = _req("mid", 4096)
        mid.is_prefill = True
        final = _req("final", 128)
        final.is_prefill = True

        rows = [
            ScheduledRow(mid, 4096, True, False),
            ScheduledRow(final, 128, True, True),
        ]
        scheduler_output = SchedulerOutput(
            scheduled_requests=[mid, final],
            is_prefill=True,
            rows=rows,
        )
        sampling_rows = LLMEngine._rows_needing_sampled_tokens(
            engine, scheduler_output
        )
        self.assertEqual(len(sampling_rows), 1)
        self.assertEqual(sampling_rows[0].request.request_id, "final")


if __name__ == "__main__":
    unittest.main()
