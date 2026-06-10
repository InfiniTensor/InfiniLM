"""Unit tests for v1 token-budget scheduler (INFINI_V1_SCHEDULER=1)."""

from __future__ import annotations

import os
import unittest

from infinilm.llm.request import InferenceRequest, RequestStatus
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler


def _req(
    request_id: str,
    prompt_len: int,
    *,
    chunk_size: int = 4096,
    max_tokens: int = 1,
) -> InferenceRequest:
    req = InferenceRequest(
        request_id=request_id,
        prompt_token_ids=list(range(1, prompt_len + 1)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )
    req.chunk_size = chunk_size
    return req


def _prime_kv(scheduler: Scheduler, req: InferenceRequest) -> None:
    """Allocate block table so running-queue decode can append_slot."""
    tokens = req.get_input_tokens()
    req.block_table, req.slot_mapping, req.num_cached_tokens = (
        scheduler.cache_manager.allocate_blocks(tokens, req.block_table)
    )
    req.num_blocks = len(req.block_table)
    req.status = RequestStatus.RUNNING


class SchedulerV1Test(unittest.TestCase):
    def setUp(self):
        os.environ["INFINI_V1_SCHEDULER"] = "1"
        os.environ["INFINI_MAX_NUM_BATCHED_TOKENS"] = "8192"
        os.environ.pop("INFINI_MAX_PREFILL_BATCH", None)
        self.scheduler = Scheduler(
            max_batch_size=4,
            num_blocks=512,
            block_size=256,
            max_prefill_batch_size=4,
            enable_prefix_cache=False,
        )

    def tearDown(self):
        os.environ.pop("INFINI_V1_SCHEDULER", None)
        os.environ.pop("INFINI_MAX_NUM_BATCHED_TOKENS", None)

    def test_mixed_prefill_4096_and_decode_one(self):
        """Running decode + new 4096 prefill → MIXED step, 4097 tokens."""
        decode = _req("decode", 4096, chunk_size=4096)
        decode.is_prefill = False
        decode.generated_token_ids = [99]
        _prime_kv(self.scheduler, decode)
        self.scheduler.running_queue.sync_q.put(decode)

        prefill = _req("prefill", 4096, chunk_size=4096)
        self.scheduler.add_request(prefill)

        out = self.scheduler.schedule()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.scheduling_mode, "MIXED")
        self.assertEqual(out.total_scheduled_tokens, 4097)
        self.assertEqual(len(out.rows), 2)
        prefill_rows = [r for r in out.rows if r.is_prefill_row]
        decode_rows = [r for r in out.rows if not r.is_prefill_row]
        self.assertEqual(len(prefill_rows), 1)
        self.assertEqual(len(decode_rows), 1)
        self.assertEqual(prefill_rows[0].num_scheduled_tokens, 4096)
        self.assertEqual(decode_rows[0].num_scheduled_tokens, 1)

    def test_budget_exhaustion_prefill_before_decode(self):
        """Budget 4096: prefill fills step; decode on next step."""
        os.environ["INFINI_MAX_NUM_BATCHED_TOKENS"] = "4096"

        self.scheduler.add_request(_req("prefill", 4096, chunk_size=4096))

        out1 = self.scheduler.schedule()
        self.assertIsNotNone(out1)
        assert out1 is not None
        self.assertEqual(out1.scheduling_mode, "PREFILL")
        self.assertEqual(out1.total_scheduled_tokens, 4096)
        self.assertEqual(len(out1.rows), 1)
        self.assertTrue(out1.rows[0].is_prefill_row)

        row = out1.rows[0]
        row.request.is_prefill = False
        row.request.generated_token_ids = [1]
        self.scheduler.requeue_running(row.request)

        out2 = self.scheduler.schedule()
        self.assertIsNotNone(out2)
        assert out2 is not None
        self.assertEqual(out2.scheduling_mode, "DECODE")
        self.assertEqual(out2.total_scheduled_tokens, 1)

    def test_seq_cap_max_batch_size_two(self):
        """Only two rows scheduled when max_batch_size=2."""
        self.scheduler.max_batch_size = 2
        for i in range(4):
            self.scheduler.add_request(_req(f"r{i}", 1024, chunk_size=1024))

        out = self.scheduler.schedule()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(len(out.rows), 2)

    def test_single_8192_mid_chunk_no_spurious_2pack(self):
        """mc=1: after first 4096 chunk, second step is 1 row (not 2-pack @8192)."""
        req = _req("solo", 8192, chunk_size=4096)
        self.scheduler.max_batch_size = 1
        self.scheduler.add_request(req)

        out1 = self.scheduler.schedule()
        self.assertIsNotNone(out1)
        assert out1 is not None
        self.assertEqual(len(out1.rows), 1)
        self.assertEqual(out1.total_scheduled_tokens, 4096)
        self.assertFalse(out1.rows[0].is_final_prefill_chunk)

        for row in out1.rows:
            row.request.chunk_prefill_offset += row.num_scheduled_tokens
        self.scheduler.complete_requests(out1.scheduled_requests)

        out2 = self.scheduler.schedule()
        self.assertIsNotNone(out2)
        assert out2 is not None
        self.assertEqual(len(out2.rows), 1)
        self.assertEqual(out2.total_scheduled_tokens, 4096)
        self.assertTrue(out2.rows[0].is_final_prefill_chunk)

    def test_four_8192_chunk4096_pack_two_rows(self):
        """4×8192 chunk4096 → one step packs 2 rows, 8192 tokens."""
        chunk_size = 4096
        for i in range(4):
            self.scheduler.add_request(_req(f"r{i}", 8192, chunk_size=chunk_size))

        out = self.scheduler.schedule()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.scheduling_mode, "PREFILL")
        self.assertEqual(len(out.rows), 2)
        self.assertEqual(out.total_scheduled_tokens, 8192)
        self.assertTrue(all(r.is_prefill_row for r in out.rows))
        self.assertTrue(
            all(r.num_scheduled_tokens == chunk_size for r in out.rows)
        )

    def test_dry_run_one_decode_one_prefill_shape(self):
        """Schedule-only: 1 decode + 1 prefill metadata shape."""
        decode = _req("d", 512, chunk_size=512)
        decode.is_prefill = False
        decode.generated_token_ids = [7]
        _prime_kv(self.scheduler, decode)
        self.scheduler.running_queue.sync_q.put(decode)

        self.scheduler.add_request(_req("p", 4096, chunk_size=4096))

        out = self.scheduler.schedule()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.scheduling_mode, "MIXED")
        self.assertEqual(out.total_scheduled_tokens, 4097)
        self.assertEqual(len(out.scheduled_requests), 2)
        self.assertEqual(len(out.rows), 2)

    def test_can_accept_rejects_third_long_seq(self):
        """3×65536-token requests with num_blocks=512 → third rejected."""
        long_len = 65536
        blocks_per_req = (long_len + 1 + 255) // 256  # max_tokens=1
        self.assertGreater(blocks_per_req * 3, 512)

        for i in range(2):
            req = _req(f"long{i}", long_len, max_tokens=1)
            _prime_kv(self.scheduler, req)
            self.scheduler.running_queue.sync_q.put(req)

        third = _req("long2", long_len, max_tokens=1)
        self.assertFalse(self.scheduler.can_accept_request(third))

    def test_can_accept_counts_allocated_blocks_not_remaining_decode(self):
        """Running 65536-token decode reserves full KV; second long seq rejected."""
        long_len = 65536
        decode = _req("decode", long_len, max_tokens=512)
        decode.is_prefill = False
        decode.generated_token_ids = [1]
        _prime_kv(self.scheduler, decode)
        self.scheduler.running_queue.sync_q.put(decode)

        second_long = _req("second", long_len, max_tokens=512)
        blocks_per = (long_len + 512 + 255) // 256
        self.assertGreater(blocks_per * 2, 512)
        self.assertFalse(self.scheduler.can_accept_request(second_long))


if __name__ == "__main__":
    unittest.main()
