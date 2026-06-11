"""Unit tests for v1 token-budget scheduler (INFINI_V1_SCHEDULER=1)."""

from __future__ import annotations

import os
import unittest

from infinilm.compile.env import long_prefill_threshold
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
        os.environ.pop("INFINI_SCHEDULE_HOMOGENEOUS", None)
        os.environ.pop("INFINI_PREFILL_NATIVE_CG", None)
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
        os.environ.pop("INFINI_SCHEDULE_HOMOGENEOUS", None)
        os.environ.pop("INFINI_PREFILL_NATIVE_CG", None)

    def _enable_homogeneous(self) -> None:
        os.environ["INFINI_SCHEDULE_HOMOGENEOUS"] = "1"
        os.environ["INFINI_PREFILL_NATIVE_CG"] = "1"

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

    def test_homogeneous_prefill_then_decode_not_mixed(self):
        """Homogeneous: 1 decode + 1 waiting prefill → PREFILL then DECODE."""
        self._enable_homogeneous()

        decode = _req("decode", 4096, chunk_size=4096)
        decode.is_prefill = False
        decode.generated_token_ids = [99]
        _prime_kv(self.scheduler, decode)
        self.scheduler.running_queue.sync_q.put(decode)

        self.scheduler.add_request(_req("prefill", 4096, chunk_size=4096))

        out1 = self.scheduler.schedule()
        self.assertIsNotNone(out1)
        assert out1 is not None
        self.assertEqual(out1.scheduling_mode, "PREFILL")
        self.assertEqual(out1.total_scheduled_tokens, 4096)
        self.assertEqual(len(out1.rows), 1)

        self.scheduler.requeue_running(out1.rows[0].request)

        out2 = self.scheduler.schedule()
        self.assertIsNotNone(out2)
        assert out2 is not None
        self.assertEqual(out2.scheduling_mode, "DECODE")
        self.assertEqual(out2.total_scheduled_tokens, 1)

    def test_homogeneous_four_decode_before_waiting_prefill(self):
        """Homogeneous: 4 decode-ready + 1 waiting prefill → decode batch=4 first."""
        self._enable_homogeneous()

        for i in range(4):
            decode = _req(f"decode{i}", 512, chunk_size=512)
            decode.is_prefill = False
            decode.generated_token_ids = [i + 1]
            _prime_kv(self.scheduler, decode)
            self.scheduler.running_queue.sync_q.put(decode)

        self.scheduler.add_request(_req("prefill", 4096, chunk_size=4096))

        out = self.scheduler.schedule()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.scheduling_mode, "DECODE")
        self.assertEqual(len(out.rows), 4)
        self.assertTrue(all(not r.is_prefill_row for r in out.rows))

    def test_homogeneous_starvation_waiting_prefill_eventually_scheduled(self):
        """Homogeneous: waiting prefill scheduled after decode-yield cap."""
        self._enable_homogeneous()
        self.scheduler.max_waiting_yields = 2

        decode = _req("decode", 512, chunk_size=512)
        decode.is_prefill = False
        decode.generated_token_ids = [1]
        _prime_kv(self.scheduler, decode)
        self.scheduler.running_queue.sync_q.put(decode)

        self.scheduler.add_request(_req("prefill", 4096, chunk_size=4096))

        modes = []
        for _ in range(4):
            out = self.scheduler.schedule()
            if out is None:
                break
            modes.append(out.scheduling_mode)
            for row in out.rows:
                if row.is_prefill_row:
                    self.scheduler.complete_requests([row.request])
                else:
                    row.request.generated_token_ids.append(2)
                    self.scheduler.requeue_running(row.request)

        self.assertIn("PREFILL", modes)

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

    def test_can_accept_small_max_tokens_long_prompt(self):
        """16k prompt + max_tokens=512 admits more concurrent reqs than a 32768 cap."""
        prompt_len = 16384
        max_tokens = 512
        block_size = self.scheduler.block_size
        blocks_per = (prompt_len + max_tokens + block_size - 1) // block_size
        max_concurrent = 512 // blocks_per

        for i in range(max_concurrent):
            req = _req(f"r{i}", prompt_len, max_tokens=max_tokens)
            self.assertTrue(self.scheduler.can_accept_request(req))
            _prime_kv(self.scheduler, req)
            self.scheduler.running_queue.sync_q.put(req)

        overflow = _req("overflow", prompt_len, max_tokens=max_tokens)
        self.assertFalse(self.scheduler.can_accept_request(overflow))

        inflated_blocks = (prompt_len + 32768 + block_size - 1) // block_size
        self.assertGreater(max_concurrent, 512 // inflated_blocks)

    def test_can_accept_running_decode_remaining_generation(self):
        """Running decode reserves remaining generation budget, not full max_tokens."""
        prompt_len = 8192
        max_tokens = 600
        block_size = self.scheduler.block_size

        decode = _req("decode", prompt_len, max_tokens=max_tokens)
        decode.is_prefill = False
        decode.generated_token_ids = list(range(300))
        _prime_kv(self.scheduler, decode)

        blocks_current = (prompt_len + 300 + block_size - 1) // block_size
        blocks_full_cap = (prompt_len + max_tokens + block_size - 1) // block_size
        blocks_remaining_cap = (
            prompt_len + (max_tokens - 300) + block_size - 1
        ) // block_size
        self.assertEqual(blocks_current, blocks_remaining_cap)
        self.assertLess(blocks_remaining_cap, blocks_full_cap)

        self.assertEqual(self.scheduler._remaining_generation_tokens(decode), 300)
        self.assertEqual(
            self.scheduler._kv_completion_tokens(decode),
            prompt_len + 300,
        )
        reservation = max(blocks_current, blocks_remaining_cap)
        self.assertLess(reservation, max(blocks_current, blocks_full_cap))


class LongPrefillThresholdTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("INFINI_LONG_PREFILL_THRESHOLD", None)
        os.environ.pop("INFINI_PREFILL_CHUNKED", None)
        os.environ.pop("INFINI_PREFILL_CHUNK_SIZE", None)

    def test_defaults_to_chunk_size_when_chunked_prefill_enabled(self):
        os.environ["INFINI_PREFILL_CHUNKED"] = "1"
        os.environ["INFINI_PREFILL_CHUNK_SIZE"] = "8192"
        self.assertEqual(long_prefill_threshold(), 8192)

    def test_explicit_env_overrides_chunked_default(self):
        os.environ["INFINI_PREFILL_CHUNKED"] = "1"
        os.environ["INFINI_PREFILL_CHUNK_SIZE"] = "8192"
        os.environ["INFINI_LONG_PREFILL_THRESHOLD"] = "4096"
        self.assertEqual(long_prefill_threshold(), 4096)

    def test_defaults_to_4096_when_chunked_prefill_disabled(self):
        os.environ.pop("INFINI_PREFILL_CHUNKED", None)
        self.assertEqual(long_prefill_threshold(), 4096)


if __name__ == "__main__":
    unittest.main()
