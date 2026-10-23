"""Unit tests for continuous-batch prefill packing in Scheduler (V1 path)."""

from __future__ import annotations

import os
import unittest

from infinilm.llm.request import InferenceRequest, RequestStatus
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler


def _req(request_id: str, prompt_len: int, *, chunk_size: int = 4096) -> InferenceRequest:
    req = InferenceRequest(
        request_id=request_id,
        prompt_token_ids=list(range(1, prompt_len + 1)),
        sampling_params=SamplingParams(max_tokens=1),
    )
    req.chunk_size = chunk_size
    return req


class SchedulerPackTest(unittest.TestCase):
    def setUp(self):
        os.environ["INFINI_V1_SCHEDULER"] = "1"
        os.environ["INFINI_MAX_NUM_BATCHED_TOKENS"] = "8192"
        os.environ.pop("INFINI_MAX_PREFILL_BATCH", None)
        os.environ.pop("INFINI_SCHEDULE_NO_MIXED", None)
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
        os.environ.pop("INFINI_SCHEDULE_NO_MIXED", None)

    def test_four_4096_chunks_no_two_pack_at_chunk_budget(self):
        """With chunk_size=4096, V1 never packs 2×4096 mid-chunks in one step.

        RUNNING-first means a mid-chunk request continues before new WAITING
        admits; assert the pack gate (one 4096 row) rather than legacy
        waiting-priority four-mid drain.
        """
        chunk_size = 4096
        for i in range(4):
            self.scheduler.add_request(_req(f"r{i}", 8192, chunk_size=chunk_size))

        out = self.scheduler.schedule()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertTrue(out.is_prefill)
        self.assertEqual(len(out.scheduled_requests), 1)
        self.assertEqual(out.total_scheduled_tokens, 4096)
        self.assertFalse(out.rows[0].is_final_prefill_chunk)
        self.assertEqual(self.scheduler.waiting_size(), 3)

        # Advance mid-chunk and requeue as RUNNING; next step finishes it (not 2-pack).
        req = out.scheduled_requests[0]
        req.chunk_prefill_offset += chunk_size
        req.num_computed_tokens = req.chunk_prefill_offset
        self.scheduler.complete_requests(out.scheduled_requests)

        out2 = self.scheduler.schedule()
        self.assertIsNotNone(out2)
        assert out2 is not None
        self.assertEqual(len(out2.scheduled_requests), 1)
        self.assertEqual(out2.total_scheduled_tokens, 4096)
        self.assertTrue(out2.rows[0].is_final_prefill_chunk)
        self.assertEqual(out2.scheduled_requests[0].request_id, req.request_id)

    def test_smaller_chunks_still_multi_pack(self):
        """Under chunk_size=4096 budget, four 2048-token prefills pack two per step."""
        chunk_size = 4096
        for i in range(4):
            self.scheduler.add_request(_req(f"r{i}", 2048, chunk_size=chunk_size))

        out1 = self.scheduler.schedule()
        self.assertIsNotNone(out1)
        assert out1 is not None
        self.assertTrue(out1.is_prefill)
        self.assertEqual(len(out1.scheduled_requests), 2)
        self.assertEqual(out1.total_scheduled_tokens, 4096)

        out2 = self.scheduler.schedule()
        self.assertIsNotNone(out2)
        assert out2 is not None
        self.assertEqual(len(out2.scheduled_requests), 2)
        self.assertEqual(out2.total_scheduled_tokens, 4096)

    def test_reject_mixed_mid_and_final(self):
        chunk_size = 4096
        mid = _req("mid", 8192, chunk_size=chunk_size)
        mid.chunk_prefill_offset = 0
        mid.num_computed_tokens = 0
        final = _req("final", 4096, chunk_size=chunk_size)
        # Prime both onto waiting; pack gate should admit only one phase per step.
        self.scheduler.add_request(mid)
        self.scheduler.add_request(final)

        out = self.scheduler.schedule()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(len(out.scheduled_requests), 1)

    def test_prefix_cache_incompatible_rejects_pack(self):
        chunk_size = 4096
        a = _req("a", 4096, chunk_size=chunk_size)
        b = _req("b", 4096, chunk_size=chunk_size)
        a.num_computed_tokens = 0
        b.num_computed_tokens = 512
        a.num_local_cached_tokens = 0
        b.num_local_cached_tokens = 512
        a.block_table = [0, 1]
        b.block_table = [2, 3]
        a.slot_mapping = list(range(4096))
        b.slot_mapping = list(range(4096))
        a.status = RequestStatus.WAITING
        b.status = RequestStatus.WAITING
        self.scheduler.waiting_append(a)
        self.scheduler.waiting_append(b)

        out = self.scheduler.schedule()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(len(out.scheduled_requests), 1)

    def test_prefill_pack_respects_capture_width(self):
        """max_batch_size=8 but capture width=4 → at most 4 requests per prefill pack."""
        wide = Scheduler(
            max_batch_size=8,
            num_blocks=512,
            block_size=256,
            max_prefill_batch_size=4,
            enable_prefix_cache=False,
        )
        chunk_size = 512
        for i in range(6):
            wide.add_request(_req(f"r{i}", 2048, chunk_size=chunk_size))

        out = wide.schedule()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertTrue(out.is_prefill)
        self.assertLessEqual(len(out.scheduled_requests), 4)

    def test_no_mixed_defers_prefill_when_decode_present(self):
        """With INFINI_SCHEDULE_NO_MIXED=1, decode-only step; mid-prefill deferred."""
        os.environ["INFINI_SCHEDULE_NO_MIXED"] = "1"
        chunk_size = 512
        prefill_req = _req("prefill", 2048, chunk_size=chunk_size)
        prefill_req.is_prefill = True
        prefill_req.status = RequestStatus.RUNNING
        prefill_req.block_table = [0, 1, 2, 3]
        prefill_req.slot_mapping = list(range(2048))
        prefill_req.num_computed_tokens = 0
        prefill_req.num_local_cached_tokens = 0
        prefill_req.chunk_prefill_offset = 0
        prefill_req.num_blocks = 4

        decode_req = _req("decode", 128, chunk_size=0)
        decode_req.is_prefill = False
        decode_req.status = RequestStatus.RUNNING
        decode_req.generated_token_ids = [99]
        decode_req.block_table, decode_req.slot_mapping, _ = (
            self.scheduler.cache_manager.allocate_blocks(
                decode_req.get_input_tokens(), []
            )
        )
        decode_req.num_blocks = len(decode_req.block_table)
        decode_req.num_local_cached_tokens = decode_req.get_total_length() - 1
        decode_req.num_computed_tokens = decode_req.num_local_cached_tokens

        self.scheduler.running_queue.sync_q.put(prefill_req)
        self.scheduler.running_queue.sync_q.put(decode_req)

        out = self.scheduler.schedule()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertFalse(out.is_prefill)
        self.assertEqual(len(out.scheduled_requests), 1)
        self.assertFalse(out.scheduled_requests[0].is_prefill)
        self.assertEqual(out.scheduled_requests[0].request_id, "decode")
        # Prefill deferred back onto running.
        self.assertEqual(self.scheduler.running_queue.sync_q.qsize(), 1)
        deferred = self.scheduler.running_queue.sync_q.get_nowait()
        self.assertEqual(deferred.request_id, "prefill")


if __name__ == "__main__":
    unittest.main()
