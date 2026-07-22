"""Unit tests for continuous-batch prefill packing in Scheduler."""

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
        os.environ.pop("INFINI_MAX_PREFILL_BATCH", None)
        self.scheduler = Scheduler(
            max_batch_size=4,
            num_blocks=512,
            block_size=256,
            max_prefill_batch_size=4,
            enable_prefix_cache=False,
        )

    def test_four_4096_chunks_no_two_pack_at_chunk_budget(self):
        """With chunk_size=4096, full mid-chunks schedule one per step (2×4096 pack forbidden)."""
        chunk_size = 4096
        for i in range(4):
            req = _req(f"r{i}", 8192, chunk_size=chunk_size)
            self.scheduler.add_request(req)

        mid_lens = []
        for _ in range(4):
            out = self.scheduler.schedule()
            self.assertIsNotNone(out)
            assert out is not None
            self.assertTrue(out.is_prefill)
            self.assertEqual(len(out.scheduled_requests), 1)
            req = out.scheduled_requests[0]
            self.assertTrue(req.is_chunking() and not req.chunk_is_last())
            total_q = Scheduler._prefill_compute_len(req)
            mid_lens.append(total_q)
            self.assertNotEqual(total_q, 8192)
            self.assertEqual(total_q, 4096)
            req.chunk_prefill_offset += chunk_size
            self.scheduler.requeue_chunking(req)

        self.assertEqual(mid_lens, [4096, 4096, 4096, 4096])

        # Final chunks also one-at-a-time under the same budget.
        for _ in range(4):
            out = self.scheduler.schedule()
            self.assertIsNotNone(out)
            assert out is not None
            self.assertEqual(len(out.scheduled_requests), 1)
            self.assertTrue(out.scheduled_requests[0].chunk_is_last())
            total_q = sum(
                Scheduler._prefill_compute_len(r) for r in out.scheduled_requests
            )
            self.assertNotEqual(total_q, 8192)

    def test_smaller_chunks_still_multi_pack(self):
        """Under chunk_size=4096 budget, four 2048-token prefills pack two per step (total_q=4096)."""
        chunk_size = 4096
        for i in range(4):
            req = _req(f"r{i}", 2048, chunk_size=chunk_size)
            self.scheduler.add_request(req)

        out1 = self.scheduler.schedule()
        self.assertIsNotNone(out1)
        assert out1 is not None
        self.assertTrue(out1.is_prefill)
        self.assertEqual(len(out1.scheduled_requests), 2)
        total_q1 = sum(
            Scheduler._prefill_compute_len(r) for r in out1.scheduled_requests
        )
        self.assertEqual(total_q1, 4096)

        out2 = self.scheduler.schedule()
        self.assertIsNotNone(out2)
        assert out2 is not None
        self.assertEqual(len(out2.scheduled_requests), 2)
        total_q2 = sum(
            Scheduler._prefill_compute_len(r) for r in out2.scheduled_requests
        )
        self.assertEqual(total_q2, 4096)

    def test_reject_mixed_mid_and_final(self):
        chunk_size = 4096
        mid = _req("mid", 8192, chunk_size=chunk_size)
        mid.chunk_prefill_offset = 0
        final = _req("final", 4096, chunk_size=chunk_size)
        self.scheduler.chunking_queue.sync_q.put(mid)
        self.scheduler.chunking_queue.sync_q.put(final)

        out = self.scheduler.schedule()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(len(out.scheduled_requests), 1)

    def test_prefix_cache_incompatible_rejects_pack(self):
        chunk_size = 4096
        a = _req("a", 4096, chunk_size=chunk_size)
        b = _req("b", 4096, chunk_size=chunk_size)
        a.num_cached_tokens = 0
        b.num_cached_tokens = 512
        a.block_table = [0, 1]
        b.block_table = [2, 3]
        a.slot_mapping = list(range(4096))
        b.slot_mapping = list(range(4096))
        a.status = RequestStatus.RUNNING
        b.status = RequestStatus.RUNNING
        self.scheduler.waiting_queue.sync_q.put(a)
        self.scheduler.waiting_queue.sync_q.put(b)

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

    def test_decode_skips_prefill_requests(self):
        """Requests still marked is_prefill must not enter decode batches."""
        chunk_size = 512
        prefill_req = _req("prefill", 2048, chunk_size=chunk_size)
        prefill_req.is_prefill = True
        prefill_req.status = RequestStatus.RUNNING
        prefill_req.block_table = [0, 1, 2, 3]
        prefill_req.slot_mapping = [0]
        prefill_req.num_cached_tokens = 511
        prefill_req.num_blocks = 4

        decode_req = _req("decode", 128, chunk_size=0)
        decode_req.is_prefill = False
        decode_req.status = RequestStatus.RUNNING
        decode_req.generated_token_ids = [99]
        decode_req.block_table = [4, 5]
        decode_req.slot_mapping = [0]
        decode_req.num_cached_tokens = 128
        decode_req.num_blocks = 2

        self.scheduler.running_queue.sync_q.put(prefill_req)
        self.scheduler.running_queue.sync_q.put(decode_req)

        out = self.scheduler.schedule()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertFalse(out.is_prefill)
        self.assertEqual(len(out.scheduled_requests), 1)
        self.assertFalse(out.scheduled_requests[0].is_prefill)
        self.assertEqual(out.scheduled_requests[0].request_id, "decode")
        self.assertEqual(self.scheduler.chunking_queue.sync_q.qsize(), 1)


if __name__ == "__main__":
    unittest.main()
