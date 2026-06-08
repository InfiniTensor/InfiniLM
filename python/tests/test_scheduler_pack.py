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

    def test_four_4096_chunks_pack_two_per_step(self):
        """4×4096-chunk @8192 prompts → 2 steps of 2-pack mid-chunks."""
        chunk_size = 4096
        for i in range(4):
            req = _req(f"r{i}", 8192, chunk_size=chunk_size)
            self.scheduler.add_request(req)

        out1 = self.scheduler.schedule()
        self.assertIsNotNone(out1)
        assert out1 is not None
        self.assertTrue(out1.is_prefill)
        self.assertEqual(len(out1.scheduled_requests), 2)
        self.assertTrue(all(r.is_chunking() and not r.chunk_is_last() for r in out1.scheduled_requests))

        out2 = self.scheduler.schedule()
        self.assertIsNotNone(out2)
        assert out2 is not None
        self.assertEqual(len(out2.scheduled_requests), 2)

        # Simulate mid-chunk advance + requeue
        for req in out1.scheduled_requests:
            req.chunk_prefill_offset += chunk_size
            self.scheduler.requeue_chunking(req)
        for req in out2.scheduled_requests:
            req.chunk_prefill_offset += chunk_size
            self.scheduler.requeue_chunking(req)

        out3 = self.scheduler.schedule()
        self.assertIsNotNone(out3)
        assert out3 is not None
        self.assertEqual(len(out3.scheduled_requests), 2)
        self.assertTrue(all(r.chunk_is_last() for r in out3.scheduled_requests))

        out4 = self.scheduler.schedule()
        self.assertIsNotNone(out4)
        assert out4 is not None
        self.assertEqual(len(out4.scheduled_requests), 2)
        self.assertTrue(all(r.chunk_is_last() for r in out4.scheduled_requests))

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


if __name__ == "__main__":
    unittest.main()
