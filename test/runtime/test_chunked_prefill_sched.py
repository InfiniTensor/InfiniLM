"""Chunked prefill scheduling behavior (no GPU required)."""

import unittest

from infinilm.llm.request import InferenceRequest, SamplingParams
from infinilm.llm.scheduler import Scheduler


def make_scheduler(**kwargs):
    defaults = dict(
        max_batch_size=4,
        num_blocks=64,
        block_size=16,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
    )
    defaults.update(kwargs)
    return Scheduler(**defaults)


def make_request(req_id, prompt_len, max_tokens=8, prefix_caching=False):
    req = InferenceRequest(
        request_id=req_id,
        prompt_token_ids=list(range(1000, 1000 + prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )
    req.initialize_block_hashes(16, prefix_caching)
    return req


def drive_prefill(s, req):
    """Run schedule/complete cycles until the last chunk is scheduled.

    Mimics the engine completion path for pending chunks: advance
    the computed boundary by the scheduled chunk before re-queuing.
    The final chunk is not completed here; returns (output, request).
    """
    while True:
        out = s.schedule()
        assert out is not None and out.is_prefill
        r = out.scheduled_requests[0]
        assert r.num_scheduled_tokens is not None and r.num_scheduled_tokens > 0
        if not r.chunked_prefill_pending:
            return out, r
        r.num_computed_tokens = r.num_local_cached_tokens + r.num_scheduled_tokens
        s.commit_computed_tokens(r, r.num_computed_tokens)
        s.complete_requests([r])


class ChunkedPrefillSchedTest(unittest.TestCase):
    def test_disabled_by_default_full_prefill(self):
        s = make_scheduler()
        req = make_request("r1", 300)
        s.add_request(req)
        out = s.schedule()
        self.assertIsNotNone(out)
        self.assertTrue(out.is_prefill)
        self.assertEqual(len(out.scheduled_requests), 1)
        r = out.scheduled_requests[0]
        self.assertFalse(r.chunked_prefill_pending)
        self.assertIsNone(r.num_scheduled_tokens)
        self.assertEqual(len(r.slot_mapping), 300)
        self.assertEqual(len(r.block_table), (300 + 15) // 16)

    def test_chunked_split_and_resume(self):
        s = make_scheduler(chunked_prefill_size=64)
        req = make_request("r1", 300)
        s.add_request(req)
        total_scheduled = 0
        steps = 0
        seen_blocks = None
        while True:
            out = s.schedule()
            self.assertIsNotNone(out, "scheduler returned None mid-chunk")
            self.assertTrue(
                out.is_prefill, f"decode scheduled while chunk pending (step {steps})"
            )
            self.assertEqual(len(out.scheduled_requests), 1)
            r = out.scheduled_requests[0]
            self.assertIsNotNone(r.num_scheduled_tokens)
            self.assertLessEqual(r.num_scheduled_tokens, 64)
            # slots cover exactly [num_computed, num_computed+scheduled)
            start = r.num_computed_tokens
            expected_slots = [
                r.block_table[(start + i) // 16] * 16 + (start + i) % 16
                for i in range(r.num_scheduled_tokens)
            ]
            self.assertEqual(r.slot_mapping, expected_slots)
            # block table only grows; earlier blocks are preserved
            if seen_blocks is not None:
                self.assertEqual(r.block_table[: len(seen_blocks)], seen_blocks)
            seen_blocks = list(r.block_table)
            total_scheduled += r.num_scheduled_tokens
            steps += 1
            if r.chunked_prefill_pending:
                self.assertLess(total_scheduled, 300)
                r.num_computed_tokens = (
                    r.num_local_cached_tokens + r.num_scheduled_tokens
                )
                s.complete_requests([r])
                continue
            self.assertEqual(total_scheduled, 300)
            self.assertEqual(r.num_computed_tokens + r.num_scheduled_tokens, 300)
            break
        self.assertEqual(steps, 5)  # 64+64+64+64+44
        # final chunk lands in the same state a full prefill would have
        self.assertEqual(len(r.block_table), (300 + 15) // 16)

    def test_chunked_completion_allows_decode(self):
        s = make_scheduler(chunked_prefill_size=64)
        req = make_request("r1", 300, max_tokens=4)
        s.add_request(req)
        out, r = drive_prefill(s, req)
        self.assertFalse(r.chunked_prefill_pending)
        # simulate engine: prompt fully computed, sample appended, decode next
        r.num_computed_tokens = r.get_prompt_length()
        r.append_generated_token_id(42)
        s.complete_requests([r])
        # next schedule must be a decode step for the request
        out = s.schedule()
        self.assertIsNotNone(out)
        self.assertFalse(out.is_prefill)
        self.assertEqual(len(out.scheduled_requests), 1)
        r = out.scheduled_requests[0]
        self.assertFalse(r.chunked_prefill_pending)
        # decode slot targets total_length = 300 prompt + 1 generated
        self.assertEqual(r.slot_mapping, [r.block_table[300 // 16] * 16 + 300 % 16])

    def test_chunk_budget_interleaves_requests(self):
        s = make_scheduler(chunked_prefill_size=64, max_num_batched_tokens=128)
        req_a = make_request("a", 200)
        req_b = make_request("b", 200)
        s.add_request(req_a)
        s.add_request(req_b)
        out = s.schedule()
        # step 1: a chunk of 64 for a, then 64 for b (two fit in 128 budget)
        self.assertTrue(out.is_prefill)
        self.assertEqual(len(out.scheduled_requests), 2)
        a, b = out.scheduled_requests
        self.assertEqual(a.num_scheduled_tokens, 64)
        self.assertTrue(a.chunked_prefill_pending)
        self.assertEqual(b.num_scheduled_tokens, 64)
        self.assertTrue(b.chunked_prefill_pending)
        for r in (a, b):
            r.num_computed_tokens = r.num_local_cached_tokens + r.num_scheduled_tokens
            s.complete_requests([r])

    def test_chunked_mixed_with_full_prefill(self):
        s = make_scheduler(chunked_prefill_size=64, max_num_batched_tokens=8192)
        small = make_request("small", 50)
        big = make_request("big", 300)
        s.add_request(small)
        s.add_request(big)
        out = s.schedule()
        self.assertTrue(out.is_prefill)
        reqs = out.scheduled_requests
        self.assertEqual(len(reqs), 2)
        first, second = reqs
        # small request scheduled whole (50 <= 64), big chunked to 64
        if first is small:
            self.assertFalse(first.chunked_prefill_pending)
            self.assertIs(second, big)
            self.assertEqual(second.num_scheduled_tokens, 64)
        else:
            self.assertIs(first, big)
            self.assertEqual(first.num_scheduled_tokens, 64)
            self.assertIs(second, small)
            self.assertFalse(second.chunked_prefill_pending)

    def test_abort_during_chunks(self):
        s = make_scheduler(chunked_prefill_size=64)
        req = make_request("r1", 300)
        s.add_request(req)
        out = s.schedule()
        r = out.scheduled_requests[0]
        self.assertTrue(r.chunked_prefill_pending)
        # abort mid-chunk: completion frees the blocks, request leaves queues
        r.mark_canceled()
        s.complete_requests([r])
        out = s.schedule()
        if out is not None:
            self.assertTrue(r not in out.scheduled_requests or r.is_finished())

    def test_misaligned_token_budget_clamped_to_block_boundary(self):
        # A token budget that is not a multiple of block_size must not
        # produce a misaligned chunk: allocate_slots raises if
        # num_computed_tokens precedes the cached block boundary on resume.
        s = make_scheduler(
            chunked_prefill_size=128, max_num_batched_tokens=100, block_size=16
        )
        self.assertEqual(s.chunked_prefill_size, 96)  # min(128, 100) floored to 96
        req = make_request("r1", 300)
        s.add_request(req)
        total_scheduled = 0
        while True:
            out = s.schedule()
            self.assertIsNotNone(out)
            self.assertTrue(out.is_prefill)
            r = out.scheduled_requests[0]
            start = r.num_computed_tokens
            expected_slots = [
                r.block_table[(start + i) // 16] * 16 + (start + i) % 16
                for i in range(r.num_scheduled_tokens)
            ]
            self.assertEqual(r.slot_mapping, expected_slots)
            total_scheduled += r.num_scheduled_tokens
            if r.chunked_prefill_pending:
                r.num_computed_tokens = (
                    r.num_local_cached_tokens + r.num_scheduled_tokens
                )
                s.complete_requests([r])
                continue
            break
        self.assertEqual(total_scheduled, 300)

    def test_misaligned_budget_below_block_size_rejected(self):
        with self.assertRaises(ValueError):
            make_scheduler(
                chunked_prefill_size=128,
                max_num_batched_tokens=8,
                block_size=16,
            )

    def test_prefix_cache_hit_then_chunked_resume(self):
        # Engine default is enable_prefix_caching=True. A prompt that
        # hits a cached prefix AND still needs chunking must resume from
        # the cache boundary: num_computed_tokens starts above zero and
        # the first chunk schedules only (cache_hit .. +chunk) tokens.
        s = make_scheduler(chunked_prefill_size=64, enable_prefix_caching=True)
        shared = list(range(1000, 1000 + 80))  # 80 tokens = 5 full blocks
        req_a = InferenceRequest(
            request_id="a",
            prompt_token_ids=shared + list(range(2000, 2000 + 220)),
            sampling_params=SamplingParams(max_tokens=8),
        )
        req_a.initialize_block_hashes(16, True)
        s.add_request(req_a)
        out, ra = drive_prefill(s, req_a)
        self.assertFalse(ra.chunked_prefill_pending)

        # Second request shares the first 80 tokens' prefix.
        req_b = InferenceRequest(
            request_id="b",
            prompt_token_ids=shared + list(range(3000, 3000 + 220)),
            sampling_params=SamplingParams(max_tokens=8),
        )
        req_b.initialize_block_hashes(16, True)
        s.add_request(req_b)
        out = s.schedule()
        rb = out.scheduled_requests[0]
        # cache hit covers the shared prefix; the chunk starts after it
        self.assertEqual(rb.num_computed_tokens, 80)
        self.assertEqual(rb.num_local_cached_tokens, 80)
        self.assertEqual(rb.num_scheduled_tokens, 64)
        self.assertTrue(rb.chunked_prefill_pending)
        # slot mapping resumes exactly at the cache boundary
        expected_slots = [
            rb.block_table[(80 + i) // 16] * 16 + (80 + i) % 16 for i in range(64)
        ]
        self.assertEqual(rb.slot_mapping, expected_slots)
        # Continue chunking. The final chunk (pending=False)
        # leaves the computed boundary at the chunk start; the
        # engine advances it in the normal completion path.
        total_b = rb.num_computed_tokens + rb.num_scheduled_tokens
        while True:
            rb.num_computed_tokens = (
                rb.num_local_cached_tokens + rb.num_scheduled_tokens
            )
            s.commit_computed_tokens(rb, rb.num_computed_tokens)
            s.complete_requests([rb])
            out = s.schedule()
            self.assertIsNotNone(out)
            self.assertTrue(out.is_prefill)
            rb = out.scheduled_requests[0]
            total_b += rb.num_scheduled_tokens
            if not rb.chunked_prefill_pending:
                break
        self.assertEqual(rb.num_computed_tokens + rb.num_scheduled_tokens, 300)
        self.assertEqual(total_b, 300)


if __name__ == "__main__":
    unittest.main()
