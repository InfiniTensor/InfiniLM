"""Chunked prefill scheduling behavior (no GPU required)."""

import unittest

from infinilm.llm.request import InferenceRequest, RequestStatus, SamplingParams
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


def run_scheduler(s, reqs, phases=None, limit=300):
    """Drive the engine loop until every request in reqs
    finishes. Mimics the engine: a chunked prefill advances
    the computed boundary by the scheduled chunk; a final
    prefill or decode step samples one token; finished
    requests are marked FINISHED. Returns False if the
    scheduler returns None while requests are unfinished
    (stall/deadlock). Records "P:ids"/"D:ids" per batch
    into `phases` if given.
    """
    for _ in range(limit):
        out = s.schedule()
        if out is None:
            return False
        if phases is not None:
            phases.append(
                ("P" if out.is_prefill else "D")
                + ":"
                + ",".join(r.request_id for r in out.scheduled_requests)
            )
        for r in out.scheduled_requests:
            if out.is_prefill:
                if r.num_scheduled_tokens is not None:
                    r.num_computed_tokens = (
                        r.num_local_cached_tokens + r.num_scheduled_tokens
                    )
                    s.commit_computed_tokens(r, r.num_computed_tokens)
                if not r.chunked_prefill_pending:
                    r.num_computed_tokens = r.get_prompt_length()
                    r.append_generated_token_id(42)
            else:
                r.append_generated_token_id(42)
            if r.get_num_generated_tokens() >= r.sampling_params.max_tokens:
                r.status = RequestStatus.FINISHED
            s.complete_requests([r])
        if all(r.is_finished() for r in reqs):
            return True
    return False


def schedule_one_chunk(s):
    """Schedule one pending chunk and re-queue the request
    the way the engine would; returns the scheduled request."""
    out = s.schedule()
    r = out.scheduled_requests[0]
    assert r.chunked_prefill_pending
    r.num_computed_tokens = r.num_local_cached_tokens + r.num_scheduled_tokens
    s.commit_computed_tokens(r, r.num_computed_tokens)
    s.complete_requests([r])
    return r


def blocks_held(s):
    """Blocks referenced by live requests (freed blocks stay
    in used_block_ids as reusable prefix-cache entries)."""
    return sum(
        1
        for bid in s.cache_manager.used_block_ids
        if s.cache_manager.blocks[bid].ref_count > 0
    )


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
        out = s.schedule()
        r = out.scheduled_requests[0]
        while r.chunked_prefill_pending:
            r.num_computed_tokens = r.num_local_cached_tokens + r.num_scheduled_tokens
            s.commit_computed_tokens(r, r.num_computed_tokens)
            s.complete_requests([r])
            out = s.schedule()
            r = out.scheduled_requests[0]
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
        out = s.schedule()
        ra = out.scheduled_requests[0]
        while ra.chunked_prefill_pending:
            ra.num_computed_tokens = (
                ra.num_local_cached_tokens + ra.num_scheduled_tokens
            )
            s.commit_computed_tokens(ra, ra.num_computed_tokens)
            s.complete_requests([ra])
            out = s.schedule()
            ra = out.scheduled_requests[0]
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
        while rb.chunked_prefill_pending:
            rb.num_computed_tokens = (
                rb.num_local_cached_tokens + rb.num_scheduled_tokens
            )
            s.commit_computed_tokens(rb, rb.num_computed_tokens)
            s.complete_requests([rb])
            out = s.schedule()
            self.assertTrue(out.is_prefill)
            rb = out.scheduled_requests[0]
            total_b += rb.num_scheduled_tokens
        self.assertEqual(rb.num_computed_tokens + rb.num_scheduled_tokens, 300)
        self.assertEqual(total_b, 300)


class ChunkedPrefillDeadlockTest(unittest.TestCase):
    """P1 regressions: multiple chunked requests can
    permanently exhaust the KV cache (reviewer's repro),
    and chunked prefill can starve decode work."""

    REPRO = dict(
        num_blocks=12,
        block_size=16,
        chunked_prefill_size=64,
        max_num_batched_tokens=8192,
    )

    def test_two_chunked_requests_cannot_exhaust_cache(self):
        # Reviewer's deterministic repro: 12 blocks; A(144+16)
        # and B(112+16) both admitted mid-chunk held all 12
        # blocks and neither could resume. With the fix both
        # complete with zero block leaks.
        s = make_scheduler(**self.REPRO)
        a, b = make_request("a", 144, 16), make_request("b", 112, 16)
        s.add_request(a)
        s.add_request(b)
        self.assertTrue(run_scheduler(s, [a, b]))
        self.assertEqual(a.get_num_generated_tokens(), 16)
        self.assertEqual(b.get_num_generated_tokens(), 16)
        self.assertEqual(blocks_held(s), 0)

    def test_admission_counts_waiting_chunked_blocks(self):
        # A holds 4 blocks pending and needs ceil((144+16)/16)-4
        # = 6 more; B needs ceil((112+16)/16) = 8. 4+6+8 = 18 > 12,
        # so B must be deferred until A completes.
        s = make_scheduler(**self.REPRO)
        a, b = make_request("a", 144, 16), make_request("b", 112, 16)
        s.add_request(a)
        ra = schedule_one_chunk(s)
        self.assertEqual(len(ra.block_table), 4)
        s.add_request(b)
        self.assertFalse(s.can_accept_request(b, 0))
        self.assertTrue(run_scheduler(s, [a, b]))

    def test_admission_boundary_exact_fit_allowed(self):
        # Same shape on an 18-block cache: 4+6+8=18 fits
        # exactly — no over-reserving, B is admitted.
        s = make_scheduler(
            num_blocks=18,
            block_size=16,
            chunked_prefill_size=64,
            max_num_batched_tokens=8192,
        )
        a, b = make_request("a", 144, 16), make_request("b", 112, 16)
        s.add_request(a)
        held = len(schedule_one_chunk(s).block_table)
        self.assertEqual((held, (144 + 16 + 15) // 16 - held), (4, 6))
        s.add_request(b)
        self.assertTrue(s.can_accept_request(b, 0))
        self.assertTrue(run_scheduler(s, [a, b]))

    def test_cancel_chunked_pending_releases_and_recovers(self):
        # Cancelling a mid-chunk request frees its blocks;
        # the other chunked request must then complete.
        s = make_scheduler(**self.REPRO)
        a, b = make_request("a", 144, 16), make_request("b", 112, 16)
        s.add_request(a)
        s.add_request(b)
        schedule_one_chunk(s)
        out = s.schedule()
        for r in out.scheduled_requests:
            if r.request_id == "a":
                r.mark_canceled()
            s.complete_requests([r])
        self.assertTrue(run_scheduler(s, [b]))
        self.assertTrue(b.is_finished())

    def test_prefill_starvation_of_decode(self):
        # While short has decodes pending, no two prefill
        # batches may run back to back (the review's linear
        # delay regression). big's trailing chunks after
        # short finishes are exempt.
        s = make_scheduler(chunked_prefill_size=64)
        short = make_request("short", 10, max_tokens=4)
        s.add_request(short)
        out = s.schedule()
        r = out.scheduled_requests[0]
        self.assertIs(r, short)
        r.num_computed_tokens = r.get_prompt_length()
        r.append_generated_token_id(42)
        s.complete_requests([r])
        out = s.schedule()  # short's first decode
        self.assertFalse(out.is_prefill)
        s.complete_requests(out.scheduled_requests)

        big = make_request("big", 320, max_tokens=4)
        s.add_request(big)
        phases = []
        self.assertTrue(run_scheduler(s, [short, big], phases=phases))
        last_short_decode = max(i for i, p in enumerate(phases) if ":short" in p)
        runs = "".join(p[0] for p in phases[: last_short_decode + 1])
        self.assertNotIn("PP", runs)

    def test_decode_priority_disabled_without_chunking(self):
        # Default path (chunked off) keeps the original
        # waiting-first behavior: a newly added prefill runs
        # before an existing request decodes.
        s = make_scheduler(num_blocks=64, block_size=16)
        short = make_request("short", 10, max_tokens=1)
        s.add_request(short)
        out = s.schedule()
        r = out.scheduled_requests[0]
        r.num_computed_tokens = r.get_prompt_length()
        r.append_generated_token_id(42)
        r.status = RequestStatus.FINISHED
        s.complete_requests([r])
        # running queue now empty: the new prefill runs first
        big = make_request("big", 300, max_tokens=4)
        s.add_request(big)
        out = s.schedule()
        self.assertTrue(out.is_prefill)
        self.assertIs(out.scheduled_requests[0], big)

    def test_prefix_cache_three_chunked_requests_no_deadlock(self):
        # Tight cache with prefix sharing: three chunked
        # requests sharing an 80-token prefix must all
        # complete; prefix-cache hits only reduce use.
        s = make_scheduler(
            num_blocks=40,
            chunked_prefill_size=64,
            enable_prefix_caching=True,
        )
        shared = list(range(1000, 1080))
        reqs = [
            InferenceRequest(
                request_id=rid,
                prompt_token_ids=shared + list(range(seed, seed + plen)),
                sampling_params=SamplingParams(max_tokens=8),
            )
            for rid, seed, plen in (
                ("a", 2000, 140),
                ("b", 3000, 140),
                ("c", 4000, 220),
            )
        ]
        for r in reqs:
            r.initialize_block_hashes(16, True)
            s.add_request(r)
        self.assertTrue(run_scheduler(s, reqs))


class ChunkedPrefillBranchMatrixTest(unittest.TestCase):
    """Branch coverage for the chunked-prefill interactions
    beyond the two P1 fixes."""

    def test_disabled_branch_flag_stays_false(self):
        # chunked off: the waiting-first behavior is intact.
        # (_last_batch_was_prefill may stay True after a prefill
        # batch, but skip_waiting's first conjunct
        # (chunked_prefill_size is not None) short-circuits it,
        # so the flag is never read on this path.)
        s = make_scheduler()  # chunked_prefill_size=None
        short = make_request("s", 10, max_tokens=2)
        s.add_request(short)
        out = s.schedule()
        r = out.scheduled_requests[0]
        self.assertIsNone(r.num_scheduled_tokens)  # never chunked
        r.num_computed_tokens = r.get_prompt_length()
        r.append_generated_token_id(42)
        s.complete_requests([r])
        out = s.schedule()  # decode even though prefill just ran
        self.assertFalse(out.is_prefill)
        s.complete_requests(out.scheduled_requests)
        out = s.schedule()  # second decode (max_tokens=2)
        r2 = out.scheduled_requests[0]
        r2.append_generated_token_id(42)
        r2.status = RequestStatus.FINISHED
        s.complete_requests([r2])
        self.assertIsNone(s.schedule())  # nothing left

    def test_flag_resets_on_none_return(self):
        # An all-defer round returns None and resets the flag,
        # so a later decode batch does not permanently
        # suppress waiting.
        s = make_scheduler(num_blocks=16, chunked_prefill_size=64)
        s.add_request(make_request("a", 400))  # 26 blocks > 16
        for _ in range(4):
            self.assertIsNone(s.schedule())
            self.assertFalse(s._last_batch_was_prefill)

    def test_mamba_cache_chunked_resume(self):
        # has_mamba_cache + chunked: the resume path passes
        # num_local_cached_tokens > 0, which the processors'
        # `resumed` branch reads for recurrent-state reuse.
        s = make_scheduler(
            has_mamba_cache=True,
            num_mamba_cache_blocks=8,
            chunked_prefill_size=64,
        )
        req = make_request("m", 300, max_tokens=4)
        s.add_request(req)
        saw_resumed = False
        for _ in range(10):
            out = s.schedule()
            self.assertTrue(out.is_prefill)
            r = out.scheduled_requests[0]
            if r.chunked_prefill_pending:
                if r.num_computed_tokens > 0:
                    saw_resumed = True
                    self.assertGreater(r.num_local_cached_tokens, 0)
                    self.assertIsNotNone(r.mamba_cache_index)
                r.num_computed_tokens = (
                    r.num_local_cached_tokens + r.num_scheduled_tokens
                )
                s.commit_computed_tokens(r, r.num_computed_tokens)
                s.complete_requests([r])
            else:
                r.num_computed_tokens = r.get_prompt_length()
                r.mamba_cache_index = 1
                r.append_generated_token_id(42)
                r.status = RequestStatus.FINISHED
                s.complete_requests([r])
                break
        self.assertTrue(saw_resumed)

    def test_connector_disabled_chunked_flag(self):
        # chunked_prefill_size is force-disabled with a
        # connector (constructor warning), so the
        # skip_waiting branch can never fire there.
        class FakeConnector:
            pass

        with self.assertLogs("infinilm.llm.scheduler", level="WARNING"):
            s = make_scheduler(connector=FakeConnector(), chunked_prefill_size=64)
        self.assertIsNone(s.chunked_prefill_size)

    def test_decode_batch_unlimited_by_token_budget(self):
        # Decode batches bypass max_num_batched_tokens (only
        # max_batch_size applies) — unchanged from baseline,
        # but must hold with chunked on.
        s = make_scheduler(
            num_blocks=64, chunked_prefill_size=64, max_num_batched_tokens=64
        )
        reqs = [make_request(f"d{i}", 10, max_tokens=4) for i in range(4)]
        for r in reqs:
            s.add_request(r)
        out = s.schedule()  # four prefills of 10 tokens fit budget 64
        self.assertTrue(out.is_prefill)
        self.assertEqual(len(out.scheduled_requests), 4)
        for r in out.scheduled_requests:
            r.num_computed_tokens = r.get_prompt_length()
            r.append_generated_token_id(42)
            s.complete_requests([r])
        out = s.schedule()  # one decode batch carries all four
        self.assertFalse(out.is_prefill)
        self.assertEqual(len(out.scheduled_requests), 4)

    def test_defer_round_robin_no_starvation(self):
        # defer-continue rotates the waiting queue
        # (round-robin instead of strict FIFO): with two
        # short requests and two oversized ones, all must
        # finish as blocks free up — no starvation.
        s = make_scheduler(
            num_blocks=32, chunked_prefill_size=64, max_num_batched_tokens=4096
        )
        reqs = [
            make_request(rid, plen)
            for rid, plen in (("c1", 100), ("c2", 100), ("a", 400), ("b", 400))
        ]
        for r in reqs:
            s.add_request(r)
        phases = []
        self.assertTrue(run_scheduler(s, reqs, phases=phases))
        ids = {rid for p in phases for rid in p[2:].split(",")}
        self.assertEqual(ids, {"c1", "c2", "a", "b"})

    def test_waiting_scan_frees_blocked_chunked_head(self):
        # Three oversized chunked requests admitted earlier
        # can each still resume as blocks free up — the
        # waiting-scan reservation must not permanently
        # reject any of them.
        s = make_scheduler(num_blocks=40, chunked_prefill_size=64)
        reqs = [make_request(f"r{i}", 300, max_tokens=8) for i in range(3)]
        for r in reqs:
            s.add_request(r)
        self.assertTrue(run_scheduler(s, reqs))
        self.assertEqual(blocks_held(s), 0)

    def test_non_chunked_low_footprint_prefills_before_decode_slot(self):
        s = make_scheduler(max_batch_size=4, num_blocks=64)
        reqs = [make_request(f"r{i}", 8, max_tokens=8) for i in range(5)]
        for r in reqs:
            s.add_request(r)
        phases = []
        self.assertTrue(run_scheduler(s, reqs, phases=phases))
        self.assertTrue(phases[0].startswith("P:r0,r1,r2,r3"))
        self.assertTrue(phases[1].startswith("P:r4"))
        self.assertEqual(s._paused_running_requests, [])

    def test_non_chunked_long_output_keeps_running_cap(self):
        s = make_scheduler(max_batch_size=4, num_blocks=64)
        reqs = [make_request(f"r{i}", 16, max_tokens=32) for i in range(5)]
        for r in reqs:
            s.add_request(r)
        phases = []
        self.assertTrue(run_scheduler(s, reqs, phases=phases))
        self.assertTrue(phases[0].startswith("P:r0,r1,r2,r3"))
        self.assertTrue(phases[1].startswith("D:r0,r1,r2,r3"))
        late_prefills = [
            index for index, phase in enumerate(phases) if phase.startswith("P:r4")
        ]
        self.assertEqual(late_prefills, [reqs[0].sampling_params.max_tokens])

    def test_non_chunked_prompt_heavy_keeps_running_cap(self):
        s = make_scheduler(max_batch_size=16, num_blocks=128)
        reqs = [make_request(f"r{i}", 64, max_tokens=8) for i in range(17)]
        for r in reqs:
            s.add_request(r)
        phases = []
        self.assertTrue(run_scheduler(s, reqs, phases=phases))
        self.assertTrue(
            phases[0].startswith("P:" + ",".join(f"r{i}" for i in range(16)))
        )
        self.assertTrue(
            phases[1].startswith("D:" + ",".join(f"r{i}" for i in range(16)))
        )
        late_prefills = [
            index for index, phase in enumerate(phases) if phase.startswith("P:r16")
        ]
        self.assertEqual(late_prefills, [reqs[0].sampling_params.max_tokens])


if __name__ == "__main__":
    unittest.main()
