"""Unit tests for RECOMPUTE preemption (INFINI_SCHEDULE_PREEMPT)."""

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
    generated: list[int] | None = None,
    max_tokens: int = 64,
    arrival_time: float = 1.0,
) -> InferenceRequest:
    req = InferenceRequest(
        request_id=request_id,
        prompt_token_ids=list(range(1, prompt_len + 1)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        arrival_time=arrival_time,
    )
    if generated:
        req.generated_token_ids = list(generated)
        req.generated_text = "x" * len(generated)
        req._text_output_offset = len(req.generated_text)
        req.is_prefill = False
        req.status = RequestStatus.RUNNING
    return req


def _prime_kv_all_tokens(scheduler: Scheduler, req: InferenceRequest) -> None:
    tokens = req.get_all_token_ids()
    req.block_table, req.slot_mapping, cached = (
        scheduler.cache_manager.allocate_blocks(tokens, req.block_table)
    )
    req.num_blocks = len(req.block_table)
    req.num_local_cached_tokens = cached
    req.num_computed_tokens = cached
    req.status = RequestStatus.RUNNING
    scheduler.cache_manager.reset_req_blocks()


class SchedulerPreemptTest(unittest.TestCase):
    def setUp(self):
        os.environ["INFINI_V1_SCHEDULER"] = "1"
        os.environ["INFINI_MAX_NUM_BATCHED_TOKENS"] = "8192"
        os.environ["INFINI_SCHEDULE_PREEMPT"] = "1"
        os.environ.pop("INFINI_SCHEDULE_NO_MIXED", None)
        os.environ.pop("INFINI_LONG_PREFILL_THRESHOLD", None)
        # Tiny KV: 3 blocks × 4 = 12 tokens total.
        self.scheduler = Scheduler(
            max_batch_size=4,
            num_blocks=3,
            block_size=4,
            max_prefill_batch_size=4,
            enable_prefix_cache=False,
        )
        self.scheduler.chunk_size = 8

    def tearDown(self):
        os.environ.pop("INFINI_V1_SCHEDULER", None)
        os.environ.pop("INFINI_MAX_NUM_BATCHED_TOKENS", None)
        os.environ.pop("INFINI_SCHEDULE_PREEMPT", None)

    def test_prepare_for_recompute_keeps_stream_state(self):
        req = _req("r", 4, generated=[10, 11, 12], arrival_time=1.0)
        req.block_table = [0, 1]
        req.slot_mapping = [0, 1, 2, 3, 4]
        req.num_local_cached_tokens = 6
        req.num_computed_tokens = 6
        req.num_blocks = 2
        stream_len = req._text_output_offset
        gen_ids = list(req.generated_token_ids)
        gen_text = req.generated_text

        req.prepare_for_recompute(chunk_size=8)

        self.assertEqual(req.status, RequestStatus.WAITING)
        self.assertTrue(req.is_prefill)
        self.assertEqual(list(req.generated_token_ids), gen_ids)
        self.assertEqual(req.generated_text, gen_text)
        self.assertEqual(req._text_output_offset, stream_len)
        self.assertEqual(req.block_table, [])
        self.assertEqual(req.num_cached_tokens, 0)
        self.assertEqual(req.num_computed_tokens, 0)
        self.assertEqual(req.chunk_size, 8)
        self.assertEqual(req.num_preemptions, 1)
        self.assertEqual(
            req.tokens_for_kv_recompute(),
            list(req.prompt_token_ids) + gen_ids,
        )
        self.assertEqual(req.get_prefill_tokens(), req.tokens_for_kv_recompute())
        self.assertEqual(req.prefill_seq_len(), 4 + 3)

    def test_waiting_prepend_ahead_of_new_admits(self):
        a = _req("a", 2, arrival_time=1.0)
        b = _req("b", 2, arrival_time=2.0)
        self.scheduler.waiting_append(a)
        self.scheduler.waiting_prepend(b)
        self.assertEqual(self.scheduler.waiting_popleft().request_id, "b")
        self.assertEqual(self.scheduler.waiting_popleft().request_id, "a")

    def test_decode_append_oom_preempts_newest_victim(self):
        """A (older) needs a new block; B (newer) is RECOMPUTE-preempted."""
        older = _req("older", 4, generated=[99], arrival_time=1.0)
        newer = _req("newer", 4, arrival_time=2.0)
        newer.is_prefill = False
        newer.generated_token_ids = []
        newer.status = RequestStatus.RUNNING

        _prime_kv_all_tokens(self.scheduler, older)
        _prime_kv_all_tokens(self.scheduler, newer)
        self.assertEqual(self.scheduler.cache_manager.get_num_free_blocks(), 0)
        self.assertEqual(len(older.block_table), 2)
        self.assertEqual(len(newer.block_table), 1)

        self.assertEqual(older.get_total_length() % self.scheduler.block_size, 1)

        self.scheduler.running_queue.sync_q.put(older)
        self.scheduler.running_queue.sync_q.put(newer)

        out = self.scheduler.schedule()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertGreater(self.scheduler.num_preemptions_total, 0)
        self.assertEqual(newer.status, RequestStatus.WAITING)
        self.assertEqual(newer.num_preemptions, 1)
        self.assertEqual(newer.block_table, [])
        self.assertEqual(self.scheduler.waiting_size(), 1)
        self.assertEqual(self.scheduler._waiting[0].request_id, "newer")
        scheduled_ids = [r.request_id for r in out.scheduled_requests]
        self.assertIn("older", scheduled_ids)
        self.assertNotIn("newer", scheduled_ids)
        self.assertEqual(
            self.scheduler.get_cache_stats()["num_preemptions_total"],
            self.scheduler.num_preemptions_total,
        )

    def test_preempt_off_defers_without_counter(self):
        os.environ["INFINI_SCHEDULE_PREEMPT"] = "0"
        older = _req("older", 4, generated=[99], arrival_time=1.0)
        newer = _req("newer", 4, arrival_time=2.0)
        newer.is_prefill = False
        newer.status = RequestStatus.RUNNING

        _prime_kv_all_tokens(self.scheduler, older)
        _prime_kv_all_tokens(self.scheduler, newer)
        self.scheduler.running_queue.sync_q.put(older)
        self.scheduler.running_queue.sync_q.put(newer)

        out = self.scheduler.schedule()
        self.assertEqual(self.scheduler.num_preemptions_total, 0)
        self.assertEqual(newer.status, RequestStatus.RUNNING)
        self.assertTrue(newer.block_table)
        self.assertEqual(older.status, RequestStatus.RUNNING)
        self.assertTrue(older.block_table)
        self.assertEqual(self.scheduler.waiting_size(), 0)
        if out is not None:
            scheduled_ids = [r.request_id for r in out.scheduled_requests]
            self.assertNotIn("older", scheduled_ids)

    def test_readmit_allocates_prompt_plus_generated(self):
        victim = _req("v", 4, generated=[7, 8], arrival_time=1.0, max_tokens=4)
        victim.prepare_for_recompute(8)
        self.assertEqual(victim.prefill_seq_len(), 6)
        self.scheduler.waiting_prepend(victim)

        self.assertGreaterEqual(
            self.scheduler.cache_manager.get_num_free_blocks(), 2
        )
        self.assertTrue(self.scheduler.can_accept_request(victim))

        out = self.scheduler.schedule()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(victim.status, RequestStatus.RUNNING)
        self.assertEqual(len(victim.block_table), 2)
        self.assertTrue(out.rows[0].is_prefill_row)
        self.assertEqual(out.rows[0].num_scheduled_tokens, 6)


if __name__ == "__main__":
    unittest.main()
