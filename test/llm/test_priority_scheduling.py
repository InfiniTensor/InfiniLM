import unittest
from unittest.mock import patch

import janus
from infinilm.llm.priority_scheduling import (
    drain_priority_ordered,
    mark_request_enqueued,
)
from infinilm.llm.request import InferenceRequest, RequestStatus
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler
from infinilm.llm.static_scheduler import StaticScheduler


class FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_request(request_id: str, priority: int = 0) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=1),
        priority=priority,
    )


class FakeConnector:
    def get_num_new_matched_tokens(self, request, num_local_computed_tokens):
        return 1, True

    def update_state_after_alloc(
        self, request, block_table, num_external_computed_tokens, block_size
    ):
        pass

    def build_connector_meta(self):
        return None


class PriorityOrderingTest(unittest.TestCase):
    def test_default_priority_preserves_fifo_order(self):
        clock = FakeClock()
        requests = [make_request("first"), make_request("second")]
        request_queue = janus.Queue()
        self.addCleanup(request_queue.close)

        for sequence, request in enumerate(requests):
            mark_request_enqueued(request, sequence, clock)
            request_queue.sync_q.put(request)

        ordered = drain_priority_ordered(request_queue.sync_q, 5.0, clock)

        self.assertEqual(
            [request.request_id for request in ordered], ["first", "second"]
        )

    def test_higher_priority_is_admitted_first(self):
        clock = FakeClock()
        low = make_request("low", priority=1)
        high = make_request("high", priority=8)
        request_queue = janus.Queue()
        self.addCleanup(request_queue.close)

        mark_request_enqueued(low, 0, clock)
        request_queue.sync_q.put(low)
        clock.now = 1.0
        mark_request_enqueued(high, 1, clock)
        request_queue.sync_q.put(high)

        ordered = drain_priority_ordered(request_queue.sync_q, 5.0, clock)

        self.assertEqual([request.request_id for request in ordered], ["high", "low"])

    def test_aging_prevents_starvation(self):
        clock = FakeClock()
        old_low = make_request("old-low", priority=0)
        request_queue = janus.Queue()
        self.addCleanup(request_queue.close)

        mark_request_enqueued(old_low, 0, clock)
        request_queue.sync_q.put(old_low)
        clock.now = 55.0
        new_high = make_request("new-high", priority=10)
        mark_request_enqueued(new_high, 1, clock)
        request_queue.sync_q.put(new_high)

        ordered = drain_priority_ordered(request_queue.sync_q, 5.0, clock)

        self.assertEqual(
            [request.request_id for request in ordered], ["old-low", "new-high"]
        )

    def test_requeue_preserves_original_age_and_fifo_position(self):
        clock = FakeClock()
        request = make_request("request")

        mark_request_enqueued(request, 3, clock)
        clock.now = 10.0
        mark_request_enqueued(request, 9, clock)

        self.assertEqual(request.scheduling_enqueue_time, 0.0)
        self.assertEqual(request.scheduling_sequence, 3)

    def test_invalid_priorities_are_rejected(self):
        for priority in (-1, 11, True, 1.5, "1"):
            with self.subTest(priority=priority), self.assertRaises(ValueError):
                make_request("invalid", priority=priority)


class SchedulerPriorityTest(unittest.TestCase):
    def test_invalid_aging_intervals_are_rejected(self):
        for interval in (0, -1, float("nan"), float("inf"), True, "5"):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                StaticScheduler(priority_aging_interval=interval)

    def test_static_scheduler_selects_high_priority_request(self):
        scheduler = StaticScheduler(enable_prefix_caching=False)
        self.addCleanup(scheduler.waiting_queue.close)
        scheduler.add_request(make_request("low", priority=1))
        scheduler.add_request(make_request("high", priority=8))

        output = scheduler.schedule()

        self.assertEqual(output.scheduled_requests[0].request_id, "high")

    def test_canceled_high_priority_request_does_not_block_queue(self):
        scheduler = StaticScheduler(enable_prefix_caching=False)
        self.addCleanup(scheduler.waiting_queue.close)
        scheduler.add_request(make_request("low", priority=1))
        canceled = make_request("canceled", priority=10)
        scheduler.add_request(canceled)
        canceled.mark_canceled()

        output = scheduler.schedule()

        self.assertEqual(output.scheduled_requests[0].request_id, "low")

    def test_timed_out_high_priority_request_does_not_block_queue(self):
        scheduler = StaticScheduler(enable_prefix_caching=False)
        self.addCleanup(scheduler.waiting_queue.close)
        scheduler.add_request(make_request("low", priority=1))
        timed_out = make_request("timed-out", priority=10)
        scheduler.add_request(timed_out)
        timed_out.mark_timeout()

        output = scheduler.schedule()

        self.assertEqual(output.scheduled_requests[0].request_id, "low")

    def test_admission_stats_track_wait_time_and_aging(self):
        clock = FakeClock()
        scheduler = StaticScheduler(
            enable_prefix_caching=False,
            priority_aging_interval=5.0,
            clock=clock,
        )
        self.addCleanup(scheduler.waiting_queue.close)
        scheduler.add_request(make_request("aged", priority=1))
        clock.now = 12.0

        scheduler.schedule()
        stats = scheduler.get_cache_stats()["priority_scheduling"]

        self.assertEqual(stats["admitted_requests"], 1)
        self.assertEqual(stats["aged_admissions"], 1)
        self.assertEqual(stats["admitted_by_priority"], {1: 1})
        self.assertEqual(stats["average_wait_time_seconds"], 12.0)
        self.assertEqual(stats["max_wait_time_seconds"], 12.0)

    def test_paged_scheduler_selects_high_priority_request(self):
        scheduler = Scheduler(
            max_batch_size=1,
            num_blocks=32,
            block_size=4,
            enable_prefix_caching=False,
        )
        self.addCleanup(scheduler.waiting_queue.close)
        self.addCleanup(scheduler.running_queue.close)
        scheduler.add_request(make_request("low", priority=1))
        scheduler.add_request(make_request("high", priority=8))

        output = scheduler.schedule()

        self.assertEqual(output.scheduled_requests[0].request_id, "high")
        self.assertEqual(scheduler.waiting_queue.sync_q.qsize(), 1)

    def test_paged_scheduler_prioritizes_limited_batch(self):
        scheduler = Scheduler(
            max_batch_size=2,
            num_blocks=64,
            block_size=4,
            enable_prefix_caching=False,
        )
        self.addCleanup(scheduler.waiting_queue.close)
        self.addCleanup(scheduler.running_queue.close)
        scheduler.add_request(make_request("low", priority=1))
        scheduler.add_request(make_request("high", priority=8))
        scheduler.add_request(make_request("medium", priority=4))

        output = scheduler.schedule()
        stats = scheduler.get_cache_stats()

        self.assertEqual(
            [request.request_id for request in output.scheduled_requests],
            ["high", "medium"],
        )
        self.assertEqual(stats["waiting_queue_size"], 1)
        self.assertEqual(
            stats["priority_scheduling"]["admitted_by_priority"], {4: 1, 8: 1}
        )

    def test_deferred_request_keeps_priority(self):
        scheduler = Scheduler(
            max_batch_size=1,
            num_blocks=32,
            block_size=4,
            enable_prefix_caching=False,
        )
        self.addCleanup(scheduler.waiting_queue.close)
        self.addCleanup(scheduler.running_queue.close)
        scheduler.add_request(make_request("low", priority=1))
        scheduler.add_request(make_request("high", priority=8))
        with patch.object(scheduler, "can_accept_request", return_value=False):
            self.assertIsNone(scheduler.schedule())

        scheduler.add_request(make_request("medium", priority=4))
        with patch.object(scheduler, "can_accept_request", return_value=True):
            output = scheduler.schedule()

        self.assertEqual(output.scheduled_requests[0].request_id, "high")

    def test_remote_kv_request_is_counted_when_admitted(self):
        scheduler = Scheduler(
            max_batch_size=1,
            num_blocks=32,
            block_size=4,
            connector=FakeConnector(),
            enable_prefix_caching=False,
        )
        self.addCleanup(scheduler.waiting_queue.close)
        self.addCleanup(scheduler.running_queue.close)
        request = make_request("remote", priority=7)
        scheduler.add_request(request)

        output = scheduler.schedule()
        stats = scheduler.get_cache_stats()["priority_scheduling"]

        self.assertEqual(output.scheduled_requests, [])
        self.assertEqual(request.status, RequestStatus.WAITING_FOR_REMOTE_KVS)
        self.assertEqual(stats["admitted_requests"], 1)
        self.assertEqual(stats["admitted_by_priority"], {7: 1})


if __name__ == "__main__":
    unittest.main()
