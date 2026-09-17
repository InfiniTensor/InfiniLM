"""CPU checks for admission reservations within and across scheduler steps."""

import unittest
from unittest.mock import patch

from cpu_modules import load_cpu_modules

modules = load_cpu_modules()
Scheduler = modules.scheduler.Scheduler
InferenceRequest = modules.request.InferenceRequest
RequestStatus = modules.request.RequestStatus
SamplingParams = modules.sampling_params.SamplingParams


def make_request(request_id, prompt_length=1, max_tokens=16):
    return InferenceRequest(
        request_id,
        prompt_token_ids=list(range(prompt_length)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def add_running(scheduler, request_id, max_tokens=32):
    request = make_request(request_id, max_tokens=max_tokens)
    request.initialize_block_hashes(scheduler.block_size, False)
    request.block_table, request.slot_mapping = scheduler.cache_manager.allocate_slots(
        request.get_prompt_length()
    )
    request.num_blocks = len(request.block_table)
    request.append_generated_token_id(42)
    request.status = RequestStatus.RUNNING
    scheduler.complete_requests([request])
    return request


def queue_ids(janus_queue):
    items = [janus_queue.sync_q.get_nowait() for _ in range(janus_queue.sync_q.qsize())]
    for item in items:
        janus_queue.sync_q.put(item)
    return [item.request_id for item in items]


class RemoteConnector:
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if request.request_id.startswith("remote"):
            return request.get_prompt_length() - num_computed_tokens, True
        return 0, False

    def update_state_after_alloc(self, *args):
        pass

    def build_connector_meta(self):
        return None

    def request_finished(self, *args):
        return False, None


class SchedulerAdmissionTest(unittest.TestCase):
    def make_scheduler(self, **kwargs):
        scheduler = Scheduler(block_size=16, enable_prefix_caching=False, **kwargs)
        self.addCleanup(scheduler.waiting_queue.close)
        self.addCleanup(scheduler.running_queue.close)
        return scheduler

    def test_running_reservations_scanned_once_and_queue_order_preserved(self):
        scheduler = self.make_scheduler(max_batch_size=8, num_blocks=128)
        for index in range(3):
            add_running(scheduler, f"running-{index}")
        for index in range(8):
            scheduler.add_request(make_request(f"waiting-{index}"))
        before = queue_ids(scheduler.running_queue)
        with patch.object(
            scheduler,
            "_get_running_required_blocks",
            wraps=scheduler._get_running_required_blocks,
        ) as scan:
            output = scheduler.schedule()
        self.assertTrue(output.is_prefill)
        self.assertEqual(output.num_requests, 8)
        self.assertEqual(scan.call_count, 1)
        self.assertEqual(queue_ids(scheduler.running_queue), before)

    def test_prefill_reservations_and_remaining_capacity_stay_live(self):
        scheduler = self.make_scheduler(num_blocks=5)
        for index in range(3):
            scheduler.add_request(make_request(str(index)))
        with self.assertLogs(modules.scheduler.logger, level="WARNING"):
            output = scheduler.schedule()
        self.assertEqual(
            [req.request_id for req in output.scheduled_requests], ["0", "1"]
        )
        self.assertEqual(queue_ids(scheduler.waiting_queue), ["2"])
        self.assertEqual(scheduler.cache_manager.get_num_free_blocks(), 3)

    def test_remote_reservations_stay_live_in_same_step(self):
        scheduler = self.make_scheduler(num_blocks=5, connector=RemoteConnector())
        scheduler.add_request(make_request("remote-0"))
        scheduler.add_request(make_request("local-1"))
        scheduler.add_request(make_request("local-2"))
        with self.assertLogs(modules.scheduler.logger, level="WARNING"):
            output = scheduler.schedule()
        self.assertEqual(
            [req.request_id for req in output.scheduled_requests], ["local-1"]
        )
        self.assertEqual(scheduler.pending_kv_decode_blocks, 1)
        self.assertEqual(set(scheduler.remote_kv_requests), {"remote-0"})
        self.assertEqual(queue_ids(scheduler.waiting_queue), ["local-2"])

    def test_snapshot_is_recomputed_on_next_step(self):
        scheduler = self.make_scheduler(num_blocks=64)
        running = add_running(scheduler, "running")
        scheduler.add_request(make_request("first"))
        with patch.object(
            scheduler, "can_accept_request", wraps=scheduler.can_accept_request
        ) as admission:
            output = scheduler.schedule()
            self.assertEqual(admission.call_args.kwargs["running_required_blocks"], 2)
            output.scheduled_requests[0].status = RequestStatus.FINISHED
            scheduler.complete_requests(output.scheduled_requests)
            scheduler.cache_manager.append_slots(running.block_table, 2, 15)
            for _ in range(15):
                running.append_generated_token_id(42)
            scheduler.add_request(make_request("second"))
            scheduler.schedule()
            self.assertEqual(admission.call_args.kwargs["running_required_blocks"], 1)

    def test_standalone_admission_recomputes_running_reservations(self):
        scheduler = self.make_scheduler(num_blocks=5)
        running = add_running(scheduler, "running")
        candidate = make_request("candidate")
        scheduler.pending_kv_decode_blocks = 1
        self.assertFalse(scheduler.can_accept_request(candidate, 0))
        for _ in range(15):
            running.append_generated_token_id(42)
        self.assertTrue(scheduler.can_accept_request(candidate, 0))

    def test_decode_only_step_does_not_scan_reservations(self):
        scheduler = self.make_scheduler(num_blocks=16)
        running = add_running(scheduler, "running")
        with patch.object(scheduler, "_get_running_required_blocks") as scan:
            output = scheduler.schedule()
        scan.assert_not_called()
        self.assertFalse(output.is_prefill)
        self.assertEqual(output.scheduled_requests, [running])

    def test_canceled_waiting_request_and_token_budget(self):
        scheduler = self.make_scheduler(num_blocks=16, max_num_batched_tokens=5)
        canceled = make_request("canceled")
        scheduler.add_request(canceled)
        canceled.status = RequestStatus.CANCELED
        scheduler.add_request(make_request("first", prompt_length=3))
        scheduler.add_request(make_request("second", prompt_length=3))
        with patch.object(
            scheduler, "can_accept_request", wraps=scheduler.can_accept_request
        ) as admission:
            output = scheduler.schedule()
        self.assertEqual(admission.call_count, 1)
        self.assertEqual(
            [req.request_id for req in output.scheduled_requests], ["first"]
        )
        self.assertEqual(queue_ids(scheduler.waiting_queue), ["second"])

    def test_mamba_rows_still_limit_admission(self):
        scheduler = self.make_scheduler(
            num_blocks=64, has_mamba_cache=True, num_mamba_cache_blocks=2
        )
        scheduler.add_request(make_request("first"))
        scheduler.add_request(make_request("second"))
        with self.assertLogs(modules.scheduler.logger, level="WARNING"):
            output = scheduler.schedule()
        self.assertEqual(
            [req.request_id for req in output.scheduled_requests], ["first"]
        )
        self.assertEqual(queue_ids(scheduler.waiting_queue), ["second"])
        self.assertEqual(scheduler.mamba_cache_manager.get_num_free_blocks(), 0)

    def test_full_mamba_pool_rejects_without_scanning_running_reservations(self):
        scheduler = self.make_scheduler(
            num_blocks=64, has_mamba_cache=True, num_mamba_cache_blocks=2
        )
        running = add_running(scheduler, "running")
        running.mamba_cache_index = scheduler.mamba_cache_manager.allocate()
        scheduler.add_request(make_request("waiting"))
        with (
            patch.object(scheduler, "_get_running_required_blocks") as scan,
            self.assertLogs(modules.scheduler.logger, level="WARNING"),
        ):
            output = scheduler.schedule()
        scan.assert_not_called()
        self.assertFalse(output.is_prefill)
        self.assertEqual(output.scheduled_requests, [running])
        self.assertEqual(queue_ids(scheduler.waiting_queue), ["waiting"])


if __name__ == "__main__":
    unittest.main()
