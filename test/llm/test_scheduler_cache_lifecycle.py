import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cache_test_support import MODULES, assert_state, chain_hashes, publish

Scheduler = MODULES["scheduler"].Scheduler
InferenceRequest = MODULES["request"].InferenceRequest
RequestStatus = MODULES["request"].RequestStatus
SamplingParams = MODULES["sampling_params"].SamplingParams
MambaCacheManager = MODULES["cache_manager"].MambaCacheManager


class DelayedConnector:
    def request_finished(self, request, block_table, block_size):
        return True, None


class SchedulerCacheTests(unittest.TestCase):
    def make_scheduler(self, **kwargs):
        scheduler = Scheduler(**kwargs)
        self.addCleanup(scheduler.waiting_queue.close)
        self.addCleanup(scheduler.running_queue.close)
        return scheduler

    @staticmethod
    def output(**connector_output):
        return SimpleNamespace(kv_connector_output=SimpleNamespace(**connector_output))

    def test_connector_metadata_survives_prefill_decode_and_idle_dispatch(self):
        metadata = object()
        connector = SimpleNamespace(
            build_connector_meta=lambda: metadata,
            get_num_new_matched_tokens=lambda req, cached: (0, False),
            update_state_after_alloc=lambda *args: None,
            request_finished=lambda *args: (False, None),
        )
        scheduler = self.make_scheduler(
            num_blocks=4, block_size=16, connector=connector
        )
        req = InferenceRequest(
            "metadata",
            prompt_token_ids=[11],
            sampling_params=SamplingParams(max_tokens=2),
        )
        scheduler.add_request(req)
        prefill = scheduler.schedule()
        self.assertTrue(prefill.is_prefill)
        self.assertIs(prefill.kv_connector_metadata, metadata)
        req.append_generated_token_id(12)
        scheduler.complete_requests([req])
        decode = scheduler.schedule()
        self.assertFalse(decode.is_prefill)
        self.assertEqual(decode.scheduled_requests, [req])
        self.assertIs(decode.kv_connector_metadata, metadata)
        req.mark_canceled()
        scheduler.complete_requests([req])
        idle = scheduler.schedule()
        self.assertEqual(idle.scheduled_requests, [])
        self.assertIs(idle.kv_connector_metadata, metadata)
        assert_state(self, scheduler.cache_manager)

    def test_send_completion_releases_exactly_once(self):
        scheduler = self.make_scheduler(
            num_blocks=1, block_size=16, connector=DelayedConnector()
        )
        manager = scheduler.cache_manager
        request = InferenceRequest("delayed", prompt_token_ids=[11] * 16)
        request.block_table, _ = manager.allocate_slots(16)
        manager.publish_computed_blocks(
            request.block_table, chain_hashes([11] * 16), 0, 16
        )
        block_id = request.block_table[0]
        request.status = RequestStatus.CANCELED

        scheduler.complete_requests([request])

        self.assertFalse(manager.try_free_blocks(1))
        self.assertEqual(manager.blocks[block_id].ref_count, 1)
        event = self.output(finished_sending={"delayed"})
        scheduler.update_from_output(event)
        scheduler.update_from_output(event)
        self.assertEqual(scheduler.pending_free_blocks, {})
        self.assertEqual(manager.get_total_usable_blocks(), 1)
        assert_state(self, manager)

    def test_terminal_status_releases_tail_and_preserves_shared_prefix(self):
        for status in (
            RequestStatus.FINISHED,
            RequestStatus.CANCELED,
            RequestStatus.FAILED,
            RequestStatus.TIMEOUT,
        ):
            with self.subTest(status=status):
                scheduler = self.make_scheduler(num_blocks=2, block_size=16)
                manager = scheduler.cache_manager
                request = InferenceRequest(status.value, prompt_token_ids=[11] * 17)
                request.block_table, _ = manager.allocate_slots(17)
                hashes = chain_hashes([11] * 16)
                manager.publish_computed_blocks(request.block_table, hashes, 0, 16)
                shared, hit = manager.get_computed_blocks(hashes, 16)
                prefix_id, tail_id = request.block_table
                request.status = status

                scheduler.complete_requests([request])

                self.assertEqual(hit, 16)
                self.assertEqual(manager.blocks[prefix_id].ref_count, 1)
                self.assertIn(prefix_id, manager.used_block_ids)
                self.assertIn(tail_id, manager.free_block_ids)
                self.assertNotIn(prefix_id, manager._evictable_blocks)
                manager.free_blocks(shared)
                self.assertIn(prefix_id, manager._evictable_blocks)
                assert_state(self, manager)

    def test_admission_failure_returns_temporary_prefix_pin(self):
        scheduler = self.make_scheduler(num_blocks=2, block_size=16)
        manager = scheduler.cache_manager
        table, hashes = publish(manager, [11] * 16)
        request = InferenceRequest(
            "too-large",
            prompt_token_ids=[11] * 16 + [12],
            sampling_params=SamplingParams(max_tokens=64),
        )
        scheduler.add_request(request)

        self.assertIsNone(scheduler.schedule())

        self.assertEqual(request.status, RequestStatus.WAITING)
        self.assertEqual(scheduler.waiting_queue.sync_q.qsize(), 1)
        self.assertEqual(manager.blocks[table[0]].ref_count, 0)
        self.assertEqual(list(manager._evictable_blocks), table)
        self.assertEqual(manager.get_total_usable_blocks(), 2)
        assert_state(self, manager)

    def test_failed_admission_refreshes_prefix_release_recency(self):
        scheduler = self.make_scheduler(num_blocks=3, block_size=16)
        manager = scheduler.cache_manager
        a_table, _ = publish(manager, [11] * 16)
        b_table, b_hashes = publish(manager, [22] * 16)
        c_table, _ = publish(manager, [33] * 16)
        request = InferenceRequest(
            "touch-a",
            prompt_token_ids=[11] * 16 + [12],
            sampling_params=SamplingParams(max_tokens=64),
        )
        scheduler.add_request(request)

        self.assertIsNone(scheduler.schedule())
        allocation = manager.allocate_slots(16)

        self.assertIsNotNone(allocation)
        self.assertEqual(list(manager._evictable_blocks), [c_table[0], a_table[0]])
        self.assertNotIn(b_hashes[0], manager.hash_to_block_ids)
        self.assertEqual(allocation[0], b_table)
        manager.free_blocks(allocation[0])
        assert_state(self, manager)

    def test_token_budget_defer_returns_second_requests_temporary_pin(self):
        scheduler = self.make_scheduler(
            max_batch_size=2,
            max_num_batched_tokens=16,
            num_blocks=8,
            block_size=16,
        )
        manager = scheduler.cache_manager
        prefix_table, _ = publish(manager, [11] * 16)
        first = InferenceRequest(
            "first",
            prompt_token_ids=[11] * 16 + [21],
            sampling_params=SamplingParams(max_tokens=1),
        )
        second = InferenceRequest(
            "second",
            prompt_token_ids=[11] * 16 + [31] * 32,
            sampling_params=SamplingParams(max_tokens=1),
        )
        scheduler.add_request(first)
        scheduler.add_request(second)

        batch = scheduler.schedule()

        self.assertEqual(batch.scheduled_requests, [first])
        self.assertEqual(first.status, RequestStatus.RUNNING)
        self.assertEqual(second.status, RequestStatus.WAITING)
        self.assertEqual(scheduler.waiting_queue.sync_q.qsize(), 1)
        self.assertEqual(manager.blocks[prefix_table[0]].ref_count, 1)
        self.assertNotIn(prefix_table[0], manager._evictable_blocks)
        first.status = RequestStatus.CANCELED
        scheduler.complete_requests([first])
        assert_state(self, manager)

    def test_prefix_disabled_releases_all_pages_as_free(self):
        scheduler = self.make_scheduler(
            num_blocks=3, block_size=16, enable_prefix_caching=False
        )
        request = InferenceRequest(
            "no-prefix",
            prompt_token_ids=[11] * 17,
            sampling_params=SamplingParams(max_tokens=1),
        )
        scheduler.add_request(request)
        batch = scheduler.schedule()
        request.status = RequestStatus.CANCELED

        scheduler.complete_requests([request])

        self.assertEqual(batch.scheduled_requests, [request])
        self.assertEqual(list(request.block_hashes), [])
        self.assertEqual(scheduler.cache_manager.get_num_free_blocks(), 3)
        self.assertEqual(scheduler.cache_manager.hash_to_block_ids, {})
        assert_state(self, scheduler.cache_manager)

    def test_receive_completion_releases_canceled_remote_request_once(self):
        scheduler = self.make_scheduler(
            num_blocks=1, block_size=16, connector=DelayedConnector()
        )
        manager = scheduler.cache_manager
        request = InferenceRequest(
            "remote",
            prompt_token_ids=[11] * 16,
            sampling_params=SamplingParams(max_tokens=1),
        )
        request.block_table, _ = manager.allocate_slots(16)
        manager.publish_computed_blocks(
            request.block_table, chain_hashes([11] * 16), 0, 16
        )
        block_id = request.block_table[0]
        scheduler.remote_kv_requests[request.request_id] = request
        scheduler.pending_kv_decode_blocks = 1
        request.status = RequestStatus.CANCELED

        scheduler.complete_requests([request])

        self.assertEqual(scheduler.pending_kv_decode_blocks, 0)
        self.assertFalse(manager.try_free_blocks(1))
        self.assertEqual(manager.blocks[block_id].ref_count, 1)
        event = self.output(finished_recving={"remote", "unknown"})
        scheduler.update_from_output(event)
        scheduler.update_from_output(event)
        self.assertEqual(scheduler.pending_free_blocks, {})
        self.assertEqual(manager.get_total_usable_blocks(), 1)
        assert_state(self, manager)

    def test_mamba_manager_and_scheduler_release_owned_rows_and_pages(self):
        mamba = MambaCacheManager(3)
        first = mamba.allocate()
        second = mamba.allocate()
        self.assertEqual((first, second), (1, 2))
        self.assertIsNone(mamba.allocate())
        mamba.free(first)
        self.assertEqual(mamba.allocate(), 1)

        scheduler = self.make_scheduler(
            num_blocks=3,
            block_size=16,
            has_mamba_cache=True,
            num_mamba_cache_blocks=2,
        )
        request = InferenceRequest(
            "mamba",
            prompt_token_ids=[11] * 17,
            sampling_params=SamplingParams(max_tokens=1),
        )
        scheduler.add_request(request)
        batch = scheduler.schedule()
        owned_row = request.mamba_cache_index
        request.status = RequestStatus.CANCELED

        scheduler.complete_requests([request])

        self.assertEqual(batch.scheduled_requests, [request])
        self.assertEqual(list(request.block_hashes), [])
        self.assertIsNotNone(owned_row)
        self.assertIsNone(request.mamba_cache_index)
        self.assertEqual(scheduler.mamba_cache_manager.get_num_free_blocks(), 1)
        self.assertEqual(scheduler.cache_manager.get_num_free_blocks(), 3)
        assert_state(self, scheduler.cache_manager)


class SegmentedSchedulerCacheTests(SchedulerCacheTests):
    def make_scheduler(self, **kwargs):
        kwargs["prefix_cache_policy"] = "slru"
        kwargs["prefix_cache_protected_ratio"] = 0.5
        return super().make_scheduler(**kwargs)

    def test_only_successful_admission_promotes_local_prefix(self):
        scheduler = self.make_scheduler(num_blocks=4, block_size=16)
        manager = scheduler.cache_manager
        table, hashes = publish(manager, [11] * 16)
        rejected = InferenceRequest(
            "rejected",
            prompt_token_ids=[11] * 16 + [12],
            sampling_params=SamplingParams(max_tokens=128),
        )
        scheduler.add_request(rejected)
        self.assertIsNone(scheduler.schedule())
        self.assertFalse(manager._protected_blocks)
        rejected.status = RequestStatus.CANCELED
        admitted = InferenceRequest(
            "admitted",
            prompt_token_ids=[11] * 16 + [13],
            sampling_params=SamplingParams(max_tokens=1),
        )
        scheduler.add_request(admitted)
        self.assertEqual(scheduler.schedule().scheduled_requests, [admitted])
        self.assertEqual(list(manager._protected_blocks), table)
        self.assertFalse(manager._protected_evictable_blocks)
        admitted.status = RequestStatus.CANCELED
        scheduler.complete_requests([admitted])
        self.assertEqual(list(manager._protected_evictable_blocks), table)
        for token in range(20, 28):
            publish(manager, [token] * 16)
        self.assertIn(hashes[0], manager.hash_to_block_ids)
        assert_state(self, manager)

    def test_token_budget_probe_does_not_promote_unrelated_prefix(self):
        scheduler = self.make_scheduler(
            num_blocks=8, block_size=16, max_batch_size=2, max_num_batched_tokens=16
        )
        manager = scheduler.cache_manager
        table, _ = publish(manager, [22] * 16)
        first = InferenceRequest(
            "first",
            prompt_token_ids=[11] * 8,
            sampling_params=SamplingParams(max_tokens=1),
        )
        second = InferenceRequest(
            "deferred",
            prompt_token_ids=[22] * 16 + [33] * 16,
            sampling_params=SamplingParams(max_tokens=1),
        )
        scheduler.add_request(first)
        scheduler.add_request(second)
        self.assertEqual(scheduler.schedule().scheduled_requests, [first])
        self.assertEqual(second.status, RequestStatus.WAITING)
        self.assertNotIn(table[0], manager._protected_blocks)
        first.status = RequestStatus.CANCELED
        scheduler.complete_requests([first])
        assert_state(self, manager)

    def test_allocation_failure_does_not_promote(self):
        scheduler = self.make_scheduler(num_blocks=4, block_size=16)
        manager = scheduler.cache_manager
        table, _ = publish(manager, [11] * 16)
        request = InferenceRequest(
            "allocation-failed",
            prompt_token_ids=[11] * 16 + [12],
            sampling_params=SamplingParams(max_tokens=1),
        )
        scheduler.add_request(request)
        with patch.object(manager, "allocate_slots", return_value=None):
            self.assertIsNone(scheduler.schedule())
        self.assertEqual(request.status, RequestStatus.WAITING)
        self.assertFalse(manager._protected_blocks)
        self.assertEqual(list(manager._evictable_blocks), table)
        assert_state(self, manager)

    def test_protected_remote_owner_is_not_evictable_until_transfer_finishes(self):
        scheduler = self.make_scheduler(
            num_blocks=4, block_size=16, connector=DelayedConnector()
        )
        manager = scheduler.cache_manager
        _, hashes = publish(manager, [11] * 16)
        table, _ = manager.get_computed_blocks(hashes, 16)
        manager.record_cache_hit(table)
        request = InferenceRequest("protected-send", prompt_token_ids=[11] * 16)
        request.block_table = table
        request.status = RequestStatus.CANCELED
        scheduler.complete_requests([request])
        self.assertEqual(list(manager._protected_blocks), table)
        self.assertFalse(manager.try_free_blocks(4))
        self.assertEqual(manager.blocks[table[0]].ref_count, 1)
        event = self.output(finished_sending={request.request_id})
        scheduler.update_from_output(event)
        scheduler.update_from_output(event)
        self.assertEqual(list(manager._protected_evictable_blocks), table)
        self.assertTrue(manager.try_free_blocks(4))
        assert_state(self, manager)


if __name__ == "__main__":
    unittest.main()
