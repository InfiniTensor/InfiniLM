import unittest

from chunk_test_support import MODULES

Scheduler = MODULES["scheduler"].Scheduler
Request = MODULES["request"].InferenceRequest
Status = MODULES["request"].RequestStatus
Sampling = MODULES["sampling_params"].SamplingParams


class ChunkSchedulerTests(unittest.TestCase):
    def scheduler(self, **kwargs):
        config = dict(num_blocks=64, block_size=16, prefill_chunk_size=16)
        config.update(kwargs)
        scheduler = Scheduler(**config)
        self.addCleanup(scheduler.waiting_queue.close)
        self.addCleanup(scheduler.running_queue.close)
        return scheduler

    @staticmethod
    def request(name, length, output=4):
        return Request(
            name,
            prompt_token_ids=[11] * length,
            sampling_params=Sampling(max_tokens=output),
        )

    def finish_step(self, scheduler, output):
        for req in output.scheduled_requests:
            if output.prefill_end is not None:
                req.num_computed_tokens = output.prefill_end
                scheduler.commit_computed_tokens(req, output.prefill_end)
                if output.prefill_end < req.prompt_length:
                    scheduler.requeue_prefill(req)
                    continue
            req.append_generated_token_id(12)
            scheduler.complete_requests([req])

    def test_slru_chunk_admission_promotes_hits_only_after_allocation(self):
        from cache_test_support import assert_state, publish

        for rejected in (False, True):
            with self.subTest(rejected=rejected):
                scheduler = self.scheduler(num_blocks=4, prefix_cache_policy="slru")
                manager = scheduler.cache_manager
                table, _ = publish(manager, [11] * 16)
                req = self.request("hit", 1000 if rejected else 33, output=1)
                scheduler.add_request(req)
                step = scheduler.schedule()
                if rejected:
                    self.assertIsNone(step)
                    self.assertFalse(manager._protected_blocks)
                    self.assertEqual(manager.blocks[table[0]].ref_count, 0)
                else:
                    self.assertEqual(step.scheduled_requests, [req])
                    self.assertIn(table[0], manager._protected_blocks)
                    req.status = Status.CANCELED
                    scheduler.complete_requests([req])
                assert_state(self, manager)

    def test_chunk_boundaries_and_last_partial_segment(self):
        scheduler = self.scheduler()
        req = self.request("long", 35)
        scheduler.add_request(req)
        for start, end in ((0, 16), (16, 32), (32, 35)):
            step = scheduler.schedule()
            self.assertTrue(step.is_prefill)
            self.assertEqual(step.prefill_end, end)
            self.assertEqual(req.num_local_cached_tokens, start)
            self.assertEqual(len(req.slot_mapping), end - start)
            self.assertEqual(req.get_num_generated_tokens(), 0)
            self.finish_step(scheduler, step)
        self.assertEqual(req.get_num_generated_tokens(), 1)
        self.assertFalse(scheduler.schedule().is_prefill)

    def test_chunk_respects_smaller_token_budget(self):
        scheduler = self.scheduler(max_num_batched_tokens=7)
        req = self.request("limited", 20)
        scheduler.add_request(req)
        self.assertEqual(scheduler.schedule().prefill_end, 7)

    def test_disabled_mode_retains_whole_prompt_dispatch(self):
        scheduler = self.scheduler(prefill_chunk_size=0, max_num_batched_tokens=16)
        req = self.request("legacy", 35)
        scheduler.add_request(req)
        step = scheduler.schedule()
        self.assertTrue(step.is_prefill)
        self.assertIsNone(step.prefill_end)
        self.assertEqual(len(req.slot_mapping), 35)
        self.assertFalse(scheduler.chunking_queue)

    def test_nonaligned_chunks_reconstruct_physical_slots(self):
        scheduler = self.scheduler(prefill_chunk_size=11)
        req = self.request("unaligned", 35)
        scheduler.add_request(req)
        for start, end in ((0, 11), (11, 22), (22, 33), (33, 35)):
            step = scheduler.schedule()
            expected = [
                req.block_table[i // 16] * 16 + i % 16 for i in range(start, end)
            ]
            self.assertEqual(req.slot_mapping, expected)
            self.finish_step(scheduler, step)

    def test_published_partial_prefix_starts_at_hit_boundary(self):
        scheduler = self.scheduler()
        first = self.request("first", 49)
        scheduler.add_request(first)
        self.finish_step(scheduler, scheduler.schedule())
        first.status = Status.CANCELED
        second = self.request("second", 49)
        scheduler.add_request(second)
        seen = False
        for _ in range(4):
            step = scheduler.schedule()
            if step.scheduled_requests == [second]:
                self.assertEqual(second.num_local_cached_tokens, 16)
                self.assertEqual(step.prefill_end, 32)
                seen = True
                break
            self.finish_step(scheduler, step)
        self.assertTrue(seen)

    def test_decode_continuation_and_admission_each_get_dispatch_opportunities(self):
        scheduler = self.scheduler(max_batch_size=1)
        active = self.request("active", 1)
        scheduler.add_request(active)
        self.finish_step(scheduler, scheduler.schedule())
        long = self.request("long", 200)
        scheduler.add_request(long)
        kinds = []
        seen = {"active"}
        for i in range(12):
            scheduler.add_request(self.request(f"new-{i}", 100))
            step = scheduler.schedule()
            req = step.scheduled_requests[0]
            kinds.append(
                "decode"
                if not step.is_prefill
                else "continue" if req.request_id in seen else "admit"
            )
            seen.add(req.request_id)
            self.finish_step(scheduler, step)
        for start in (3, 6, 9):
            self.assertEqual(
                set(kinds[start : start + 3]), {"decode", "continue", "admit"}
            )

    def test_cancelled_middle_chunk_is_released_without_requeue(self):
        scheduler = self.scheduler()
        req = self.request("canceled", 64)
        scheduler.add_request(req)
        self.finish_step(scheduler, scheduler.schedule())
        req.status = Status.CANCELED
        self.assertIsNone(scheduler.schedule())
        self.assertFalse(scheduler.chunking_queue)
        self.assertEqual(scheduler.cache_manager.get_total_usable_blocks(), 64)

    def test_prefix_disabled_still_advances(self):
        scheduler = self.scheduler(enable_prefix_caching=False)
        req = self.request("no-cache", 33)
        scheduler.add_request(req)
        for end in (16, 32, 33):
            step = scheduler.schedule()
            self.assertEqual(step.prefill_end, end)
            self.finish_step(scheduler, step)
        self.assertFalse(scheduler.cache_manager.hash_to_block_ids)

    def test_partial_requests_reserve_future_decode_capacity(self):
        scheduler = self.scheduler(num_blocks=6)
        long = self.request("long", 48, output=32)
        scheduler.add_request(long)
        self.finish_step(scheduler, scheduler.schedule())
        other = self.request("other", 32, output=16)
        scheduler.add_request(other)
        for _ in range(2):
            step = scheduler.schedule()
            self.assertNotIn(other, step.scheduled_requests)
            self.finish_step(scheduler, step)
        self.assertEqual(other.status, Status.WAITING)

    def test_rejected_admission_returns_temporary_prefix_reference(self):
        scheduler = self.scheduler(num_blocks=5)
        req = self.request("long", 48, output=16)
        scheduler.add_request(req)
        self.finish_step(scheduler, scheduler.schedule())
        other = self.request("too-large", 48, output=128)
        scheduler.add_request(other)
        before = [b.ref_count for b in scheduler.cache_manager.blocks]
        step = scheduler.schedule()
        self.assertNotIn(other, step.scheduled_requests)
        self.assertEqual(before, [b.ref_count for b in scheduler.cache_manager.blocks])

    def test_shared_prefix_stays_pinned_until_both_owners_release(self):
        scheduler = self.scheduler()
        first = self.request("first", 64)
        second = self.request("second", 64)
        scheduler.add_request(first)
        self.finish_step(scheduler, scheduler.schedule())
        scheduler.add_request(second)
        for _ in range(3):
            step = scheduler.schedule()
            self.finish_step(scheduler, step)
            if second.status == Status.RUNNING:
                break
        shared = first.block_table[0]
        self.assertEqual(second.block_table[0], shared)
        self.assertEqual(scheduler.cache_manager.blocks[shared].ref_count, 2)
        first.mark_canceled()
        scheduler.schedule()
        self.assertEqual(scheduler.cache_manager.blocks[shared].ref_count, 1)
        # The previous dispatch may have removed the second request for execution.
        second.mark_canceled()
        scheduler.complete_requests([second])
        self.assertTrue(all(b.ref_count == 0 for b in scheduler.cache_manager.blocks))

    def test_multimodal_requests_fail_before_ownership_is_acquired(self):
        scheduler = self.scheduler()
        req = self.request("image", 16)
        req.has_multimodal_inputs = True
        with self.assertRaisesRegex(ValueError, "multimodal"):
            scheduler.add_request(req)
        self.assertEqual(scheduler.waiting_queue.sync_q.qsize(), 0)
