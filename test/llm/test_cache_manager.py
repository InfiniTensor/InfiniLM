"""CPU-only cache lifecycle and scheduler regression tests.

Run with: python -m unittest discover -s test/llm -v
"""

import importlib.util
import random
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


def load_control_plane():
    """Load real control-plane modules without the GPU package initializer."""
    directory = Path(__file__).resolve().parents[2] / "python/infinilm/llm"
    modules = {}
    missing = object()
    originals = {}
    try:
        for name in (
            "prefix_cache",
            "sampling_params",
            "request",
            "cache_manager",
            "scheduler",
        ):
            qualified_name = f"infinilm.llm.{name}"
            spec = importlib.util.spec_from_file_location(
                qualified_name, directory / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            originals[qualified_name] = sys.modules.get(qualified_name, missing)
            sys.modules[qualified_name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    finally:
        # Restore only our modules; dependencies may contain non-reloadable extensions.
        for name, original in originals.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return SimpleNamespace(**modules)


modules = load_control_plane()
BlockManager = modules.cache_manager.BlockManager


class ScanningBlockManager(BlockManager):
    """The previous capacity query, used as a scheduler decision oracle."""

    def get_total_usable_blocks(self):
        return len(self.free_block_ids) + sum(
            self.blocks[block_id].ref_count == 0 for block_id in self.used_block_ids
        )


class BlockManagerTest(unittest.TestCase):
    def assert_state(self, manager, owners):
        references = Counter(block for table in owners for block in table)
        self.assertEqual(
            manager.get_total_usable_blocks(), manager.num_blocks - len(references)
        )
        self.assertEqual(
            [block.ref_count for block in manager.blocks],
            [references[i] for i in range(manager.num_blocks)],
        )
        free = set(manager.free_block_ids)
        self.assertEqual(len(free), len(manager.free_block_ids))
        self.assertFalse(free & manager.used_block_ids)
        self.assertEqual(free | manager.used_block_ids, set(range(manager.num_blocks)))
        indexed = {}
        for block in manager.blocks:
            if block.hash:
                self.assertIn(block.block_id, manager.used_block_ids)
                indexed.setdefault(block.hash, set()).add(block.block_id)
        self.assertEqual(manager.hash_to_block_ids, indexed)

    def test_shared_prefix_counts_physical_pages_once(self):
        manager = BlockManager(4, 4)
        table, _ = manager.allocate_slots(8)
        hashes = [b"a" * 16, b"b" * 16]
        manager.publish_computed_blocks(table, hashes, 0, 8)
        shared, hit = manager.get_computed_blocks(hashes, 8)
        self.assertEqual(hit, 8)
        self.assert_state(manager, [table, shared])
        manager.free_blocks(table)
        self.assert_state(manager, [shared])
        manager.free_blocks(shared)
        self.assert_state(manager, [])
        repinned, hit = manager.get_computed_blocks(hashes, 4)
        self.assertEqual(hit, 4)
        self.assert_state(manager, [repinned])
        manager.free_blocks(repinned)
        self.assertTrue(manager.try_free_blocks(4))
        self.assert_state(manager, [])

    def test_failed_allocation_after_reclaim_and_retry(self):
        manager = BlockManager(3, 4)
        live, _ = manager.allocate_slots(4)
        released, _ = manager.allocate_slots(8)
        manager.free_blocks(released)
        self.assertIsNone(manager.allocate_slots(12))
        self.assert_state(manager, [live])
        replacement, _ = manager.allocate_slots(8)
        self.assert_state(manager, [live, replacement])
        with self.assertRaisesRegex(RuntimeError, "No available"):
            manager.append_slots(replacement, 9, 1)
        self.assert_state(manager, [live, replacement])

    def test_prefix_reuse_then_decode_crosses_block_boundary(self):
        manager = BlockManager(4, 4)
        prefix, _ = manager.allocate_slots(4)
        hashes = [b"a" * 16]
        manager.publish_computed_blocks(prefix, hashes, 0, 4)
        manager.free_blocks(prefix)
        reused, hit = manager.get_computed_blocks(hashes, 4)
        table, _ = manager.allocate_slots(8, hit, reused)
        self.assert_state(manager, [table])
        self.assertEqual(manager.get_total_usable_blocks(), 1)
        table, slot = manager.append_slot(table, 13)
        self.assertEqual(slot, table[-1] * 4)
        self.assert_state(manager, [table])
        self.assertEqual(manager.get_total_usable_blocks(), 0)
        manager.free_blocks(table)
        self.assert_state(manager, [])

    def test_speculative_rollback_and_rejected_truncation(self):
        manager = BlockManager(6, 4)
        table, _ = manager.allocate_slots(5)
        table, _ = manager.append_slots(table, 6, 8)
        self.assert_state(manager, [table])
        table = manager.truncate_blocks(table, 6)
        self.assertEqual(len(table), 2)
        self.assert_state(manager, [table])
        manager.publish_computed_blocks(table, [b"a" * 16, b"b" * 16], 0, 8)
        with self.assertRaisesRegex(RuntimeError, "must not be published"):
            manager.truncate_blocks(table, 4)
        self.assert_state(manager, [table])
        manager.free_blocks(table)
        self.assert_state(manager, [])

    def test_duplicate_hash_survives_other_copy_eviction(self):
        manager = BlockManager(2, 4)
        tables = [manager.allocate_slots(4)[0] for _ in range(2)]
        hashes = [b"a" * 16]
        for table in tables:
            manager.publish_computed_blocks(table, hashes, 0, 4)
        manager.free_blocks(tables[0])
        self.assertTrue(manager.try_free_blocks(1))
        self.assertEqual(manager.hash_to_block_ids[hashes[0]], set(tables[1]))
        shared, _ = manager.get_computed_blocks(hashes, 4)
        self.assert_state(manager, [tables[1], shared])

    def test_randomized_lifecycles(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                manager = BlockManager(23, 4)
                owners = []
                for _ in range(1000):
                    action = rng.randrange(7)
                    if action == 0 or not owners:
                        tokens = rng.randint(1, 20)
                        result = manager.allocate_slots(tokens)
                        if result is not None:
                            table, _ = result
                            owners.append([table, tokens, False])
                    elif action == 1:
                        table, _, _ = owners.pop(rng.randrange(len(owners)))
                        manager.free_blocks(table)
                    elif action == 2:
                        owner = rng.choice(owners)
                        table, tokens, published = owner
                        if not published and tokens >= 4:
                            hashes = [
                                bytes([i % 4 + 1]) * 16 for i in range(tokens // 4)
                            ]
                            manager.publish_computed_blocks(table, hashes, 0, tokens)
                            owner[2] = True
                    elif action == 3:
                        hashes = [
                            bytes([i % 4 + 1]) * 16 for i in range(rng.randint(1, 6))
                        ]
                        table, tokens = manager.get_computed_blocks(
                            hashes, rng.randint(0, 24)
                        )
                        if table:
                            owners.append([table, tokens, True])
                    elif action == 4:
                        owner = rng.choice(owners)
                        if not owner[2]:
                            extra = rng.randint(0, 9)
                            try:
                                table, _ = manager.append_slots(
                                    owner[0], owner[1] + 1, extra
                                )
                            except RuntimeError:
                                pass
                            else:
                                owner[:2] = [table, owner[1] + extra]
                    elif action == 5:
                        owner = rng.choice(owners)
                        if not owner[2]:
                            tokens = rng.randint(1, owner[1])
                            owner[:2] = [
                                manager.truncate_blocks(owner[0], tokens),
                                tokens,
                            ]
                    else:
                        manager.try_free_blocks(rng.randint(1, 23))
                    self.assert_state(manager, [owner[0] for owner in owners])


class SchedulerCapacityTest(unittest.TestCase):
    def run_lifecycle(self, manager_type):
        scheduler = modules.scheduler.Scheduler(
            max_batch_size=4, num_blocks=16, block_size=4
        )
        scheduler.cache_manager = manager_type(16, 4)
        scheduler.speculative_cache_ops = modules.scheduler.SpeculativeCacheOps(
            scheduler.cache_manager
        )
        self.addCleanup(scheduler.waiting_queue.close)
        self.addCleanup(scheduler.running_queue.close)
        trace = []
        hits = 0
        for step in range(40):
            if step in (0, 5, 10, 15):
                for i in range(3):
                    req = modules.request.InferenceRequest(
                        request_id=f"{step}-{i}",
                        prompt_token_ids=[(i + step // 10) % 2] * 9,
                        sampling_params=modules.sampling_params.SamplingParams(
                            max_tokens=3
                        ),
                    )
                    scheduler.add_request(req)
            output = scheduler.schedule()
            if output is None:
                continue
            trace.append(
                (
                    output.is_prefill,
                    [
                        (
                            req.request_id,
                            tuple(req.block_table),
                            tuple(req.slot_mapping),
                            req.num_local_cached_tokens,
                        )
                        for req in output.scheduled_requests
                    ],
                )
            )
            for req in output.scheduled_requests:
                if output.is_prefill:
                    hits += req.num_local_cached_tokens > 0
                scheduler.commit_computed_tokens(req, req.get_total_length())
                req.append_generated_token_id(7)
                if req.get_num_generated_tokens() == 3:
                    req.mark_finished(modules.request.FinishReason.LENGTH)
            scheduler.complete_requests(output.scheduled_requests)
            manager = scheduler.cache_manager
            self.assertEqual(
                manager.get_total_usable_blocks(),
                ScanningBlockManager.get_total_usable_blocks(manager),
            )
            trace.append(
                (manager.get_total_usable_blocks(), tuple(manager.free_block_ids))
            )
        self.assertTrue(scheduler.waiting_queue.sync_q.empty())
        self.assertTrue(scheduler.running_queue.sync_q.empty())
        self.assertGreater(hits, 0)
        self.assertEqual(scheduler.cache_manager.get_total_usable_blocks(), 16)
        return trace

    def test_scheduler_decisions_match_scanning_capacity(self):
        self.assertEqual(
            self.run_lifecycle(BlockManager), self.run_lifecycle(ScanningBlockManager)
        )

    def test_remote_transfer_delays_release(self):
        scheduler = modules.scheduler.Scheduler(num_blocks=4, block_size=4)
        self.addCleanup(scheduler.waiting_queue.close)
        self.addCleanup(scheduler.running_queue.close)
        scheduler.connector = SimpleNamespace(
            request_finished=lambda *args: (True, None)
        )
        request = modules.request.InferenceRequest(
            request_id="remote", prompt_token_ids=[1] * 4
        )
        request.block_table, _ = scheduler.cache_manager.allocate_slots(4)
        request.mark_canceled()
        scheduler.complete_requests([request])
        self.assertEqual(scheduler.cache_manager.get_total_usable_blocks(), 3)
        output = SimpleNamespace(
            kv_connector_output=SimpleNamespace(finished_sending={"remote"})
        )
        scheduler.update_from_output(output)
        self.assertEqual(scheduler.cache_manager.get_total_usable_blocks(), 4)
        scheduler.update_from_output(output)
        self.assertEqual(scheduler.cache_manager.get_total_usable_blocks(), 4)


if __name__ == "__main__":
    unittest.main()
