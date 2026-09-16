import random
import unittest

from cache_test_support import BlockManager, assert_state, publish


class SegmentedCacheTests(unittest.TestCase):
    def make_manager(self, blocks=4, ratio=0.5):
        return BlockManager(
            blocks,
            16,
            prefix_cache_policy="slru",
            prefix_cache_protected_ratio=ratio,
        )

    def reuse(self, manager, hashes):
        table, _ = manager.get_computed_blocks(hashes, len(hashes) * 16)
        manager.record_cache_hit(table)
        manager.free_blocks(table)

    def test_admitted_hot_prefix_survives_one_off_scan(self):
        for policy, retained in (("lru", False), ("slru", True)):
            with self.subTest(policy=policy):
                manager = BlockManager(4, 16, prefix_cache_policy=policy)
                _, hot = publish(manager, [11] * 16)
                self.reuse(manager, hot)
                for token in range(20, 30):
                    publish(manager, [token] * 16)
                    assert_state(self, manager)
                self.assertEqual(hot[0] in manager.hash_to_block_ids, retained)

    def test_probe_does_not_promote(self):
        manager = self.make_manager()
        table, hashes = publish(manager, [11] * 16)
        probe, _ = manager.get_computed_blocks(hashes, 16)
        manager.free_blocks(probe)
        self.assertNotIn(table[0], manager._protected_blocks)
        for token in range(20, 25):
            publish(manager, [token] * 16)
        self.assertNotIn(hashes[0], manager.hash_to_block_ids)

    def test_protected_capacity_demotes_old_hotspot(self):
        manager = self.make_manager(ratio=0.25)
        old_table, old = publish(manager, [11] * 16)
        self.reuse(manager, old)
        new_table, new = publish(manager, [22] * 16)
        self.reuse(manager, new)
        self.assertEqual(list(manager._protected_blocks), new_table)
        self.assertIn(old_table[0], manager._evictable_blocks)
        for token in range(30, 35):
            publish(manager, [token] * 16)
        self.assertNotIn(old[0], manager.hash_to_block_ids)
        self.assertIn(new[0], manager.hash_to_block_ids)
        assert_state(self, manager)

    def test_pinned_demotion_never_releases_shared_owner(self):
        manager = self.make_manager(ratio=0.25)
        table, hashes = publish(manager, [11] * 16)
        first, _ = manager.get_computed_blocks(hashes, 16)
        second, _ = manager.get_computed_blocks(hashes, 16)
        manager.record_cache_hit(first)
        _, other = publish(manager, [22] * 16)
        self.reuse(manager, other)
        self.assertNotIn(table[0], manager._protected_blocks)
        self.assertEqual(manager.blocks[table[0]].ref_count, 2)
        self.assertFalse(manager.try_free_blocks(4))
        manager.free_blocks(first)
        self.assertFalse(manager.try_free_blocks(4))
        manager.free_blocks(second)
        self.assertTrue(manager.try_free_blocks(4))
        assert_state(self, manager)

    def test_tail_loses_protection_before_prefix_when_cap_is_small(self):
        manager = self.make_manager(ratio=0.25)
        table, hashes = publish(manager, [11] * 16 + [22] * 16)
        self.reuse(manager, hashes)
        self.assertEqual(list(manager._protected_blocks), table[:1])
        self.assertTrue(manager.try_free_blocks(3))
        self.assertIn(hashes[0], manager.hash_to_block_ids)
        self.assertNotIn(hashes[1], manager.hash_to_block_ids)

    def test_protected_tail_evicted_before_prefix_when_no_probation_remains(self):
        manager = self.make_manager(ratio=0.75)
        _, hashes = publish(manager, [11] * 16 + [22] * 16)
        self.reuse(manager, hashes)
        self.assertTrue(manager.try_free_blocks(3))
        self.assertIn(hashes[0], manager.hash_to_block_ids)
        self.assertNotIn(hashes[1], manager.hash_to_block_ids)

    def test_eviction_clears_protection_before_block_id_reuse(self):
        manager = self.make_manager()
        _, hot = publish(manager, [11] * 16)
        self.reuse(manager, hot)
        self.assertTrue(manager.try_free_blocks(4))
        self.assertFalse(manager._protected_blocks)
        self.assertFalse(manager._protected_evictable_blocks)
        for token in range(20, 24):
            publish(manager, [token] * 16)
        self.assertFalse(manager._protected_blocks)
        assert_state(self, manager)

    def test_one_block_pool_has_no_permanent_protection(self):
        manager = self.make_manager(blocks=1)
        _, hashes = publish(manager, [11] * 16)
        self.reuse(manager, hashes)
        self.assertFalse(manager._protected_blocks)
        self.assertTrue(manager.try_free_blocks(1))
        assert_state(self, manager)

    def test_invalid_policy_configuration(self):
        for policy, ratio in (
            ("unknown", 0.5),
            ("slru", 0),
            ("slru", 1),
            ("slru", float("nan")),
        ):
            with self.subTest(policy=policy, ratio=ratio):
                with self.assertRaises(ValueError):
                    BlockManager(
                        4,
                        16,
                        prefix_cache_policy=policy,
                        prefix_cache_protected_ratio=ratio,
                    )

    def test_randomized_shared_lifetimes_preserve_capacity(self):
        rng = random.Random(71)
        manager = self.make_manager(blocks=12)
        held = []
        for _ in range(400):
            action = rng.randrange(4)
            if action == 0 and manager.hash_to_block_ids:
                key = rng.choice(list(manager.hash_to_block_ids))
                table, _ = manager.get_computed_blocks([key], 16)
                if rng.choice((True, False)):
                    manager.record_cache_hit(table)
                held.append(table)
            elif action == 1 and held:
                manager.free_blocks(held.pop(rng.randrange(len(held))))
            elif action == 2 and manager.get_total_usable_blocks():
                publish(manager, [rng.randrange(10, 1000)] * 16)
            else:
                manager.try_free_blocks(rng.randrange(1, 13))
            assert_state(self, manager)
            self.assertLessEqual(len(manager._protected_blocks), 6)
        for table in held:
            manager.free_blocks(table)
        self.assertTrue(manager.try_free_blocks(12))
        assert_state(self, manager)
