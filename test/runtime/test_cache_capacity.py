"""Capacity accounting across prefix reuse, eviction and speculative rollback."""

import random
import unittest

from infinilm.llm.cache_manager import BlockManager


class CacheCapacityTest(unittest.TestCase):
    def check_capacity(self, manager):
        reclaimable = {
            bid for bid in manager.used_block_ids
            if manager.blocks[bid].ref_count == 0
        }
        self.assertEqual(manager._reclaimable_block_ids, reclaimable)
        self.assertEqual(manager.get_total_usable_blocks(),
                         len(manager.free_block_ids) + len(reclaimable))
        self.assertFalse(reclaimable.intersection(manager.free_block_ids))
        self.assertTrue(all(b.ref_count >= 0 for b in manager.blocks))

    def test_prefix_reuse_shared_owners_and_eviction(self):
        m = BlockManager(8, 16)
        table, _ = m.allocate_slots(33)
        hashes = [b'a' * 16, b'b' * 16]
        m.publish_computed_blocks(table, hashes, 0, 32)
        m.free_blocks(table)
        self.check_capacity(m)
        self.assertEqual(m.get_total_usable_blocks(), 8)
        first, count = m.get_computed_blocks(hashes, 32)
        self.assertEqual(count, 32)
        second, _ = m.get_computed_blocks(hashes, 32)
        self.check_capacity(m)
        self.assertEqual(first, second)
        m.free_blocks(first)
        self.assertEqual(m.get_total_usable_blocks(), 6)
        m.free_blocks(second)
        self.check_capacity(m)
        self.assertTrue(m.try_free_blocks(8))
        self.check_capacity(m)
        self.assertFalse(m.hash_to_block_ids)
        self.assertFalse(m._reclaimable_block_ids)

    def test_speculative_rollback(self):
        m = BlockManager(8, 16)
        table, _ = m.allocate_slots(15)
        before = m.get_total_usable_blocks()
        table, _ = m.append_slots(table, 16, 20)
        self.check_capacity(m)
        table = m.truncate_blocks_many([table], [15])[0]
        self.check_capacity(m)
        self.assertEqual(m.get_total_usable_blocks(), before)
        m.free_blocks(table)
        self.check_capacity(m)
        self.assertEqual(m.get_total_usable_blocks(), 8)

    def test_overlapping_rollback_is_atomic(self):
        m = BlockManager(8, 16)
        table, _ = m.allocate_slots(33)
        before = m.get_total_usable_blocks()
        with self.assertRaises(RuntimeError):
            m.truncate_blocks_many([table, table], [16, 16])
        self.assertEqual(m.get_total_usable_blocks(), before)
        self.check_capacity(m)
        m.free_blocks(table)
        self.check_capacity(m)

    def test_randomized_allocate_release_and_publish(self):
        rng = random.Random(20261124)
        m = BlockManager(32, 16)
        live = []
        for i in range(2000):
            if live and rng.random() < 0.5:
                m.free_blocks(live.pop(rng.randrange(len(live))))
            else:
                length = rng.randrange(1, 97)
                result = m.allocate_slots(length)
                if result is not None:
                    table, _ = result
                    hashes = [(i * 8 + j + 1).to_bytes(16, 'little')
                              for j in range(length // 16)]
                    m.publish_computed_blocks(table, hashes, 0, length)
                    live.append(table)
                    if hashes:
                        reused, _ = m.get_computed_blocks(hashes, length)
                        live.append(reused)
            self.check_capacity(m)
        for table in live:
            m.free_blocks(table)
            self.check_capacity(m)
        self.assertEqual(m.get_total_usable_blocks(), 32)

    def test_capacity_does_not_scan_used_blocks(self):
        class NoIterationSet(set):
            def __iter__(self):
                raise AssertionError('Capacity lookup scanned the used block set.')
        m = BlockManager(8, 16)
        table, _ = m.allocate_slots(33)
        m.free_blocks(table)
        m.used_block_ids = NoIterationSet(m.used_block_ids)
        self.assertEqual(m.get_total_usable_blocks(), 8)


if __name__ == '__main__':
    unittest.main()
