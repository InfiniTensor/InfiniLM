"""CPU slot-mapping tests: python -m unittest discover -s test/llm -v."""

import random
import unittest

from cpu_modules import load_cpu_modules

BlockManager = load_cpu_modules().cache_manager.BlockManager


def reference_slots(table, size, start, end):
    return [table[i // size] * size + i % size for i in range(start, end)]


class SlotMappingTest(unittest.TestCase):
    def test_all_intervals_across_reordered_pages(self):
        for size in (1, 2, 3, 16, 256):
            manager = BlockManager(8, size)
            table = [7, 2, 5, 0]
            boundaries = {0, 1, size - 1, size, size + 1, 3 * size - 1, 4 * size}
            for start in boundaries:
                for end in boundaries:
                    with self.subTest(size=size, start=start, end=end):
                        self.assertEqual(
                            manager.update_blocks_slot(table, start, end),
                            reference_slots(table, size, start, end),
                        )

    def test_random_ragged_intervals(self):
        rng = random.Random(20260917)
        for _ in range(2000):
            size = rng.choice((1, 3, 16, 64, 256))
            table = rng.sample(range(100), rng.randint(1, 30))
            start = rng.randrange(len(table) * size + 1)
            end = rng.randint(start, len(table) * size)
            self.assertEqual(
                BlockManager(100, size).update_blocks_slot(table, start, end),
                reference_slots(table, size, start, end),
            )

    def test_allocation_with_partial_remote_and_local_prefix(self):
        manager = BlockManager(12, 4)
        prefix, _ = manager.allocate_slots(8)
        manager.publish_computed_blocks(prefix, [b"a" * 16, b"b" * 16], 0, 8)
        manager.free_blocks(prefix)
        cached, hit = manager.get_computed_blocks([b"a" * 16, b"b" * 16], 8)
        table, slots = manager.allocate_slots(
            7, num_computed_tokens=hit + 3, cached_block_table=cached
        )
        self.assertEqual(table[:2], prefix)
        self.assertEqual(slots, reference_slots(table, 4, 11, 18))
        self.assertEqual(
            manager.update_blocks_slot(table, 9, 18), reference_slots(table, 4, 9, 18)
        )

    def test_speculative_append_and_rollback(self):
        for size in (1, 3, 16):
            manager = BlockManager(20, size)
            length = size + 1
            table, slots = manager.allocate_slots(length)
            self.assertEqual(slots, reference_slots(table, size, 0, length))
            before = list(table)
            self.assertEqual(manager.append_slots(table, length + 1, 0), (before, []))
            table, slots = manager.append_slots(table, length + 1, size * 2 + 1)
            self.assertEqual(
                slots, reference_slots(table, size, length, length + size * 2 + 1)
            )
            table = manager.truncate_blocks(table, length)
            self.assertEqual(table, before)
            table, slots = manager.append_slots(table, length + 1, size + 1)
            self.assertEqual(
                slots, reference_slots(table, size, length, length + size + 1)
            )

    def test_failure_leaves_owned_pages_unchanged(self):
        manager = BlockManager(2, 4)
        table, _ = manager.allocate_slots(8)
        self.assertIsNone(manager.allocate_slots(1))
        with self.assertRaisesRegex(RuntimeError, "No available"):
            manager.append_slots(table, 9, 1)
        self.assertEqual(table, [0, 1])
        self.assertEqual([page.ref_count for page in manager.blocks], [1, 1])


if __name__ == "__main__":
    unittest.main()
