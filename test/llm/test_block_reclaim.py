"""CPU checks for bounded KV page reclamation."""

import random
import unittest

from cpu_modules import load_cpu_modules

BlockManager = load_cpu_modules().cache_manager.BlockManager


class ReferenceBlockManager(BlockManager):
    def try_free_blocks(self, num_required):
        to_free = [
            block_id
            for block_id in self.used_block_ids
            if self.blocks[block_id].ref_count == 0
        ]
        for block_id in to_free:
            self._deallocate_block(block_id)
            if self.can_allocate(num_required):
                return True
        return self.can_allocate(num_required)


def make_manager(manager_type, ref_counts, num_free=0):
    manager = manager_type(len(ref_counts) + num_free, 16)
    for index, ref_count in enumerate(ref_counts):
        block = manager._allocate_block()
        manager.publish_computed_blocks(
            [block.block_id], [(index % 3 + 1).to_bytes(16, "little")], 0, 16
        )
        block.ref_count = ref_count
    return manager


def snapshot(manager):
    return (
        list(manager.free_block_ids),
        manager.used_block_ids.copy(),
        {key: value.copy() for key, value in manager.hash_to_block_ids.items()},
        [(block.ref_count, block.hash) for block in manager.blocks],
    )


class BlockReclaimTest(unittest.TestCase):
    def assert_matches_reference(self, ref_counts, num_free, num_required):
        actual = make_manager(BlockManager, ref_counts, num_free)
        expected = make_manager(ReferenceBlockManager, ref_counts, num_free)
        self.assertEqual(
            actual.try_free_blocks(num_required),
            expected.try_free_blocks(num_required),
        )
        self.assertEqual(snapshot(actual), snapshot(expected))
        return actual

    def test_capacity_boundaries_and_existing_free_pages(self):
        for refs in ([], [0], [1], [0, 1, 0, 2, 0], [1, 2, 3], [0] * 8):
            for num_free in (0, 1, 4):
                if not refs and not num_free:
                    continue
                for required in range(len(refs) + num_free + 3):
                    with self.subTest(refs=refs, free=num_free, required=required):
                        self.assert_matches_reference(refs, num_free, required)

    def test_duplicate_hash_keeps_live_peer(self):
        manager = make_manager(BlockManager, [0, 1, 1, 2])
        shared_hash = manager.blocks[0].hash
        self.assertTrue(manager.try_free_blocks(1))
        self.assertEqual(manager.hash_to_block_ids[shared_hash], {3})
        self.assertEqual(manager.blocks[3].ref_count, 2)
        self.assertEqual(manager.blocks[0].hash, b"")
        self.assertEqual(manager._allocate_block().block_id, 0)
        self.assertEqual(manager.blocks[0].ref_count, 1)

    def test_failure_reclaims_all_available_pages(self):
        manager = self.assert_matches_reference([0, 2, 0, 1, 0], 1, 5)
        self.assertEqual(list(manager.free_block_ids), [5, 0, 2, 4])
        self.assertEqual(manager.used_block_ids, {1, 3})

    def test_stops_reading_pages_after_enough_candidates(self):
        class CountingBlocks(list):
            reads = 0

            def __getitem__(self, index):
                self.reads += 1
                return super().__getitem__(index)

        manager = make_manager(BlockManager, [0] * 1024)
        manager.blocks = CountingBlocks(manager.blocks)
        self.assertTrue(manager.try_free_blocks(2))
        # Two candidate reads plus two deallocations; the tail is untouched.
        self.assertEqual(manager.blocks.reads, 4)
        self.assertEqual(len(manager.used_block_ids), 1022)

    def test_randomized_reclaim_and_reallocate_sequences(self):
        rng = random.Random(725)
        for _ in range(200):
            size = rng.randint(1, 128)
            used = rng.randint(0, size)
            refs = [rng.randrange(4) for _ in range(used)]
            actual = make_manager(BlockManager, refs, size - used)
            expected = make_manager(ReferenceBlockManager, refs, size - used)
            for _ in range(20):
                required = rng.randrange(size + 3)
                self.assertEqual(
                    actual.try_free_blocks(required),
                    expected.try_free_blocks(required),
                )
                self.assertEqual(snapshot(actual), snapshot(expected))
                count = rng.randrange(len(actual.free_block_ids) + 1)
                for _ in range(count):
                    self.assertEqual(
                        actual._allocate_block().block_id,
                        expected._allocate_block().block_id,
                    )
                for block_id in list(actual.used_block_ids):
                    if actual.blocks[block_id].ref_count and rng.random() < 0.3:
                        actual.free_blocks([block_id])
                        expected.free_blocks([block_id])
                self.assertEqual(snapshot(actual), snapshot(expected))


if __name__ == "__main__":
    unittest.main()
