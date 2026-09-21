import random
import unittest

from cache_test_support import (
    BlockManager,
    assert_state,
    chain_hashes,
    publish,
)


class CachePolicyTests(unittest.TestCase):
    def test_recently_reused_prefix_survives_pressure(self):
        manager = BlockManager(3, 16)
        _, a = publish(manager, [11] * 16)
        _, b = publish(manager, [22] * 16)
        _, c = publish(manager, [33] * 16)
        touched, hit = manager.get_computed_blocks(a, 16)
        self.assertEqual(hit, 16)
        manager.free_blocks(touched)
        self.assertIsNotNone(manager.allocate_slots(16))
        self.assertIn(a[0], manager.hash_to_block_ids)
        self.assertNotIn(b[0], manager.hash_to_block_ids)
        self.assertIn(c[0], manager.hash_to_block_ids)
        assert_state(self, manager)

    def test_request_tail_is_evicted_before_its_prefix(self):
        manager = BlockManager(2, 16)
        _, hashes = publish(manager, [11] * 16 + [22] * 16)
        self.assertIsNotNone(manager.allocate_slots(16))
        pinned, hit = manager.get_computed_blocks(hashes, 32)
        self.assertEqual(hit, 16)
        self.assertNotIn(hashes[1], manager.hash_to_block_ids)
        manager.free_blocks(pinned)
        assert_state(self, manager)


class CacheStateTests(unittest.TestCase):
    def test_shared_pin_survives_until_last_release(self):
        manager = BlockManager(2, 16)
        _, hashes = publish(manager, [11] * 16)
        first, _ = manager.get_computed_blocks(hashes, 16)
        second, _ = manager.get_computed_blocks(hashes, 16)
        pressure, _ = manager.allocate_slots(16)

        manager.free_blocks(first)
        self.assertFalse(manager.try_free_blocks(1))
        self.assertEqual(manager.blocks[second[0]].ref_count, 1)

        manager.free_blocks(second)
        self.assertTrue(manager.try_free_blocks(1))
        self.assertNotIn(hashes[0], manager.hash_to_block_ids)
        manager.free_blocks(pressure)
        assert_state(self, manager)

    def test_duplicate_hash_keeps_other_block_indexed(self):
        manager = BlockManager(2, 16)
        first, _ = manager.allocate_slots(16)
        second, _ = manager.allocate_slots(16)
        block_hash = chain_hashes([7] * 16)
        manager.publish_computed_blocks(first, block_hash, 0, 16)
        manager.publish_computed_blocks(second, block_hash, 0, 16)

        manager.free_blocks(first)
        self.assertTrue(manager.try_free_blocks(1))
        self.assertEqual(manager.hash_to_block_ids[block_hash[0]], {second[0]})
        manager.free_blocks(second)
        assert_state(self, manager)

    def test_unpublished_release_returns_free_capacity(self):
        manager = BlockManager(2, 16)
        table, _ = manager.allocate_slots(17)
        manager.free_blocks(table)

        self.assertEqual(manager.get_num_free_blocks(), 2)
        self.assertEqual(manager.used_block_ids, set())
        self.assertEqual(manager.hash_to_block_ids, {})
        self.assertEqual(manager._evictable_blocks, {})
        assert_state(self, manager)

    def test_partial_publish_retains_only_full_block(self):
        manager = BlockManager(2, 16)
        table, _ = manager.allocate_slots(17)
        hashes = chain_hashes([11] * 16 + [22] * 16)
        manager.publish_computed_blocks(table, hashes, 0, 17)
        manager.free_blocks(table)

        self.assertEqual(manager.get_num_free_blocks(), 1)
        missed, missed_tokens = manager.get_computed_blocks(hashes, 15)
        self.assertEqual((missed, missed_tokens), ([], 0))
        pinned, hit_tokens = manager.get_computed_blocks(hashes, 17)
        self.assertEqual(hit_tokens, 16)
        manager.free_blocks(pinned)
        assert_state(self, manager)

    def test_insufficient_capacity_preserves_pins(self):
        manager = BlockManager(2, 16)
        _, hashes = publish(manager, [11] * 16)
        private, _ = manager.allocate_slots(16)
        private_id = private[0]

        self.assertFalse(manager.try_free_blocks(2))
        self.assertNotIn(hashes[0], manager.hash_to_block_ids)
        self.assertEqual(manager.blocks[private_id].ref_count, 1)
        self.assertEqual(manager.blocks[private_id].block_id, private_id)
        self.assertEqual(manager.get_num_free_blocks(), 1)
        manager.free_blocks(private)
        assert_state(self, manager)

    def test_sufficient_capacity_does_not_evict(self):
        manager = BlockManager(2, 16)
        _, hashes = publish(manager, [11] * 16)

        self.assertTrue(manager.try_free_blocks(1))
        self.assertTrue(manager.try_free_blocks(0))
        self.assertIn(hashes[0], manager.hash_to_block_ids)
        assert_state(self, manager)

    def test_speculative_truncate_releases_private_tail(self):
        manager = BlockManager(3, 16)
        table, _ = manager.allocate_slots(17)
        table, _ = manager.append_slots(table, 18, 16)
        self.assertEqual(len(table), 3)

        retained = manager.truncate_blocks(table, 17)
        self.assertEqual(len(retained), 2)
        self.assertEqual(manager.get_num_free_blocks(), 1)
        self.assertEqual(manager._evictable_blocks, {})
        manager.free_blocks(retained)
        assert_state(self, manager)

    def test_invalid_truncate_is_atomic(self):
        for invalid_kind in (
            "discarded_published",
            "discarded_shared",
            "retained_published",
            "retained_shared",
        ):
            with self.subTest(invalid_kind=invalid_kind):
                manager = BlockManager(4, 16)
                table, _ = manager.allocate_slots(48)
                hashes = chain_hashes([11] * 16 + [22] * 16 + [33] * 16)
                second_owner = []
                if invalid_kind == "discarded_published":
                    manager.publish_computed_blocks(table, hashes, 0, 48)
                    keep_tokens = 16
                elif invalid_kind == "discarded_shared":
                    second_owner = [table[-1]]
                    manager.blocks[second_owner[0]].ref_count += 1
                    keep_tokens = 16
                elif invalid_kind == "retained_published":
                    manager.publish_computed_blocks(table, hashes, 0, 32)
                    keep_tokens = 17
                else:
                    second_owner = [table[1]]
                    manager.blocks[second_owner[0]].ref_count += 1
                    keep_tokens = 17
                before = self._snapshot(manager, table)

                with self.assertRaises(RuntimeError):
                    manager.truncate_blocks(table, keep_tokens)

                self.assertEqual(self._snapshot(manager, table), before)
                manager.free_blocks(table)
                if second_owner:
                    manager.free_blocks(second_owner)
                assert_state(self, manager)

    def test_append_slot_uses_lru_at_boundary(self):
        manager = BlockManager(2, 16)
        old_table, old_hashes = publish(manager, [11] * 16)
        table, _ = manager.allocate_slots(16)

        table, slot = manager.append_slot(table, 17)
        self.assertEqual(slot, old_table[0] * 16)
        self.assertNotIn(old_hashes[0], manager.hash_to_block_ids)
        self.assertEqual(len(table), 2)
        manager.free_blocks(table)
        assert_state(self, manager)

    def test_bounded_legal_operation_sequences_preserve_invariants(self):
        for seed in range(20):
            rng = random.Random(seed)
            manager = BlockManager(8, 16)
            owners = []
            published_sequences = []
            next_token = 1
            history = []
            for step in range(500):
                operation = rng.choice(
                    ("allocate", "publish", "pin", "release", "free")
                )
                try:
                    if operation == "allocate":
                        blocks = rng.randint(1, 3)
                        allocation = manager.allocate_slots(blocks * 16)
                        if allocation is not None:
                            owners.append(allocation[0])
                    elif operation == "publish":
                        candidates = [
                            table
                            for table in owners
                            if all(
                                manager.blocks[block_id].ref_count == 1
                                and not manager.blocks[block_id].hash
                                for block_id in table
                            )
                        ]
                        if candidates:
                            table = rng.choice(candidates)
                            tokens = list(
                                range(next_token, next_token + len(table) * 16)
                            )
                            next_token += len(tokens)
                            hashes = chain_hashes(tokens)
                            manager.publish_computed_blocks(
                                table, hashes, 0, len(tokens)
                            )
                            published_sequences.append(hashes)
                    elif operation == "pin" and published_sequences:
                        hashes = rng.choice(published_sequences)
                        table, _ = manager.get_computed_blocks(hashes, len(hashes) * 16)
                        if table:
                            owners.append(table)
                    elif operation == "release" and owners:
                        owner = owners.pop(rng.randrange(len(owners)))
                        manager.free_blocks(owner)
                    elif operation == "free":
                        manager.try_free_blocks(rng.randint(0, 9))
                    history.append(operation)
                    assert_state(self, manager)
                except Exception as error:
                    self.fail(
                        f"seed={seed} step={step} operation={operation} "
                        f"history={history[-30:]} error={error!r}"
                    )
            for owner in owners:
                manager.free_blocks(owner)
            self.assertEqual(manager.get_total_usable_blocks(), 8, f"seed={seed}")
            assert_state(self, manager)

    @staticmethod
    def _snapshot(manager, table):
        return (
            list(table),
            list(manager.free_block_ids),
            set(manager.used_block_ids),
            list(manager._evictable_blocks),
            {key: set(value) for key, value in manager.hash_to_block_ids.items()},
            [(block.ref_count, block.hash) for block in manager.blocks],
        )


if __name__ == "__main__":
    unittest.main()
