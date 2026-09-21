import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

# Keep native NumPy modules alive when restoring the isolated package imports.
import numpy as np  # noqa: F401


def load_modules():
    source = Path(__file__).resolve().parents[2] / "python/infinilm/llm"
    modules = {}
    with patch.dict(sys.modules):
        for name in (
            "prefix_cache",
            "sampling_params",
            "request",
            "cache_manager",
            "scheduler",
        ):
            fullname = f"infinilm.llm.{name}"
            spec = importlib.util.spec_from_file_location(
                fullname, source / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module
            spec.loader.exec_module(module)
            modules[name] = module
    return modules


MODULES = load_modules()
BlockManager = MODULES["cache_manager"].BlockManager


def chain_hashes(tokens, block_size=16):
    hashes = []
    parent = MODULES["prefix_cache"].EMPTY_BLOCK_HASH
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = MODULES["prefix_cache"].hash_block_tokens(
            tokens[start : start + block_size], parent
        )
        hashes.append(parent)
    return hashes


def publish(manager, tokens):
    table, _ = manager.allocate_slots(len(tokens))
    hashes = chain_hashes(tokens, manager.block_size)
    manager.publish_computed_blocks(table, hashes, 0, len(tokens))
    manager.free_blocks(table)
    return table, hashes


def assert_state(case, manager):
    free = list(manager.free_block_ids)
    used = set(manager.used_block_ids)
    case.assertEqual(len(free), len(set(free)))
    case.assertFalse(set(free) & used)
    case.assertEqual(set(free) | used, set(range(manager.num_blocks)))
    evictable = set()
    index = {}
    for block in manager.blocks:
        case.assertGreaterEqual(block.ref_count, 0)
        if block.block_id in free:
            case.assertEqual((block.ref_count, block.hash), (0, b""))
        elif block.ref_count == 0:
            case.assertNotEqual(block.hash, b"")
            evictable.add(block.block_id)
        if block.hash:
            case.assertIn(block.block_id, used)
            index.setdefault(block.hash, set()).add(block.block_id)
    probationary = set(manager._evictable_blocks)
    protected = set(manager._protected_evictable_blocks)
    case.assertFalse(probationary & protected)
    case.assertEqual(probationary | protected, evictable)
    case.assertTrue(protected <= set(manager._protected_blocks) <= used)
    case.assertFalse(probationary & set(manager._protected_blocks))
    case.assertLessEqual(len(manager._protected_blocks), manager._protected_capacity)
    for block_id in manager._protected_blocks:
        case.assertTrue(manager.blocks[block_id].hash)
        case.assertEqual(manager.blocks[block_id].ref_count == 0, block_id in protected)
    case.assertEqual(manager.hash_to_block_ids, index)
    case.assertEqual(manager.get_total_usable_blocks(), len(free) + len(evictable))
