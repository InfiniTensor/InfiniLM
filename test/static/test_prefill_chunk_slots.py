"""Exercise the Python chunk metadata without loading a device extension."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[2]


def load_method(path, class_name, name):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == name)
    method.decorator_list = []
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace[name]


advance = load_method("python/infinilm/llm/llm.py", "LLMEngine", "_advance_chunk_progress")
append_metadata = load_method(
    "python/infinilm/processors/basic_llm_processor.py",
    "BasicLLMProcessor", "_append_prefill_row_metadata",
)


class PrefillChunkSlotsTest(unittest.TestCase):
    def make_request(self, length, cached=0, offset=0):
        # Nonconsecutive physical blocks catch accidental logical-slot indexing.
        slots = [(7, 2, 11, 4, 13, 1, 9, 5)[i // 256] * 256 + i % 256
                 for i in range(length)]
        request = SimpleNamespace(
            num_local_cached_tokens=cached, num_computed_tokens=cached,
            chunk_prefill_offset=offset, slot_mapping=slots[cached:],
            get_prefill_tokens=lambda: list(range(length)),
        )
        return request, slots

    def check_chunks(self, length, cached, offset, chunks):
        request, slots = self.make_request(length, cached, offset)
        engine = SimpleNamespace(scheduler=SimpleNamespace(commit_computed_tokens=Mock()))
        start = offset or cached
        for index, count in enumerate(chunks):
            end = start + count
            metadata = {name: [] for name in (
                "tokens", "seq_lens", "slot_mapping", "cached_lens",
                "position_ids", "is_final_prefill_chunk",
            )}
            metadata.update(seq_offsets=[0], cu_seqlens=[0])
            final = index == len(chunks) - 1
            row = SimpleNamespace(request=request, num_scheduled_tokens=count,
                                  is_final_prefill_chunk=final)
            append_metadata(row, **metadata)
            self.assertEqual(metadata["slot_mapping"], slots[start:end])
            self.assertEqual(metadata["position_ids"], list(range(start, end)))
            self.assertEqual(metadata["cached_lens"], [start])
            self.assertEqual(metadata["tokens"], list(range(start, end)))
            if not final:
                advance(engine, request, count)
                self.assertEqual(request.chunk_prefill_offset, end)
                self.assertEqual(request.num_local_cached_tokens, end)
                self.assertEqual(request.num_computed_tokens, end)
                self.assertEqual(request.slot_mapping, slots[end:])
                engine.scheduler.commit_computed_tokens.assert_called_with(request, end)
            start = end
        self.assertEqual(start, length)

    def test_successive_chunks_keep_distinct_physical_slots(self):
        self.check_chunks(1400, 0, 0, [512, 512, 376])

    def test_cached_prefix_with_budget_limited_first_chunk(self):
        self.check_chunks(900, 300, 0, [127, 256, 217])

    def test_resumed_chunk_with_cached_prefix(self):
        self.check_chunks(1400, 512, 512, [511, 377])

    def test_single_prefill_chunk_preserves_slot_mapping(self):
        self.check_chunks(300, 128, 0, [172])


if __name__ == "__main__":
    unittest.main()
