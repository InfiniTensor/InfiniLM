"""Successive prefill chunks must populate distinct KV slots, including prefix hits."""
import unittest
from types import SimpleNamespace
from infinilm.processors.basic_llm_processor import BasicLLMProcessor
from infinilm.llm.llm import LLMEngine

class PrefillSlotMapping(unittest.TestCase):
    def check_chunks(self, prefix, chunks):
        length = prefix + sum(chunks)
        physical_slots = [700 + i * 3 for i in range(length - prefix)]
        req = SimpleNamespace(num_local_cached_tokens=prefix,
                              num_computed_tokens=prefix,
                              chunk_prefill_offset=prefix,
                              slot_mapping=physical_slots,
                              get_prefill_tokens=lambda: list(range(length)))
        engine = SimpleNamespace(scheduler=SimpleNamespace(commit_computed_tokens=lambda *_: None))
        seen = []
        for i, size in enumerate(chunks):
            meta = dict(tokens=[], seq_lens=[], seq_offsets=[0], slot_mapping=[],
                        cached_lens=[], position_ids=[], cu_seqlens=[0], is_final_prefill_chunk=[])
            start = prefix + sum(chunks[:i])
            row = SimpleNamespace(request=req, num_scheduled_tokens=size,
                                  is_final_prefill_chunk=i == len(chunks) - 1)
            BasicLLMProcessor._append_prefill_row_metadata(row, **meta)
            self.assertEqual(meta['tokens'], list(range(start, start + size)))
            self.assertEqual(meta['position_ids'], list(range(start, start + size)))
            self.assertEqual(meta['cached_lens'], [start])
            self.assertEqual(meta['seq_lens'], [start + size])
            expected = physical_slots[sum(chunks[:i]):sum(chunks[:i+1])]
            self.assertEqual(meta['slot_mapping'], expected)
            seen.extend(meta['slot_mapping'])
            LLMEngine._advance_chunk_progress(engine, req, size)
        self.assertEqual(seen, physical_slots)
        self.assertEqual(len(set(seen)), len(seen))

    def test_uncached_three_chunks(self): self.check_chunks(0, [4, 4, 2])
    def test_cached_prefix_three_chunks(self): self.check_chunks(3, [4, 4, 2])
    def test_single_chunk(self): self.check_chunks(0, [10])
    def test_single_chunk_cached_prefix(self): self.check_chunks(3, [10])

if __name__ == '__main__': unittest.main()
