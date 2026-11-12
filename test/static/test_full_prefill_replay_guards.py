"""Static contracts supplement the retained long-context GPU replay diagnostic."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'csrc/engine/compiler/piecewise_prefill_compiler.cpp'


class FullPrefillReplayGuardsTest(unittest.TestCase):
    def test_capture_records_actual_attention_limits(self):
        text = SOURCE.read_text()
        self.assertIn('graphs.full_max_query_length = capture_meta.max_query_length;', text)
        self.assertIn('graphs.full_max_sequence_length = capture_meta.max_sequence_length;', text)

    def test_length_guard_precedes_device_replay(self):
        text = SOURCE.read_text().split('auto &candidate = compiled_.at(graph_bucket);', 1)[1]
        guard = text.index('runtime_meta.max_query_length > candidate.full_max_query_length')
        replay = text.index('std::get<0>(candidate.full_compiled)->run()')
        self.assertIn('runtime_meta.max_sequence_length > candidate.full_max_sequence_length', text[guard:replay])
        self.assertIn('return std::nullopt;', text[guard:replay])

    def test_query_and_kv_tails_are_independent(self):
        text = SOURCE.read_text()
        self.assertIn('std::vector<int32_t> cu_pad(off_tail, last_kv);', text)
        self.assertIn('cu_tail->data(), cu_pad.data(), cu_pad.size()', text)

    def test_full_padding_is_invalid_and_output_rows_are_live(self):
        text = SOURCE.read_text().split('auto &candidate = compiled_.at(graph_bucket);', 1)[1]
        replay = text.index('std::get<0>(candidate.full_compiled)->run()')
        self.assertIn('set_minus_one(unused_slots);', text[:replay])
        self.assertIn('for (size_t req = 0; req < runtime_n_req; ++req)', text[replay:])


if __name__ == '__main__':
    unittest.main()
