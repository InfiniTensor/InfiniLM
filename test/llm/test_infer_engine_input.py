"""Check the Python/native input boundary without constructing a model."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config_test_support import load_module


class InputConversionTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        calls = self.calls

        class NativeEngine:
            @staticmethod
            def Input(input_ids, **kwargs):
                return SimpleNamespace(input_ids=input_ids, **kwargs)

            def forward(self, inputs):
                calls.append(inputs)
                return SimpleNamespace(output_ids=77, logits=88, hidden_states=99)

        replacements = {
            "infinicore": SimpleNamespace(Tensor=lambda value: value),
            "infinilm.cache": SimpleNamespace(PagedKVCacheConfig=object),
            "infinilm.distributed": SimpleNamespace(DistConfig=object),
            "infinilm.lib": SimpleNamespace(
                _infinilm=SimpleNamespace(InferEngine=NativeEngine)
            ),
            "infinilm.exception_utils": SimpleNamespace(
                handle_oom_and_exit=lambda e: None
            ),
            "infinilm.modeling_utils": SimpleNamespace(parse_dtype=object),
        }
        with patch.dict(sys.modules, replacements):
            cls = load_module("infinilm.infer_engine", "infer_engine.py").InferEngine
        self.engine = cls.__new__(cls)

    def test_forward_preserves_tensor_metadata_and_output_modes(self):
        tensor = SimpleNamespace(_underlying=object())
        for prefill_only in (False, True):
            with self.subTest(prefill_only=prefill_only):
                output = self.engine.forward(
                    tensor,
                    position_ids=tensor,
                    past_kv_lengths=tensor,
                    total_kv_lengths=tensor,
                    input_offsets=tensor,
                    cu_seqlens=tensor,
                    block_tables=tensor,
                    slot_mapping=tensor,
                    mamba_init_state_indices=tensor,
                    mamba_final_state_indices=tensor,
                    target_hidden_states=tensor,
                    pixel_values=[tensor],
                    image_bound=tensor,
                    tgt_sizes=[],
                    image_grid_thw=None,
                    image_req_ids=[0],
                    visual_token_ranges=[(0, 1)],
                    temperature=0.0,
                    top_k=3,
                    top_p=0.9,
                    prefill_only=prefill_only,
                )
                call = self.calls[-1]
                for name in (
                    "input_ids",
                    "position_ids",
                    "past_sequence_lengths",
                    "total_sequence_lengths",
                    "input_offsets",
                    "cu_seqlens",
                    "block_tables",
                    "slot_mapping",
                    "mamba_init_state_indices",
                    "mamba_final_state_indices",
                    "target_hidden_states",
                ):
                    self.assertIs(getattr(call, name), tensor._underlying)
                self.assertEqual(call.pixel_values, [tensor._underlying])
                self.assertEqual(call.image_bound, [tensor._underlying])
                self.assertIsNone(call.tgt_sizes)
                self.assertIsNone(call.image_grid_thw)
                self.assertEqual(
                    (call.image_req_ids, call.visual_token_ranges), ([0], [(0, 1)])
                )
                self.assertEqual(
                    (call.temperature, call.top_k, call.top_p), (0.0, 3, 0.9)
                )
                self.assertEqual(call.prefill_only, prefill_only)
                self.assertFalse(call.sample_all_positions)
                self.assertEqual(output, None if prefill_only else 77)

    def test_forward_and_raw_keep_default_sampling_and_output_contracts(self):
        tensor = SimpleNamespace(_underlying=object())
        self.assertEqual(self.engine.forward(tensor), 77)
        self.assertEqual(
            self.engine.forward_raw(tensor),
            {"output_ids": 77, "logits": 88, "hidden_states": 99},
        )
        for call in self.calls:
            self.assertIs(call.input_ids, tensor._underlying)
            self.assertEqual((call.temperature, call.top_k, call.top_p), (1.0, 1, 1.0))
            self.assertFalse(call.prefill_only)
        self.assertFalse(self.calls[0].sample_all_positions)
        self.assertTrue(self.calls[1].sample_all_positions)


if __name__ == "__main__":
    unittest.main()
