import json
import tempfile
import unittest
from pathlib import Path

import torch

from infinilm.infer_engine import InferEngine, model_uses_mamba_cache
from infinilm.modeling_utils import _remap_rwkv5
from infinilm.processors import AutoInfinilmProcessor
from infinilm.processors.rwkv5_processor import RWKV5Processor, RWKVWorldTokenizer
from scripts.convert_rwkv5_checkpoint import _infer_config


class RWKV5CheckpointTest(unittest.TestCase):
    def _state_dict(self, time_decay, *, gated=False):
        state_dict = {
            "emb.weight": torch.zeros(16, 8),
            "blocks.0.att.time_decay": time_decay,
            "blocks.0.att.time_first": torch.log(
                torch.linspace(0.5, 0.75, time_decay.shape[0])
            ),
            "blocks.0.ffn.key.weight": torch.zeros(16, 8),
        }
        if gated:
            state_dict["blocks.0.att.gate.weight"] = torch.zeros(8, 8)
            state_dict["blocks.0.att.time_mix_g"] = torch.zeros(1, 1, 8)
        return state_dict

    def test_infers_rwkv50_scalar_per_head_layout(self):
        config = _infer_config(
            self._state_dict(torch.tensor([-2.0, -3.0])), context_length=1024
        )

        self.assertEqual(config["rwkv_version"], "5.0")
        self.assertEqual(config["num_attention_heads"], 2)
        self.assertEqual(config["head_dim"], 4)
        self.assertFalse(config["use_attention_gate"])

    def test_infers_rwkv52_per_channel_layout(self):
        config = _infer_config(
            self._state_dict(torch.zeros(2, 4), gated=True), context_length=2048
        )

        self.assertEqual(config["rwkv_version"], "5.2")
        self.assertEqual(config["num_attention_heads"], 2)
        self.assertEqual(config["head_dim"], 4)
        self.assertTrue(config["use_attention_gate"])

    def test_infers_rwkv51_gated_scalar_per_head_layout(self):
        config = _infer_config(
            self._state_dict(torch.zeros(2), gated=True), context_length=2048
        )

        self.assertEqual(config["rwkv_version"], "5.1")
        self.assertTrue(config["use_attention_gate"])

    def test_remaps_and_expands_rwkv50_time_parameters(self):
        state_dict = self._state_dict(torch.tensor([-2.0, -3.0]))
        remapped = _remap_rwkv5(
            state_dict,
            {"num_attention_heads": 2, "head_dim": 4},
        )

        self.assertEqual(remapped["model.blocks.0.att.time_decay"].shape, (2, 4))
        torch.testing.assert_close(
            remapped["model.blocks.0.att.time_decay"][:, 0],
            torch.tensor([-2.0, -3.0]),
        )
        expected_first = torch.linspace(0.5, 0.75, 2)
        torch.testing.assert_close(
            remapped["model.blocks.0.att.time_faaaa"][:, 0], expected_first
        )
        self.assertEqual(remapped["model.embeddings.weight"].shape, (16, 8))

    def test_rejects_invalid_time_parameter_size(self):
        with self.assertRaisesRegex(ValueError, "expected 2 or 8"):
            _remap_rwkv5(
                {"blocks.0.att.time_decay": torch.zeros(3)},
                {"num_attention_heads": 2, "head_dim": 4},
            )


class RWKV5ProcessorTest(unittest.TestCase):
    VOCAB = """1 b'a' 1
2 b'b' 1
3 b'ab' 2
4 '\u4f60' 3
"""

    def test_world_tokenizer_prefers_longest_byte_match(self):
        with tempfile.TemporaryDirectory() as directory:
            vocab_path = Path(directory) / "rwkv_vocab_v20230424.txt"
            vocab_path.write_text(self.VOCAB, encoding="utf-8")
            tokenizer = RWKVWorldTokenizer(vocab_path)

            self.assertEqual(tokenizer.encode("ab\u4f60"), [3, 4])
            self.assertEqual(tokenizer.decode([3, 4]), "ab\u4f60")

    def test_auto_processor_uses_registered_rwkv5_processor(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "rwkv5"}), encoding="utf-8"
            )
            (model_dir / "rwkv_vocab_v20230424.txt").write_text(
                self.VOCAB, encoding="utf-8"
            )

            processor = AutoInfinilmProcessor.from_pretrained(str(model_dir))

            self.assertIsInstance(processor, RWKV5Processor)
            self.assertEqual(processor("ab")["input_ids"], [3])

    def test_rwkv5_uses_request_state_cache(self):
        self.assertTrue(model_uses_mamba_cache({"model_type": "rwkv5"}))

    def test_zero_is_a_valid_eos_token_id(self):
        engine = type("EngineConfigStub", (), {})()
        engine.hf_generation_config = {"eos_token_id": 0}
        engine.hf_config = {"eos_token_id": 7}

        self.assertEqual(InferEngine.eos_token_id.fget(engine), [0])

    def test_eos_is_not_decoded_as_a_byte_token(self):
        with tempfile.TemporaryDirectory() as directory:
            vocab_path = Path(directory) / "rwkv_vocab_v20230424.txt"
            vocab_path.write_text(self.VOCAB, encoding="utf-8")
            tokenizer = RWKVWorldTokenizer(vocab_path)

            self.assertEqual(tokenizer.decode([tokenizer.eos_token_id]), "")


if __name__ == "__main__":
    unittest.main()
