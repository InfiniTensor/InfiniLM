import unittest
from types import SimpleNamespace

from infinilm.infer_engine import InferEngine


class EosConfigTest(unittest.TestCase):
    def eos(self, config, generation=None):
        engine = SimpleNamespace(
            hf_config=config, hf_generation_config=generation or {}
        )
        return InferEngine.eos_token_id.fget(engine)

    def test_nested_multimodal_config(self):
        self.assertEqual(self.eos({"text_config": {"eos_token_id": 248044}}), [248044])

    def test_generation_config_has_priority(self):
        self.assertEqual(
            self.eos(
                {"eos_token_id": 1, "text_config": {"eos_token_id": 2}},
                {"eos_token_id": [3, 4]},
            ),
            [3, 4],
        )

    def test_top_level_config_has_priority(self):
        self.assertEqual(
            self.eos({"eos_token_id": 1, "text_config": {"eos_token_id": 2}}), [1]
        )

    def test_missing_config(self):
        self.assertEqual(self.eos({}), [])
        self.assertEqual(self.eos({"text_config": None}), [])


if __name__ == "__main__":
    unittest.main()
