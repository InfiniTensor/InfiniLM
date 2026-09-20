import unittest

from run_infinilm_lfm2_real import _compare_reference, resolve_attention_backend


class Lfm2RunConfigTest(unittest.TestCase):
    def test_default_backend_matches_cache_layout(self):
        self.assertEqual(resolve_attention_backend("static", "default"), "static-attn")
        self.assertEqual(resolve_attention_backend("paged", "default"), "paged-attn")
        self.assertEqual(resolve_attention_backend("paged", "flash-attn"), "flash-attn")

    def test_invalid_backend_pairs_are_rejected(self):
        for cache_type, backend in (
            ("paged", "static-attn"),
            ("static", "paged-attn"),
            ("static", "flash-attn"),
            ("paged", "unknown"),
            ("unknown", "default"),
        ):
            with self.subTest(cache_type=cache_type, backend=backend):
                with self.assertRaises(ValueError):
                    resolve_attention_backend(cache_type, backend)

    def test_each_reference_prompt_is_checked_independently(self):
        runs = [
            {"prompt": "A", "prompt_token_ids": [1, 2], "generated_token_ids": [3, 4]},
            {"prompt": "B", "prompt_token_ids": [1, 5], "generated_token_ids": [6, 7]},
        ]
        good = {"prompt": "A", "input_ids": [1, 2], "generated_token_ids": [3, 4]}
        bad = {"prompt": "B", "input_ids": [1, 5], "generated_token_ids": [6, 8]}
        self.assertTrue(_compare_reference(good, runs)["passed"])
        self.assertFalse(_compare_reference(bad, runs)["passed"])
        self.assertFalse(_compare_reference({"prompt": "missing"}, runs)["passed"])
        self.assertIsNone(_compare_reference(None, runs))


if __name__ == "__main__":
    unittest.main()
