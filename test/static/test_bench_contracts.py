import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class BenchContractsTest(unittest.TestCase):
    def test_paged_warmup_reuses_the_benchmark_cache(self) -> None:
        source = (ROOT / "examples/bench.py").read_text(encoding="utf-8")
        warmup = source[
            source.index("#                                Warmup") : source.index(
                "#                                Warmup done"
            )
        ]

        self.assertNotIn("PagedKVCacheConfig(", warmup)
        self.assertIn("if not enable_paged_attn:", warmup)
        self.assertEqual(warmup.count("test.model.reset_cache("), 2)


if __name__ == "__main__":
    unittest.main()
