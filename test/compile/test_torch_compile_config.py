# Copyright (c) 2025, InfiniCore
"""Unit tests for PRD-04 TorchCompileConfig ladder and pad table (no GPU)."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest

_COMPILE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../python/infinilm/compile")
)


def _load_compile_module(name: str):
    path = os.path.join(_COMPILE_ROOT, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"infinilm.compile.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[f"infinilm.compile.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


class TorchCompileConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # env.py has no infinilm/infinicore deps; config.py imports env only.
        _load_compile_module("env")
        cls.config = _load_compile_module("config")

    def test_ladder_and_pad_table_8448(self):
        TorchCompileConfig = self.config.TorchCompileConfig
        env = sys.modules["infinilm.compile.env"]
        build_bs_to_padded_bucket = env.build_bs_to_padded_bucket
        compile_buckets = env.compile_buckets
        padded_bucket_for_seq_len = env.padded_bucket_for_seq_len

        max_seq = 8448
        expected = compile_buckets(max_seq)
        self.assertIn(512, expected)
        self.assertIn(8192, expected)
        self.assertIn(8448, expected)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = TorchCompileConfig(
                model_path=os.path.join(
                    os.path.dirname(__file__),
                    "../../..",
                ),
                max_seq_len=max_seq,
                cache_root=tmp,
            )
            self.assertEqual(sorted(cfg.compile_sizes), sorted(expected))

            table = build_bs_to_padded_bucket(list(cfg.compile_sizes))
            self.assertEqual(
                padded_bucket_for_seq_len(128, table, fallback=max_seq), 512
            )
            self.assertEqual(
                padded_bucket_for_seq_len(513, table, fallback=max_seq), 1024
            )
            self.assertEqual(
                padded_bucket_for_seq_len(8192, table, fallback=max_seq), 8192
            )
            self.assertEqual(
                padded_bucket_for_seq_len(8193, table, fallback=max_seq), 8448
            )
            self.assertEqual(
                padded_bucket_for_seq_len(8448, table, fallback=max_seq), 8448
            )


if __name__ == "__main__":
    unittest.main()
