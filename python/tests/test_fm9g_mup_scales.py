"""Unit tests for FM9G MuP scale computation (no GPU, no infinicore)."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

_MUP_PATH = (
    Path(__file__).resolve().parents[1] / "infinilm" / "torch_llama" / "mup.py"
)
_spec = importlib.util.spec_from_file_location("torch_llama_mup", _MUP_PATH)
assert _spec and _spec.loader
_mup = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mup
_spec.loader.exec_module(_mup)
Fm9gMupScales = _mup.Fm9gMupScales


class Fm9gMupScalesTest(unittest.TestCase):
    def test_from_fm9g_config_dict(self):
        cfg = {
            "model_type": "fm9g",
            "scale_depth": 2.0,
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "dim_model_base": 256,
        }
        scales = Fm9gMupScales.from_config(cfg)
        self.assertIsNotNone(scales)
        assert scales is not None
        self.assertAlmostEqual(scales.proj_alpha, 2.0 / math.sqrt(32))
        self.assertAlmostEqual(scales.lm_head_alpha, 256.0 / 4096.0)

    def test_non_fm9g_without_mup_fields_returns_none(self):
        self.assertIsNone(Fm9gMupScales.from_config({"model_type": "llama"}))

    def test_llama_with_mup_fields_applies_scales(self):
        cfg = {
            "model_type": "llama",
            "scale_depth": 2.0,
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "dim_model_base": 256,
        }
        scales = Fm9gMupScales.from_config(cfg)
        self.assertIsNotNone(scales)
        assert scales is not None
        self.assertAlmostEqual(scales.proj_alpha, 2.0 / math.sqrt(32))
        self.assertAlmostEqual(scales.lm_head_alpha, 256.0 / 4096.0)

    def test_fm9g7b_model_type(self):
        cfg = {
            "model_type": "fm9g7b",
            "scale_depth": 1.0,
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "dim_model_base": 256,
        }
        scales = Fm9gMupScales.from_config(cfg)
        self.assertIsNotNone(scales)
        assert scales is not None
        self.assertAlmostEqual(scales.lm_head_alpha, 256.0 / 4096.0)


if __name__ == "__main__":
    unittest.main()
