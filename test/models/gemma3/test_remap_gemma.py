import json
import math
import os
import tempfile
import unittest

import torch
from infinilm.modeling_utils import _get_scale_emb, _remap_gemma


def _make_state_dict():
    return {
        "model.embed_tokens.weight": torch.randn(8, 4),
        "model.norm.weight": torch.zeros(4),
        "model.layers.0.input_layernorm.weight": torch.zeros(4),
        "model.layers.0.post_attention_layernorm.weight": torch.zeros(4),
        "model.layers.0.pre_feedforward_layernorm.weight": torch.zeros(4),
        "model.layers.0.post_feedforward_layernorm.weight": torch.zeros(4),
        "model.layers.0.self_attn.q_norm.weight": torch.zeros(4),
        "model.layers.0.self_attn.k_norm.weight": torch.zeros(4),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(6, 4),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(5, 4),
        "lm_head.weight": torch.randn(8, 4),
    }


class RemapGemmaTest(unittest.TestCase):
    def test_norm_weights_shifted_by_one(self):
        sd = _make_state_dict()
        out = _remap_gemma(sd, None)
        for key in (
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.pre_feedforward_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
        ):
            self.assertTrue(torch.allclose(out[key], torch.ones(4)), key)

    def test_non_norm_weights_untouched(self):
        sd = _make_state_dict()
        out = _remap_gemma(sd, None)
        self.assertTrue(
            torch.allclose(
                out["model.layers.0.self_attn.q_proj.weight"],
                sd["model.layers.0.self_attn.q_proj.weight"],
            )
        )
        self.assertTrue(
            torch.allclose(
                out["model.layers.0.mlp.gate_proj.weight"],
                sd["model.layers.0.mlp.gate_proj.weight"],
            )
        )
        self.assertTrue(
            torch.allclose(
                out["model.embed_tokens.weight"], sd["model.embed_tokens.weight"]
            )
        )

    def test_tied_lm_head_not_shifted(self):
        sd = _make_state_dict()
        out = _remap_gemma(sd, None)
        self.assertTrue(torch.allclose(out["lm_head.weight"], sd["lm_head.weight"]))


class ScaleEmbTest(unittest.TestCase):
    def _write_config(self, config):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(config, f)
        return d

    def test_gemma2_uses_sqrt_hidden(self):
        path = self._write_config({"model_type": "gemma2", "hidden_size": 2304})
        self.assertAlmostEqual(_get_scale_emb(path), math.sqrt(2304))

    def test_gemma3_text_uses_sqrt_hidden(self):
        path = self._write_config({"model_type": "gemma3_text", "hidden_size": 1152})
        self.assertAlmostEqual(_get_scale_emb(path), math.sqrt(1152))

    def test_other_models_default_to_one(self):
        path = self._write_config({"model_type": "llama", "hidden_size": 4096})
        self.assertEqual(_get_scale_emb(path), 1.0)


if __name__ == "__main__":
    unittest.main()
