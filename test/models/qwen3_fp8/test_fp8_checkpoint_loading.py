import os
import tempfile
import unittest

import torch
from infinilm.modeling_utils import (
    _cast_fp8_scales_to_fp32,
    _resolve_preserve_config,
    load_state_dict,
)
from safetensors.torch import save_file

QWEN3_FP8_HF_CONFIG = {
    "model_type": "qwen3",
    "torch_dtype": "bfloat16",
    "quantization_config": {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    },
}


class ResolvePreserveConfigTest(unittest.TestCase):
    def test_fp8_quant_config_preserves_float8_and_scales(self):
        fp32_suffixes, dtype_suffixes, dtypes = _resolve_preserve_config(
            QWEN3_FP8_HF_CONFIG
        )

        self.assertIn(torch.float8_e4m3fn, dtypes)
        self.assertIn(torch.float8_e5m2, dtypes)
        self.assertIn(".weight_scale_inv", dtype_suffixes)
        self.assertIn(".weight_scale", dtype_suffixes)
        self.assertEqual(fp32_suffixes, (".e_score_correction_bias",))

    def test_plain_config_keeps_default_behavior(self):
        fp32_suffixes, dtype_suffixes, dtypes = _resolve_preserve_config(
            {"model_type": "qwen3"}
        )

        self.assertEqual(fp32_suffixes, (".e_score_correction_bias",))
        self.assertEqual(dtype_suffixes, ())
        self.assertEqual(dtypes, ())



class FP8CheckpointLoadingTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(
            self.temp_dir.name, "model-00001-of-00001.safetensors"
        )

        self.weight = torch.tensor(
            [[1.0, -2.0, 0.5, 0.25], [3.0, -4.0, 5.5, -6.0]],
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn)
        self.scale_inv = torch.tensor([[0.015625]], dtype=torch.float32)
        self.state_dict = {
            "model.layers.0.mlp.gate_proj.weight": self.weight,
            "model.layers.0.mlp.gate_proj.weight_scale_inv": self.scale_inv,
            "model.layers.0.input_layernorm.weight": torch.tensor(
                [0.1, 0.2], dtype=torch.float32
            ),
        }
        save_file(self.state_dict, self.checkpoint_path, metadata={"format": "pt"})

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_fp8_config_preserves_weight_bits_and_scale_dtype(self):
        fp32_suffixes, dtype_suffixes, dtypes = _resolve_preserve_config(
            QWEN3_FP8_HF_CONFIG
        )
        loaded = load_state_dict(
            self.checkpoint_path,
            dtype=torch.bfloat16,
            preserve_fp32_suffixes=fp32_suffixes,
            preserve_dtype_suffixes=dtype_suffixes,
            preserve_dtypes=dtypes,
        )

        weight = loaded["model.layers.0.mlp.gate_proj.weight"]
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        self.assertTrue(
            torch.equal(weight.view(torch.uint8), self.weight.view(torch.uint8))
        )

        scale = loaded["model.layers.0.mlp.gate_proj.weight_scale_inv"]
        self.assertEqual(scale.dtype, torch.float32)
        self.assertTrue(torch.equal(scale, self.scale_inv))

        # Non-quantized float tensors still follow the model compute dtype.
        self.assertEqual(
            loaded["model.layers.0.input_layernorm.weight"].dtype, torch.bfloat16
        )

    def test_default_path_still_casts_float8_weights(self):
        loaded = load_state_dict(self.checkpoint_path, dtype=torch.bfloat16)

        self.assertEqual(
            loaded["model.layers.0.mlp.gate_proj.weight"].dtype, torch.bfloat16
        )
        self.assertEqual(
            loaded["model.layers.0.mlp.gate_proj.weight_scale_inv"].dtype,
            torch.bfloat16,
        )


class CastFP8ScalesToFP32Test(unittest.TestCase):
    def test_bf16_scales_widened_to_fp32(self):
        # Real Qwen3-8B-FP8 checkpoints store weight_scale_inv as BF16.
        scale_inv = torch.tensor([[0.015625, 0.5], [-0.25, 1.0]], dtype=torch.bfloat16)
        weight = torch.zeros(2, 2, dtype=torch.float32).to(torch.float8_e4m3fn)
        params = {
            "model.layers.0.mlp.gate_proj.weight": weight,
            "model.layers.0.mlp.gate_proj.weight_scale_inv": scale_inv,
            "model.layers.0.input_layernorm.weight": torch.tensor(
                [0.1, 0.2], dtype=torch.bfloat16
            ),
        }

        out = _cast_fp8_scales_to_fp32(params, QWEN3_FP8_HF_CONFIG)

        scale = out["model.layers.0.mlp.gate_proj.weight_scale_inv"]
        self.assertEqual(scale.dtype, torch.float32)
        self.assertTrue(torch.equal(scale, scale_inv.to(torch.float32)))
        # FP8 weight bits and unrelated tensors are left untouched.
        self.assertIs(out["model.layers.0.mlp.gate_proj.weight"], weight)
        self.assertEqual(
            out["model.layers.0.input_layernorm.weight"].dtype, torch.bfloat16
        )

    def test_weight_scale_suffix_also_cast(self):
        scale = torch.tensor([0.5], dtype=torch.bfloat16)
        params = {"model.layers.0.self_attn.q_proj.weight_scale": scale}

        out = _cast_fp8_scales_to_fp32(params, QWEN3_FP8_HF_CONFIG)

        self.assertEqual(
            out["model.layers.0.self_attn.q_proj.weight_scale"].dtype,
            torch.float32,
        )

    def test_fp32_scale_kept_as_is(self):
        scale = torch.tensor([[0.015625]], dtype=torch.float32)
        params = {"model.layers.0.mlp.gate_proj.weight_scale_inv": scale}

        out = _cast_fp8_scales_to_fp32(params, QWEN3_FP8_HF_CONFIG)

        self.assertIs(out["model.layers.0.mlp.gate_proj.weight_scale_inv"], scale)

    def test_non_fp8_config_is_noop(self):
        scale = torch.tensor([[0.015625]], dtype=torch.bfloat16)
        params = {"model.layers.0.mlp.gate_proj.weight_scale_inv": scale}

        out = _cast_fp8_scales_to_fp32(params, {"model_type": "qwen3"})

        self.assertIs(out["model.layers.0.mlp.gate_proj.weight_scale_inv"], scale)
        self.assertEqual(scale.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
