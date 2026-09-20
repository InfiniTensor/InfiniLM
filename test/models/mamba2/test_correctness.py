import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import infinicore
from infinilm.cache import PagedKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file


RUN_GPU_TESTS = os.getenv("INFINILM_RUN_GPU_TESTS") == "1"


def make_tiny_checkpoint(seed=20260920):
    generator = torch.Generator().manual_seed(seed)
    config = {
        "architectures": ["Mamba2ForCausalLM"],
        "model_type": "mamba2",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "state_size": 4,
        "conv_kernel": 4,
        "num_heads": 2,
        "head_dim": 8,
        "num_groups": 1,
        "rms_norm_eps": 1e-5,
        "use_bias": False,
        "use_conv_bias": True,
        "torch_dtype": "float32",
        "max_position_embeddings": 128,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
    }
    vocab = config["vocab_size"]
    hidden = config["hidden_size"]
    inner = config["intermediate_size"]
    heads = config["num_heads"]
    state_size = config["state_size"]
    conv_dim = inner + 2 * state_size
    projection_size = inner + conv_dim + heads

    def randn(*shape, scale=0.08):
        return torch.randn(*shape, generator=generator) * scale

    mixer = "backbone.layers.0.mixer"
    weights = {
        "backbone.embedding.weight": randn(vocab, hidden),
        "backbone.norm_f.weight": torch.ones(hidden),
        "lm_head.weight": randn(vocab, hidden),
        "backbone.layers.0.norm.weight": torch.ones(hidden),
        f"{mixer}.norm.weight": torch.ones(config["head_dim"]),
        f"{mixer}.in_proj.weight": randn(projection_size, hidden),
        f"{mixer}.out_proj.weight": randn(hidden, inner),
        f"{mixer}.conv1d.weight": randn(conv_dim, 1, config["conv_kernel"]),
        f"{mixer}.conv1d.bias": randn(conv_dim, scale=0.01),
        f"{mixer}.A_log": torch.zeros(heads),
        f"{mixer}.D": torch.ones(heads),
        f"{mixer}.dt_bias": torch.zeros(heads),
    }
    return weights, config


class Mamba2Reference:
    def __init__(self, weights, config):
        self.w = weights
        self.config = config
        self.conv_states = {}
        self.ssm_states = {}

    def _state(self, state_id):
        inner = self.config["intermediate_size"]
        dstate = self.config["state_size"]
        heads = self.config["num_heads"]
        head_dim = self.config["head_dim"]
        conv_dim = inner + 2 * dstate
        self.conv_states.setdefault(
            state_id,
            torch.zeros(conv_dim, self.config["conv_kernel"] - 1),
        )
        self.ssm_states.setdefault(
            state_id,
            torch.zeros(heads, head_dim, dstate),
        )
        return self.conv_states[state_id], self.ssm_states[state_id]

    def _rms_norm(self, x, weight):
        eps = self.config["rms_norm_eps"]
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight

    def forward(self, token_ids, state_id):
        c = self.config
        mixer = "backbone.layers.0.mixer"
        conv_state, ssm_state = self._state(state_id)
        x = F.embedding(torch.tensor(token_ids), self.w["backbone.embedding.weight"])
        residual = x
        x = self._rms_norm(x, self.w["backbone.layers.0.norm.weight"])

        projected = F.linear(x, self.w[f"{mixer}.in_proj.weight"])
        inner = c["intermediate_size"]
        dstate = c["state_size"]
        heads = c["num_heads"]
        head_dim = c["head_dim"]
        z = projected[:, :inner]
        xbc = projected[:, inner : inner + inner + 2 * dstate]
        dt = projected[:, inner + inner + 2 * dstate :]

        conv_weight = self.w[f"{mixer}.conv1d.weight"].squeeze(1)
        conv_bias = self.w[f"{mixer}.conv1d.bias"]
        conv_outputs = []
        for token in xbc:
            window = torch.cat((conv_state, token[:, None]), dim=1)
            conv_outputs.append((window * conv_weight).sum(dim=1) + conv_bias)
            conv_state = window[:, 1:].clone()
        conv_outputs = F.silu(torch.stack(conv_outputs))
        x_part = conv_outputs[:, :inner].view(-1, heads, head_dim)
        b_part = conv_outputs[:, inner : inner + dstate]
        c_part = conv_outputs[:, inner + dstate :]

        ys = []
        a = -torch.exp(self.w[f"{mixer}.A_log"])
        d = self.w[f"{mixer}.D"]
        dt_bias = self.w[f"{mixer}.dt_bias"]
        for token_idx in range(len(token_ids)):
            dt_token = F.softplus(dt[token_idx] + dt_bias)
            ssm_state = (
                torch.exp(dt_token[:, None, None] * a[:, None, None]) * ssm_state
                + dt_token[:, None, None]
                * x_part[token_idx][:, :, None]
                * b_part[token_idx][None, None, :]
            )
            y = (ssm_state * c_part[token_idx][None, None, :]).sum(dim=-1)
            y = y + x_part[token_idx] * d[:, None]
            y = self._rms_norm(y, self.w[f"{mixer}.norm.weight"])
            ys.append((y * F.silu(z[token_idx].view(heads, head_dim))).reshape(inner))

        mixed = F.linear(torch.stack(ys), self.w[f"{mixer}.out_proj.weight"])
        self.conv_states[state_id] = conv_state
        self.ssm_states[state_id] = ssm_state
        hidden = residual + mixed
        hidden = self._rms_norm(hidden, self.w["backbone.norm_f.weight"])
        return F.linear(hidden, self.w["lm_head.weight"])


@unittest.skipUnless(
    RUN_GPU_TESTS and torch.cuda.is_available(),
    "set INFINILM_RUN_GPU_TESTS=1 on an NVIDIA host",
)
class Mamba2GPUCorrectnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        model_dir = Path(cls.temp_dir.name)
        cls.weights, cls.config = make_tiny_checkpoint()
        save_file(cls.weights, model_dir / "model.safetensors", metadata={"format": "pt"})
        (model_dir / "config.json").write_text(json.dumps(cls.config))
        cls.engine = InferEngine(
            str(model_dir),
            device=infinicore.device("cuda", 0),
            distributed_config=DistConfig(1),
            cache_config=PagedKVCacheConfig(
                num_blocks=16, block_size=16, max_batch_size=4
            ),
            attention_backend="paged-attn",
            weight_load_mode="sync",
        )
        load_model_state_dict_by_file(
            cls.engine, str(model_dir), dtype=infinicore.float32
        )

    @classmethod
    def tearDownClass(cls):
        del cls.engine
        cls.temp_dir.cleanup()

    def forward(self, requests, initial_rows, final_rows, past_lengths):
        flat_tokens = [token for request in requests for token in request]
        offsets = [0]
        for request in requests:
            offsets.append(offsets[-1] + len(request))
        total_lengths = [
            past + len(request) for past, request in zip(past_lengths, requests)
        ]
        positions = [
            position
            for past, request in zip(past_lengths, requests)
            for position in range(past, past + len(request))
        ]
        cu_seqlens = [0]
        for total_length in total_lengths:
            cu_seqlens.append(cu_seqlens[-1] + total_length)
        output = self.engine.forward_raw(
            infinicore.from_list([flat_tokens], dtype=infinicore.int64),
            position_ids=infinicore.from_list(positions, dtype=infinicore.int64),
            past_kv_lengths=infinicore.from_list(past_lengths, dtype=infinicore.int32),
            total_kv_lengths=infinicore.from_list(total_lengths, dtype=infinicore.int32),
            input_offsets=infinicore.from_list(offsets, dtype=infinicore.int32),
            cu_seqlens=infinicore.from_list(cu_seqlens, dtype=infinicore.int32),
            mamba_init_state_indices=infinicore.from_list(
                initial_rows, dtype=infinicore.int32
            ),
            mamba_final_state_indices=infinicore.from_list(
                final_rows, dtype=infinicore.int32
            ),
            sample_all_positions=True,
        )
        return torch.from_numpy(np.asarray(output["logits"].to_numpy())).reshape(
            len(flat_tokens), -1
        )

    def test_prefill_decode_and_state_isolation(self):
        reference = Mamba2Reference(self.weights, self.config)

        actual = self.forward([[1, 2, 3]], [0], [1], [0])
        expected = reference.forward([1, 2, 3], 1)
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)

        actual = self.forward([[4]], [1], [1], [3])
        expected = reference.forward([4], 1)
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)

        actual = self.forward([[5, 6], [7, 8]], [0, 0], [2, 3], [0, 0])
        expected = torch.cat(
            [reference.forward([5, 6], 2), reference.forward([7, 8], 3)]
        )
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)

        actual = self.forward([[9], [10]], [2, 3], [2, 3], [2, 2])
        expected = torch.cat([reference.forward([9], 2), reference.forward([10], 3)])
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
