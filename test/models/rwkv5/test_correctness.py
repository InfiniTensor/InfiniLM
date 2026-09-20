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


def _make_tiny_checkpoint(seed=20260920):
    generator = torch.Generator().manual_seed(seed)
    vocab_size = 32
    hidden_size = 8
    intermediate_size = 16
    num_heads = 2
    head_dim = hidden_size // num_heads
    num_layers = 2

    def randn(*shape, scale=0.08):
        return torch.randn(*shape, generator=generator) * scale

    state = {
        "emb.weight": randn(vocab_size, hidden_size),
        "blocks.0.ln0.weight": 1 + randn(hidden_size, scale=0.02),
        "blocks.0.ln0.bias": randn(hidden_size, scale=0.02),
        "ln_out.weight": 1 + randn(hidden_size, scale=0.02),
        "ln_out.bias": randn(hidden_size, scale=0.02),
        "head.weight": randn(vocab_size, hidden_size),
    }
    for layer_idx in range(num_layers):
        block = f"blocks.{layer_idx}"
        state.update(
            {
                f"{block}.ln1.weight": 1 + randn(hidden_size, scale=0.02),
                f"{block}.ln1.bias": randn(hidden_size, scale=0.02),
                f"{block}.ln2.weight": 1 + randn(hidden_size, scale=0.02),
                f"{block}.ln2.bias": randn(hidden_size, scale=0.02),
                f"{block}.att.time_mix_k": torch.rand(
                    1, 1, hidden_size, generator=generator
                ),
                f"{block}.att.time_mix_v": torch.rand(
                    1, 1, hidden_size, generator=generator
                ),
                f"{block}.att.time_mix_r": torch.rand(
                    1, 1, hidden_size, generator=generator
                ),
                f"{block}.att.time_mix_g": torch.rand(
                    1, 1, hidden_size, generator=generator
                ),
                f"{block}.att.time_decay": randn(
                    num_heads, head_dim, scale=0.4
                )
                - 2.0,
                f"{block}.att.time_faaaa": torch.rand(
                    num_heads, head_dim, generator=generator
                ),
                f"{block}.att.key.weight": randn(hidden_size, hidden_size),
                f"{block}.att.value.weight": randn(hidden_size, hidden_size),
                f"{block}.att.receptance.weight": randn(
                    hidden_size, hidden_size
                ),
                f"{block}.att.gate.weight": randn(hidden_size, hidden_size),
                f"{block}.att.output.weight": randn(hidden_size, hidden_size),
                f"{block}.att.ln_x.weight": 1
                + randn(hidden_size, scale=0.02),
                f"{block}.att.ln_x.bias": randn(hidden_size, scale=0.02),
                f"{block}.ffn.time_mix_k": torch.rand(
                    1, 1, hidden_size, generator=generator
                ),
                f"{block}.ffn.time_mix_r": torch.rand(
                    1, 1, hidden_size, generator=generator
                ),
                f"{block}.ffn.key.weight": randn(
                    intermediate_size, hidden_size
                ),
                f"{block}.ffn.value.weight": randn(
                    hidden_size, intermediate_size
                ),
                f"{block}.ffn.receptance.weight": randn(
                    hidden_size, hidden_size
                ),
            }
        )

    config = {
        "architectures": ["RWKV5ForCausalLM"],
        "model_type": "rwkv5",
        "rwkv_version": "5.2",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "max_position_embeddings": 128,
        "layer_norm_eps": 1e-5,
        "group_norm_eps": 64e-5,
        "use_attention_gate": True,
        "torch_dtype": "float32",
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
    }
    return state, config


class TorchRWKV5Reference:
    def __init__(self, weights, config):
        self.weights = weights
        self.config = config
        self.states = {}

    def _new_state(self):
        h = self.config["num_attention_heads"]
        n = self.config["head_dim"]
        c = self.config["hidden_size"]
        return [
            {
                "att_prev": torch.zeros(c),
                "wkv": torch.zeros(h, n, n),
                "ffn_prev": torch.zeros(c),
            }
            for _ in range(self.config["num_hidden_layers"])
        ]

    @staticmethod
    def _mix(previous, current, amount):
        return torch.lerp(previous, current, amount)

    def forward(self, token_ids, state_id):
        state = self.states.setdefault(state_id, self._new_state())
        w = self.weights
        c = self.config["hidden_size"]
        h = self.config["num_attention_heads"]
        n = self.config["head_dim"]
        eps = self.config["layer_norm_eps"]
        group_eps = self.config["group_norm_eps"]

        x = F.embedding(torch.tensor(token_ids), w["emb.weight"])
        x = F.layer_norm(
            x,
            (c,),
            w["blocks.0.ln0.weight"],
            w["blocks.0.ln0.bias"],
            eps,
        )

        for layer_idx, layer_state in enumerate(state):
            block = f"blocks.{layer_idx}"
            att = f"{block}.att"
            ffn = f"{block}.ffn"

            xx = F.layer_norm(
                x, (c,), w[f"{block}.ln1.weight"], w[f"{block}.ln1.bias"], eps
            )
            previous = torch.cat([layer_state["att_prev"][None], xx[:-1]], dim=0)
            layer_state["att_prev"] = xx[-1].clone()
            k = F.linear(
                self._mix(previous, xx, w[f"{att}.time_mix_k"].reshape(c)),
                w[f"{att}.key.weight"],
            )
            v = F.linear(
                self._mix(previous, xx, w[f"{att}.time_mix_v"].reshape(c)),
                w[f"{att}.value.weight"],
            )
            r = F.linear(
                self._mix(previous, xx, w[f"{att}.time_mix_r"].reshape(c)),
                w[f"{att}.receptance.weight"],
            )
            g = F.silu(
                F.linear(
                    self._mix(previous, xx, w[f"{att}.time_mix_g"].reshape(c)),
                    w[f"{att}.gate.weight"],
                )
            )

            decay = torch.exp(-torch.exp(w[f"{att}.time_decay"]))[:, :, None]
            first = w[f"{att}.time_faaaa"][:, :, None]
            time_out = []
            for token_idx in range(len(token_ids)):
                rt = r[token_idx].reshape(h, n)
                kt = k[token_idx].reshape(h, n)
                vt = v[token_idx].reshape(h, n)
                kv = kt[:, :, None] * vt[:, None, :]
                yt = torch.matmul(
                    rt[:, None, :], first * kv + layer_state["wkv"]
                ).squeeze(1)
                time_out.append(yt.reshape(c))
                layer_state["wkv"] = kv + decay * layer_state["wkv"]
            time_out = torch.stack(time_out)
            time_out = F.group_norm(
                time_out,
                num_groups=h,
                weight=w[f"{att}.ln_x.weight"],
                bias=w[f"{att}.ln_x.bias"],
                eps=group_eps,
            )
            x = x + F.linear(time_out * g, w[f"{att}.output.weight"])

            xx = F.layer_norm(
                x, (c,), w[f"{block}.ln2.weight"], w[f"{block}.ln2.bias"], eps
            )
            previous = torch.cat([layer_state["ffn_prev"][None], xx[:-1]], dim=0)
            layer_state["ffn_prev"] = xx[-1].clone()
            k = F.linear(
                self._mix(previous, xx, w[f"{ffn}.time_mix_k"].reshape(c)),
                w[f"{ffn}.key.weight"],
            )
            r = F.linear(
                self._mix(previous, xx, w[f"{ffn}.time_mix_r"].reshape(c)),
                w[f"{ffn}.receptance.weight"],
            )
            x = x + torch.sigmoid(r) * F.linear(
                torch.relu(k).square(), w[f"{ffn}.value.weight"]
            )

        x = F.layer_norm(
            x, (c,), w["ln_out.weight"], w["ln_out.bias"], eps
        )
        return F.linear(x, w["head.weight"])


@unittest.skipUnless(
    RUN_GPU_TESTS and torch.cuda.is_available(),
    "set INFINILM_RUN_GPU_TESTS=1 on an NVIDIA host",
)
class RWKV5GPUCorrectnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        model_dir = Path(cls.temp_dir.name)
        cls.weights, cls.config = _make_tiny_checkpoint()
        save_file(cls.weights, model_dir / "model.safetensors", metadata={"format": "pt"})
        (model_dir / "config.json").write_text(
            json.dumps(cls.config), encoding="utf-8"
        )
        (model_dir / "generation_config.json").write_text(
            json.dumps({"eos_token_id": 0}), encoding="utf-8"
        )

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

    def _forward(self, requests, initial_rows, final_rows, past_lengths):
        flat_tokens = [token for request in requests for token in request]
        offsets = [0]
        for request in requests:
            offsets.append(offsets[-1] + len(request))
        total_lengths = [past + len(request) for past, request in zip(past_lengths, requests)]
        cu_seqlens = [0]
        for total_length in total_lengths:
            cu_seqlens.append(cu_seqlens[-1] + total_length)

        output = self.engine.forward_raw(
            infinicore.from_list([flat_tokens], dtype=infinicore.int64),
            position_ids=infinicore.from_list(
                [
                    position
                    for past, request in zip(past_lengths, requests)
                    for position in range(past, past + len(request))
                ],
                dtype=infinicore.int64,
            ),
            past_kv_lengths=infinicore.from_list(
                past_lengths, dtype=infinicore.int32
            ),
            total_kv_lengths=infinicore.from_list(
                total_lengths, dtype=infinicore.int32
            ),
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

    def test_prefill_decode_and_concurrent_state_isolation(self):
        reference = TorchRWKV5Reference(self.weights, self.config)

        prompt = [3, 7, 11, 5]
        actual = self._forward([prompt], [0], [1], [0])
        expected = reference.forward(prompt, state_id=1)
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)

        actual = self._forward([[9]], [1], [1], [len(prompt)])
        expected = reference.forward([9], state_id=1)
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)

        prompts = [[2, 4, 6], [13, 17]]
        actual = self._forward(prompts, [0, 0], [2, 3], [0, 0])
        expected = torch.cat(
            [
                reference.forward(prompts[0], state_id=2),
                reference.forward(prompts[1], state_id=3),
            ]
        )
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)

        actual = self._forward([[8], [19]], [2, 3], [2, 3], [3, 2])
        expected = torch.cat(
            [
                reference.forward([8], state_id=2),
                reference.forward([19], state_id=3),
            ]
        )
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)


if __name__ == "__main__":
    unittest.main()
