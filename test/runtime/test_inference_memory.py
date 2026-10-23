"""Opt-in CUDA regressions for component-based prefill and TP weight sharding.

Run with INFINILM_RUN_CUDA_TESTS=1 and one or two visible CUDA devices.
INFINILM_TEST_TP_SIZE defaults to 2. No checkpoint or Torch is required.
"""

import ctypes
import gc
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(
    os.environ.get("INFINILM_RUN_CUDA_TESTS") == "1", "Opt-in CUDA test."
)
class InferenceMemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from infinilm import _core
        from infinilm.cache import PagedKVCacheConfig
        from infinilm.distributed import DistConfig
        from infinilm.infer_engine import InferEngine

        cls.core = _core
        cls.cache_type = PagedKVCacheConfig
        cls.dist_type = DistConfig
        cls.engine_type = InferEngine
        cls.tp = int(os.environ.get("INFINILM_TEST_TP_SIZE", "2"))
        if cls.tp <= 0:
            raise ValueError("This regression requires a positive TP size.")

    def make_engine(
        self,
        directory,
        chunk,
        shard,
        fp32_head=False,
        vocab=256,
        hidden=512,
        family="qwen3",
        embedding_shard=None,
        backend="paged-attn",
    ):
        config = {
            "model_type": family,
            "architectures": [
                {
                    "qwen3": "Qwen3ForCausalLM",
                    "qwen2": "Qwen2ForCausalLM",
                    "llama": "LlamaForCausalLM",
                    "fm9g": "FM9GForCausalLM",
                }[family]
            ],
            "torch_dtype": "bfloat16",
            "hidden_size": hidden,
            "intermediate_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": vocab,
            "max_position_embeddings": 4096,
            "max_window_layers": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "hidden_act": "silu",
            "attention_bias": family == "qwen2",
            "tie_word_embeddings": False,
            "infinilm_prefill_chunk_size": chunk,
            "infinilm_tp_vocab_sharding": shard,
        }
        if family == "fm9g":
            config["dim_model_base"] = hidden * 2
        if embedding_shard is not None:
            config["infinilm_tp_embedding_sharding"] = embedding_shard
        if fp32_head:
            config["lm_head_output_dtype"] = "float32"
        (Path(directory) / "config.json").write_text(json.dumps(config))
        engine = self.engine_type(
            directory,
            device=self.core.device("cuda"),
            distributed_config=self.dist_type(self.tp),
            cache_config=self.cache_type(32, 256, 2),
            enable_graph_compiling=False,
            attention_backend=backend,
        )
        random = np.random.default_rng(773)

        def bf16(values):
            bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
            words = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(
                np.uint16
            )
            return self.core.from_blob(
                words.ctypes.data,
                list(words.shape),
                dtype=self.core.bfloat16,
                device="cpu",
                owner=words,
            )

        weights = {
            "model.embed_tokens.weight": bf16(random.normal(0, 0.03, (vocab, hidden))),
            "lm_head.weight": bf16(random.normal(0, 0.03, (vocab, hidden))),
            "model.norm.weight": bf16(np.ones(hidden)),
        }
        for layer in range(2):
            prefix = f"model.layers.{layer}."
            projections = {
                "self_attn.q_proj.weight": (512, hidden),
                "self_attn.k_proj.weight": (256, hidden),
                "self_attn.v_proj.weight": (256, hidden),
                "self_attn.o_proj.weight": (hidden, 512),
                "mlp.gate_proj.weight": (1024, hidden),
                "mlp.up_proj.weight": (1024, hidden),
                "mlp.down_proj.weight": (hidden, 1024),
            }
            for name, shape in projections.items():
                weights[prefix + name] = bf16(random.normal(0, 0.02, shape))
            if family == "qwen2":
                for name, size in (("q_proj", 512), ("k_proj", 256), ("v_proj", 256)):
                    weights[prefix + f"self_attn.{name}.bias"] = bf16(np.zeros(size))
            for name, size in {
                "input_layernorm.weight": hidden,
                "post_attention_layernorm.weight": hidden,
                "self_attn.q_norm.weight": 128,
                "self_attn.k_norm.weight": 128,
            }.items():
                if family != "qwen3" and name in (
                    "self_attn.q_norm.weight",
                    "self_attn.k_norm.weight",
                ):
                    continue
                weights[prefix + name] = bf16(np.ones(size))
        self.assertEqual(set(weights), set(engine.state_dict_keyname()))
        engine.load_state_dict(weights)
        engine.process_weights_after_loading()
        for rank in engine.state_dict():
            divisor = self.tp if shard else 1
            embedding_divisor = (
                self.tp
                if (shard if embedding_shard is None else embedding_shard)
                else 1
            )
            embedding_divisor = (
                embedding_divisor if hidden % embedding_divisor == 0 else 1
            )
            head_divisor = divisor if not fp32_head and vocab % divisor == 0 else 1
            self.assertEqual(
                list(rank["model.embed_tokens.weight"].shape),
                [vocab, hidden // embedding_divisor],
            )
            self.assertEqual(
                list(rank["lm_head.weight"].shape),
                [vocab // head_divisor, hidden],
            )
        return engine

    def snapshot(self, tensor):
        tensor = tensor.to("cpu").contiguous()
        if tensor.dtype == self.core.bfloat16:
            words = np.empty(tensor.shape, dtype=np.uint16)
            ctypes.memmove(words.ctypes.data, tensor.data_ptr(), words.nbytes)
            return (words.astype(np.uint32) << 16).view(np.float32)
        return tensor.to_numpy().copy()

    def forward(self, engine, sequences, past=None, all_positions=False):
        core = self.core
        past = past or [0] * len(sequences)
        lengths = [len(sequence) for sequence in sequences]
        totals = [p + n for p, n in zip(past, lengths)]
        positions = np.concatenate(
            [np.arange(p, total, dtype=np.int64) for p, total in zip(past, totals)]
        )
        slots = np.concatenate(
            [
                np.arange(2048 * i + p, 2048 * i + total, dtype=np.int64)
                for i, (p, total) in enumerate(zip(past, totals))
            ]
        )
        output = engine.forward_raw(
            core.from_numpy(np.concatenate(sequences).reshape(1, -1).astype(np.int64)),
            position_ids=core.from_numpy(positions.reshape(1, -1)),
            past_kv_lengths=core.from_numpy(np.array(past, np.int32)),
            total_kv_lengths=core.from_numpy(np.array(totals, np.int32)),
            input_offsets=core.from_numpy(np.cumsum([0] + lengths, dtype=np.int32)),
            cu_seqlens=core.from_numpy(np.cumsum([0] + totals, dtype=np.int32)),
            block_tables=core.from_numpy(
                np.array(
                    [list(range(i * 8, (i + 1) * 8)) for i in range(len(sequences))],
                    np.int32,
                )
            ),
            slot_mapping=core.from_numpy(slots),
            sample_all_positions=all_positions,
        )
        return {key: self.snapshot(output[key]) for key in ("hidden_states", "logits")}

    def collect(
        self,
        chunk,
        shard,
        fp32_head=False,
        vocab=256,
        hidden=512,
        family="qwen3",
        embedding_shard=None,
        backend="paged-attn",
    ):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.make_engine(
                directory,
                chunk,
                shard,
                fp32_head,
                vocab,
                hidden,
                family,
                embedding_shard,
                backend,
            )
            random = np.random.default_rng(1063)
            records = {}
            try:
                for length in (127, 128, 129, 257, 513):
                    engine.reset_cache(self.cache_type(32, 256, 2))
                    sequence = random.integers(0, vocab, length, dtype=np.int64)
                    sequence[-1] = vocab - 1
                    records[f"prefill-{length}"] = self.forward(
                        engine, [sequence], all_positions=True
                    )
                    records[f"decode-{length}"] = self.forward(
                        engine, [np.array([7], np.int64)], past=[length]
                    )
                engine.reset_cache(self.cache_type(32, 256, 2))
                records["packed"] = self.forward(
                    engine,
                    [random.integers(0, vocab, n, dtype=np.int64) for n in (129, 257)],
                )
                records["packed-decode"] = self.forward(
                    engine, [np.array([17]), np.array([29])], past=[129, 257]
                )
                engine.reset_cache(self.cache_type(32, 256, 2))
                self.forward(
                    engine,
                    [random.integers(0, vocab, n, dtype=np.int64) for n in (128, 256)],
                )
                records["mixed-prefill-decode"] = self.forward(
                    engine,
                    [np.array([13]), random.integers(0, vocab, 129, dtype=np.int64)],
                    past=[128, 256],
                )
                engine.reset_cache(self.cache_type(32, 256, 2))
                sequence = random.integers(0, vocab, 770, dtype=np.int64)
                self.forward(engine, [sequence[:257]])
                records["continuation"] = self.forward(
                    engine, [sequence[257:]], past=[257], all_positions=True
                )
                engine.reset_cache(self.cache_type(32, 256, 2))
                records["cold"] = self.forward(engine, [sequence], all_positions=True)
            finally:
                del engine
                gc.collect()
            return records

    def assert_close(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for name, values in expected.items():
            for key, reference in values.items():
                with self.subTest(case=name, output=key):
                    candidate = actual[name][key]
                    self.assertEqual(reference.shape, candidate.shape)
                    self.assertTrue(np.isfinite(candidate).all())
                    x = reference.astype(np.float64).ravel()
                    y = candidate.astype(np.float64).ravel()
                    relative_l2 = np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30)
                    cosine = np.dot(x, y) / max(
                        np.linalg.norm(x) * np.linalg.norm(y), 1e-30
                    )
                    self.assertLess(relative_l2, 0.02)
                    self.assertGreater(cosine, 0.999)

    def test_boundaries_packed_prefix_and_opt_out(self):
        reference = self.collect(0, False)
        self.assert_close(reference, self.collect(0, True))
        self.assert_close(reference, self.collect(128, False))
        self.assert_close(reference, self.collect(128, True))

    def test_float32_head_keeps_replicated_storage(self):
        reference = self.collect(0, False, fp32_head=True)
        candidate = self.collect(128, True, fp32_head=True)
        self.assert_close(reference, candidate)

    def test_indivisible_shapes_keep_compatible_storage(self):
        for vocab, hidden in ((257, 512), (256, 513)):
            with self.subTest(vocab=vocab, hidden=hidden):
                reference = self.collect(0, False, vocab=vocab, hidden=hidden)
                candidate = self.collect(128, True, vocab=vocab, hidden=hidden)
                self.assert_close(reference, candidate)

    def test_component_paths_across_families(self):
        for family in ("llama", "qwen2", "fm9g"):
            with self.subTest(family=family):
                reference = self.collect(0, False, family=family)
                candidate = self.collect(128, True, family=family)
                self.assert_close(reference, candidate)

    def test_independent_embedding_and_head_sharding(self):
        reference = self.collect(0, False)
        self.assert_close(reference, self.collect(0, False, embedding_shard=True))
        self.assert_close(reference, self.collect(0, True, embedding_shard=False))

    def test_weight_sharding_without_paged_attention(self):
        reference = self.collect(0, False, family="llama", backend="flash-attn")
        candidate = self.collect(128, True, family="llama", backend="flash-attn")
        self.assert_close(reference, candidate)

    def test_multi_request_batches(self):
        def collect_batches(chunk, shard):
            records = {}
            with tempfile.TemporaryDirectory() as directory:
                engine = self.make_engine(directory, chunk, shard, family="qwen2")
                random = np.random.default_rng(1063)
                try:
                    for batch in (1, 8, 16):
                        engine.reset_cache(self.cache_type(128, 256, 16))
                        sequences = [
                            random.integers(0, 256, 129, dtype=np.int64)
                            for _ in range(batch)
                        ]
                        records[f"prefill-b{batch}"] = self.forward(engine, sequences)
                        records[f"decode-b{batch}"] = self.forward(
                            engine,
                            [np.array([7], np.int64) for _ in range(batch)],
                            past=[129] * batch,
                        )
                finally:
                    del engine
                    gc.collect()
            return records

        self.assert_close(collect_batches(0, False), collect_batches(128, True))


if __name__ == "__main__":
    unittest.main()
