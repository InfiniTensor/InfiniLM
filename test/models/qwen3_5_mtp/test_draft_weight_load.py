#!/usr/bin/env python3
"""Draft weight injection for tied and untied Qwen3.5 checkpoints.

The embedded MTP draft shares the target's embedding table, and its output head
follows the target's ``tie_word_embeddings`` setting: when the target ties the
embedding and the head, the checkpoint stores a single embedding matrix, and
when the target does not tie them, the checkpoint stores ``lm_head.weight`` next
to ``model.language_model.embed_tokens.weight``.

Both layouts are exercised through the production loader
(``load_model_state_dict_by_file``) on a real draft engine built from a tiny
synthetic checkpoint, so the injection is verified by what the draft actually
holds after loading rather than by the remap dictionary alone.
"""

import json
import os
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "llama"))

import infinicore  # noqa: E402
from infinilm.cache.cache import StaticKVCacheConfig  # noqa: E402
from infinilm.infer_engine import InferEngine  # noqa: E402
from infinilm.modeling_utils import (  # noqa: E402
    _remap_qwen3_5_mtp,
    load_model_state_dict_by_file,
)
from utils import infinicore_to_torch_tensor  # noqa: E402

# Tiny draft dimensions; the draft layer is always full attention.
HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 16
INTERMEDIATE_SIZE = 128
VOCAB_SIZE = 128
EMBED_KEY = "model.language_model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"


def build_tiny_text_config():
    """Minimal Qwen3.5 text config mirroring the released checkpoint fields."""
    return {
        "attn_output_gate": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "dtype": "bfloat16",
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "layer_types": ["full_attention"],
        "max_position_embeddings": 128,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": NUM_HEADS,
        "num_hidden_layers": 1,
        "num_key_value_heads": NUM_KV_HEADS,
        "rms_norm_eps": 1e-06,
        "use_cache": True,
        "vocab_size": VOCAB_SIZE,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.5,
        },
    }


def build_tiny_config(tie_word_embeddings):
    """Draft config as the fixture builder writes it.

    The untied layout mirrors the released checkpoints: only the top-level
    ``tie_word_embeddings`` flag is present.
    """
    text_config = build_tiny_text_config()
    if tie_word_embeddings:
        text_config["tie_word_embeddings"] = True
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5_mtp",
        "torch_dtype": "bfloat16",
        "text_config": text_config,
        "tie_word_embeddings": tie_word_embeddings,
    }


def build_tiny_mtp_weights(seed):
    """The 15 embedded ``mtp.*`` tensors with released-checkpoint shapes."""
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.bfloat16

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(dtype)

    q_out = 2 * NUM_HEADS * HEAD_DIM  # attn_output_gate doubles q_proj
    kv_out = NUM_KV_HEADS * HEAD_DIM
    return {
        "mtp.fc.weight": randn(HIDDEN_SIZE, 2 * HIDDEN_SIZE),
        "mtp.pre_fc_norm_embedding.weight": randn(HIDDEN_SIZE),
        "mtp.pre_fc_norm_hidden.weight": randn(HIDDEN_SIZE),
        "mtp.norm.weight": randn(HIDDEN_SIZE),
        "mtp.layers.0.input_layernorm.weight": randn(HIDDEN_SIZE),
        "mtp.layers.0.post_attention_layernorm.weight": randn(HIDDEN_SIZE),
        "mtp.layers.0.self_attn.q_proj.weight": randn(q_out, HIDDEN_SIZE),
        "mtp.layers.0.self_attn.k_proj.weight": randn(kv_out, HIDDEN_SIZE),
        "mtp.layers.0.self_attn.v_proj.weight": randn(kv_out, HIDDEN_SIZE),
        "mtp.layers.0.self_attn.o_proj.weight": randn(
            HIDDEN_SIZE, NUM_HEADS * HEAD_DIM
        ),
        "mtp.layers.0.self_attn.q_norm.weight": randn(HEAD_DIM),
        "mtp.layers.0.self_attn.k_norm.weight": randn(HEAD_DIM),
        "mtp.layers.0.mlp.gate_proj.weight": randn(INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "mtp.layers.0.mlp.up_proj.weight": randn(INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "mtp.layers.0.mlp.down_proj.weight": randn(HIDDEN_SIZE, INTERMEDIATE_SIZE),
    }


def build_tiny_checkpoint(root, tie_word_embeddings):
    """Write a two-shard draft checkpoint; returns the source tensors.

    The embedding/head shard is split from the ``mtp.*`` shards exactly like the
    released multi-shard checkpoints, whose first shard carries the target
    embedding and head while the MTP tensors live in later shards.
    """
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(build_tiny_config(tie_word_embeddings), f)

    generator = torch.Generator().manual_seed(7)
    embed = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, generator=generator).to(torch.bfloat16)
    lm_head = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, generator=generator).to(
        torch.bfloat16
    )

    embedding_shard = "model-00001-of-00002.safetensors"
    embedding_weights = {EMBED_KEY: embed}
    if not tie_word_embeddings:
        embedding_weights[LM_HEAD_KEY] = lm_head
    save_file(embedding_weights, os.path.join(root, embedding_shard))

    draft_shard = "model-00002-of-00002.safetensors"
    draft_weights = build_tiny_mtp_weights(seed=11)
    save_file(draft_weights, os.path.join(root, draft_shard))

    weight_map = {key: embedding_shard for key in embedding_weights}
    weight_map.update({key: draft_shard for key in draft_weights})
    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)

    return {"embed": embed, "lm_head": lm_head}


class Qwen35MtpDraftWeightLoadTest(unittest.TestCase):
    """End-to-end draft loading for both embedding layouts."""

    def setUp(self):
        self.fixtures = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for path in self.fixtures:
            for root, _, files in os.walk(path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                os.rmdir(root)

    def make_checkpoint(self, tie_word_embeddings):
        root = tempfile.mkdtemp(prefix="infinilm_mtp_weight_load_")
        self.fixtures.append(root)
        sources = build_tiny_checkpoint(root, tie_word_embeddings)
        return root, sources

    def load_draft_engine(self, checkpoint_dir):
        engine = InferEngine(
            model_path=checkpoint_dir,
            device=infinicore.device("cpu", 0),
            cache_config=StaticKVCacheConfig(max_batch_size=1, max_cache_len=16),
            attention_backend="default",
        )
        load_model_state_dict_by_file(engine, checkpoint_dir, dtype=engine.dtype)
        return engine

    def read_parameter(self, engine, name):
        state_dict = engine.state_dict()[0]
        self.assertIn(name, state_dict)
        return infinicore_to_torch_tensor(state_dict[name], torch.empty(0)).float()

    def test_tied_checkpoint_shares_the_embedding_with_the_head(self):
        root, sources = self.make_checkpoint(tie_word_embeddings=True)
        self.assertNotIn(LM_HEAD_KEY, self.index_keys(root))

        engine = self.load_draft_engine(root)

        embed = self.read_parameter(engine, "model.embed_tokens.weight")
        lm_head = self.read_parameter(engine, "lm_head.weight")
        self.assertEqual(tuple(embed.shape), (VOCAB_SIZE, HIDDEN_SIZE))
        self.assertTrue(torch.equal(embed, sources["embed"].float()))
        self.assertTrue(torch.equal(lm_head, sources["embed"].float()))

    def test_untied_checkpoint_injects_both_weights(self):
        root, sources = self.make_checkpoint(tie_word_embeddings=False)
        keys = self.index_keys(root)
        self.assertIn(LM_HEAD_KEY, keys)
        self.assertIn(EMBED_KEY, keys)

        engine = self.load_draft_engine(root)

        embed = self.read_parameter(engine, "model.embed_tokens.weight")
        lm_head = self.read_parameter(engine, "lm_head.weight")
        self.assertEqual(tuple(embed.shape), (VOCAB_SIZE, HIDDEN_SIZE))
        self.assertEqual(tuple(lm_head.shape), (VOCAB_SIZE, HIDDEN_SIZE))
        self.assertTrue(torch.equal(embed, sources["embed"].float()))
        self.assertTrue(torch.equal(lm_head, sources["lm_head"].float()))
        # The head must come from lm_head.weight, not from the embedding.
        self.assertFalse(torch.equal(lm_head, embed))

    def index_keys(self, root):
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            return set(json.load(f)["weight_map"])


class Qwen35MtpDraftWeightRemapTest(unittest.TestCase):
    """Shard-level injection: each shard contributes only what it stores."""

    def setUp(self):
        self.config = build_tiny_config(tie_word_embeddings=False)

    def test_untied_shard_maps_embedding_and_head(self):
        embed = torch.arange(VOCAB_SIZE * HIDDEN_SIZE).reshape(VOCAB_SIZE, HIDDEN_SIZE)
        lm_head = torch.full((VOCAB_SIZE, HIDDEN_SIZE), 3.0)
        remapped = _remap_qwen3_5_mtp(
            {EMBED_KEY: embed, LM_HEAD_KEY: lm_head, "mtp.norm.weight": torch.ones(4)},
            self.config,
        )

        self.assertIs(remapped["model.embed_tokens.weight"], embed)
        self.assertIs(remapped["lm_head.weight"], lm_head)
        self.assertTrue(
            torch.equal(remapped["model.norm.weight"], torch.full((4,), 2.0))
        )

    def test_untied_shards_inject_independently(self):
        embed = torch.zeros(VOCAB_SIZE, HIDDEN_SIZE)
        lm_head = torch.ones(VOCAB_SIZE, HIDDEN_SIZE)
        embedding_shard = _remap_qwen3_5_mtp({EMBED_KEY: embed}, self.config)
        draft_shard = _remap_qwen3_5_mtp(
            {"mtp.fc.weight": torch.zeros(4, 8)}, self.config
        )
        head_shard = _remap_qwen3_5_mtp({LM_HEAD_KEY: lm_head}, self.config)

        self.assertEqual(set(embedding_shard), {"model.embed_tokens.weight"})
        self.assertEqual(set(draft_shard), {"model.fc.weight"})
        self.assertEqual(set(head_shard), {"lm_head.weight"})

    def test_untied_shard_without_head_does_not_fall_back_to_tied(self):
        # An untied checkpoint with no head must fail loudly in the loader
        # rather than silently reuse the embedding.
        embed = torch.zeros(VOCAB_SIZE, HIDDEN_SIZE)
        remapped = _remap_qwen3_5_mtp({EMBED_KEY: embed}, self.config)

        self.assertEqual(set(remapped), {"model.embed_tokens.weight"})

    def test_tie_flag_read_from_nested_text_config(self):
        embed = torch.zeros(VOCAB_SIZE, HIDDEN_SIZE)
        remapped = _remap_qwen3_5_mtp(
            {EMBED_KEY: embed}, {"text_config": {"tie_word_embeddings": True}}
        )

        self.assertEqual(set(remapped), {"model.embed_tokens.weight", "lm_head.weight"})
        self.assertIs(remapped["lm_head.weight"], embed)


if __name__ == "__main__":
    unittest.main()
