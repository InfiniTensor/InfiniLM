#!/usr/bin/env python3
"""Draft weight injection for MiMo checkpoints.

A MiMo checkpoint publishes its draft block inside the target's own shards under
``model.mtp_layers.<depth>.``; the draft holds its own copy of the target's
embedding table and head rather than a tied one. Both halves are exercised
through the production loader (``load_model_state_dict_by_file``) on a draft
engine built from a tiny synthetic checkpoint, so what the draft holds after
loading is what decides.
"""

import dataclasses
import json
import os
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "llama"))
sys.path.insert(0, _TEST_DIR)

import infinicore  # noqa: E402
from infinilm.cache.cache import StaticKVCacheConfig  # noqa: E402
from infinilm.draft_spec import (  # noqa: E402
    DRAFT_MODEL_SPECS,
    ConcatOrder,
    EmbeddingAtPositionZero,
    LayerSource,
    UnsupportedDraftError,
    _check_description,
    get_draft_weight_remapper,
    register_draft_model_spec,
    resolve_draft,
)
from infinilm.infer_engine import InferEngine  # noqa: E402
from test_forward_validation import (  # noqa: E402
    build_engine,
    build_tiny_config,
    build_tiny_weights,
    remove_tree,
    write_tiny_checkpoint,
)
from utils import infinicore_to_torch_tensor  # noqa: E402

# The published draft namespace and the canonical names it maps onto.
PUBLISHED_PREFIX = "model.mtp_layers.0."
EXPECTED_MAPPING = {
    "token_layernorm.weight": "model.pre_fc_norm_embedding.weight",
    "hidden_layernorm.weight": "model.pre_fc_norm_hidden.weight",
    "input_proj.weight": "model.fc.weight",
    "input_layernorm.weight": "model.input_layernorm.weight",
    "post_attention_layernorm.weight": "model.post_attention_layernorm.weight",
    "final_layernorm.weight": "model.norm.weight",
    "self_attn.q_proj.weight": "model.self_attn.q_proj.weight",
    "self_attn.q_proj.bias": "model.self_attn.q_proj.bias",
    "self_attn.k_proj.weight": "model.self_attn.k_proj.weight",
    "self_attn.k_proj.bias": "model.self_attn.k_proj.bias",
    "self_attn.v_proj.weight": "model.self_attn.v_proj.weight",
    "self_attn.v_proj.bias": "model.self_attn.v_proj.bias",
    "self_attn.o_proj.weight": "model.self_attn.o_proj.weight",
    "mlp.gate_proj.weight": "model.mlp.gate_proj.weight",
    "mlp.up_proj.weight": "model.mlp.up_proj.weight",
    "mlp.down_proj.weight": "model.mlp.down_proj.weight",
}


class MimoDraftDescriptionTest(unittest.TestCase):
    """What the description says about the family, and its key mapping."""

    def setUp(self):
        self.spec = DRAFT_MODEL_SPECS["mimo_mtp"]

    def test_description_records_the_released_semantics(self):
        self.assertEqual(self.spec.draft_model_type, "mimo_mtp")
        self.assertEqual(self.spec.target_model_types, ("mimo",))
        self.assertTrue(self.spec.embedded)
        self.assertTrue(self.spec.is_available)
        self.assertEqual(self.spec.depth_keys, ("num_nextn_predict_layers",))
        self.assertEqual(self.spec.concat_order, ConcatOrder.HIDDEN_FIRST)
        self.assertEqual(
            self.spec.embedding_at_position_zero,
            EmbeddingAtPositionZero.ZEROED,
        )
        # The published fusion norms are ordinary RMSNorm scales: the released
        # checkpoint stores them directly, unlike the zero-centered convention
        # of the other MTP family in this repository.
        self.assertEqual(self.spec.weight_map.zero_centered_keys, ())
        self.assertEqual(self.spec.weight_map.zero_centered_norms, ())

    def test_remapper_moves_every_published_tensor_onto_its_draft_parameter(self):
        weights = build_tiny_weights()
        remap = get_draft_weight_remapper("mimo_mtp")
        remapped = remap(weights, config=build_tiny_config())

        for published, canonical in EXPECTED_MAPPING.items():
            self.assertIn(canonical, remapped)
            self.assertTrue(
                torch.equal(remapped[canonical], weights[PUBLISHED_PREFIX + published]),
                f"{published} did not reach {canonical} unchanged",
            )
        # The embedding and the head come from the checkpoint's own target
        # tensors, not from a share of the draft's own table.
        self.assertTrue(
            torch.equal(
                remapped["model.embed_tokens.weight"],
                weights["model.embed_tokens.weight"],
            )
        )
        self.assertTrue(
            torch.equal(remapped["lm_head.weight"], weights["lm_head.weight"])
        )
        self.assertFalse(
            torch.equal(
                remapped["model.embed_tokens.weight"], remapped["lm_head.weight"]
            )
        )
        self.assertEqual(len(remapped), len(EXPECTED_MAPPING) + 2)

    def test_remapper_leaves_other_shards_alone(self):
        remap = get_draft_weight_remapper("mimo_mtp")
        remapped = remap(
            {"model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4)},
            config=build_tiny_config(),
        )
        self.assertEqual(remapped, {})


class MimoDraftWeightLoadTest(unittest.TestCase):
    """The loaded draft, built and filled by the production path."""

    def setUp(self):
        self.checkpoint = tempfile.mkdtemp(prefix="infinilm_mimo_mtp_load_")
        self.weights = build_tiny_weights()
        write_tiny_checkpoint(self.checkpoint, self.weights)
        self.fixture = None

    def tearDown(self):
        if self.fixture is not None:
            remove_tree(self.fixture)
        remove_tree(self.checkpoint)

    def load(self):
        engine, self.fixture = build_engine(self.checkpoint)
        return engine

    def read_parameter(self, engine, name):
        state_dict = engine.state_dict()[0]
        self.assertIn(name, state_dict)
        return infinicore_to_torch_tensor(state_dict[name], torch.empty(0)).float()

    def test_loaded_draft_holds_the_published_tensors(self):
        engine = self.load()

        for published, canonical in EXPECTED_MAPPING.items():
            expected = self.weights[PUBLISHED_PREFIX + published].float()
            actual = self.read_parameter(engine, canonical)
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertTrue(
                torch.equal(actual, expected),
                f"{canonical} differs from {published}",
            )

    def test_norm_scales_are_loaded_without_a_zero_centered_offset(self):
        engine = self.load()

        for canonical in (
            "model.pre_fc_norm_embedding.weight",
            "model.pre_fc_norm_hidden.weight",
            "model.input_layernorm.weight",
            "model.post_attention_layernorm.weight",
            "model.norm.weight",
        ):
            published = next(
                key for key, value in EXPECTED_MAPPING.items() if value == canonical
            )
            expected = self.weights[PUBLISHED_PREFIX + published].float()
            actual = self.read_parameter(engine, canonical)
            self.assertTrue(torch.equal(actual, expected))
            self.assertFalse(
                torch.allclose(actual, expected + 1.0, rtol=1e-3, atol=1e-3),
                f"{canonical} was shifted as if it were stored around zero",
            )

    def test_attention_biases_reach_the_qkv_projections(self):
        engine = self.load()

        for name in ("q_proj", "k_proj", "v_proj"):
            expected = self.weights[f"{PUBLISHED_PREFIX}self_attn.{name}.bias"].float()
            actual = self.read_parameter(engine, f"model.self_attn.{name}.bias")
            self.assertTrue(torch.equal(actual, expected))
        # The released checkpoint publishes no o_proj bias.
        self.assertNotIn("model.self_attn.o_proj.bias", engine.state_dict()[0])


class MimoDraftResolutionTest(unittest.TestCase):
    """Resolution of the checkpoint against the description."""

    def setUp(self):
        self.checkpoint = tempfile.mkdtemp(prefix="infinilm_mimo_mtp_resolve_")
        self.weights = build_tiny_weights()
        write_tiny_checkpoint(self.checkpoint, self.weights)
        self.fixtures = []

    def tearDown(self):
        for fixture in self.fixtures:
            remove_tree(fixture)
        remove_tree(self.checkpoint)

    def resolve(self, config=None):
        if config is not None:
            write_tiny_checkpoint(self.checkpoint, self.weights, config=config)
        checkpoint = resolve_draft(self.checkpoint)
        self.assertIsNotNone(checkpoint)
        self.fixtures.append(checkpoint.engine_path)
        return checkpoint

    def test_depth_comes_from_the_checkpoint_field(self):
        checkpoint = self.resolve()
        self.assertEqual(checkpoint.published_depth, 1)
        self.assertEqual(checkpoint.spec.family, "mimo_mtp")
        self.assertEqual(checkpoint.depth_source, "weights")

    def test_a_non_positive_depth_is_reported_readably(self):
        # A hand-written draft config can carry a count that is not a positive
        # whole number. The engine must name the value that was written rather
        # than the unsigned wraparound of it, and must not build the block.
        for depth in (-1, 0):
            with self.subTest(depth=depth):
                root = tempfile.mkdtemp(prefix="infinilm_mimo_mtp_depth_")
                self.fixtures.append(root)
                config = build_tiny_config()
                config["model_type"] = "mimo_mtp"
                config["text_config"] = dict(config, num_hidden_layers=depth)
                with open(os.path.join(root, "config.json"), "w") as f:
                    json.dump(config, f)

                with self.assertRaisesRegex(RuntimeError, f"got {depth}") as caught:
                    InferEngine(
                        model_path=root,
                        device=infinicore.device("cpu", 0),
                        cache_config=StaticKVCacheConfig(
                            max_batch_size=1, max_cache_len=128
                        ),
                        attention_backend="default",
                    )
                self.assertIn("positive whole number", str(caught.exception))

    def test_checkpoint_without_draft_weights_is_rejected_by_name(self):
        config = build_tiny_config()
        config["num_nextn_predict_layers"] = 0
        with open(os.path.join(self.checkpoint, "config.json"), "w") as f:
            json.dump(config, f)
        shard = "model-00001-of-00001.safetensors"
        save_file(
            {
                key: value
                for key, value in self.weights.items()
                if not key.startswith(PUBLISHED_PREFIX)
            },
            os.path.join(self.checkpoint, shard),
        )
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {
                key: shard
                for key in self.weights
                if not key.startswith(PUBLISHED_PREFIX)
            },
        }
        with open(
            os.path.join(self.checkpoint, "model.safetensors.index.json"), "w"
        ) as f:
            json.dump(index, f)

        with self.assertRaisesRegex(
            UnsupportedDraftError, "does not publish draft weights"
        ):
            resolve_draft(self.checkpoint)

    def test_a_second_published_depth_is_reported_by_the_engine(self):
        # A checkpoint that publishes more than one depth gives each depth its
        # own weights, which this build's single draft block does not carry; the
        # engine must say so instead of running one depth for all of them.
        weights = dict(self.weights)
        weights["model.mtp_layers.1.input_proj.weight"] = torch.zeros(64, 128)
        config = build_tiny_config()
        config["num_nextn_predict_layers"] = 2
        write_tiny_checkpoint(self.checkpoint, weights, config=config)

        checkpoint = resolve_draft(self.checkpoint)
        self.assertIsNotNone(checkpoint)
        self.fixtures.append(checkpoint.engine_path)
        self.assertEqual(checkpoint.published_depth, 2)
        with self.assertRaisesRegex(RuntimeError, "one published depth"):
            InferEngine(
                model_path=checkpoint.engine_path,
                device=infinicore.device("cpu", 0),
                cache_config=StaticKVCacheConfig(max_batch_size=1, max_cache_len=128),
                attention_backend="default",
            )

    def test_a_depth_mismatch_between_config_and_weights_is_reported(self):
        # The weights publish two depths while the config counts one. The
        # description resolves the depth from the weights and warns about the
        # mismatch; the engine must still refuse to build a block that composes
        # one depth, instead of loading both depths onto the same parameters.
        weights = dict(self.weights)
        weights["model.mtp_layers.1.input_proj.weight"] = torch.zeros(64, 128)
        write_tiny_checkpoint(self.checkpoint, weights, config=build_tiny_config())

        checkpoint = resolve_draft(self.checkpoint)
        self.assertIsNotNone(checkpoint)
        self.fixtures.append(checkpoint.engine_path)
        self.assertEqual(checkpoint.published_depth, 2)
        self.assertEqual(checkpoint.depth_source, "weights")
        with self.assertRaisesRegex(RuntimeError, "one published depth"):
            InferEngine(
                model_path=checkpoint.engine_path,
                device=infinicore.device("cpu", 0),
                cache_config=StaticKVCacheConfig(max_batch_size=1, max_cache_len=128),
                attention_backend="default",
            )


class MimoDraftDescriptionGateTest(unittest.TestCase):
    """The concatenation-order gate only opens for a family's own block.

    A family whose description says the two input streams concatenate the hidden
    state first needs a draft block that runs that order. The families that
    reuse this build's shared draft block keep the embedding-first order, and
    asking one of them for the other order must still be refused.
    """

    def setUp(self):
        self.checkpoint = tempfile.mkdtemp(prefix="infinilm_mimo_mtp_gate_")
        self.weights = build_tiny_weights()
        write_tiny_checkpoint(self.checkpoint, self.weights)
        self.fixtures = []

    def tearDown(self):
        for fixture in self.fixtures:
            remove_tree(fixture)
        remove_tree(self.checkpoint)

    def test_available_families_still_pass_their_own_checks(self):
        for family in ("qwen3_5_mtp", "minicpm_eagle", "mimo_mtp"):
            with self.subTest(family=family):
                _check_description(DRAFT_MODEL_SPECS[family])

    def test_described_families_without_a_block_still_fail(self):
        for family in ("qwen_moe_mtp", "deepseek_v3_mtp"):
            with self.subTest(family=family):
                with self.assertRaisesRegex(
                    UnsupportedDraftError, "cannot construct it yet"
                ):
                    _check_description(DRAFT_MODEL_SPECS[family])

    def test_a_shared_block_family_cannot_ask_for_the_other_order(self):
        spec = DRAFT_MODEL_SPECS["qwen3_5_mtp"]
        self.assertEqual(spec.layer_source, LayerSource.REUSE_TARGET)
        variant = dataclasses.replace(spec, concat_order=ConcatOrder.HIDDEN_FIRST)

        with self.assertRaisesRegex(UnsupportedDraftError, "draft block of its own"):
            _check_description(variant)

    def test_the_gate_keys_on_the_family_block_not_on_the_order_alone(self):
        # The pair pins the gate from both sides: reverting it to "reject every
        # non-embedding-first description" fails the first assertion, widening
        # it to "accept that order for everyone" fails the second.
        hidden_first_shared_block = dataclasses.replace(
            DRAFT_MODEL_SPECS["qwen3_5_mtp"], concat_order=ConcatOrder.HIDDEN_FIRST
        )
        hidden_first_own_block = dataclasses.replace(
            DRAFT_MODEL_SPECS["mimo_mtp"], concat_order=ConcatOrder.HIDDEN_FIRST
        )

        with self.assertRaisesRegex(UnsupportedDraftError, "draft block of its own"):
            _check_description(hidden_first_shared_block)
        _check_description(hidden_first_own_block)

    def test_resolution_refuses_a_family_without_its_own_block(self):
        spec = DRAFT_MODEL_SPECS["mimo_mtp"]
        variant = dataclasses.replace(spec, layer_source=LayerSource.REUSE_TARGET)
        register_draft_model_spec(variant)
        try:
            with self.assertRaisesRegex(
                UnsupportedDraftError, "draft block of its own"
            ):
                resolve_draft(self.checkpoint)
        finally:
            register_draft_model_spec(spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
