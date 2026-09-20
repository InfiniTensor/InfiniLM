#!/usr/bin/env python3
"""MiMo target-side loading: the embedded draft head is dropped, nothing else.

A MiMo checkpoint publishes its MTP tensors next to the target weights under
``model.mtp_layers.<depth>.``, and the target engine loads the checkpoint as a
whole, so its load-time mapping removes them. Those tensors are not discarded:
the draft engine extracts the same shards through the family description, which
is a mapping of its own.

The cases below run on the shard dictionaries the loader passes in, so what the
target mapping does to each key is what decides.
"""

import unittest

import torch
from infinilm.draft_spec import get_embedded_draft_target_remapper
from infinilm.modeling_utils import (
    _WEIGHT_REMAPPER,
    _remap_mimo,
    get_weight_remapper,
)

DRAFT_PREFIX = "model.mtp_layers.0."
SECOND_DEPTH_PREFIX = "model.mtp_layers.1."


def build_fake_state_dict():
    """Shard-style state dict: target keys + the published draft keys.

    Shapes follow the released checkpoint (hidden_size=4 for cheap asserts).
    """
    return {
        "model.embed_tokens.weight": torch.arange(3 * 4, dtype=torch.float32).reshape(
            3, 4
        ),
        "lm_head.weight": torch.full((3, 4), 2.0),
        "model.norm.weight": torch.full((4,), 1.5),
        "model.layers.0.input_layernorm.weight": torch.full((4,), 0.25),
        "model.layers.0.self_attn.q_proj.weight": torch.full((4, 4), 1.0),
        "model.layers.0.self_attn.q_proj.bias": torch.full((4,), 0.5),
        "model.layers.0.mlp.up_proj.weight": torch.full((8, 4), 1.0),
        DRAFT_PREFIX + "token_layernorm.weight": torch.full((4,), 0.75),
        DRAFT_PREFIX + "hidden_layernorm.weight": torch.full((4,), 0.75),
        DRAFT_PREFIX + "input_proj.weight": torch.full((4, 8), 1.0),
        DRAFT_PREFIX + "final_layernorm.weight": torch.full((4,), 1.25),
        DRAFT_PREFIX + "self_attn.q_proj.bias": torch.full((4,), 0.25),
        DRAFT_PREFIX + "mlp.down_proj.weight": torch.full((4, 8), 1.0),
    }


def target_keys(state_dict):
    return [key for key in state_dict if not key.startswith("model.mtp_layers.")]


class MimoTargetWeightRemapTest(unittest.TestCase):
    """The registered MiMo entry drops the draft head and leaves the rest alone."""

    def test_registered_in_remapper(self):
        self.assertIs(_WEIGHT_REMAPPER["mimo"], _remap_mimo)
        self.assertIs(get_weight_remapper("mimo"), _remap_mimo)

    def test_drops_every_published_draft_key(self):
        state_dict = build_fake_state_dict()

        remapped = _remap_mimo(state_dict)

        self.assertEqual(
            [key for key in remapped if key.startswith("model.mtp_layers.")], []
        )
        self.assertEqual(len(remapped), len(target_keys(state_dict)))

    def test_keeps_the_target_tensors_unchanged(self):
        state_dict = build_fake_state_dict()

        remapped = _remap_mimo(state_dict)

        for key in target_keys(state_dict):
            self.assertIn(key, remapped)
            self.assertIs(remapped[key], state_dict[key])
        # A MiMo checkpoint stores its RMSNorm scales directly; a shift as if
        # they were stored around zero would corrupt every target layer.
        self.assertTrue(
            torch.equal(remapped["model.norm.weight"], torch.full((4,), 1.5))
        )
        self.assertTrue(
            torch.equal(
                remapped["model.layers.0.input_layernorm.weight"],
                torch.full((4,), 0.25),
            )
        )

    def test_keeps_the_embedding_and_the_head(self):
        # The draft shares these two with the target, so the target mapping
        # must not remove or retie them.
        state_dict = build_fake_state_dict()

        remapped = _remap_mimo(state_dict)

        self.assertIs(
            remapped["model.embed_tokens.weight"],
            state_dict["model.embed_tokens.weight"],
        )
        self.assertIs(remapped["lm_head.weight"], state_dict["lm_head.weight"])
        self.assertFalse(
            torch.equal(
                remapped["lm_head.weight"], remapped["model.embed_tokens.weight"]
            )
        )

    def test_drops_a_second_published_depth(self):
        state_dict = build_fake_state_dict()
        state_dict[SECOND_DEPTH_PREFIX + "input_proj.weight"] = torch.full((4, 8), 1.0)

        remapped = _remap_mimo(state_dict)

        self.assertEqual(
            [key for key in remapped if key.startswith("model.mtp_layers.")], []
        )
        self.assertEqual(len(remapped), len(target_keys(state_dict)))

    def test_a_key_outside_the_draft_namespace_survives(self):
        # The drop is scoped to the family's published namespace; a target key
        # that merely spells a similar name must reach the target model.
        state_dict = {
            "model.mtp_layers_extra.weight": torch.ones(4),
            "model.layers.0.mtp_layers.weight": torch.ones(4),
            "model.layers.0.mtp_layers.0.input_proj.weight": torch.ones(4),
        }

        remapped = _remap_mimo(state_dict)

        self.assertEqual(set(remapped), set(state_dict))

    def test_a_shard_without_draft_keys_is_unchanged(self):
        shard = {
            "model.layers.1.mlp.up_proj.weight": torch.zeros(8, 4),
            "model.layers.1.self_attn.o_proj.weight": torch.zeros(4, 4),
        }

        remapped = _remap_mimo(shard)

        self.assertEqual(set(remapped), set(shard))
        for key, tensor in shard.items():
            self.assertIs(remapped[key], tensor)

    def test_the_loader_config_argument_is_not_required(self):
        # The loader calls a table entry with the checkpoint config; this
        # mapping is key-only, so it must also work without one.
        state_dict = build_fake_state_dict()

        self.assertEqual(
            set(_remap_mimo(state_dict, None)), set(_remap_mimo(state_dict))
        )


class MimoTargetPathUnchangedTest(unittest.TestCase):
    """Regression guard: the target path keeps removing the embedded draft head."""

    def test_target_remap_still_drops_the_draft_keys(self):
        state_dict = build_fake_state_dict()

        remapped = get_weight_remapper("mimo")(state_dict, config={})

        self.assertEqual(
            [key for key in remapped if key.startswith("model.mtp_layers.")], []
        )
        # Non-draft keys survive unchanged.
        self.assertIn("model.embed_tokens.weight", remapped)
        self.assertIn("model.layers.0.mlp.up_proj.weight", remapped)

    def test_target_remap_applies_no_other_transformation(self):
        # Adding the entry must not rename, add or re-scale anything: the
        # mapping is exactly the input minus the draft namespace.
        state_dict = build_fake_state_dict()

        remapped = get_weight_remapper("mimo")(state_dict, config={})

        self.assertEqual(set(remapped), set(target_keys(state_dict)))
        for key in target_keys(state_dict):
            self.assertTrue(torch.equal(remapped[key], state_dict[key]))


class MimoTargetSideDescriptionTest(unittest.TestCase):
    """The family description answers for the target side as well.

    The registered entry keeps priority, so the description's answer is what a
    family without one would get. Here the two must agree tensor for tensor,
    which is what makes the description a stand-in for the entry rather than a
    second, divergent rule.
    """

    def setUp(self):
        self.entry = _WEIGHT_REMAPPER.pop("mimo")
        self.addCleanup(_WEIGHT_REMAPPER.__setitem__, "mimo", self.entry)

    def test_the_description_answers_for_the_target(self):
        self.assertIs(
            get_weight_remapper("mimo"), get_embedded_draft_target_remapper("mimo")
        )

    def test_the_derived_mapping_matches_the_registered_entry(self):
        # The shard also carries keys just outside the draft namespace, so a
        # derived rule that reached past the family's own tensors would show up
        # as a difference from the entry rather than passing unnoticed.
        state_dict = build_fake_state_dict()
        state_dict.update(
            {
                "model.mtp_layers_extra.weight": torch.ones(4),
                "model.layers.0.mtp_layers.0.input_proj.weight": torch.ones(4),
                "model.mtp_layers.1.input_proj.weight": torch.full((4, 8), 3.0),
            }
        )

        from_entry = self.entry(dict(state_dict))
        from_description = get_weight_remapper("mimo")(dict(state_dict), config={})

        self.assertEqual(set(from_entry), set(from_description))
        for key, tensor in from_entry.items():
            self.assertIs(from_description[key], tensor)

    def test_the_derived_mapping_keeps_keys_outside_the_draft_namespace(self):
        # The drop is scoped to the family's published namespace; a target key
        # that merely spells a similar name must reach the target model.
        state_dict = {
            "model.mtp_layers_extra.weight": torch.ones(4),
            "model.layers.0.mtp_layers.weight": torch.ones(4),
            "model.layers.0.mtp_layers.0.input_proj.weight": torch.ones(4),
        }

        remapped = get_weight_remapper("mimo")(state_dict, config={})

        self.assertEqual(set(remapped), set(state_dict))

    def test_the_derived_mapping_does_not_require_config(self):
        # The loader calls a mapper with the checkpoint config; this mapping is
        # key-only, so it must also work without one.
        state_dict = build_fake_state_dict()

        self.assertEqual(
            set(get_weight_remapper("mimo")(dict(state_dict))),
            set(get_weight_remapper("mimo")(dict(state_dict), config={})),
        )


if __name__ == "__main__":
    unittest.main()
