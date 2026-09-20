import unittest

import torch
from infinilm.draft_spec import get_embedded_draft_target_remapper
from infinilm.modeling_utils import (
    _WEIGHT_REMAPPER,
    _remap_qwen3_5,
    _remap_qwen3_5_mtp,
    get_weight_remapper,
)


def build_fake_config():
    """Minimal config mirroring the Qwen3.5-2B checkpoint layout."""
    return {
        "tie_word_embeddings": True,
        "text_config": {
            "hidden_size": 4,
            "tie_word_embeddings": True,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
        },
    }


def build_fake_state_dict():
    """Shard-style state dict: target keys + the 15 embedded mtp.* keys.

    Shapes follow the real checkpoint (hidden_size=4 for cheap asserts).
    """
    embed = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    state_dict = {
        "model.language_model.embed_tokens.weight": embed,
        "model.language_model.norm.weight": torch.ones(4),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.zeros(2, 4),
        "mtp.fc.weight": torch.full((4, 8), 2.0),
        "mtp.pre_fc_norm_embedding.weight": torch.full((4,), 0.5),
        "mtp.pre_fc_norm_hidden.weight": torch.full((4,), 0.5),
        "mtp.norm.weight": torch.full((4,), 1.5),
        "mtp.layers.0.input_layernorm.weight": torch.full((4,), 0.25),
        "mtp.layers.0.post_attention_layernorm.weight": torch.full((4,), 0.25),
        "mtp.layers.0.self_attn.q_proj.weight": torch.full((8, 4), 1.0),
        "mtp.layers.0.self_attn.k_proj.weight": torch.full((4, 4), 1.0),
        "mtp.layers.0.self_attn.v_proj.weight": torch.full((4, 4), 1.0),
        "mtp.layers.0.self_attn.o_proj.weight": torch.full((4, 4), 1.0),
        "mtp.layers.0.self_attn.q_norm.weight": torch.full((2,), 0.5),
        "mtp.layers.0.self_attn.k_norm.weight": torch.full((2,), 0.5),
        "mtp.layers.0.mlp.gate_proj.weight": torch.full((2, 4), 1.0),
        "mtp.layers.0.mlp.up_proj.weight": torch.full((2, 4), 1.0),
        "mtp.layers.0.mlp.down_proj.weight": torch.full((4, 2), 1.0),
    }
    return state_dict


# Every published mtp.* tensor and the draft parameter it has to reach. Listed in
# full rather than sampled: a renamed key that the mapping gets wrong would
# otherwise keep the key count at 15 and slip through.
EXPECTED_MAPPING = {
    "mtp.fc.weight": "model.fc.weight",
    "mtp.pre_fc_norm_embedding.weight": "model.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight": "model.pre_fc_norm_hidden.weight",
    "mtp.norm.weight": "model.norm.weight",
    "mtp.layers.0.input_layernorm.weight": "model.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight": (
        "model.layers.0.post_attention_layernorm.weight"
    ),
    "mtp.layers.0.self_attn.q_proj.weight": "model.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight": "model.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight": "model.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight": "model.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight": "model.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight": "model.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.mlp.gate_proj.weight": "model.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight": "model.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight": "model.layers.0.mlp.down_proj.weight",
}

# The renamed keys whose stored value the mapping adjusts by the zero-centered
# norm offset (everything else is carried over verbatim).
ZERO_CENTERED_DRAFT_KEYS = {
    "model.norm.weight",
    "model.pre_fc_norm_embedding.weight",
    "model.pre_fc_norm_hidden.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.self_attn.q_norm.weight",
    "model.layers.0.self_attn.k_norm.weight",
}


class Qwen35MtpWeightRemapTest(unittest.TestCase):
    def setUp(self):
        self.config = build_fake_config()

    def test_registered_in_remapper(self):
        self.assertIs(_WEIGHT_REMAPPER["qwen3_5_mtp"], _remap_qwen3_5_mtp)

    def test_renames_all_mtp_keys_to_draft_names(self):
        state_dict = build_fake_state_dict()
        remapped = _remap_qwen3_5_mtp(state_dict, self.config)

        # The whole mapping, not a sample: a key renamed to the wrong draft
        # parameter keeps the count at 15, so only the names can catch it.
        for published, canonical in EXPECTED_MAPPING.items():
            self.assertIn(canonical, remapped)
            self.assertNotIn(published, remapped)
        expected_draft_keys = set(EXPECTED_MAPPING.values())
        renamed = set(remapped) - {"model.embed_tokens.weight", "lm_head.weight"}
        self.assertEqual(renamed, expected_draft_keys)
        # Values are carried over unchanged; the only value edit the mapping
        # makes is the zero-centered norm offset, checked on its own below.
        for published, canonical in EXPECTED_MAPPING.items():
            if canonical in ZERO_CENTERED_DRAFT_KEYS:
                continue
            self.assertTrue(
                torch.equal(remapped[canonical], state_dict[published]),
                f"{published} did not reach {canonical} unchanged",
            )

    def test_applies_zero_centered_norm_offset(self):
        remapped = _remap_qwen3_5_mtp(build_fake_state_dict(), self.config)

        self.assertTrue(
            torch.equal(remapped["model.norm.weight"], torch.full((4,), 2.5))
        )
        self.assertTrue(
            torch.equal(
                remapped["model.pre_fc_norm_embedding.weight"], torch.full((4,), 1.5)
            )
        )
        self.assertTrue(
            torch.equal(
                remapped["model.layers.0.self_attn.q_norm.weight"],
                torch.full((2,), 1.5),
            )
        )
        # Linear weights must stay untouched.
        self.assertTrue(
            torch.equal(remapped["model.fc.weight"], torch.full((4, 8), 2.0))
        )

    def test_completes_tied_embedding_and_lm_head(self):
        remapped = _remap_qwen3_5_mtp(build_fake_state_dict(), self.config)

        embed = build_fake_state_dict()["model.language_model.embed_tokens.weight"]
        self.assertTrue(torch.equal(remapped["model.embed_tokens.weight"], embed))
        self.assertTrue(torch.equal(remapped["lm_head.weight"], embed))

    def test_drops_target_weights(self):
        remapped = _remap_qwen3_5_mtp(build_fake_state_dict(), self.config)

        self.assertNotIn("model.language_model.norm.weight", remapped)
        self.assertNotIn(
            "model.language_model.layers.0.self_attn.q_proj.weight", remapped
        )

    def test_handles_shard_without_draft_keys(self):
        shard = {
            "model.language_model.layers.1.mlp.up_proj.weight": torch.zeros(2, 4),
        }
        remapped = _remap_qwen3_5_mtp(shard, self.config)
        self.assertEqual(remapped, {})

    def test_handles_shard_with_only_embedding(self):
        shard = {"model.language_model.embed_tokens.weight": torch.zeros(3, 4)}
        remapped = _remap_qwen3_5_mtp(shard, self.config)
        self.assertEqual(set(remapped), {"model.embed_tokens.weight", "lm_head.weight"})

    def test_rejects_dedicated_draft_embeddings(self):
        config = build_fake_config()
        config["text_config"]["mtp_use_dedicated_embeddings"] = True
        with self.assertRaisesRegex(NotImplementedError, "dedicated draft embeddings"):
            _remap_qwen3_5_mtp(build_fake_state_dict(), config)


class Qwen35TargetPathUnchangedTest(unittest.TestCase):
    """Regression guard: the target remap must keep dropping mtp.* weights."""

    def test_target_remap_still_drops_mtp_keys(self):
        state_dict = build_fake_state_dict()
        remapped = _remap_qwen3_5(
            dict(state_dict),
            {"text_config": {"linear_key_head_dim": 2, "linear_num_key_heads": 1}},
        )

        self.assertEqual([k for k in remapped if k.startswith("mtp.")], [])
        # Non-MTP keys survive unchanged.
        self.assertIn("model.language_model.embed_tokens.weight", remapped)
        self.assertIn("model.language_model.norm.weight", remapped)
        self.assertTrue(
            torch.equal(
                remapped["model.language_model.embed_tokens.weight"],
                state_dict["model.language_model.embed_tokens.weight"],
            )
        )

    def test_target_remap_behavior_is_independent_of_mtp_remap(self):
        # Adding the qwen3_5_mtp entry must not alter the qwen3_5 mapping.
        from infinilm import modeling_utils

        self.assertIs(modeling_utils._WEIGHT_REMAPPER["qwen3_5"], _remap_qwen3_5)
        self.assertIsNot(
            modeling_utils._WEIGHT_REMAPPER["qwen3_5"],
            modeling_utils._WEIGHT_REMAPPER["qwen3_5_mtp"],
        )


class Qwen35TargetSideDescriptionTest(unittest.TestCase):
    """The family description also answers for the checkpoint's target side.

    The registered entry keeps priority, so this pins what the family would get
    without one: exactly the embedded draft tensors removed and every target
    tensor handed over untouched. On the published layout of this fixture the
    entry itself performs no other transformation — the norm offset, the fused
    linear-attention split and the tied head all apply to layouts this fixture
    does not carry — so the two mappings are comparable key for key.
    """

    TARGET_CONFIG = {
        "text_config": {"linear_key_head_dim": 2, "linear_num_key_heads": 1}
    }

    def setUp(self):
        self.entry = _WEIGHT_REMAPPER.pop("qwen3_5")
        self.addCleanup(_WEIGHT_REMAPPER.__setitem__, "qwen3_5", self.entry)

    def test_the_derived_target_mapping_matches_the_entry(self):
        state_dict = build_fake_state_dict()

        from_entry = self.entry(dict(state_dict), self.TARGET_CONFIG)
        from_description = get_weight_remapper("qwen3_5")(
            dict(state_dict), config=self.TARGET_CONFIG
        )

        self.assertIs(
            get_weight_remapper("qwen3_5"),
            get_embedded_draft_target_remapper("qwen3_5"),
        )
        self.assertEqual(set(from_entry), set(from_description))
        for key, tensor in from_entry.items():
            self.assertTrue(
                torch.equal(from_description[key], tensor),
                f"{key} differs between the entry and the description",
            )

    def test_the_derived_target_mapping_removes_only_the_draft_tensors(self):
        state_dict = build_fake_state_dict()

        remapped = get_weight_remapper("qwen3_5")(
            dict(state_dict), config=self.TARGET_CONFIG
        )

        self.assertEqual(
            set(state_dict) - set(remapped),
            {key for key in state_dict if key.startswith("mtp.")},
        )
        for key in remapped:
            self.assertIs(remapped[key], state_dict[key])


if __name__ == "__main__":
    unittest.main()
