#!/usr/bin/env python3
"""Draft descriptions: registration, resolution, weight mapping, runner wiring.

A new draft family must enter through a description plus a registration entry,
so these tests pin both ends: the registry resolves structural checkpoints
(including families this build cannot run yet) and the speculative runner's
source mentions no family name or family metadata key. The runner-level cases
run on CPU against tiny synthetic checkpoints that use the released key
layouts.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "llama"))
sys.path.insert(0, _TEST_DIR)

import infinicore  # noqa: E402
from infinilm.cache.cache import PagedKVCacheConfig, StaticKVCacheConfig  # noqa: E402
from infinilm.config.engine_config import EngineConfig  # noqa: E402
from infinilm.draft_spec import (  # noqa: E402
    DRAFT_MODEL_SPECS,
    ConcatOrder,
    DraftLayerKind,
    DraftModelSpec,
    DraftWeightMap,
    EmbeddingSharing,
    PositionIdLayout,
    UnsupportedDraftError,
    _literal_prefix,
    draft_position_ids,
    explain_missing_draft,
    get_draft_model_spec,
    get_draft_weight_remapper,
    get_embedded_draft_target_remapper,
    list_draft_model_specs,
    register_draft_model_spec,
    resolve_draft,
    resolve_embedded_draft,
)
from infinilm.llm.model_runner.speculative_runner import (  # noqa: E402
    SpeculativeRunner,
)
from infinilm.llm.static_scheduler import StaticSchedulerOutput  # noqa: E402
from infinilm.modeling_utils import get_weight_remapper  # noqa: E402
from test_draft_weight_load import (  # noqa: E402
    EMBED_KEY,
    HIDDEN_SIZE,
    LM_HEAD_KEY,
    VOCAB_SIZE,
    build_tiny_mtp_weights,
    build_tiny_text_config,
)

MODELS_PATH = os.path.join(_TEST_DIR, "..", "..", "..", "MODELS.md")
RUNNER_SOURCE = os.path.join(
    _TEST_DIR,
    "..",
    "..",
    "..",
    "python",
    "infinilm",
    "llm",
    "model_runner",
    "speculative_runner.py",
)
# Names the runner's judgement used to hardcode; the registry-derived names are
# added by the guard below, so a family registered later is covered too.
FAMILY_TOKENS = ("qwen", "minicpm", "deepseek", "mimo")


def family_tokens():
    """Every name that would make the runner's judgement family-specific."""
    tokens = list(FAMILY_TOKENS)
    for spec in list_draft_model_specs():
        tokens.append(spec.family)
        if spec.draft_model_type:
            tokens.append(spec.draft_model_type)
        tokens.extend(spec.target_model_types)
        tokens.extend(spec.depth_keys)
        if spec.weight_map:
            tokens.extend(
                prefix
                for prefix in (
                    _literal_prefix(spec.weight_map.family_keys),
                    _literal_prefix(spec.weight_map.key_pattern),
                )
                if prefix
            )
    return sorted(set(tokens))


SECOND_FAMILY = "synthetic_second_family"
SECOND_FAMILY_TARGET = "synthetic_target"


def rand(shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)).to(
        torch.bfloat16
    )


def write_checkpoint(root, hf_config, shards):
    """Write config.json plus sharded safetensors and their index."""
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(hf_config, f)
    weight_map = {}
    for shard, tensors in shards.items():
        save_file(tensors, os.path.join(root, shard))
        weight_map.update({key: shard for key in tensors})
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)
    return weight_map


def tiny_qwen_text_config(depth_key="mtp_num_hidden_layers", depth=1):
    text = build_tiny_text_config()
    text.pop("mtp_num_hidden_layers", None)
    if depth_key is not None:
        text[depth_key] = depth
    return text


def qwen_target_config(
    depth_key="mtp_num_hidden_layers", depth=1, tie=True, extra=None
):
    text = tiny_qwen_text_config(depth_key, depth)
    text.update(extra or {})
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "torch_dtype": "bfloat16",
        "text_config": text,
        "tie_word_embeddings": tie,
    }


def qwen_target_shards(tie=True, depth_groups=1, embed_seed=7):
    embedding = {EMBED_KEY: rand((VOCAB_SIZE, HIDDEN_SIZE), embed_seed)}
    if not tie:
        embedding[LM_HEAD_KEY] = rand((VOCAB_SIZE, HIDDEN_SIZE), embed_seed + 1)
    draft = {}
    for group in range(depth_groups):
        for key, tensor in build_tiny_mtp_weights(seed=11 + group).items():
            draft[key.replace("mtp.layers.0.", f"mtp.layers.{group}.")] = tensor
    return {
        "model-00001-of-00002.safetensors": embedding,
        "model-00002-of-00002.safetensors": draft,
    }


def deepseek_shards():
    """The tensors DeepSeek-V3 publishes for its MTP block, scaled down."""
    return {
        "model-00001-of-00001.safetensors": {
            "model.embed_tokens.weight": rand((VOCAB_SIZE, HIDDEN_SIZE), 3),
            "lm_head.weight": rand((VOCAB_SIZE, HIDDEN_SIZE), 4),
            "model.layers.2.enorm.weight": rand((HIDDEN_SIZE,), 5),
            "model.layers.2.hnorm.weight": rand((HIDDEN_SIZE,), 6),
            "model.layers.2.eh_proj.weight": rand((HIDDEN_SIZE, 2 * HIDDEN_SIZE), 7),
            "model.layers.2.shared_head.norm.weight": rand((HIDDEN_SIZE,), 8),
            "model.layers.2.shared_head.head.weight": rand(
                (VOCAB_SIZE, HIDDEN_SIZE), 9
            ),
            "model.layers.2.self_attn.q_a_proj.weight": rand(
                (HIDDEN_SIZE, HIDDEN_SIZE), 10
            ),
        }
    }


class DraftSpecTestCase(unittest.TestCase):
    def setUp(self):
        self.fixtures = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for path in self.fixtures:
            shutil.rmtree(path, ignore_errors=True)

    def make_root(self):
        root = tempfile.mkdtemp(prefix="infinilm_draft_spec_")
        self.fixtures.append(root)
        return root

    def make_qwen_checkpoint(self, **kwargs):
        shards = kwargs.pop("shards", None)
        root = self.make_root()
        write_checkpoint(
            root, qwen_target_config(**kwargs), shards or qwen_target_shards()
        )
        return root


class RunnerSourceTest(unittest.TestCase):
    """The runner must not carry a family's judgement."""

    def test_runner_names_no_family(self):
        with open(RUNNER_SOURCE) as f:
            source = f.read()
        present = [token for token in family_tokens() if token in source]
        self.assertEqual(
            present,
            [],
            "the speculative runner must reach draft families through "
            "infinilm.draft_spec; found family tokens: " + ", ".join(present),
        )


class RegistryDocumentationTest(unittest.TestCase):
    """The description registry and the model guide must agree.

    The guide's family table is what an integrator reads before touching the
    code, so a family that exists in only one of the two places is a defect.
    """

    @classmethod
    def setUpClass(cls):
        with open(MODELS_PATH, encoding="utf-8") as f:
            cls.guide = f.read()

    def table_rows(self):
        rows = []
        for line in self.guide.splitlines():
            if not line.startswith("| `") and not line.startswith("| \u2014 "):
                continue
            rows.append([cell.strip() for cell in line.strip("|").split("|")])
        return rows

    def test_every_described_family_is_in_the_family_table(self):
        listed = {
            row[0].strip("`") for row in self.table_rows() if row[0].startswith("`")
        }
        described = {spec.family for spec in list_draft_model_specs()}
        self.assertEqual(described - listed, set())

    def test_the_table_states_the_same_availability_as_the_registry(self):
        rows = {
            row[0].strip("`"): row
            for row in self.table_rows()
            if row[0].startswith("`")
        }
        for spec in list_draft_model_specs():
            row = rows[spec.family]
            status = row[4]
            row_text = " | ".join(row)
            for target in spec.target_model_types:
                self.assertIn(target, row_text)
            if spec.is_available:
                self.assertIn("block implemented here", status)
            else:
                self.assertIn("block not implemented here", status)
                # The primary missing piece is quoted from the registry.
                self.assertIn(spec.unimplemented[0], status)

    def test_families_the_guide_names_are_registered(self):
        described = {spec.family for spec in list_draft_model_specs()}
        for row in self.table_rows():
            if row[0].startswith("`"):
                self.assertIn(row[0].strip("`"), described)
            else:
                # A surveyed family without a description must say so.
                self.assertIn("not described yet", row[4])

    def description_section(self):
        section = self.guide.split("### 6.2 The description", 1)[1]
        return section.split("### 6.3", 1)[0]

    def test_every_description_field_is_documented(self):
        # The field reference is what an integrator fills in, so a field that
        # exists in code but not in the guide is a defect in either of them.
        documented = self.description_section()
        for dataclass in (DraftModelSpec, DraftWeightMap):
            for name in dataclass.__dataclass_fields__:
                self.assertIn(f"`{name}`", documented)

    def test_registry_splits_available_and_pending_families(self):
        available = [
            spec.family for spec in list_draft_model_specs() if spec.is_available
        ]
        pending = [
            spec.family for spec in list_draft_model_specs() if not spec.is_available
        ]
        self.assertEqual(available, ["qwen3_5_mtp", "minicpm_eagle", "mimo_mtp"])
        self.assertEqual(pending, ["qwen_moe_mtp", "deepseek_v3_mtp"])


class DescriptorResolutionTest(DraftSpecTestCase):
    def test_qwen_checkpoint_resolves_to_its_family(self):
        root = self.make_qwen_checkpoint(depth_key="mtp_num_hidden_layers", depth=1)

        checkpoint = resolve_embedded_draft(root)

        self.assertEqual(checkpoint.spec.family, "qwen3_5_mtp")
        self.assertEqual(checkpoint.published_depth, 1)
        self.assertEqual(checkpoint.depth_source, "weights")
        self.assertEqual(checkpoint.layer_kinds, (DraftLayerKind.FULL_ATTENTION,))
        with open(os.path.join(checkpoint.engine_path, "config.json")) as f:
            fixture = json.load(f)
        self.assertEqual(fixture["model_type"], "qwen3_5_mtp")
        self.assertEqual(fixture["text_config"]["num_hidden_layers"], 1)
        self.assertEqual(fixture["text_config"]["layer_types"], ["full_attention"])

    def test_config_depth_decides_when_the_weights_carry_no_layer_index(self):
        # A family whose head is not split into per-depth layers has no depth to
        # count, so the depth key names the answer. The depth key is spelled
        # differently from the Qwen3.5 one on purpose.
        spec = register_draft_model_spec(
            DraftModelSpec(
                family="synthetic_flat_family",
                draft_model_type="qwen3_5_mtp",
                target_model_types=("synthetic_flat_target",),
                embedded=True,
                depth_keys=("num_nextn_predict_layers",),
                layer_kinds=(DraftLayerKind.FULL_ATTENTION,),
                weight_map=DraftWeightMap(
                    family_keys=r"^mtp\.(?:fc|norm)\.",
                    key_pattern=r"^mtp\.(?P<rest>.+)$",
                    layer_key_pattern=None,
                    zero_centered_keys=("model.norm.weight",),
                    embedding_keys=(EMBED_KEY,),
                    head_keys=(LM_HEAD_KEY,),
                ),
            )
        )
        self.addCleanup(DRAFT_MODEL_SPECS.pop, spec.family, None)
        root = self.make_root()
        config = qwen_target_config(depth_key="num_nextn_predict_layers", depth=1)
        config["model_type"] = "synthetic_flat_target"
        write_checkpoint(
            root,
            config,
            {
                "model.safetensors": {
                    EMBED_KEY: rand((VOCAB_SIZE, HIDDEN_SIZE), 5),
                    "mtp.fc.weight": rand((HIDDEN_SIZE, 2 * HIDDEN_SIZE), 6),
                    "mtp.norm.weight": rand((HIDDEN_SIZE,), 7),
                }
            },
        )

        checkpoint = resolve_embedded_draft(root)

        self.assertEqual(checkpoint.published_depth, 1)
        self.assertEqual(
            checkpoint.depth_source, "config key 'num_nextn_predict_layers'"
        )

    def test_depth_uses_the_published_layer_groups_and_reports_a_mismatch(self):
        # Mirrors the checkpoints whose counting key disagrees with the weights:
        # the weights are authoritative and the mismatch is reported.
        root = self.make_qwen_checkpoint(
            depth_key="mtp_num_hidden_layers",
            depth=1,
            shards=qwen_target_shards(depth_groups=2),
        )

        with self.assertLogs("infinilm.draft_spec", level="WARNING") as logs:
            checkpoint = resolve_embedded_draft(root)

        self.assertEqual(checkpoint.published_depth, 2)
        self.assertEqual(checkpoint.depth_source, "weights")
        self.assertIn("the weights publish 2", " ".join(logs.output))

    def test_declared_depth_covers_a_checkpoint_without_a_counting_key(self):
        # The Qwen3-Next shape: the config carries no MTP depth key at all, so
        # the registration supplies the depth and the resolution proceeds to
        # the block check.
        root = self.make_root()
        config = qwen_target_config(depth_key=None, depth=1)
        config["model_type"] = "qwen3_next"
        config["text_config"]["model_type"] = "qwen3_next_text"
        write_checkpoint(root, config, qwen_target_shards())

        with self.assertRaisesRegex(UnsupportedDraftError, "MoE MLP"):
            resolve_embedded_draft(root)

    def test_deepseek_description_names_what_this_build_is_missing(self):
        root = self.make_root()
        config = {
            "model_type": "deepseek_v3",
            "text_config": {
                "model_type": "deepseek_v3",
                "hidden_size": HIDDEN_SIZE,
                "num_hidden_layers": 2,
                "num_nextn_predict_layers": 1,
                "vocab_size": VOCAB_SIZE,
            },
        }
        write_checkpoint(root, config, deepseek_shards())

        with self.assertRaisesRegex(UnsupportedDraftError, "MLA \\+ MoE draft block"):
            resolve_embedded_draft(root)

    def test_undeclared_family_with_draft_keys_fails_loudly(self):
        # A checkpoint that publishes a draft head, but whose model type no
        # description declares, must not be attributed to a described family.
        root = self.make_root()
        config = qwen_target_config(depth_key="num_nextn_predict_layers", depth=1)
        config["model_type"] = "undeclared_family"
        shards = {
            "model-00001-of-00001.safetensors": {
                "mtp.enorm.weight": rand((HIDDEN_SIZE,), 11),
                "mtp.hnorm.weight": rand((HIDDEN_SIZE,), 12),
                "mtp.eh_proj.weight": rand((HIDDEN_SIZE, 2 * HIDDEN_SIZE), 13),
                "mtp.layers.0.input_layernorm.weight": rand((HIDDEN_SIZE,), 14),
                "mtp.layers.1.input_layernorm.weight": rand((HIDDEN_SIZE,), 15),
            }
        }
        write_checkpoint(root, config, shards)

        with self.assertRaises(UnsupportedDraftError) as caught:
            resolve_draft(root)

        message = str(caught.exception)
        self.assertIn("undeclared_family", message)
        self.assertIn("no draft description", message)
        self.assertNotIn("is a 'qwen3_5_mtp' checkpoint", message)

    def test_generic_checkpoint_is_not_attributed_to_a_family(self):
        # A plain transformer layout must not look like a family's draft head:
        # the failure has to say "no draft weights", not name a family.
        root = self.make_root()
        config = {"model_type": "llama", "text_config": {"num_hidden_layers": 2}}
        write_checkpoint(
            root,
            config,
            {
                "model.safetensors": {
                    "model.embed_tokens.weight": rand((VOCAB_SIZE, HIDDEN_SIZE), 21),
                    "model.layers.0.self_attn.q_proj.weight": rand(
                        (HIDDEN_SIZE, HIDDEN_SIZE), 22
                    ),
                    "model.layers.1.mlp.up_proj.weight": rand(
                        (2 * HIDDEN_SIZE, HIDDEN_SIZE), 23
                    ),
                }
            },
        )

        self.assertIsNone(resolve_draft(root))
        explanation = explain_missing_draft(root, "llama")
        self.assertIn("no draft description", explanation)
        # The message lists the registered families, but must not claim the
        # checkpoint carries any of their draft heads.
        self.assertNotIn("resembling", explanation)
        self.assertNotIn("deepseek_v3_mtp", explanation)

    def test_standalone_draft_directory_resolves_to_no_fixture(self):
        # A directory whose own model type is a registered draft type is the
        # standalone draft; it must not be re-derived from a description.
        root = self.make_root()
        config = qwen_target_config(depth_key=None, depth=1)
        config["model_type"] = "qwen3_5_mtp"
        config["text_config"]["num_hidden_layers"] = 1
        write_checkpoint(root, config, qwen_target_shards())

        self.assertIsNone(resolve_draft(root))
        self.assertIsNone(resolve_embedded_draft(root))

    def test_materialised_fixture_is_accepted_as_a_draft_model(self):
        # A fixture produced by this module is a standalone draft directory,
        # which is what a user gets when pointing --draft-model at one.
        target = self.make_qwen_checkpoint()
        fixture = resolve_embedded_draft(target).engine_path
        self.fixtures.append(fixture)

        self.assertIsNone(resolve_draft(fixture))

    def test_standalone_directory_of_an_unimplemented_family_names_the_block(self):
        root = self.make_root()
        config = {
            "model_type": "deepseek_v3_mtp",
            "text_config": {"num_hidden_layers": 1},
        }
        write_checkpoint(
            root,
            config,
            {
                "model.safetensors": {
                    "enorm.weight": rand((HIDDEN_SIZE,), 61),
                    "hnorm.weight": rand((HIDDEN_SIZE,), 62),
                }
            },
        )

        with self.assertRaises(UnsupportedDraftError) as caught:
            resolve_draft(root)

        message = str(caught.exception)
        self.assertIn("MLA + MoE draft block", message)
        self.assertNotIn("no draft description", message)

    def test_recorded_semantics_are_rejected_for_standalone_descriptions_too(self):
        # The "recorded but not executed" fields must fail on every path that
        # uses the description, not only when a checkpoint embeds its head.
        spec = register_draft_model_spec(
            DraftModelSpec(
                family="synthetic_recorded_family",
                draft_model_type="synthetic_recorded_draft",
                embedded=False,
                runtime_depth=1,
            )
        )
        self.addCleanup(DRAFT_MODEL_SPECS.pop, spec.family, None)
        root = self.make_root()
        config = {"model_type": "synthetic_recorded_draft", "text_config": {}}
        write_checkpoint(
            root,
            config,
            {"model.safetensors": {"weight": rand((HIDDEN_SIZE,), 71)}},
        )

        with self.assertRaisesRegex(UnsupportedDraftError, "runtime depth"):
            resolve_draft(root)

    def test_checkpoint_without_draft_weights_resolves_to_nothing(self):
        root = self.make_root()
        shards = qwen_target_shards()
        shards["model-00002-of-00002.safetensors"] = {
            "mtp_free.weight": torch.zeros(2, 2)
        }
        write_checkpoint(root, qwen_target_config(), shards)

        self.assertIsNone(resolve_embedded_draft(root))

    def test_missing_draft_weights_are_explained(self):
        root = self.make_root()
        shards = {
            "model-00001-of-00001.safetensors": {
                "model.embed_tokens.weight": rand((VOCAB_SIZE, HIDDEN_SIZE), 1)
            }
        }
        write_checkpoint(root, qwen_target_config(depth_key=None), shards)

        explanation = explain_missing_draft(root, "qwen3_5")

        self.assertIn("criterion C1", explanation)
        self.assertIn("qwen3_5_mtp", explanation)

    def test_unknown_model_type_is_explained(self):
        root = self.make_root()
        config = qwen_target_config(depth_key=None)
        config["model_type"] = "llama"
        shards = {
            "model-00001-of-00001.safetensors": {
                "model.embed_tokens.weight": rand((VOCAB_SIZE, HIDDEN_SIZE), 1)
            }
        }
        write_checkpoint(root, config, shards)

        explanation = explain_missing_draft(root, "llama")

        self.assertIn("no draft description", explanation)

    def test_position_ids_follow_the_description(self):
        qwen = get_draft_model_spec("qwen3_5_mtp")
        eagle = get_draft_model_spec("minicpm_eagle")

        self.assertEqual(tuple(draft_position_ids(qwen, [1, 2, 3]).shape), (3, 3))
        self.assertEqual(tuple(draft_position_ids(eagle, [1, 2, 3]).shape), (3, 1))


class DraftWeightMapTest(unittest.TestCase):
    def test_deepseek_keys_map_onto_the_canonical_tree(self):
        shards = deepseek_shards()
        state_dict = shards["model-00001-of-00001.safetensors"]
        config = {
            "text_config": {"num_hidden_layers": 2, "tie_word_embeddings": False},
        }

        remapped = get_draft_weight_remapper("deepseek_v3_mtp")(state_dict, config)

        self.assertEqual(
            set(remapped),
            {
                "model.enorm.weight",
                "model.hnorm.weight",
                "model.eh_proj.weight",
                "model.shared_head.norm.weight",
                "model.shared_head.head.weight",
                "model.self_attn.q_a_proj.weight",
                "model.embed_tokens.weight",
                "lm_head.weight",
            },
        )
        # Target layers other than the published draft block stay out.
        self.assertNotIn("model.layers.2.enorm.weight", remapped)

    def test_deepseek_block_of_another_layer_is_ignored(self):
        state_dict = {
            "model.layers.1.enorm.weight": torch.ones(HIDDEN_SIZE),
            "model.layers.2.enorm.weight": torch.ones(HIDDEN_SIZE),
        }

        remapped = get_draft_weight_remapper("deepseek_v3_mtp")(
            state_dict, {"text_config": {"num_hidden_layers": 2}}
        )

        self.assertEqual(len(remapped), 1)
        self.assertTrue(
            torch.equal(remapped["model.enorm.weight"], torch.ones(HIDDEN_SIZE))
        )

    def test_loader_prefers_a_registered_table_entry(self):
        from infinilm import modeling_utils

        # Registered table entries keep priority, so the historical
        # qwen3_5_mtp mapper is still the one the loader calls.
        self.assertIs(
            get_weight_remapper("qwen3_5_mtp"), modeling_utils._remap_qwen3_5_mtp
        )

    def test_dedicated_draft_embeddings_still_fail_loudly(self):
        config = qwen_target_config()
        config["text_config"]["mtp_use_dedicated_embeddings"] = True

        with self.assertRaisesRegex(NotImplementedError, "dedicated draft embeddings"):
            get_draft_weight_remapper("qwen3_5_mtp")(
                build_tiny_mtp_weights(seed=1), config
            )


def deepseek_layout(block_layers=(2,), plain_layers=(0, 1)):
    """The released DeepSeek-V3 layout, scaled down.

    This family publishes its draft block as one extra decoder layer per
    published depth, after the target's own layers. Besides the tensors its key
    feature names, each of those layers carries a whole transformer block and a
    per-depth copy of the embedding, while the target keeps its own embedding
    and head at the top level.
    """

    def tensor(value, shape=(4,)):
        return torch.full(shape, float(value))

    state_dict = {
        "model.embed_tokens.weight": tensor(1.0, (8, 4)),
        "lm_head.weight": tensor(2.0, (8, 4)),
        "model.norm.weight": tensor(3.0),
    }
    for layer in plain_layers:
        prefix = f"model.layers.{layer}."
        state_dict[prefix + "input_layernorm.weight"] = tensor(10 + layer)
        state_dict[prefix + "self_attn.q_a_proj.weight"] = tensor(20 + layer, (4, 4))
        state_dict[prefix + "mlp.gate_proj.weight"] = tensor(30 + layer, (4, 4))
    for depth, layer in enumerate(block_layers):
        prefix = f"model.layers.{layer}."
        for offset, (suffix, shape) in enumerate(
            (
                ("enorm.weight", (4,)),
                ("hnorm.weight", (4,)),
                ("eh_proj.weight", (4, 8)),
                ("shared_head.norm.weight", (4,)),
                ("shared_head.head.weight", (8, 4)),
                ("embed_tokens.weight", (8, 4)),
                ("input_layernorm.weight", (4,)),
                ("post_attention_layernorm.weight", (4,)),
                ("self_attn.q_a_proj.weight", (4, 4)),
                ("mlp.gate_proj.weight", (4, 4)),
            )
        ):
            state_dict[prefix + suffix] = tensor(40 + 10 * depth + offset, shape)
    return state_dict


class TargetSideDerivationTest(unittest.TestCase):
    """A target whose checkpoint embeds a described draft head is answered too.

    The target engine loads the checkpoint as a whole, so the embedded draft
    tensors have to leave before the target module tree sees them. The loader's
    table keeps priority; the cases below pin the answer an embedded family gets
    from its description alone, and the properties that keep that answer to the
    family's own tensors: only an embedded family gives one, a table entry
    always wins, the removal covers everything the family publishes (its key
    feature plus the draft layers that feature locates in the weights), and it
    reaches nothing else.
    """

    def register(
        self,
        family,
        embedded,
        targets=(),
        weight_map=True,
        layer_key_pattern=None,
        **overrides,
    ):
        spec = register_draft_model_spec(
            DraftModelSpec(
                family=family,
                embedded=embedded,
                target_model_types=targets,
                weight_map=(
                    DraftWeightMap(
                        family_keys=r"^draft_head\.",
                        key_pattern=r"^draft_head\.(?P<rest>.+)$",
                        layer_key_pattern=layer_key_pattern,
                    )
                    if weight_map
                    else None
                ),
                **overrides,
            )
        )
        self.addCleanup(DRAFT_MODEL_SPECS.pop, family, None)
        return spec

    def test_a_described_family_answers_for_its_target_type(self):
        self.register(
            "synthetic_target_family", True, targets=("synthetic_target_model",)
        )
        draft_key = "draft_head.fc.weight"
        target_key = "model.layers.0.self_attn.q_proj.weight"
        state_dict = {
            target_key: torch.full((4, 4), 1.0),
            draft_key: torch.full((4, 8), 2.0),
        }

        remapper = get_weight_remapper("synthetic_target_model")

        self.assertIs(
            remapper, get_embedded_draft_target_remapper("synthetic_target_model")
        )
        remapped = remapper(state_dict, config={})
        self.assertEqual(set(remapped), {target_key})
        # The kept tensors are handed over untouched: this mapping removes, it
        # does not rename, rescale or add.
        self.assertIs(remapped[target_key], state_dict[target_key])

    def test_the_target_answer_does_not_wait_for_the_draft_block(self):
        # A family can be described before this build can run its block; loading
        # its checkpoint as a target is independent of that gap.
        self.register(
            "synthetic_pending_family",
            True,
            targets=("synthetic_pending_target",),
            unimplemented=("a draft block this build cannot compose",),
        )

        self.assertIsNotNone(get_weight_remapper("synthetic_pending_target"))

    def test_every_target_type_of_a_family_gets_the_same_mapper(self):
        self.assertIs(
            get_weight_remapper("deepseek_v3"), get_weight_remapper("deepseek_v32")
        )

    def test_only_an_embedded_family_answers_for_its_target(self):
        # The two descriptions differ in `embedded` alone, so the pair separates
        # the embedded gate from the weight-map gate.
        self.register(
            "synthetic_embedded_gate",
            True,
            targets=("synthetic_embedded_gate_target",),
        )
        self.register(
            "synthetic_standalone_gate",
            False,
            targets=("synthetic_standalone_gate_target",),
        )

        self.assertIsNotNone(get_weight_remapper("synthetic_embedded_gate_target"))
        self.assertIsNone(get_weight_remapper("synthetic_standalone_gate_target"))

    def test_a_family_without_a_weight_map_answers_for_nothing(self):
        self.register(
            "synthetic_mapless_gate",
            True,
            targets=("synthetic_mapless_gate_target",),
            weight_map=False,
        )

        self.assertIsNone(get_weight_remapper("synthetic_mapless_gate_target"))

    def test_the_standalone_family_keeps_both_sides_unchanged(self):
        # The one registered family that is not embedded: no target-side answer,
        # and its draft side stays exactly as it was.
        self.assertIsNone(get_weight_remapper("minicpm_eagle"))
        self.assertIsNone(get_draft_weight_remapper("minicpm_eagle"))

    def test_a_registered_table_entry_still_wins(self):
        from infinilm import modeling_utils

        self.assertIs(get_weight_remapper("mimo"), modeling_utils._remap_mimo)
        entry = modeling_utils._WEIGHT_REMAPPER.pop("mimo")
        self.addCleanup(modeling_utils._WEIGHT_REMAPPER.__setitem__, "mimo", entry)

        self.assertIsNot(get_weight_remapper("mimo"), entry)
        self.assertIs(
            get_weight_remapper("mimo"), get_embedded_draft_target_remapper("mimo")
        )

    def test_a_draft_block_that_reuses_the_layer_layout_is_dropped_whole(self):
        # The sharpest layout: this family publishes its draft block as one extra
        # decoder layer, so most of its tensors sit in the namespace every
        # checkpoint has. The layer the family's own keys locate is the draft
        # layer, and everything under it goes with it.
        state_dict = deepseek_layout(block_layers=(2,), plain_layers=(0, 1))
        block_keys = {key for key in state_dict if key.startswith("model.layers.2.")}

        remapped = get_weight_remapper("deepseek_v3")(state_dict, config={})

        self.assertEqual(set(state_dict) - set(remapped), block_keys)
        # Every target tensor survives and is handed over untouched.
        self.assertEqual(
            set(remapped),
            {key for key in state_dict if key not in block_keys},
        )
        for key, tensor in remapped.items():
            self.assertIs(tensor, state_dict[key])

    def test_every_published_draft_layer_is_dropped_not_only_the_first(self):
        # A family may publish more than one depth; each one is a whole reused
        # layer of its own. Dropping only the first located layer would leave the
        # remaining depths behind and fail the load.
        state_dict = deepseek_layout(block_layers=(2, 3), plain_layers=(0, 1))
        block_keys = {
            key
            for key in state_dict
            if key.startswith("model.layers.2.") or key.startswith("model.layers.3.")
        }

        remapped = get_weight_remapper("deepseek_v3")(state_dict, config={})

        self.assertEqual(set(state_dict) - set(remapped), block_keys)
        self.assertEqual(
            set(remapped), {key for key in state_dict if key not in block_keys}
        )
        for key, tensor in remapped.items():
            self.assertIs(tensor, state_dict[key])

    def test_a_description_that_cannot_name_its_layer_index_is_reported(self):
        # Locating a draft layer needs the layer index, which the description
        # field is documented to expose in a `depth` group. A description that
        # does not must name that requirement, whatever the shard holds, rather
        # than failing inside a weight load with a bare group lookup.
        self.register(
            "synthetic_groupless_family",
            True,
            targets=("synthetic_groupless_target",),
            layer_key_pattern=r"^draft_head\.layers\.\d+\.",
        )
        state_dict = {"model.layers.0.self_attn.q_proj.weight": torch.full((4, 4), 1.0)}

        with self.assertRaisesRegex(
            ValueError, "synthetic_groupless_family.*group named 'depth'"
        ):
            get_weight_remapper("synthetic_groupless_target")(state_dict, config={})

    def test_a_located_layer_does_not_swallow_a_longer_layer_index(self):
        # Layer 6 carries the family's keys while layer 61 is an ordinary target
        # layer: a layer prefix that stopped before the separator would remove
        # both.
        state_dict = {
            "model.layers.6.enorm.weight": torch.full((4,), 1.0),
            "model.layers.6.input_layernorm.weight": torch.full((4,), 2.0),
            "model.layers.61.self_attn.q_a_proj.weight": torch.full((4, 4), 3.0),
            "model.layers.61.mlp.gate_proj.weight": torch.full((8, 4), 4.0),
        }

        remapped = get_weight_remapper("deepseek_v3")(state_dict, config={})

        self.assertEqual(
            set(remapped),
            {
                "model.layers.61.self_attn.q_a_proj.weight",
                "model.layers.61.mlp.gate_proj.weight",
            },
        )

    def test_target_tensors_outside_the_located_layer_survive(self):
        # Names that merely resemble the family's, target tensors of other
        # layers, and the shared embedding and head all reach the target; only
        # the layer the family's own keys locate goes.
        state_dict = {
            "model.embed_tokens.weight": torch.full((4,), 1.0),
            "lm_head.weight": torch.full((4,), 2.0),
            "model.layers.3.enorm_extra.weight": torch.full((4,), 3.0),
            "model.layers.3.mtp_layers.weight": torch.full((4,), 4.0),
            "model.layers.3.self_attn.q_a_proj.weight": torch.full((4, 4), 5.0),
            "model.layers.2.enorm.weight": torch.full((4,), 6.0),
        }

        remapped = get_weight_remapper("deepseek_v3")(state_dict, config={})

        self.assertEqual(
            set(remapped),
            {key for key in state_dict if key != "model.layers.2.enorm.weight"},
        )

    def test_a_located_layer_removes_nothing_from_another_family(self):
        # The three families happen to use the same layer index in their own
        # namespaces; the layer this family locates must not reach into another
        # family's namespace or into the other layers of its own.
        state_dict = {
            "mtp.layers.2.fc.weight": torch.full((4, 8), 1.0),
            "mtp.fc.weight": torch.full((4, 8), 2.0),
            "model.mtp_layers.2.input_proj.weight": torch.full((4, 8), 3.0),
            "model.layers.2.enorm.weight": torch.full((4,), 4.0),
            "model.layers.20.self_attn.q_a_proj.weight": torch.full((4, 4), 5.0),
        }

        remapped = get_weight_remapper("deepseek_v3")(state_dict, config={})

        self.assertEqual(
            set(remapped),
            {
                "mtp.layers.2.fc.weight",
                "mtp.fc.weight",
                "model.mtp_layers.2.input_proj.weight",
                "model.layers.20.self_attn.q_a_proj.weight",
            },
        )

    def test_a_generic_checkpoint_loses_nothing(self):
        state_dict = {
            "model.embed_tokens.weight": torch.full((4,), 1.0),
            "model.layers.0.self_attn.q_proj.weight": torch.full((4, 4), 1.0),
            "model.layers.1.mlp.up_proj.weight": torch.full((8, 4), 1.0),
            "model.norm.weight": torch.full((4,), 1.0),
        }

        remapped = get_weight_remapper("deepseek_v3")(state_dict, config={})

        self.assertEqual(set(remapped), set(state_dict))

    def test_one_family_does_not_remove_another_family_s_tensors(self):
        from infinilm import modeling_utils

        state_dict = {
            "mtp.fc.weight": torch.full((4, 8), 1.0),
            "model.mtp_layers.0.input_proj.weight": torch.full((4, 8), 2.0),
            "model.layers.2.enorm.weight": torch.full((4,), 3.0),
        }

        deepseek = get_weight_remapper("deepseek_v3")(state_dict, config={})
        self.assertEqual(
            set(deepseek),
            {"mtp.fc.weight", "model.mtp_layers.0.input_proj.weight"},
        )

        entry = modeling_utils._WEIGHT_REMAPPER.pop("mimo")
        self.addCleanup(modeling_utils._WEIGHT_REMAPPER.__setitem__, "mimo", entry)
        mimo = get_weight_remapper("mimo")(state_dict, config={})
        self.assertEqual(set(mimo), {"mtp.fc.weight", "model.layers.2.enorm.weight"})

    def test_the_target_answer_is_not_the_draft_selection(self):
        # The two answers are different mappings: the draft selection keeps only
        # the draft tensors, the target answer keeps everything else.
        self.assertIsNot(
            get_weight_remapper("deepseek_v3"),
            get_draft_weight_remapper("deepseek_v3_mtp"),
        )

    def test_model_types_outside_both_namespaces_answer_for_nothing(self):
        for model_type in ("llama", "minicpm", "qwen3_moe"):
            with self.subTest(model_type=model_type):
                self.assertIsNone(get_weight_remapper(model_type))


class _StubTargetEngine:
    """The runner reads the target's cache config and hf config only."""

    def __init__(self, hf_config, cache_config):
        self.hf_config = hf_config
        self._cache_config = cache_config

    def get_cache_config(self):
        return self._cache_config


def register_second_family():
    """Register a second family that reuses the draft block this build has.

    No other family with a runnable block and published weights exists, so the
    registration proof uses a description of a family whose metadata differs
    from Qwen3.5's (its own model type, depth key and position layout) while
    its block is the one already implemented.
    """
    return register_draft_model_spec(
        DraftModelSpec(
            family=SECOND_FAMILY,
            draft_model_type="qwen3_5_mtp",
            target_model_types=(SECOND_FAMILY_TARGET,),
            embedded=True,
            depth_keys=("num_nextn_predict_layers",),
            layer_kinds=(DraftLayerKind.FULL_ATTENTION,),
            concat_order=ConcatOrder.EMBEDDING_FIRST,
            embedding_sharing=EmbeddingSharing.SHARED_WITH_TARGET,
            position_ids=PositionIdLayout.STANDARD,
            weight_map=DraftWeightMap(
                family_keys=r"^mtp\.(?:fc|norm|layers)\.",
                key_pattern=r"^mtp\.(?P<rest>.+)$",
                layer_key_pattern=r"^mtp\.layers\.(?P<depth>\d+)\.",
                embedding_keys=(EMBED_KEY,),
                head_keys=(LM_HEAD_KEY,),
            ),
        )
    )


class RunnerConstructionTest(DraftSpecTestCase):
    def build_config(self, draft_model_path, cache_type="paged", **overrides):
        options = dict(
            model_path=draft_model_path,
            draft_model_path=draft_model_path,
            device="cpu",
            dtype="bfloat16",
            cache_type=cache_type,
            max_batch_size=1,
            num_blocks=8,
            block_size=64,
            max_cache_len=1024,
            num_draft_tokens=1,
        )
        options.update(overrides)
        return EngineConfig(**options)

    def target_engine(self, cache_config, hf_config=None):
        return _StubTargetEngine(hf_config or qwen_target_config(), cache_config)

    def test_second_family_is_built_from_its_description(self):
        register_second_family()
        self.addCleanup(DRAFT_MODEL_SPECS.pop, SECOND_FAMILY, None)
        root = self.make_root()
        config = qwen_target_config(depth_key="num_nextn_predict_layers", depth=1)
        config["model_type"] = SECOND_FAMILY_TARGET
        write_checkpoint(root, config, qwen_target_shards())

        engine = SpeculativeRunner(
            self.build_config(root),
            self.target_engine(PagedKVCacheConfig(num_blocks=8, block_size=64)),
            infinicore.device("cpu", 0),
        )

        self.assertEqual(engine.draft_spec.family, SECOND_FAMILY)
        self.assertEqual(engine.draft_model_type, "qwen3_5_mtp")
        self.assertEqual(engine._cache_block_size, 64)
        weights = engine.draft_model_engine.state_dict()[0]
        self.assertIn("model.fc.weight", weights)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", weights)

    def test_static_cache_constructs_and_keeps_drafting_off(self):
        root = self.make_qwen_checkpoint()
        cache_config = StaticKVCacheConfig(max_batch_size=1, max_cache_len=64)

        with self.assertLogs(
            "infinilm.llm.model_runner.speculative_runner", level="WARNING"
        ) as logs:
            runner = SpeculativeRunner(
                self.build_config(root, cache_type="static"),
                self.target_engine(cache_config),
                infinicore.device("cpu", 0),
            )

        self.assertIsNone(runner._cache_block_size)
        self.assertIn("non-speculatively", " ".join(logs.output))
        # The static scheduler hands out no speculative cache ops, which is why
        # the runner's forward stays on the plain target path.
        self.assertFalse(
            hasattr(StaticSchedulerOutput([], False), "speculative_cache_ops")
        )

    def test_fixture_directory_can_be_used_as_a_draft_model(self):
        target = self.make_qwen_checkpoint()
        fixture = resolve_embedded_draft(target).engine_path
        self.fixtures.append(fixture)

        engine = SpeculativeRunner(
            self.build_config(fixture),
            self.target_engine(PagedKVCacheConfig(num_blocks=8, block_size=64)),
            infinicore.device("cpu", 0),
        )

        self.assertEqual(engine.draft_spec.family, "qwen3_5_mtp")
        self.assertEqual(engine.draft_model_type, "qwen3_5_mtp")
        # Loading happens against the fixture, which reuses the checkpoint
        # shards through symlinks.
        weights = engine.draft_model_engine.state_dict()[0]
        self.assertIn("model.fc.weight", weights)

    def test_undeclared_family_fails_at_construction(self):
        root = self.make_root()
        config = qwen_target_config(depth_key="num_nextn_predict_layers", depth=1)
        config["model_type"] = "undeclared_family"
        shards = {
            "model-00001-of-00001.safetensors": {
                "mtp.fc.weight": rand((HIDDEN_SIZE, 2 * HIDDEN_SIZE), 31),
                "mtp.layers.0.input_layernorm.weight": rand((HIDDEN_SIZE,), 32),
            }
        }
        write_checkpoint(root, config, shards)

        with self.assertRaisesRegex(UnsupportedDraftError, "no draft description"):
            SpeculativeRunner(
                self.build_config(root),
                self.target_engine(PagedKVCacheConfig(num_blocks=8, block_size=64)),
                infinicore.device("cpu", 0),
            )

    def test_checkpoint_without_draft_weights_fails_at_construction(self):
        root = self.make_root()
        shards = {
            "model-00001-of-00001.safetensors": {
                "model.embed_tokens.weight": rand((VOCAB_SIZE, HIDDEN_SIZE), 1)
            }
        }
        write_checkpoint(root, qwen_target_config(depth_key=None), shards)

        with self.assertRaisesRegex(UnsupportedDraftError, "criterion C1"):
            SpeculativeRunner(
                self.build_config(root),
                self.target_engine(PagedKVCacheConfig(num_blocks=8, block_size=64)),
                infinicore.device("cpu", 0),
            )

    def test_draft_with_a_different_vocabulary_is_rejected(self):
        root = self.make_qwen_checkpoint()
        target_config = qwen_target_config()
        target_config["text_config"]["vocab_size"] = VOCAB_SIZE * 2

        with self.assertRaisesRegex(UnsupportedDraftError, "criterion C5"):
            SpeculativeRunner(
                self.build_config(root),
                self.target_engine(
                    PagedKVCacheConfig(num_blocks=8, block_size=64), target_config
                ),
                infinicore.device("cpu", 0),
            )


if __name__ == "__main__":
    unittest.main()
