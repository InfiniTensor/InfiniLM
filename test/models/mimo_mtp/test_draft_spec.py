#!/usr/bin/env python3
"""MiMo MTP draft descriptions: registration, resolution, runner wiring.

A checkpoint that embeds its draft head enters through a description plus a
registration entry, so these tests pin the MiMo end of that contract: the
registry resolves the published ``model.mtp_layers.<depth>.`` layout with the
semantics the family records, the loader reaches the family's draft mapping
without a table entry of its own, and the speculative runner builds the draft
engine from the checkpoint through the description alone.

The resolution cases run on tiny synthetic checkpoints that use the released key
layout. The state-pool case records why the multi-token verification hand-off
has nothing to manage on this family.
"""

import os
import shutil
import sys
import tempfile
import unittest

import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "llama"))
sys.path.insert(0, _TEST_DIR)

import infinicore  # noqa: E402
from infinilm.cache.cache import PagedKVCacheConfig  # noqa: E402
from infinilm.config.engine_config import EngineConfig  # noqa: E402
from infinilm.draft_spec import (  # noqa: E402
    ConcatOrder,
    EmbeddingAtPositionZero,
    get_draft_weight_remapper,
    resolve_draft,
)
from infinilm.infer_engine import model_uses_mamba_cache  # noqa: E402
from infinilm.llm.model_runner.speculative_runner import (  # noqa: E402
    SpeculativeRunner,
)
from infinilm.modeling_utils import get_weight_remapper  # noqa: E402
from test_forward_validation import (  # noqa: E402
    HIDDEN_SIZE,
    VOCAB_SIZE,
    build_tiny_config,
    build_tiny_weights,
    write_tiny_checkpoint,
)
from utils import infinicore_to_torch_tensor  # noqa: E402


class _StubTargetEngine:
    """The runner reads the target's cache config and hf config only."""

    def __init__(self, hf_config, cache_config):
        self.hf_config = hf_config
        self._cache_config = cache_config

    def get_cache_config(self):
        return self._cache_config


class DraftSpecTestCase(unittest.TestCase):
    def setUp(self):
        self.fixtures = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for path in self.fixtures:
            shutil.rmtree(path, ignore_errors=True)

    def make_root(self):
        root = tempfile.mkdtemp(prefix="infinilm_mimo_draft_spec_")
        self.fixtures.append(root)
        return root

    def make_checkpoint(self, config=None, weights=None):
        """A tiny checkpoint carrying the target and its embedded draft head."""
        root = self.make_root()
        write_tiny_checkpoint(
            root,
            build_tiny_weights() if weights is None else weights,
            config=build_tiny_config() if config is None else config,
        )
        return root


class RegistryDispatchTest(unittest.TestCase):
    """The loader derives the MiMo draft mapping from the description."""

    def test_the_mapping_is_derived_without_a_loader_table_entry(self):
        import infinilm.modeling_utils as modeling_utils

        self.assertIsNotNone(get_draft_weight_remapper("mimo_mtp"))
        # The target-side entry and the draft mapping are separate: the loader
        # table names the target model type, the draft mapping comes from the
        # family description.
        self.assertNotIn("mimo_mtp", modeling_utils._WEIGHT_REMAPPER)
        self.assertIs(
            get_weight_remapper("mimo_mtp"), get_draft_weight_remapper("mimo_mtp")
        )


class DescriptorResolutionTest(DraftSpecTestCase):
    def test_mimo_checkpoint_resolves_with_its_recorded_semantics(self):
        # The description reads the depth from the weights. Both config shapes are
        # accepted: the released MiMo config keeps its fields at the top level,
        # while a nested text_config (the shape other families publish) reaches
        # the same resolution.
        flat = self.make_checkpoint()
        nested = self.make_checkpoint(
            config={
                "model_type": "mimo",
                "text_config": {
                    "num_nextn_predict_layers": 1,
                    "hidden_size": HIDDEN_SIZE,
                    "num_hidden_layers": 2,
                    "vocab_size": VOCAB_SIZE,
                },
            }
        )

        for root in (flat, nested):
            with self.subTest(config=os.path.basename(root)):
                checkpoint = resolve_draft(root)

                self.assertEqual(checkpoint.spec.family, "mimo_mtp")
                self.assertEqual(checkpoint.published_depth, 1)
                self.assertEqual(checkpoint.depth_source, "weights")
                self.assertEqual(checkpoint.spec.concat_order, ConcatOrder.HIDDEN_FIRST)
                self.assertEqual(
                    checkpoint.spec.embedding_at_position_zero,
                    EmbeddingAtPositionZero.ZEROED,
                )


class TargetStateMechanismTest(unittest.TestCase):
    """The multi-token verification hand-off has no state rows to manage here.

    The hand-off only engages for a target that declares a recurrent state pool;
    a pure-attention target never allocates one, so the runner's scratch-row
    bookkeeping stays out of the loop. The run-time half of this statement is
    reported by the speculative losslessness check, which counts the state rows
    borrowed while real verifications are in flight.
    """

    def test_a_mimo_target_declares_no_recurrent_state_pool(self):
        config = build_tiny_config()
        # The pool decision reads the declared layer types and the
        # linear-attention block; this family publishes neither.
        self.assertNotIn("layer_types", config)
        self.assertFalse(model_uses_mamba_cache(config))


class RunnerConstructionTest(DraftSpecTestCase):
    """The speculative runner builds the MiMo draft through the description."""

    def build_config(self, draft_model_path, **overrides):
        options = dict(
            model_path=draft_model_path,
            draft_model_path=draft_model_path,
            device="cpu",
            dtype="bfloat16",
            cache_type="paged",
            max_batch_size=1,
            num_blocks=8,
            block_size=64,
            max_cache_len=1024,
            num_draft_tokens=1,
        )
        options.update(overrides)
        return EngineConfig(**options)

    def target_engine(self):
        return _StubTargetEngine(
            build_tiny_config(), PagedKVCacheConfig(num_blocks=8, block_size=64)
        )

    def build_runner(self, draft_model_path):
        return SpeculativeRunner(
            self.build_config(draft_model_path),
            self.target_engine(),
            infinicore.device("cpu", 0),
        )

    def test_runner_builds_the_draft_from_the_checkpoint(self):
        root = self.make_checkpoint()
        source = build_tiny_weights()

        engine = self.build_runner(root)

        self.assertEqual(engine.draft_spec.family, "mimo_mtp")
        self.assertEqual(engine.draft_model_type, "mimo_mtp")
        self.assertEqual(engine._cache_block_size, 64)
        weights = engine.draft_model_engine.state_dict()[0]
        for name in (
            "model.fc.weight",
            "model.pre_fc_norm_embedding.weight",
            "model.pre_fc_norm_hidden.weight",
            "model.norm.weight",
            "model.self_attn.q_proj.bias",
            "lm_head.weight",
        ):
            self.assertIn(name, weights)
        # The published namespace belongs to the target checkpoint, not to the
        # draft engine that was built from it.
        self.assertEqual(
            [key for key in weights if key.startswith("model.mtp_layers.")], []
        )
        # Every parameter holds the published tensor the description maps onto
        # it, so a swapped or missing rename cannot pass by leaving the canonical
        # names in place. The tiny checkpoint is float32, so the comparison is
        # exact; following the loaded dtype keeps it meaningful if that changes.
        for published, canonical in (
            ("input_proj.weight", "model.fc.weight"),
            ("token_layernorm.weight", "model.pre_fc_norm_embedding.weight"),
            ("hidden_layernorm.weight", "model.pre_fc_norm_hidden.weight"),
            ("final_layernorm.weight", "model.norm.weight"),
            ("self_attn.q_proj.bias", "model.self_attn.q_proj.bias"),
        ):
            actual = infinicore_to_torch_tensor(weights[canonical], torch.empty(0))
            expected = source[f"model.mtp_layers.0.{published}"].to(actual.dtype)
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertTrue(
                torch.equal(actual, expected),
                f"{published} did not reach {canonical}",
            )


if __name__ == "__main__":
    unittest.main()
