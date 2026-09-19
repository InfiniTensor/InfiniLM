#!/usr/bin/env python3
"""Component validation of the MiMo MTP draft block.

The reference follows the released MiMo MTP rollout, not the checkpoint's own
module definitions: the next-token embedding and the target hidden state are
normalized separately, the embedding of position 0 is masked before its norm,
the hidden state is concatenated first, the fused input runs through one
full-attention decoder layer whose q/k/v projections carry biases, and the final
norm produces both the logits and the hidden state the next draft step consumes.

Everything runs on a tiny synthetic checkpoint that publishes the released
checkpoint's own ``model.mtp_layers.0.*`` tensor names, so the published-key to
draft-parameter mapping is exercised together with the forward, at float32 so
that a wrong draft fails by a wide margin instead of hiding inside rounding.

Every semantic is covered in both directions: the reference that follows it
agrees with the C++ draft, and the reference that breaks it one way diverges
beyond tolerance. The last test swaps two entries of the description's rename
map and asserts that the alignment does fail, so the comparison itself is shown
to be sensitive to a wrong draft.
"""

import dataclasses
import json
import os
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

try:
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        Qwen2MLP,
        Qwen2RMSNorm,
        Qwen2RotaryEmbedding,
    )
except ImportError as error:  # pragma: no cover - environment guard
    print(f"Error: Required packages not found. Please install: {error}")
    sys.exit(1)

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "llama"))

import infinicore  # noqa: E402
from infinilm.cache.cache import StaticKVCacheConfig  # noqa: E402
from infinilm.draft_spec import (  # noqa: E402
    DRAFT_MODEL_SPECS,
    register_draft_model_spec,
    resolve_draft,
)
from infinilm.infer_engine import InferEngine  # noqa: E402
from infinilm.modeling_utils import load_model_state_dict_by_file  # noqa: E402
from utils import infinicore_to_torch_tensor, tensor_all_close  # noqa: E402

# Tiny draft dimensions; the draft block is always one full-attention layer.
HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 16
INTERMEDIATE_SIZE = 128
VOCAB_SIZE = 128
SEQ_LEN = 6
ROPE_THETA = 640000
RMS_NORM_EPS = 1e-5
MAX_POSITIONS = 128
SEED = 20260918
# float32 on both sides: the reference and the C++ draft run the same values
# through different kernels, so their difference is accumulation order alone.
RTOL = 1e-4
ATOL = 1e-4


def build_tiny_config(head_dim=HEAD_DIM):
    """Minimal MiMo config mirroring the released checkpoint fields."""
    return {
        "architectures": ["MiMoForCausalLM"],
        "attention_bias": True,
        "attention_dropout": 0.0,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "max_position_embeddings": MAX_POSITIONS,
        "model_type": "mimo",
        "num_attention_heads": NUM_HEADS,
        "num_hidden_layers": 2,
        "num_key_value_heads": NUM_KV_HEADS,
        "num_nextn_predict_layers": 1,
        "rms_norm_eps": RMS_NORM_EPS,
        "rope_theta": ROPE_THETA,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": VOCAB_SIZE,
    }


def build_tiny_weights(seed=SEED, head_dim=HEAD_DIM):
    """The released draft-key layout: 16 mtp_layers.0 tensors plus embed/head."""
    generator = torch.Generator().manual_seed(seed)

    def weight(*shape):
        return (torch.randn(*shape, generator=generator) * 0.05).to(torch.float32)

    def scale():
        return (1.0 + torch.randn(HIDDEN_SIZE, generator=generator) * 0.1).to(
            torch.float32
        )

    q_out = NUM_HEADS * head_dim
    kv_out = NUM_KV_HEADS * head_dim
    prefix = "model.mtp_layers.0."
    return {
        "model.embed_tokens.weight": weight(VOCAB_SIZE, HIDDEN_SIZE),
        "lm_head.weight": weight(VOCAB_SIZE, HIDDEN_SIZE),
        prefix + "token_layernorm.weight": scale(),
        prefix + "hidden_layernorm.weight": scale(),
        prefix + "input_proj.weight": weight(HIDDEN_SIZE, 2 * HIDDEN_SIZE),
        prefix + "input_layernorm.weight": scale(),
        prefix + "post_attention_layernorm.weight": scale(),
        prefix + "self_attn.q_proj.weight": weight(q_out, HIDDEN_SIZE),
        prefix + "self_attn.q_proj.bias": weight(q_out),
        prefix + "self_attn.k_proj.weight": weight(kv_out, HIDDEN_SIZE),
        prefix + "self_attn.k_proj.bias": weight(kv_out),
        prefix + "self_attn.v_proj.weight": weight(kv_out, HIDDEN_SIZE),
        prefix + "self_attn.v_proj.bias": weight(kv_out),
        prefix + "self_attn.o_proj.weight": weight(HIDDEN_SIZE, q_out),
        prefix + "mlp.gate_proj.weight": weight(INTERMEDIATE_SIZE, HIDDEN_SIZE),
        prefix + "mlp.up_proj.weight": weight(INTERMEDIATE_SIZE, HIDDEN_SIZE),
        prefix + "mlp.down_proj.weight": weight(HIDDEN_SIZE, INTERMEDIATE_SIZE),
        prefix + "final_layernorm.weight": scale(),
    }


def write_tiny_checkpoint(root, weights, config=None):
    """Write the checkpoint the released layout describes, with an index."""
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(build_tiny_config() if config is None else config, f)
    shard = "model-00001-of-00001.safetensors"
    save_file(weights, os.path.join(root, shard))
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {key: shard for key in weights},
    }
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)


def remove_tree(path):
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full):
            remove_tree(full)
        else:
            os.remove(full)
    os.rmdir(path)


def build_engine(checkpoint_dir):
    """Build the draft engine the way the speculative runner does.

    ``resolve_draft`` applies the family description and materialises the
    standalone draft config next to the checkpoint's own shards; the loader
    derives the weight mapping from the same description.
    """
    checkpoint = resolve_draft(checkpoint_dir)
    if checkpoint is None:
        raise AssertionError("the description did not resolve the checkpoint")
    fixture = checkpoint.engine_path
    engine = InferEngine(
        model_path=fixture,
        device=infinicore.device("cpu", 0),
        cache_config=StaticKVCacheConfig(max_batch_size=1, max_cache_len=MAX_POSITIONS),
        attention_backend="default",
    )
    load_model_state_dict_by_file(engine, fixture, dtype=engine.dtype)
    return engine, fixture


def draft_forward(
    engine, input_ids, positions, target_hidden, past_kv=0, per_token_positions=False
):
    """One draft forward; ``past_kv`` continues a draft chain step by step."""
    seq_len = len(input_ids)
    position_list = (
        [[position] for position in positions]
        if per_token_positions
        else list(positions)
    )
    output = engine.forward_raw(
        input_ids=infinicore.from_list([list(input_ids)], dtype=infinicore.int64),
        position_ids=infinicore.from_list(position_list, dtype=infinicore.int64),
        past_kv_lengths=infinicore.from_list([past_kv], dtype=infinicore.int32),
        total_kv_lengths=infinicore.from_list(
            [past_kv + seq_len], dtype=infinicore.int32
        ),
        input_offsets=infinicore.from_list([0, seq_len], dtype=infinicore.int32),
        cu_seqlens=infinicore.from_list([0, past_kv + seq_len], dtype=infinicore.int32),
        target_hidden_states=infinicore.from_torch(
            target_hidden.reshape(1, seq_len, HIDDEN_SIZE).contiguous()
        ),
        temperature=1.0,
        top_k=1,
        top_p=1.0,
    )
    logits = infinicore_to_torch_tensor(output["logits"], torch.empty(0)).float()
    hidden = infinicore_to_torch_tensor(output["hidden_states"], torch.empty(0)).float()
    return logits, hidden


def build_reference_config(head_dim=HEAD_DIM):
    """Qwen2 layer config with the MiMo fields the draft block depends on."""
    config = Qwen2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=head_dim,
        rms_norm_eps=RMS_NORM_EPS,
        max_position_embeddings=MAX_POSITIONS,
        attention_dropout=0.0,
        use_sliding_window=False,
        tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA},
    )
    config._attn_implementation = "eager"
    return config


class TorchMimoMtpReference(torch.nn.Module):
    """MiMo draft block built from the HF Qwen2 layers the family publishes.

    ``forward`` takes one knob per recorded semantic. The defaults are the
    released behaviour; a test flips one knob and expects the comparison to
    fail, which is what makes the alignment check falsifiable.
    """

    def __init__(self, config, embed_weight, lm_head_weight):
        super().__init__()
        self.config = config
        self.embed_tokens = torch.nn.Parameter(embed_weight, requires_grad=False)
        self.lm_head = torch.nn.Parameter(lm_head_weight, requires_grad=False)
        self.token_layernorm = Qwen2RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)
        self.hidden_layernorm = Qwen2RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)
        self.input_proj = torch.nn.Linear(2 * HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.input_layernorm = Qwen2RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)
        self.post_attention_layernorm = Qwen2RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)
        self.final_layernorm = Qwen2RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)
        self.self_attn = Qwen2Attention(config, layer_idx=0)
        self.mlp = Qwen2MLP(config)
        self.rope = Qwen2RotaryEmbedding(config)

    def load_mtp_weights(self, weights):
        """Load the published draft tensors by their checkpoint names."""
        prefix = "model.mtp_layers.0."
        mapping = {
            "token_layernorm.weight": self.token_layernorm.weight,
            "hidden_layernorm.weight": self.hidden_layernorm.weight,
            "input_proj.weight": self.input_proj.weight,
            "input_layernorm.weight": self.input_layernorm.weight,
            "post_attention_layernorm.weight": self.post_attention_layernorm.weight,
            "final_layernorm.weight": self.final_layernorm.weight,
        }
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mapping[f"self_attn.{proj}.weight"] = getattr(self.self_attn, proj).weight
        for proj in ("q_proj", "k_proj", "v_proj"):
            mapping[f"self_attn.{proj}.bias"] = getattr(self.self_attn, proj).bias
        for proj in ("gate_proj", "up_proj", "down_proj"):
            mapping[f"mlp.{proj}.weight"] = getattr(self.mlp, proj).weight

        for name, parameter in mapping.items():
            parameter.data = weights[prefix + name].clone()

    def draft_layer(self, hidden_states, position_ids, *, use_attention_bias=True):
        """One pre-norm decoder layer, as the published block defines it."""
        seq_len = hidden_states.shape[1]
        if not use_attention_bias:
            # A build that drops the published q/k/v biases sums the same
            # projections without them. This changes the module, so callers run
            # each variant on a module of its own (`fresh_reference`).
            for proj in ("q_proj", "k_proj", "v_proj"):
                getattr(self.self_attn, proj).bias = None

        causal = torch.full(
            (1, 1, seq_len, seq_len),
            torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).triu(diagonal=1)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        cos_sin = self.rope(hidden_states, position_ids)
        attn_out, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=cos_sin,
            attention_mask=causal,
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward(
        self,
        input_ids,
        target_hidden,
        position_ids,
        *,
        concat_order="hidden_first",
        norm_targets="separate",
        mask_position_zero=True,
        use_attention_bias=True,
        apply_final_norm=True,
        logits_from_final_layer=True,
    ):
        embeds = torch.nn.functional.embedding(input_ids, self.embed_tokens)
        if mask_position_zero:
            embeds = embeds.masked_fill(
                (position_ids == 0).unsqueeze(-1).to(embeds.device), 0.0
            )
        if norm_targets == "separate":
            normed_embed = self.token_layernorm(embeds)
            normed_hidden = self.hidden_layernorm(target_hidden)
        elif norm_targets == "one_norm":
            normed_embed = self.hidden_layernorm(embeds)
            normed_hidden = self.hidden_layernorm(target_hidden)
        elif norm_targets == "swapped":
            normed_embed = self.hidden_layernorm(embeds)
            normed_hidden = self.token_layernorm(target_hidden)
        else:
            raise ValueError(norm_targets)

        if concat_order == "hidden_first":
            fused = torch.cat([normed_hidden, normed_embed], dim=-1)
        else:
            fused = torch.cat([normed_embed, normed_hidden], dim=-1)

        hidden_states = self.input_proj(fused)
        hidden_states = self.draft_layer(
            hidden_states, position_ids, use_attention_bias=use_attention_bias
        )
        # The published block normalizes its output; a build without that norm
        # hands the block output to the head and to the next step unchanged.
        out_hidden = (
            self.final_layernorm(hidden_states) if apply_final_norm else hidden_states
        )
        head_input = (
            out_hidden
            if (apply_final_norm and logits_from_final_layer)
            else hidden_states
        )
        logits = torch.nn.functional.linear(head_input, self.lm_head)
        return logits, out_hidden


class MimoDraftHarness(unittest.TestCase):
    """Shared synthetic checkpoint, draft engine and torch reference."""

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = tempfile.mkdtemp(prefix="infinilm_mimo_mtp_")
        cls.weights = build_tiny_weights()
        write_tiny_checkpoint(cls.checkpoint, cls.weights)

        cls.input_ids = torch.randint(
            0, VOCAB_SIZE, (1, SEQ_LEN), generator=torch.Generator().manual_seed(SEED)
        )
        cls.target_hidden = (
            torch.randn(
                1,
                SEQ_LEN,
                HIDDEN_SIZE,
                generator=torch.Generator().manual_seed(SEED + 1),
            )
            * 0.5
        )
        cls.position_ids = torch.arange(SEQ_LEN).view(1, -1)

        cls.reference = cls.fresh_reference()
        with torch.no_grad():
            cls.reference_logits, cls.reference_hidden = cls.reference(
                cls.input_ids, cls.target_hidden, cls.position_ids
            )
        cls.reference_logits = cls.reference_logits.float()
        cls.reference_hidden = cls.reference_hidden.float()

        cls.engine, cls.fixture = build_engine(cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        remove_tree(cls.fixture)
        remove_tree(cls.checkpoint)

    @classmethod
    def fresh_reference(cls):
        """A reference module of its own.

        A variant knob that changes the module (dropping the attention biases,
        for instance) must not reach the shared reference or the other variants
        through it, so every variant builds its own module from the same
        tensors.
        """
        reference = TorchMimoMtpReference(
            build_reference_config(),
            cls.weights["model.embed_tokens.weight"],
            cls.weights["lm_head.weight"],
        )
        reference.load_mtp_weights(cls.weights)
        reference.eval()
        return reference

    def draft(self, **kwargs):
        """A draft forward over the whole sequence, as the component path."""
        return draft_forward(
            self.engine,
            self.input_ids[0].tolist(),
            list(range(SEQ_LEN)),
            self.target_hidden,
            **kwargs,
        )

    def assert_matches(self, reference, actual):
        ok, stats = tensor_all_close(reference, actual, rtol=RTOL, atol=ATOL)
        self.assertTrue(
            ok,
            "reference and draft differ: "
            f"max_abs_diff={stats['max_abs_diff']:.6f} "
            f"mean_abs_diff={stats['mean_abs_diff']:.6f}",
        )
        return stats

    def assert_diverges(self, reference, actual):
        ok, stats = tensor_all_close(reference, actual, rtol=RTOL, atol=ATOL)
        self.assertFalse(
            ok,
            "a broken reference still matched the draft; the check is not "
            f"sensitive to this semantic (max_abs_diff={stats['max_abs_diff']:.6f})",
        )
        return stats


class MimoDraftForwardTest(MimoDraftHarness):
    """The draft forward and one falsifiable contrast per recorded semantic."""

    def test_draft_matches_the_reference(self):
        logits, hidden = self.draft()
        self.assert_matches(self.reference_logits, logits)
        self.assert_matches(self.reference_hidden, hidden)

    def test_concat_order_is_hidden_first(self):
        reference = self.fresh_reference()
        with torch.no_grad():
            swapped_logits, swapped_hidden = reference(
                self.input_ids,
                self.target_hidden,
                self.position_ids,
                concat_order="embedding_first",
            )
        self.assert_diverges(swapped_logits, self.reference_logits)
        _, hidden = self.draft()
        # The other family's order is a different model, not this block.
        self.assert_diverges(swapped_hidden, hidden)

    def test_both_fusion_norms_apply_to_their_own_stream(self):
        _, hidden = self.draft()
        for norm_targets in ("one_norm", "swapped"):
            reference = self.fresh_reference()
            with torch.no_grad():
                logits, variant_hidden = reference(
                    self.input_ids,
                    self.target_hidden,
                    self.position_ids,
                    norm_targets=norm_targets,
                )
            self.assert_diverges(logits, self.reference_logits)
            self.assert_diverges(variant_hidden, hidden)

    def test_position_zero_embedding_is_masked(self):
        reference = self.fresh_reference()
        with torch.no_grad():
            logits, variant_hidden = reference(
                self.input_ids,
                self.target_hidden,
                self.position_ids,
                mask_position_zero=False,
            )
        self.assert_diverges(logits, self.reference_logits)
        # Masking position 0 changes that row's key/value, so later rows move
        # too; the masked row itself must move as well.
        row_diff = (variant_hidden - self.reference_hidden)[:, 0, :].abs().max()
        self.assertGreater(row_diff.item(), ATOL)
        _, hidden = self.draft()
        self.assert_diverges(variant_hidden, hidden)

    def test_attention_bias_is_applied(self):
        # This variant changes its module, so it runs on a module of its own.
        reference = self.fresh_reference()
        with torch.no_grad():
            logits, variant_hidden = reference(
                self.input_ids,
                self.target_hidden,
                self.position_ids,
                use_attention_bias=False,
            )
        self.assert_diverges(logits, self.reference_logits)
        _, hidden = self.draft()
        self.assert_diverges(variant_hidden, hidden)

    def test_final_norm_is_inside_the_block(self):
        # A block that returns its pre-norm output (the Eagle convention) has a
        # different hidden state and, through the head, different logits.
        reference = self.fresh_reference()
        with torch.no_grad():
            pre_norm_logits, _ = reference(
                self.input_ids,
                self.target_hidden,
                self.position_ids,
                logits_from_final_layer=False,
            )
            _, pre_norm_hidden = reference(
                self.input_ids,
                self.target_hidden,
                self.position_ids,
                apply_final_norm=False,
            )
        self.assert_diverges(pre_norm_logits, self.reference_logits)
        _, hidden = self.draft()
        self.assert_diverges(pre_norm_hidden, hidden)


class MimoDraftRolloutTest(MimoDraftHarness):
    """The serial draft chain.

    Every step runs the same block at the next position and consumes the hidden
    state the previous step returned, while the draft's own key/value entries
    from the earlier steps stay in the cache. The reference reproduces that by
    running the block over the tokens drafted so far, each row fed the hidden
    state its own step received.
    """

    STEPS = 3
    START = 1

    def source_token(self):
        return int(self.input_ids[0, self.START])

    def source_hidden(self):
        return self.target_hidden[:, self.START : self.START + 1, :].clone()

    def engine_rollout(self, steps):
        """The draft chain the runner builds, feeding back what it returns."""
        token = self.source_token()
        hidden = self.source_hidden()
        tokens, hiddens = [], []
        for step in range(steps):
            logits, hidden = draft_forward(
                self.engine,
                [token],
                [self.START + step],
                hidden,
                past_kv=step,
                per_token_positions=True,
            )
            token = int(logits[0, -1].argmax())
            tokens.append(token)
            hiddens.append(hidden.clone())
        return tokens, hiddens

    def reference_rollout(self, steps, **kwargs):
        """The same chain, with the earlier draft steps inside the attention."""
        reference = self.fresh_reference()
        tokens = [self.source_token()]
        positions = [self.START]
        hidden_inputs = [self.source_hidden()]
        drafted, hiddens = [], []
        for _ in range(steps):
            with torch.no_grad():
                logits, hidden = reference(
                    torch.tensor([tokens]),
                    torch.cat(hidden_inputs, dim=1),
                    torch.tensor([positions]),
                    **kwargs,
                )
            token = int(logits[0, -1].argmax())
            last = hidden[:, -1:, :]
            drafted.append(token)
            hiddens.append(last.clone())
            tokens.append(token)
            positions.append(positions[-1] + 1)
            hidden_inputs.append(last)
        return drafted, hiddens

    def test_recycled_hidden_is_the_normed_block_output(self):
        tokens, hiddens = self.engine_rollout(self.STEPS)
        reference_tokens, reference_hiddens = self.reference_rollout(self.STEPS)

        self.assertEqual(reference_tokens, tokens)
        for step in range(self.STEPS):
            self.assert_matches(reference_hiddens[step], hiddens[step])

    def test_recycling_the_pre_norm_hidden_diverges(self):
        _, hiddens = self.engine_rollout(2)
        # A block without the final norm recycles its pre-norm output (the Eagle
        # convention) and reaches a different second step.
        _, pre_norm = self.reference_rollout(2, apply_final_norm=False)
        self.assert_diverges(pre_norm[1], hiddens[1])


class MimoDraftMappingSensitivityTest(unittest.TestCase):
    """A wrong published-key mapping must be caught by the alignment check."""

    def setUp(self):
        self.checkpoint = tempfile.mkdtemp(prefix="infinilm_mimo_mtp_swap_")
        self.weights = build_tiny_weights()
        write_tiny_checkpoint(self.checkpoint, self.weights)
        self.input_ids = torch.randint(
            0, VOCAB_SIZE, (1, SEQ_LEN), generator=torch.Generator().manual_seed(SEED)
        )
        self.target_hidden = (
            torch.randn(
                1,
                SEQ_LEN,
                HIDDEN_SIZE,
                generator=torch.Generator().manual_seed(SEED + 1),
            )
            * 0.5
        )

    def tearDown(self):
        remove_tree(self.checkpoint)

    def test_swapped_norm_mapping_fails_the_alignment(self):
        spec = DRAFT_MODEL_SPECS["mimo_mtp"]
        swapped = dataclasses.replace(
            spec,
            weight_map=dataclasses.replace(
                spec.weight_map,
                renames=(
                    ("token_layernorm.", "pre_fc_norm_hidden."),
                    ("hidden_layernorm.", "pre_fc_norm_embedding."),
                    ("input_proj.", "fc."),
                    ("final_layernorm.", "norm."),
                ),
            ),
        )
        register_draft_model_spec(swapped)
        fixture = None
        try:
            engine, fixture = build_engine(self.checkpoint)
            logits, _ = draft_forward(
                engine,
                self.input_ids[0].tolist(),
                list(range(SEQ_LEN)),
                self.target_hidden,
            )
        finally:
            register_draft_model_spec(spec)
            if fixture is not None:
                remove_tree(fixture)

        reference = TorchMimoMtpReference(
            build_reference_config(),
            self.weights["model.embed_tokens.weight"],
            self.weights["lm_head.weight"],
        )
        reference.load_mtp_weights(self.weights)
        reference.eval()
        with torch.no_grad():
            reference_logits, _ = reference(
                self.input_ids,
                self.target_hidden,
                torch.arange(SEQ_LEN).view(1, -1),
            )
        ok, stats = tensor_all_close(
            reference_logits.float(), logits, rtol=RTOL, atol=ATOL
        )
        self.assertFalse(
            ok,
            "swapping the two fusion norms still matched; the alignment check "
            "does not see the published-key mapping",
        )
        self.assertGreater(stats["max_abs_diff"], ATOL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
