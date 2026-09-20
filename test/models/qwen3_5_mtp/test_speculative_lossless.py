#!/usr/bin/env python3
"""
Losslessness test for Qwen3.5 MTP speculative decoding.

Greedy-decodes a fixed prompt set twice with the InfiniLM engine: once with
speculation off (no --draft-model) and once with MTP speculation on (the
target checkpoint passed as --draft-model). The generated token id sequences
must match token by token; the speculative run also reports the draft
acceptance counters and how many tokens were accepted per verification.

Without --model the check builds a tiny synthetic Qwen3.5 checkpoint that
publishes the released ``mtp.*`` key layout, so the whole mechanism — the
hybrid target's linear-attention state, the verification, the partial-accept
replay and the full-accept hand-off — runs on any machine with a GPU and needs
no downloaded weights. The tiny checkpoint's own weights are random, so the
draft head is not expected to predict the target's continuation and the
mechanism is driven the way the losslessness claim needs: what the draft
proposes is replaced, in a separate run for each verification outcome, by
tokens whose acceptance is known (section 3 partial), and the
run asserts that the outcome really happened instead of assuming it. A fully
accepted verification is not constructed: it happens on its own in the K sweep
above, whose per-prompt comparison covers it, and the state hand-off behind it
is pinned separately by test_verify_handoff.py.

Point --model at a checkpoint other than the tiny one this file writes; only
K=1 is a lossless setting there, so the sweep is narrowed to it and the
multi-token comparison against plain decoding is reported instead of asserted:
a verification that drafts several tokens can round a step's near-tied top-2
logits differently from the single-token decode step and flip its argmax. That
scope follows the checkpoint, not the flag: --model pointing at the tiny
checkpoint this file writes is the same checkpoint as the default run and gets
the same criteria.

The number of draft tokens verified per target step is configurable via
--num-draft-tokens. A multi-token verification is only lossless if a partially
accepted one leaves the target's recurrent linear-attention state matching the
accepted prefix, so the forced-partial run corrupts the drafted tail wherever
the draft budget allows it: once per configured budget, each raised to two
because a single drafted token has no tail to reject, and twice per budget with
two different rejected tails. Its own criteria are that the engine really
verified at the requested budget, that a verification which drafted the full
budget still accepted fewer tokens than it drafted, and that no round fell back
for want of a state row. The two rejected tails must also emit identical
tokens, on either checkpoint: a rejected tail only reaches the committed state
if the hand-off carried it there, and comparing two runs that differ only in
that tail stays inside the verification shape, so a near-tied step rounds the
same way in both. The state-level equivalence behind the mechanism is covered
separately by test_verify_handoff.py.
"""

import argparse
import gc
import json
import os
import sys
import tempfile

try:
    import torch
except ImportError as e:
    print(f"Error: Required packages not found. Please install: {e}")
    sys.exit(1)

try:
    from infinilm.llm.llm import LLM
    from infinilm.llm.sampling_params import SamplingParams
except ImportError as e:
    print("Error: InfiniLM package not found. Please install it:")
    print(f"  Error: {e}")
    sys.exit(1)

# The fixture builders and the reference module are shared with the other
# checks of this family, which is why the directory joins the path first.
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "llama"))
sys.path.insert(0, _TEST_DIR)

from test_draft_weight_load import (  # noqa: E402
    VOCAB_SIZE,
    build_tiny_text_config,
)

DEFAULT_MODEL_DIR = os.path.expanduser("~/models/Qwen3.5-2B")
DEFAULT_DEVICE = "cuda"
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_DRAFT_TOKEN_COUNTS = (1, 2, 3, 4)
# The paged-attention prefill kernel refuses a small head_dim, and the chunked
# delta-rule kernels accept a key/value head dim of 64 or 128 only.
HEAD_DIM = 256
LINEAR_HEAD_DIM = 128
LINEAR_HEADS = 2
LINEAR_CONV_KERNEL_DIM = 4
# A handful of layers: enough for the recurrent state to matter, small enough
# that the target, the draft and the paged cache fit on a consumer GPU.
TARGET_LAYERS = 2
FULL_ATTENTION_LAYERS = 1
MAX_POSITIONS = 2048
MAX_CACHE_LEN = 512
BLOCK_SIZE = 16
# max(2, NUM_BLOCKS // 4) state rows must leave a free row for the scratch of a
# multi-token verification, so the pool is several times the row count in use.
NUM_BLOCKS = 32
# Tokens the target is not expected to predict: the forced-partial section runs
# once per rejected tail, and the two runs must emit the same tokens.
FORCED_OFFSETS = (7, 11)
DEFAULT_PROMPTS = [
    "1 + 1 =",
    "def fibonacci(n):",
    "SELECT * FROM users WHERE",
    "The meaning of life is",
    "The Eiffel Tower is located in",
    "The following is a list of prime numbers: 2, 3, 5, 7,",
]


def build_engine(model_dir, draft_model_dir, device, max_new_tokens, num_draft_tokens):
    """Build the LLM engine; hybrid qwen3.5 needs paged attn, no prefix cache."""
    # Small cache footprint so the target and draft engines fit alongside
    # each other on a single consumer GPU.
    return LLM(
        model_path=model_dir,
        draft_model_path=draft_model_dir,
        num_draft_tokens=num_draft_tokens,
        device=device,
        dtype="bfloat16",
        cache_type="paged",
        attn_backend="paged-attn",
        enable_prefix_caching=False,
        max_batch_size=1,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_cache_len=MAX_CACHE_LEN,
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
    )


def write_tokenizer(root):
    """A character-level tokenizer, so prompts are fixed token id sequences."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    vocab = {chr(index): index for index in range(32, 127)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="~"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.Fuse()
    tokenizer.save(os.path.join(root, "tokenizer.json"))
    with open(os.path.join(root, "tokenizer_config.json"), "w") as f:
        json.dump(
            {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 512}, f
        )


def tiny_checkpoint_config():
    """The config.json of the tiny checkpoint this check writes.

    Doubles as the identity test for a --model directory: a checkpoint
    publishing exactly this config is the checkpoint the default run builds, so
    it is covered by the same comparison whichever way it was passed.
    """
    config = json.loads(json.dumps(build_tiny_text_config()))
    # The shared fixture describes the draft block alone (one full-attention
    # layer); the target side adds the linear-attention fields the released
    # hybrid target carries and the layer split.
    config["linear_conv_kernel_dim"] = LINEAR_CONV_KERNEL_DIM
    config["linear_key_head_dim"] = LINEAR_HEAD_DIM
    config["linear_value_head_dim"] = LINEAR_HEAD_DIM
    config["linear_num_key_heads"] = LINEAR_HEADS
    config["linear_num_value_heads"] = LINEAR_HEADS
    config["num_hidden_layers"] = TARGET_LAYERS
    config["layer_types"] = ["linear_attention"] * (
        TARGET_LAYERS - FULL_ATTENTION_LAYERS
    ) + ["full_attention"] * FULL_ATTENTION_LAYERS
    config["head_dim"] = HEAD_DIM
    # The section sum has to equal head_dim * partial_rotary_factor / 2.
    config["rope_parameters"]["partial_rotary_factor"] = 0.25
    config["rope_parameters"]["mrope_section"] = [12, 12, 8]
    config["max_position_embeddings"] = MAX_POSITIONS
    # The released dense checkpoints tie the embedding and the head; the draft
    # loader derives which shard tensors to inject from this flag.
    config["tie_word_embeddings"] = True
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": config,
        "tie_word_embeddings": config["tie_word_embeddings"],
        "torch_dtype": "bfloat16",
    }


def is_tiny_fixture(model_dir):
    """Whether a checkpoint directory publishes this file's tiny config."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            return json.load(f) == tiny_checkpoint_config()
    except (OSError, ValueError):
        return False


def write_checkpoint(root):
    """A checkpoint carrying the target and its embedded draft head.

    The config mirrors the released layout field for field, with a few layers
    and small head counts, so the released key names, the hybrid
    full/linear-attention split and the ``mtp.*`` draft block are all exercised
    by a checkpoint small enough to build here.
    """
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(tiny_checkpoint_config(), f)

    from safetensors.torch import save_file

    weights = build_tiny_target_weights()
    shard = "model-00001-of-00001.safetensors"
    save_file(weights, os.path.join(root, shard))
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {key: shard for key in weights},
    }
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)


def build_tiny_target_weights(seed=7):
    """Target tensors plus the embedded draft head, in the released layout.

    Every shape is derived from the same config the checkpoint publishes, so
    the head dim and the layer split above are the only knobs.
    """
    config = json.loads(json.dumps(build_tiny_text_config()))
    config["linear_conv_kernel_dim"] = LINEAR_CONV_KERNEL_DIM
    config["linear_key_head_dim"] = LINEAR_HEAD_DIM
    config["linear_value_head_dim"] = LINEAR_HEAD_DIM
    config["linear_num_key_heads"] = LINEAR_HEADS
    config["linear_num_value_heads"] = LINEAR_HEADS
    config["head_dim"] = HEAD_DIM
    config["num_hidden_layers"] = TARGET_LAYERS
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    intermediate = config["intermediate_size"]
    vocab = config["vocab_size"]
    key_heads = config["linear_num_key_heads"]
    value_heads = config["linear_num_value_heads"]
    generator = torch.Generator().manual_seed(seed)

    def weight(*shape):
        return (torch.randn(*shape, generator=generator) * 0.05).to(torch.bfloat16)

    def scale(*shape):
        return (1.0 + torch.randn(*shape, generator=generator) * 0.1).to(torch.bfloat16)

    # attn_output_gate doubles q_proj, exactly as in the released checkpoints.
    q_out = 2 * heads * HEAD_DIM
    kv_out = kv_heads * HEAD_DIM
    linear_key_dim = key_heads * LINEAR_HEAD_DIM
    linear_value_dim = value_heads * LINEAR_HEAD_DIM
    conv_dim = 2 * linear_key_dim + linear_value_dim
    weights = {
        "model.language_model.embed_tokens.weight": weight(vocab, hidden),
        "model.language_model.norm.weight": scale(hidden),
    }
    for layer in range(TARGET_LAYERS):
        prefix = f"model.language_model.layers.{layer}."
        weights[prefix + "input_layernorm.weight"] = scale(hidden)
        weights[prefix + "post_attention_layernorm.weight"] = scale(hidden)
        weights[prefix + "mlp.gate_proj.weight"] = weight(intermediate, hidden)
        weights[prefix + "mlp.up_proj.weight"] = weight(intermediate, hidden)
        weights[prefix + "mlp.down_proj.weight"] = weight(hidden, intermediate)
        if layer >= TARGET_LAYERS - FULL_ATTENTION_LAYERS:
            weights[prefix + "self_attn.q_proj.weight"] = weight(q_out, hidden)
            weights[prefix + "self_attn.k_proj.weight"] = weight(kv_out, hidden)
            weights[prefix + "self_attn.v_proj.weight"] = weight(kv_out, hidden)
            weights[prefix + "self_attn.o_proj.weight"] = weight(
                hidden, heads * HEAD_DIM
            )
            weights[prefix + "self_attn.q_norm.weight"] = scale(HEAD_DIM)
            weights[prefix + "self_attn.k_norm.weight"] = scale(HEAD_DIM)
        else:
            weights[prefix + "linear_attn.conv1d.weight"] = weight(
                conv_dim, 1, config["linear_conv_kernel_dim"]
            )
            weights[prefix + "linear_attn.in_proj_qkv.weight"] = weight(
                conv_dim, hidden
            )
            weights[prefix + "linear_attn.in_proj_z.weight"] = weight(
                linear_value_dim, hidden
            )
            weights[prefix + "linear_attn.in_proj_b.weight"] = weight(
                value_heads, hidden
            )
            weights[prefix + "linear_attn.in_proj_a.weight"] = weight(
                value_heads, hidden
            )
            weights[prefix + "linear_attn.out_proj.weight"] = weight(
                hidden, linear_value_dim
            )
            # The gated norm sits on the value head dim, not on hidden_size.
            weights[prefix + "linear_attn.norm.weight"] = scale(LINEAR_HEAD_DIM)
            weights[prefix + "linear_attn.dt_bias"] = torch.ones(
                value_heads, dtype=torch.bfloat16
            )
            weights[prefix + "linear_attn.A_log"] = torch.log(
                torch.empty(value_heads).uniform_(0.01, 16, generator=generator)
            ).to(torch.bfloat16)
    # The embedded draft head: the released 15 ``mtp.*`` tensors, one
    # full-attention layer that reuses the target's own layer class.
    weights.update(
        {
            "mtp.fc.weight": weight(hidden, 2 * hidden),
            "mtp.pre_fc_norm_embedding.weight": scale(hidden),
            "mtp.pre_fc_norm_hidden.weight": scale(hidden),
            "mtp.norm.weight": scale(hidden),
            "mtp.layers.0.input_layernorm.weight": scale(hidden),
            "mtp.layers.0.post_attention_layernorm.weight": scale(hidden),
            "mtp.layers.0.self_attn.q_proj.weight": weight(q_out, hidden),
            "mtp.layers.0.self_attn.k_proj.weight": weight(kv_out, hidden),
            "mtp.layers.0.self_attn.v_proj.weight": weight(kv_out, hidden),
            "mtp.layers.0.self_attn.o_proj.weight": weight(hidden, heads * HEAD_DIM),
            "mtp.layers.0.self_attn.q_norm.weight": scale(HEAD_DIM),
            "mtp.layers.0.self_attn.k_norm.weight": scale(HEAD_DIM),
            "mtp.layers.0.mlp.gate_proj.weight": weight(intermediate, hidden),
            "mtp.layers.0.mlp.up_proj.weight": weight(intermediate, hidden),
            "mtp.layers.0.mlp.down_proj.weight": weight(hidden, intermediate),
        }
    )
    return weights


def remove_tree(path):
    """Delete a directory tree; the fixtures hold files only."""
    for root, _, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        os.rmdir(root)


def cleanup_built_fixtures(before):
    """Remove the draft fixtures this run added, once its engines are closed.

    Building the draft engine materialises a directory of symlinks next to the
    checkpoint's shards, and nothing in the library removes it: the directory
    has to outlive the engine (the shards are read through it), but not the
    process. Callers pass the set of names seen before the run so only this
    run's directories go, after every engine that could still be reading them
    has been closed.
    """
    pattern = "infinilm_draft_fixture_"
    added = [
        name for name in os.listdir(tempfile.gettempdir()) if name.startswith(pattern)
    ]
    removed = 0
    for name in added:
        if name in before:
            continue
        try:
            remove_tree(os.path.join(tempfile.gettempdir(), name))
            removed += 1
        except OSError as error:  # a directory another process is reading
            print(f"   (left {name} in place: {error})")
    return removed


def built_fixtures():
    """Names of the draft fixture directories that already exist."""
    pattern = "infinilm_draft_fixture_"
    return set(
        name for name in os.listdir(tempfile.gettempdir()) if name.startswith(pattern)
    )


def generate_outputs(engine, prompts, max_new_tokens):
    """Greedy-decode every prompt and return (prompt_ids, token_ids) pairs."""
    # Only max_tokens and ignore_eos take effect at request level.
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=True,
    )
    results = []
    for prompt in prompts:
        output = engine.generate(
            prompts=[prompt], sampling_params=sampling_params, use_tqdm=False
        )[0]
        results.append(
            (list(output.prompt_token_ids), list(output.outputs[0].token_ids))
        )
    return results


def runner_stats(engine):
    """Snapshot of the speculative runner's counters, or None if unreachable."""
    runner = getattr(engine.engine.model_runner, "speculative_runner", None)
    if runner is None:
        return None
    return {
        "accepted": runner.eagle_accept_count,
        "total": runner.eagle_total_count,
        "histogram": dict(runner.accepted_count_histogram),
        "budget": runner.num_draft_tokens,
        "scratch_exhausted": runner.verify_scratch_exhausted,
        "state_rows": runner._mamba_cache,
        "family": runner.draft_spec.family if runner.draft_spec else None,
    }


def report_accept_stats(engine):
    """Print the runner's acceptance counters; returns the stats snapshot."""
    stats = runner_stats(engine)
    if stats is None:
        print("   (no speculative runner found)")
        return None
    total = stats["total"]
    accepted = stats["accepted"]
    rate = accepted / total if total else 0.0
    print(
        f"   accepted {accepted}/{total} drafted tokens"
        f" ({100.0 * rate:.1f}% acceptance)"
    )
    histogram = stats["histogram"]
    if histogram:
        rounds = sum(histogram.values())
        counts = ", ".join(f"{size}:{histogram[size]}" for size in sorted(histogram))
        print(f"   verification rounds {rounds}, accepted per round {{{counts}}}")
        if stats["budget"] > 1:
            partial = sum(
                count for size, count in histogram.items() if size < stats["budget"]
            )
            print(
                "   rounds that accepted fewer than"
                f" {stats['budget']} drafted tokens: {partial}/{rounds}"
            )
    if stats["scratch_exhausted"]:
        print(f"   ✗ state-row exhaustion fallbacks: {stats['scratch_exhausted']}")
    return stats


def force_drafted_tokens(engine, vocab_size, offset):
    """Replace every drafted token but the first with one the target will not
    predict, to drive the partially accepted verification path.

    A verification only accepts while the drafted token equals the target's own
    greedy token. The first drafted token is the target's own previous token and
    is always accepted; replacing the rest makes the verification accept exactly
    one token out of the ones it drafted. That is index-independent, unlike
    proposing a specific continuation, and the offset picks which rejected tail
    is proposed.

    Returns the number of tokens the draft proposed in each verification round,
    which is what tells a round that accepted fewer than it drafted from a round
    whose budget the end of the generation cut short.
    """
    runner = engine.engine.model_runner.speculative_runner
    original = runner._draft_eagle_tokens_batch
    lengths = []

    def partial_draft(jobs):
        results = original(jobs)
        for job, tokens in zip(jobs, results):
            target = int(job["target_token"])
            for index in range(len(tokens)):
                tokens[index] = target if index == 0 else (target + offset) % vocab_size
            lengths.append(len(tokens))
        return results

    runner._draft_eagle_tokens_batch = partial_draft
    return lengths


def compare_outputs(
    prompts,
    reference_results,
    actual_results,
    reference_name="baseline",
    actual_name="speculative",
    report_only=False,
):
    """Print the per-prompt comparison; returns whether every prompt matched.

    A report-only call marks the outcome without the pass/fail glyphs the
    criteria use: its result does not reach the exit code, so it must not read
    as a criterion that failed. None comes back for a result-count mismatch:
    that is a defect in this check rather than a comparison outcome, and it
    stays fatal wherever it is called.
    """
    if len(reference_results) != len(prompts) or len(actual_results) != len(prompts):
        print(
            "✗ Internal error: expected one result per prompt"
            f" ({reference_name}={len(reference_results)},"
            f" {actual_name}={len(actual_results)})"
        )
        return None
    all_match = True
    diverged = 0
    for prompt, (_, reference_ids), (_, actual_ids) in zip(
        prompts, reference_results, actual_results
    ):
        # Two empty sequences would compare equal; treat them as a failure.
        match = (
            bool(reference_ids) and bool(actual_ids) and (reference_ids == actual_ids)
        )
        all_match = all_match and match
        if report_only:
            print(
                f"   report: {prompt!r}"
                f" {'matches' if match else 'diverges'}"
                f" ({len(actual_ids)} tokens)"
            )
        else:
            print(f"   {'✓' if match else '✗'} {prompt!r} ({len(actual_ids)} tokens)")
        if not match:
            diverged += 1
            if len(reference_ids) != len(actual_ids):
                print(
                    f"       length mismatch: {reference_name}={len(reference_ids)},"
                    f" {actual_name}={len(actual_ids)}"
                )
            first_div = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(reference_ids, actual_ids))
                    if a != b
                ),
                min(len(reference_ids), len(actual_ids)),
            )
            print(f"       first divergence at position {first_div}")
            print(f"       {reference_name}: {reference_ids}")
            print(f"       {actual_name}: {actual_ids}")
    if report_only:
        print(
            f"   report: {diverged}/{len(prompts)} prompt(s) diverge from the"
            f" {reference_name}; this comparison is not a criterion here"
        )
    return all_match


def check_draft_budget(stats, requested):
    """Assert the engine verified at the requested draft budget.

    The acceptance counters come from the runner, so a configured budget that
    quietly shrinks would leave every comparison above unchanged and the
    section's own label as the only place the budget appears.
    """
    reported = stats["budget"] if stats is not None else None
    ok = reported == requested
    print(
        f"   {'✓' if ok else '✗'} verified draft budget: requested {requested},"
        f" runner reports {reported}"
    )
    return ok


def run_forced_generation(
    engine, prompts, max_new_tokens, vocab_size, requested_budget, offset
):
    """Run one forced-partial generation, with the criteria that belong to it.

    The partially accepted path really has to have been taken: this checks that
    the engine verified at the requested budget, that at least one round drafted
    that budget in full, that a round which drafted it accepted fewer tokens
    than it drafted, and that no round fell back for want of a state row.

    Returns the generated results for the caller to compare, and the criteria.
    """
    draft_lengths = force_drafted_tokens(engine, vocab_size, offset)
    results = generate_outputs(engine, prompts, max_new_tokens)
    stats = runner_stats(engine)
    if stats is None:
        print("   ✗ no speculative runner to read the verification outcome from")
        return results, False
    counts = stats["histogram"]
    accepted_below_budget = sum(
        count for size, count in counts.items() if size < requested_budget
    )
    # A round that drafted fewer than the budget accepted fewer by construction,
    # so it explains one of those rounds; what remains can only come from a round
    # that drafted the budget and lost a token.
    truncated = [length for length in draft_lengths if length < requested_budget]
    partial_rounds = accepted_below_budget - len(truncated)
    triggered = partial_rounds > 0
    drafted_full = bool(draft_lengths) and max(draft_lengths) == requested_budget
    state_clean = stats["scratch_exhausted"] == 0
    print(f"   drafted per round {draft_lengths}")
    print(f"   accepted per round {counts}")
    budget_ok = check_draft_budget(stats, requested_budget)
    print(
        f"   {'✓' if drafted_full else '✗'} a round drafted the full budget:"
        f" most drafted {max(draft_lengths) if draft_lengths else 0}"
    )
    print(
        f"   {'✓' if triggered else '✗'} partially accepted verifications beyond"
        f" truncated rounds: {partial_rounds} (accepted<{requested_budget}:"
        f" {accepted_below_budget}, truncated rounds: {len(truncated)})"
    )
    if not state_clean:
        print(f"   ✗ state-row exhaustion fall-backs: {stats['scratch_exhausted']}")
    return results, budget_ok and drafted_full and triggered and state_clean


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5 MTP speculative losslessness test (greedy, token-exact)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a Qwen3.5 checkpoint other than the tiny one this check "
        "writes; without it that tiny checkpoint, carrying the target and its "
        "draft head, is built here",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device for both runs, e.g. cpu or cuda (default: %(default)s)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Greedy tokens generated per prompt (default: %(default)s)",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        nargs="+",
        default=list(DEFAULT_DRAFT_TOKEN_COUNTS),
        help="Drafted tokens verified per target step; the lossless comparison "
        "runs once per value (default: 1 2 3 4), and the forced-partial section "
        "once per value raised to two. A checkpoint other than the tiny fixture "
        "is only lossless at K=1, so --model narrows this to 1",
    )
    args = parser.parse_args()

    model_dir = args.model
    # The scope of the multi-token comparison follows the checkpoint, not how it
    # was passed: --model pointing at this file's own fixture is the checkpoint
    # the default run writes, and keeps the same criteria.
    external_checkpoint = model_dir is not None and not is_tiny_fixture(model_dir)
    # Validate before anything narrows the request, so an impossible value is
    # rejected instead of being quietly replaced by the external-checkpoint rule.
    if not args.num_draft_tokens or any(count < 1 for count in args.num_draft_tokens):
        print("✗ --num-draft-tokens must be >= 1")
        return 1
    # A repeated value would run the same sweep twice; the help says once each.
    args.num_draft_tokens = list(dict.fromkeys(args.num_draft_tokens))

    print("=" * 70)
    print("Qwen3.5 MTP Speculative Decoding Losslessness Test")
    print("=" * 70)
    print(f"Model: {model_dir or 'synthetic checkpoint'}")
    print(f"Device: {args.device}")
    print(
        f"Prompts: {len(DEFAULT_PROMPTS)} fixed inputs, {args.max_new_tokens} new tokens each"
    )
    if external_checkpoint:
        counts = [count for count in args.num_draft_tokens if count == 1]
        if len(counts) != len(args.num_draft_tokens):
            print(
                "   NOTE: a checkpoint other than this file's tiny fixture is "
                "lossless at K=1 only; the multi-token values in "
                "--num-draft-tokens are ignored"
            )
        args.num_draft_tokens = [1]
    print(f"Num draft tokens: {' '.join(str(c) for c in args.num_draft_tokens)}")
    print("=" * 70)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("✗ CUDA device requested but torch.cuda is not available")
        return 1
    if args.max_new_tokens < 32:
        print("✗ --max-new-tokens must be >= 32 for the losslessness check")
        return 1

    root = None
    ok = True
    fixtures_before = built_fixtures()
    if model_dir is None:
        root = tempfile.mkdtemp(prefix="infinilm_qwen35_lossless_")
        write_checkpoint(root)
        write_tokenizer(root)
        model_dir = root
        print(f"\n0. Wrote the synthetic checkpoint ({root})...")

    def drop_fixtures():
        """Remove the fixture directories the engine built just now created."""
        removed = cleanup_built_fixtures(fixtures_before)
        if removed:
            print(f"   (removed {removed} draft fixture director(ies))")

    try:
        print("\n1. Baseline run (speculation off)...")
        baseline = build_engine(
            model_dir, None, args.device, args.max_new_tokens, args.num_draft_tokens[0]
        )
        try:
            baseline_results = generate_outputs(
                baseline, DEFAULT_PROMPTS, args.max_new_tokens
            )
        finally:
            # Release the engine before the speculative run: the target and draft
            # engines must not share the GPU with the previous run's weights.
            baseline.close()
            del baseline
            gc.collect()
        print(f"   ✓ {len(baseline_results)} prompts generated")

        for num_draft_tokens in args.num_draft_tokens:
            print(f"\n2.{num_draft_tokens} Speculative run (K={num_draft_tokens})...")
            engine = build_engine(
                model_dir,
                model_dir,
                args.device,
                args.max_new_tokens,
                num_draft_tokens,
            )
            try:
                speculative_results = generate_outputs(
                    engine, DEFAULT_PROMPTS, args.max_new_tokens
                )
                print(f"   ✓ {len(speculative_results)} prompts generated")
                print("   acceptance stats:")
                stats = report_accept_stats(engine)
                print("   comparison:")
                ok &= (
                    compare_outputs(
                        DEFAULT_PROMPTS, baseline_results, speculative_results
                    )
                    is True
                )
                ok &= check_draft_budget(stats, num_draft_tokens)
                ok &= stats is not None and stats["scratch_exhausted"] == 0
            finally:
                engine.close()
                del engine
                gc.collect()
                drop_fixtures()

        # One forced run per configured budget, each raised to two: a single
        # drafted token has no tail to reject, so K=1 cannot reach the partially
        # accepted path. An external checkpoint's sweep was narrowed to K=1
        # above, which leaves its one forced budget at K=2.
        first_offset, second_offset = FORCED_OFFSETS
        for budget in sorted({max(2, count) for count in args.num_draft_tokens}):
            print(
                f"\n3.{budget} Forced partial acceptance (K={budget}),"
                " the hand-off path..."
            )
            results = []
            for offset in FORCED_OFFSETS:
                print(f"   rejected tail +{offset}:")
                engine = build_engine(
                    model_dir, model_dir, args.device, args.max_new_tokens, budget
                )
                try:
                    run_results, run_ok = run_forced_generation(
                        engine,
                        DEFAULT_PROMPTS,
                        args.max_new_tokens,
                        VOCAB_SIZE,
                        budget,
                        offset,
                    )
                finally:
                    engine.close()
                    del engine
                    gc.collect()
                    drop_fixtures()
                results.append(run_results)
                ok &= run_ok
            # A rejected tail must not reach the committed state, so the two
            # runs have to emit the same tokens. Comparing them keeps the
            # comparison inside the verification shape, where a near-tied step
            # rounds the same way in both runs.
            print("   rejected-tail comparison:")
            ok &= (
                compare_outputs(
                    DEFAULT_PROMPTS,
                    results[0],
                    results[1],
                    reference_name=f"tail +{first_offset}",
                    actual_name=f"tail +{second_offset}",
                )
                is True
            )
            print("   plain-baseline comparison:")
            matched = compare_outputs(
                DEFAULT_PROMPTS,
                baseline_results,
                results[0],
                report_only=external_checkpoint,
            )
            # Reported, not decided: on an external checkpoint the multi-token
            # boundary is what this comparison shows. A None return is an
            # internal error and stays fatal.
            if not external_checkpoint or matched is None:
                ok &= matched is True

        print("\n4. Target state pool...")
        engine = build_engine(model_dir, model_dir, args.device, args.max_new_tokens, 2)
        try:
            target_engine = engine.engine.model_runner.model_engine
            # A hybrid target must have a recurrent-state pool; its absence
            # would mean the runs above never exercised the hand-off at all.
            pool = target_engine.has_mamba_cache
            ok &= pool
            print(
                f"   {'✓' if pool else '✗'} target has_mamba_cache={pool}"
                " (a hybrid target needs the pool the verification borrows"
                " rows from)"
            )
            stats = runner_stats(engine)
            print(f"   draft description: {stats['family']}")
            # Run one generation and read the pool afterwards: a round that
            # could not get a state row falls back to single-token drafting,
            # which is the silent degradation this counter exists to catch.
            generate_outputs(engine, DEFAULT_PROMPTS[:1], args.max_new_tokens)
            stats = runner_stats(engine)
            rows_clean = stats["scratch_exhausted"] == 0
            ok &= rows_clean
            # A diagnostic line, not a criterion: what is asserted is that no
            # round fell back for want of a state row. Whether the runner still
            # holds the pool is printed because it is the only readable signal
            # for that side of the bookkeeping.
            print(
                f"   {'✓' if rows_clean else '✗'} state rows with a generation in"
                f" flight: held={stats['state_rows'] is not None} (reported, not"
                f" asserted), exhaustion fall-backs={stats['scratch_exhausted']}"
            )
        finally:
            engine.close()
            del engine
            gc.collect()
            drop_fixtures()
    finally:
        # Every engine is closed by now, so any fixture this run created can go;
        # a released checkpoint passed with --model is left untouched.
        drop_fixtures()
        if root is not None:
            remove_tree(root)

    print("\n" + "=" * 70)
    if ok:
        if external_checkpoint:
            print("✓ Losslessness passed for the claimed setting: K=1, batch size 1")
            print(
                "   the multi-token comparison against plain decoding is reported"
                " above, not asserted (an external checkpoint is not claimed"
                " lossless at K>1)"
            )
        else:
            print("✓ Losslessness passed: speculative output matches the baseline")
    else:
        print("✗ Losslessness failed: a criterion above did not hold")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
