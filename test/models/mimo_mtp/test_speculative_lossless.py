#!/usr/bin/env python3
"""Speculative losslessness for MiMo MTP drafts (synthetic checkpoint, no weights).

With greedy decoding, every run is compared token by token against the same run
with speculation off: the same fixed prompt set, the same generation length, the
same paged cache and the same in-process baseline as the Qwen3.5 check. The
number of drafted tokens verified per target step is configurable, and
``--deterministic-partial`` corrupts the drafted tail so a multi-token
verification has to take the partially accepted path.

Two parts of the comparison are specific to this family:

  * the checkpoint is synthetic — the released MiMo checkpoint is 15 GB and is
    not part of this repository, so the tiny checkpoint publishes the released
    tensor names and the draft is resolved through the family description
    exactly as a released checkpoint is. The target and the draft head live in
    one directory;
  * a MiMo target is pure attention, so the recurrent-state bookkeeping of the
    multi-token verification stays out of the loop. The report states that from
    the runner while verifications are in flight (section 3) and from the built
    engines (section 5).

The paged KV cache the speculative path needs has no CPU implementation in
InfiniCore (``paged_caching`` reports "Device Type Not Supported"), so this
check needs a GPU, like the other speculative checks in this repository.
"""

import argparse
import gc
import json
import os
import sys
import tempfile

import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "llama"))
sys.path.insert(0, _TEST_DIR)

from infinilm.llm.llm import LLM  # noqa: E402
from infinilm.llm.sampling_params import SamplingParams  # noqa: E402
from test_forward_validation import (  # noqa: E402
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_HEADS,
    NUM_KV_HEADS,
    VOCAB_SIZE,
    build_tiny_config,
    build_tiny_weights,
    remove_tree,
    write_tiny_checkpoint,
)

DEFAULT_DEVICE = "cuda"
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_DRAFT_TOKEN_COUNTS = (1, 2, 3, 4)
# The scheduler requires max_num_batched_tokens >= 1024 and a matching
# max_position_embeddings; both are tiny models in every other respect.
MAX_POSITIONS = 2048
MAX_CACHE_LEN = 512
BLOCK_SIZE = 16
NUM_BLOCKS = 8
TARGET_LAYERS = 2
# The paged-attention prefill kernel refuses the small head_dim the
# component-level fixture uses, so this check runs a wider one.
PAGED_HEAD_DIM = 64
# A token the target is not expected to predict, used for the forced tail.
FORCED_OFFSET = 7
DEFAULT_PROMPTS = [
    "1 + 1 =",
    "def fibonacci(n):",
    "SELECT * FROM users WHERE",
    "The meaning of life is",
    "The Eiffel Tower is located in",
    "The following is a list of prime numbers: 2, 3, 5, 7,",
]
# The released checkpoint ships its own configuration class and an auto_map that
# points at it; transformers does not know the "mimo" model type on its own, so
# the synthetic checkpoint publishes the same pair.
CONFIG_MODULE = """from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class MiMoConfig(Qwen2Config):
    model_type = "mimo"

    def __init__(self, *args, num_nextn_predict_layers=0, **kwargs):
        self.num_nextn_predict_layers = num_nextn_predict_layers
        super().__init__(*args, **kwargs)
"""


def target_weights(seed=7, head_dim=PAGED_HEAD_DIM):
    """Target tensors of the released layout, on top of the draft tensors."""
    generator = torch.Generator().manual_seed(seed)

    def weight(*shape):
        return (torch.randn(*shape, generator=generator) * 0.05).to(torch.float32)

    def scale():
        return (1.0 + torch.randn(HIDDEN_SIZE, generator=generator) * 0.1).to(
            torch.float32
        )

    q_out = NUM_HEADS * head_dim
    kv_out = NUM_KV_HEADS * head_dim
    weights = {
        "model.embed_tokens.weight": weight(VOCAB_SIZE, HIDDEN_SIZE),
        "lm_head.weight": weight(VOCAB_SIZE, HIDDEN_SIZE),
        "model.norm.weight": scale(),
    }
    for layer in range(TARGET_LAYERS):
        prefix = f"model.layers.{layer}."
        weights[prefix + "input_layernorm.weight"] = scale()
        weights[prefix + "post_attention_layernorm.weight"] = scale()
        for proj, dim in (("q_proj", q_out), ("k_proj", kv_out), ("v_proj", kv_out)):
            weights[prefix + f"self_attn.{proj}.weight"] = weight(dim, HIDDEN_SIZE)
            weights[prefix + f"self_attn.{proj}.bias"] = weight(dim)
        weights[prefix + "self_attn.o_proj.weight"] = weight(HIDDEN_SIZE, q_out)
        weights[prefix + "mlp.gate_proj.weight"] = weight(
            INTERMEDIATE_SIZE, HIDDEN_SIZE
        )
        weights[prefix + "mlp.up_proj.weight"] = weight(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        weights[prefix + "mlp.down_proj.weight"] = weight(
            HIDDEN_SIZE, INTERMEDIATE_SIZE
        )
    return weights


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
            {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 128}, f
        )


def write_checkpoint(root):
    """A checkpoint carrying both the target and its embedded draft head."""
    config = build_tiny_config(head_dim=PAGED_HEAD_DIM)
    config["num_hidden_layers"] = TARGET_LAYERS
    config["max_position_embeddings"] = MAX_POSITIONS
    # The NVIDIA paged-attention prefill kernel has no float32 path.
    config["torch_dtype"] = "bfloat16"
    config["auto_map"] = {"AutoConfig": "configuration_mimo.MiMoConfig"}
    weights = target_weights()
    weights.update(build_tiny_weights(head_dim=PAGED_HEAD_DIM))
    write_tiny_checkpoint(root, weights, config=config)
    write_tokenizer(root)
    with open(os.path.join(root, "configuration_mimo.py"), "w") as f:
        f.write(CONFIG_MODULE)
    return weights


def build_engine(model_dir, draft_model_dir, device, max_new_tokens, num_draft_tokens):
    """The engine the library builds for a MiMo checkpoint with MTP drafting."""
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
    runner = getattr(engine.engine.model_runner, "speculative_runner", None)
    if runner is None:
        return None
    return {
        "accepted": runner.eagle_accept_count,
        "total": runner.eagle_total_count,
        "histogram": dict(runner.accepted_count_histogram),
        "budget": runner.num_draft_tokens,
        "scratch_exhausted": runner.verify_scratch_exhausted,
        "mamba_cache": runner._mamba_cache,
        "spec": runner.draft_spec,
    }


def report_accept_stats(engine):
    """Read the speculative runner's acceptance counters, if reachable."""
    runner = getattr(engine.engine.model_runner, "speculative_runner", None)
    if runner is None:
        print("   (no speculative runner found)")
        return
    total = runner.eagle_total_count
    accepted = runner.eagle_accept_count
    rate = accepted / total if total else 0.0
    print(
        f"   accepted {accepted}/{total} drafted tokens"
        f" ({100.0 * rate:.1f}% acceptance)"
    )
    histogram = runner.accepted_count_histogram
    if histogram:
        rounds = sum(histogram.values())
        counts = ", ".join(f"{size}:{histogram[size]}" for size in sorted(histogram))
        print(f"   verification rounds {rounds}, accepted per round {{{counts}}}")
        if runner.num_draft_tokens > 1:
            partial = sum(
                count
                for size, count in histogram.items()
                if size < runner.num_draft_tokens
            )
            print(
                "   rounds that accepted fewer than"
                f" {runner.num_draft_tokens} drafted tokens: {partial}/{rounds}"
            )
    if runner.verify_scratch_exhausted:
        print(f"   ✗ state-row exhaustion fallbacks: {runner.verify_scratch_exhausted}")


def force_partial_acceptance(engine):
    """Make the drafted tail disagree with the target so acceptance is partial.

    The first drafted token stays as drafted (the target's own previous token,
    always accepted); every later one is replaced by the previous token, which
    cannot match the target's greedy continuation.
    """
    runner = getattr(engine.engine.model_runner, "speculative_runner", None)
    if runner is None:
        raise RuntimeError("no speculative runner to patch for partial acceptance")
    original = runner._draft_eagle_tokens_batch

    def corrupt_tail(jobs):
        results = original(jobs)
        for tokens in results:
            for index in range(1, len(tokens)):
                tokens[index] = tokens[index - 1]
        return results

    runner._draft_eagle_tokens_batch = corrupt_tail
    print("   forced partial acceptance: drafted tail replaced (positions >= 1)")


def force_drafted_tokens(engine, partial, target_sequences):
    """Replace what the draft proposes, to drive both verification paths.

    A verification only accepts while the drafted token equals the target's own
    greedy token, so the partial variant keeps the first drafted token (the
    target's token for that position) and proposes a token the target is not
    expected to predict afterwards. The full variant proposes exactly the tokens
    the target itself generated at those positions, which every verification
    accepts.

    Returns the number of tokens the draft proposed in each verification round,
    which is what tells a round that accepted fewer than it drafted from a round
    whose budget the end of the generation cut short.
    """
    runner = engine.engine.model_runner.speculative_runner
    original = runner._draft_eagle_tokens_batch
    lengths = []

    def sequence_for(job):
        return target_sequences[tuple(job["req"].prompt_token_ids)]

    def partial_draft(jobs):
        results = original(jobs)
        for job, tokens in zip(jobs, results):
            target = int(job["target_token"])
            for index in range(len(tokens)):
                tokens[index] = (
                    target if index == 0 else (target + FORCED_OFFSET) % VOCAB_SIZE
                )
            lengths.append(len(tokens))
        return results

    def full_draft(jobs):
        results = original(jobs)
        for job, tokens in zip(jobs, results):
            sequence = sequence_for(job)
            # The first drafted token follows the request's last input token.
            base = int(job["source_position"]) + 1
            for index in range(len(tokens)):
                tokens[index] = sequence[(base + index) % len(sequence)]
            lengths.append(len(tokens))
        return results

    runner._draft_eagle_tokens_batch = partial_draft if partial else full_draft
    return lengths


def compare_outputs(baseline_results, speculative_results):
    """Print the per-prompt comparison; returns whether every prompt matched."""
    if len(baseline_results) != len(DEFAULT_PROMPTS) or len(speculative_results) != len(
        DEFAULT_PROMPTS
    ):
        print(
            "✗ Internal error: expected one result per prompt"
            f" (baseline={len(baseline_results)},"
            f" speculative={len(speculative_results)})"
        )
        return False
    all_match = True
    for prompt, (_, baseline_ids), (_, speculative_ids) in zip(
        DEFAULT_PROMPTS, baseline_results, speculative_results
    ):
        # Two empty sequences would compare equal; treat them as a failure.
        match = (
            bool(baseline_ids)
            and bool(speculative_ids)
            and (baseline_ids == speculative_ids)
        )
        all_match = all_match and match
        status = "✓" if match else "✗"
        print(f"   {status} {prompt!r} ({len(speculative_ids)} tokens)")
        if not match:
            if len(baseline_ids) != len(speculative_ids):
                print(
                    f"       length mismatch: baseline={len(baseline_ids)},"
                    f" speculative={len(speculative_ids)}"
                )
            first_div = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(baseline_ids, speculative_ids))
                    if a != b
                ),
                min(len(baseline_ids), len(speculative_ids)),
            )
            print(f"       first divergence at position {first_div}")
            print(f"       baseline:    {baseline_ids}")
            print(f"       speculative: {speculative_ids}")
    return all_match


def main():
    parser = argparse.ArgumentParser(
        description="MiMo MTP speculative losslessness test (greedy, token-exact)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a released MiMo checkpoint; without it a tiny synthetic "
        "checkpoint carrying the target and its draft head is used",
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
        "runs once per value (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--deterministic-partial",
        action="store_true",
        help="Corrupt the drafted tail so every verification whose draft budget "
        "exceeds one token takes the partially accepted path",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MiMo MTP Speculative Decoding Losslessness Test")
    print("=" * 70)
    print(f"Model: {args.model or 'synthetic checkpoint'}")
    print(f"Device: {args.device}")
    print(
        f"Prompts: {len(DEFAULT_PROMPTS)} fixed inputs, {args.max_new_tokens} new tokens each"
    )
    print(f"Num draft tokens: {args.num_draft_tokens}")
    if args.deterministic_partial:
        print("Partial acceptance: forced on every multi-token verification")
    print("=" * 70)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("✗ CUDA device requested but torch.cuda is not available")
        print("  drafting needs the paged KV cache, which has no CPU backend")
        return 1
    if args.max_new_tokens < 32:
        print("✗ --max-new-tokens must be >= 32 for the losslessness check")
        return 1
    if any(count < 1 for count in args.num_draft_tokens):
        print("✗ --num-draft-tokens must be >= 1")
        return 1
    if args.deterministic_partial and min(args.num_draft_tokens) < 2:
        print("✗ --deterministic-partial needs --num-draft-tokens >= 2")
        return 1

    root = None
    model_dir = args.model
    ok = True
    if model_dir is None:
        root = tempfile.mkdtemp(prefix="infinilm_mimo_lossless_")
        write_checkpoint(root)
        model_dir = root
        print(f"\n0. Wrote the synthetic checkpoint ({root})...")

    try:
        print("\n1. Baseline run (speculation off)...")
        # The baseline has no draft model, so its draft budget is inert; it is
        # passed the run's first value to mirror the Qwen3.5 script's shape.
        baseline_engine = build_engine(
            model_dir, None, args.device, args.max_new_tokens, args.num_draft_tokens[0]
        )
        try:
            baseline_results = generate_outputs(
                baseline_engine, DEFAULT_PROMPTS, args.max_new_tokens
            )
        finally:
            # Release the engine before the speculative run: the target and draft
            # engines must not share the GPU with the previous run's weights.
            baseline_engine.close()
            del baseline_engine
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
                if args.deterministic_partial:
                    force_partial_acceptance(engine)
                speculative_results = generate_outputs(
                    engine, DEFAULT_PROMPTS, args.max_new_tokens
                )
                print(f"   ✓ {len(speculative_results)} prompts generated")
                print("   acceptance stats:")
                report_accept_stats(engine)
                print("   comparison:")
                ok &= compare_outputs(baseline_results, speculative_results)
            finally:
                engine.close()
                del engine
                gc.collect()

        print("\n3. Forced partial acceptance (K=2)...")
        target_sequences = {
            tuple(prompt_ids): prompt_ids + tokens
            for prompt_ids, tokens in baseline_results
        }
        engine = build_engine(model_dir, model_dir, args.device, args.max_new_tokens, 2)
        try:
            draft_lengths = force_drafted_tokens(engine, True, target_sequences)
            speculative_results = generate_outputs(
                engine, DEFAULT_PROMPTS, args.max_new_tokens
            )
            stats = runner_stats(engine)
            same = compare_outputs(baseline_results, speculative_results)
            budget = stats["budget"]
            accepted_below_budget = sum(
                count for size, count in stats["histogram"].items() if size < budget
            )
            # A round that drafted fewer than the budget accepted fewer by
            # construction, so it explains one of those rounds; what remains can
            # only come from a round that drafted the budget and lost a token.
            truncated = [length for length in draft_lengths if length < budget]
            partial_rounds = accepted_below_budget - len(truncated)
            triggered = partial_rounds > 0
            state_clean = (
                stats["mamba_cache"] is None and stats["scratch_exhausted"] == 0
            )
            ok &= same and triggered and state_clean
            print(
                f"   drafted per round {draft_lengths},"
                f" accepted per round {stats['histogram']}"
            )
            print(
                f"   {'✓' if triggered else '✗'} partially accepted verifications"
                f" beyond truncated rounds: {partial_rounds} (accepted<{budget}:"
                f" {accepted_below_budget}, truncated rounds: {len(truncated)})"
            )
            print(
                f"   {'✓' if state_clean else '✗'} state rows after"
                f" {len(draft_lengths)} verification rounds:"
                f" retained={stats['mamba_cache']},"
                f" scratch fallbacks={stats['scratch_exhausted']}"
            )
        finally:
            engine.close()
            del engine
            gc.collect()

        print("\n4. Forced full acceptance (K=2), for the other verification path...")
        engine = build_engine(model_dir, model_dir, args.device, args.max_new_tokens, 2)
        try:
            force_drafted_tokens(engine, False, target_sequences)
            speculative_results = generate_outputs(
                engine, DEFAULT_PROMPTS, args.max_new_tokens
            )
            stats = runner_stats(engine)
            same = compare_outputs(baseline_results, speculative_results)
            full_rounds = stats["histogram"].get(stats["budget"], 0)
            ok &= same and full_rounds > 0
            print(
                f"   {'✓' if full_rounds else '✗'} fully accepted verifications:"
                f" {full_rounds}, accepted per round {stats['histogram']}"
            )
        finally:
            engine.close()
            del engine
            gc.collect()

        print("\n5. Target state pool (build-time facts)...")
        engine = build_engine(model_dir, model_dir, args.device, args.max_new_tokens, 2)
        try:
            target_engine = engine.engine.model_runner.model_engine
            stats = runner_stats(engine)
            clean = not target_engine.has_mamba_cache and stats["mamba_cache"] is None
            ok &= clean
            print(
                f"   {'✓' if clean else '✗'} target has_mamba_cache="
                f"{target_engine.has_mamba_cache},"
                f" runner state rows={stats['mamba_cache']}"
            )
            print(
                "   (the run-time half of this check is section 3: its verification"
                " rounds finished with no state row borrowed)"
            )
            print(f"   draft description: {stats['spec'].family}")
        finally:
            engine.close()
            del engine
            gc.collect()
    finally:
        if root is not None:
            remove_tree(root)

    print("\n" + "=" * 70)
    if ok:
        print("✓ Losslessness passed: speculative output matches the baseline")
    else:
        print("✗ Losslessness failed: outputs diverge")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
