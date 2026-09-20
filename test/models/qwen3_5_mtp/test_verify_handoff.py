#!/usr/bin/env python3
"""
Mechanism checks for the multi-token verification state hand-off (Qwen3.5).

The speculative runner verifies several draft tokens per target step. For a
hybrid target the recurrent linear-attention state must end up matching the
accepted prefix only: the verified batch advances a temporary state row while
the committed row is left untouched, a fully accepted verification takes over
the temporary row, and a partially accepted one re-advances the committed row
over the accepted tokens through the regular decode path.

These checks drive a single target engine directly over one fixed token
sequence and compare, for K = 2, 3 and 4:

  1. a K-token verification batch against K single-token steps, which is the
     decomposition the state hand-off assumes;
  2. the committed row after a partially accepted verification (replayed over
     the kept tokens in the decode shape) against the next plain-decoded token;
  3. the temporary row of a fully accepted verification against the plain step
     that follows all K tokens.

Every check compares sampled greedy tokens, so a state row that ran ahead of, or
fell behind, the kept sequence shows up as a differing prediction. Also
reported: the cost of the state-row bookkeeping and of stating the batch shape.
"""

import argparse
import gc
import os
import sys
import time
from types import SimpleNamespace

try:
    import torch
except ImportError as e:
    print(f"Error: Required packages not found. Please install: {e}")
    sys.exit(1)

try:
    import infinicore
    from infinilm.cache.cache import PagedKVCacheConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.llm.cache_manager import MambaCacheManager
    from infinilm.llm.model_runner.speculative_runner import SpeculativeRunner
    from infinilm.modeling_utils import load_model_state_dict_by_file
except ImportError as e:
    print("Error: InfiniLM package not found. Please install it:")
    print(f"  Error: {e}")
    sys.exit(1)

DEFAULT_MODEL_DIR = os.path.expanduser("~/models/Qwen3.5-2B")
DEFAULT_DEVICE = "cuda"
DEFAULT_PROMPT = (
    "The capital of France is Paris. The largest planet in the solar system is"
)
DEFAULT_SEQUENCE_LENGTH = 5
BLOCK_SIZE = 256
# 32 blocks give a state pool of max(2, 32 // 4) = 8 rows: row 0 is the zero
# state, rows 1..7 are the state rows these checks use.
NUM_BLOCKS = 32


# State rows used by these checks, kept distinct so a collision is loud.
SEQUENTIAL = 1
COMMITTED = 2
TEMPORARY = 3


def position_ids(positions, axes):
    return infinicore.from_list([list(positions)] * axes, dtype=infinicore.int64)


PAGED_BATCH_KEYS = (
    "input_ids",
    "position_ids",
    "past_kv_lengths",
    "total_kv_lengths",
    "input_offsets",
    "cu_seqlens",
    "block_tables",
    "slot_mapping",
)


class CheckFailed(Exception):
    """A failed check.

    Raised instead of an `assert` statement so the failure survives `python -O`,
    which strips assertions at compile time; this script's checks are the only
    gate on the K=2..4 state hand-off.
    """


def check(condition, message):
    if not condition:
        raise CheckFailed(message)


def check_paged_batch(batch):
    """Every paged batch writes its key/value entries, so these keys are needed.

    The attention backends assert `slot_mapping` at the forward entry, so a
    batch missing it stops the worker instead of failing here.
    """
    missing = [key for key in PAGED_BATCH_KEYS if key not in batch]
    check(not missing, f"batch is missing required keys: {missing}")


class TargetHarness:
    """Single target engine driven through the raw forward interface."""

    def __init__(self, model_dir, device):
        self.engine = InferEngine(
            model_path=model_dir,
            device=infinicore.device(device, 0),
            cache_config=PagedKVCacheConfig(
                num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE, max_batch_size=1
            ),
            attention_backend="paged-attn",
        )
        load_model_state_dict_by_file(self.engine, model_dir, dtype=self.engine.dtype)
        self.axes = self.engine.position_id_axes
        # A stand-in for the speculative runner that lends only its batch
        # builders, so the verification and replay batches are built by the code
        # production uses and their key sets stay in the test.
        self.runner = SpeculativeRunner.__new__(SpeculativeRunner)
        self.runner._mamba_cache = None
        self.runner._cache_block_size = BLOCK_SIZE
        # Two blocks cover the prompt (well inside block 0) and every position
        # these checks write, so no written position aliases another.
        self.block_table = [0, 1]

    def _run(
        self,
        tokens,
        positions,
        past_lens,
        seq_lens,
        offsets,
        cu_seqlens,
        slots,
        init_index,
        final_index,
        multi_token_batch=False,
    ):
        return self.engine.forward_raw(
            infinicore.from_list([list(tokens)], dtype=infinicore.int64),
            position_ids=position_ids(positions, self.axes),
            past_kv_lengths=infinicore.from_list(past_lens, dtype=infinicore.int32),
            total_kv_lengths=infinicore.from_list(seq_lens, dtype=infinicore.int32),
            input_offsets=infinicore.from_list(offsets, dtype=infinicore.int32),
            cu_seqlens=infinicore.from_list(cu_seqlens, dtype=infinicore.int32),
            block_tables=infinicore.from_list(
                [self.block_table], dtype=infinicore.int32
            ),
            slot_mapping=infinicore.from_list(slots, dtype=infinicore.int64),
            mamba_init_state_indices=infinicore.from_list(
                [init_index], dtype=infinicore.int32
            ),
            mamba_final_state_indices=infinicore.from_list(
                [final_index], dtype=infinicore.int32
            ),
            mamba_multi_token_batch=multi_token_batch,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
        )

    def prefill(self, token_ids, final_state_index):
        """Prompt pass: the single request occupies the whole packed batch.

        A prefill batch carries one offset per request plus the trailing total,
        which is what tells the model this batch is not the decode shape.
        """
        return self._run(
            token_ids,
            range(len(token_ids)),
            [0],
            [len(token_ids)],
            [0, len(token_ids)],
            [0, len(token_ids)],
            list(range(len(token_ids))),
            0,
            final_state_index,
        )

    def step(self, token, position, past_len, state_index):
        """One single-token step against a request's own state row."""
        return self._run(
            [token],
            [position],
            [past_len],
            [past_len + 1],
            [0, 1],
            [0, past_len + 1],
            [past_len],
            state_index,
            state_index,
        )

    def replay(self, token_ids, base_len, state_index):
        """A packed multi-token decode, built by the runner's replay builder."""
        batch = self.runner._build_state_replay_batch_input(
            [
                {
                    "req": SimpleNamespace(
                        block_table=list(self.block_table),
                        mamba_cache_index=state_index,
                    ),
                    "base_len": base_len,
                    "draft_tokens": list(token_ids),
                    "accepted": len(token_ids),
                    "scratch_index": state_index,
                }
            ]
        )
        check_paged_batch(batch)
        print(
            "   [INFO] replay batch slot_mapping:"
            f" {batch['slot_mapping'].to_numpy().tolist()}"
        )
        return self.engine.forward_raw(**batch)

    def verify(self, token_ids, first_position, init_index, final_index):
        """One verification batch reading init and writing final.

        Built by the runner's own verification builder, so the batch carries
        exactly the keys production sends.
        """
        count = len(token_ids)
        batch = self.runner._build_paged_verify_batch_input(
            [
                {
                    "req": SimpleNamespace(
                        block_table=list(self.block_table),
                        mamba_cache_index=init_index,
                    ),
                    "base_len": first_position,
                    "draft_tokens": list(token_ids),
                    "slot_mapping": [first_position + step for step in range(count)],
                    "scratch_index": (
                        final_index if final_index != init_index else None
                    ),
                }
            ]
        )
        check_paged_batch(batch)
        return self.engine.forward_raw(**batch)


def argmax_ids(output):
    return [int(token) for token in output["output_ids"].to_numpy().tolist()]


def run_check(label, expected, actual):
    ok = expected == actual
    print(f"   [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"          expected {expected}")
        print(f"          actual   {actual}")
    return ok


def plain_decode_step(harness, previous, position, state_index):
    """One token of a plain decode chain, returning the predicted next token."""
    return argmax_ids(harness.step(previous, position, position, state_index))[0]


def check_verification(harness, prompt_ids, count):
    """Compare a K-token verification batch against plain decoding.

    A verification batch is the anchor token plus the drafted tokens that follow
    it. The anchor is the token the target produced from the prompt, so nothing
    but the prompt is cached when the batch starts. Decoding the same tokens one
    at a time on a separate state row gives the prediction every batch position
    has to reproduce; both sides start from the prompt, so no other work has to
    be kept in step.
    """
    base_len = len(prompt_ids)
    first_position = base_len

    # Plain decode chain: step i feeds the token the previous step predicted,
    # so prediction[i] is the token each batch position must reproduce.
    harness.prefill(prompt_ids, COMMITTED)
    prediction = []
    for step in range(count + 2):
        previous = prompt_ids[-1] if step == 0 else prediction[step - 1]
        prediction.append(
            argmax_ids(
                harness.step(
                    previous,
                    first_position + step - 1,
                    first_position + step - 2,
                    COMMITTED,
                )
            )[0]
        )

    payload = prediction[:count]
    harness.prefill(prompt_ids, COMMITTED)
    verify_ids = argmax_ids(
        harness.verify(payload, first_position, COMMITTED, TEMPORARY)
    )
    check(
        verify_ids == prediction[1 : count + 1],
        f"K={count} verification disagrees with plain decoding:"
        f" {verify_ids} vs {prediction[1 : count + 1]}",
    )

    # Partially accepted batch: the committed row still holds the state the
    # batch read, so re-advancing it over the kept tokens in the decode shape
    # must predict what that prefix predicts.
    kept = count - 1
    replayed = argmax_ids(harness.replay(payload[:kept], first_position - 1, COMMITTED))
    check(
        replayed[-1] == prediction[kept],
        f"K={count} committed row after replaying {kept} kept token(s) predicts"
        f" {replayed[-1]}, plain decoding predicts {prediction[kept]}",
    )
    continued = argmax_ids(
        harness.step(
            prediction[count],
            first_position + count,
            first_position + count - 1,
            TEMPORARY,
        )
    )
    check(
        continued[0] == prediction[count + 1],
        f"K={count} temporary row after the full verification predicts"
        f" {continued[0]}, plain decoding predicts {prediction[count + 1]}",
    )
    print(
        f"   [PASS] K={count}: {count} batch predictions, {kept}-token replay and"
        " full-accept hand-off all match plain decoding"
    )


def check_shape_declaration(harness, prompt_ids):
    """The batch states its shape, and a single-token batch states the decode shape.

    The batch builder must always state whether requests hold more than one
    token: a single-token verification is structurally a decode step, so it has
    to declare the decode shape and take the decode kernel. Leaving the shape
    unstated would fall back to inferring it from the packed layout, which is
    what stating it explicitly exists to avoid.
    """
    base_len = len(prompt_ids)

    def build(draft_counts):
        candidates = []
        for index, count in enumerate(draft_counts):
            candidates.append(
                {
                    "req": SimpleNamespace(
                        block_table=list(harness.block_table),
                        mamba_cache_index=COMMITTED,
                    ),
                    "base_len": base_len,
                    "draft_tokens": [48017] * count,
                    "slot_mapping": [base_len + step for step in range(count)],
                    "scratch_index": TEMPORARY if count > 1 else None,
                }
            )
        return harness.runner._build_paged_verify_batch_input(candidates)

    single = build([1])
    check(
        "mamba_multi_token_batch" in single,
        "batch builder left the shape unstated; the model would infer it from"
        " the packed layout",
    )
    check(
        single["mamba_multi_token_batch"] is False,
        "a single-token verification batch declared the multi-token shape, which"
        " routes it to the batched kernel instead of the decode kernel",
    )

    multi = build([2])
    check(
        multi["mamba_multi_token_batch"] is True,
        "a two-token verification batch did not declare the multi-token shape",
    )

    mixed = build([1, 2])
    check(
        mixed["mamba_multi_token_batch"] is True,
        "a batch holding a two-token request did not declare the multi-token shape",
    )
    print(
        "   [PASS] batch shape stated: single-token batch -> decode shape,"
        " two-token and mixed batches -> multi-token shape"
    )

    # The single-token verification must also produce the decode step's token.
    tokens = [48017]
    harness.prefill(prompt_ids, COMMITTED)
    verified = argmax_ids(harness.verify(tokens, base_len, COMMITTED, COMMITTED))
    harness.prefill(prompt_ids, COMMITTED)
    decoded = argmax_ids(harness.step(tokens[0], base_len, base_len - 1, COMMITTED))
    check(
        verified == decoded,
        f"single-token verification {verified} differs from the decode step {decoded}",
    )
    print(
        "   [PASS] single-token verification produces the decode step's token:"
        f" {verified}"
    )


def check_bookkeeping(rounds):
    """Cost of one borrow/swap/release cycle, the per-round orchestration work."""
    manager = MambaCacheManager(NUM_BLOCKS)
    committed = manager.borrow_slot()
    start = time.perf_counter()
    for _ in range(rounds):
        # One accepted verification: borrow a row, hand it to the request and
        # give the replaced row back. The swap itself moves no state data.
        scratch = manager.borrow_slot()
        manager.swap_slots(committed, scratch)
        manager.release_slot(committed)
        committed = scratch
    elapsed = time.perf_counter() - start
    print(
        f"   [INFO] borrow + swap + release: {1e6 * elapsed / rounds:.2f} us"
        f" per accepted round ({rounds} rounds)"
    )


def check_marker_cost(harness, prompt_ids, reps):
    """Cost of stating the batch shape, against letting the model infer it.

    Both forms must take the decode kernel for a single-token batch, so the two
    numbers are expected to agree; a gap here would mean the declared shape sent
    the batch down a different kernel than a plain decode step uses.
    """
    base_len = len(prompt_ids)
    tokens = [48017]

    def timing(stated):
        start = time.perf_counter()
        if stated:
            harness.verify(tokens, base_len, COMMITTED, TEMPORARY)
        else:
            harness.step(tokens[0], base_len, base_len - 1, COMMITTED)
        return (time.perf_counter() - start) * 1000.0

    for stated, label in (
        (True, "K=1 shape stated (decode shape)"),
        (False, "K=1 shape inferred (decode shape)"),
    ):
        timing(stated)
        samples = sorted(timing(stated) for _ in range(reps))
        print(
            f"   [INFO] {label}: median {samples[len(samples) // 2]:.2f} ms (n={reps})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-token verification state hand-off checks (Qwen3.5)"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=15,
        help="Timed calls per shape for the marker cost (default: %(default)s)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Anchor tokens decoded before the checks (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("✗ CUDA device requested but torch.cuda is not available")
        return 1
    if args.sequence_length < 4:
        print("✗ --sequence-length must be at least 4")
        return 1

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    prompt_ids = list(tokenizer.encode(args.prompt))
    if len(prompt_ids) < 2:
        print("✗ --prompt must tokenize to at least two tokens")
        return 1

    print("=" * 70)
    print("Multi-token verification state hand-off checks (Qwen3.5 hybrid)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Prompt: {args.prompt!r} ({len(prompt_ids)} tokens)")
    print(f"State rows: {NUM_BLOCKS // 4} (row 0 is the zero state)")

    harness = TargetHarness(args.model, args.device)
    ok = True
    try:
        print("\n1. Batched verification against plain decoding...")
        for count in (2, 3, 4):
            check_verification(harness, prompt_ids, count)

        print("\n2. Batch shape declaration...")
        check_shape_declaration(harness, prompt_ids)

        print("\n3. State-row bookkeeping cost...")
        check_bookkeeping(2000)

        print("\n4. Cost of stating the batch shape...")
        check_marker_cost(harness, prompt_ids, args.repetitions)
    except (CheckFailed, AssertionError) as failure:
        # AssertionError too: a lower layer that asserts should read as a failed
        # check here rather than escaping as a traceback.
        ok = False
        print(f"   [FAIL] {failure}")
    finally:
        del harness
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    if ok:
        print("✓ Mechanism checks passed: verification, partial-accept replay and")
        print("  full-accept hand-off predict identically to sequential decoding")
    else:
        print("✗ Mechanism checks failed: see the mismatching checks above")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
