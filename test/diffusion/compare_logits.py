"""Measure how far saved logits are from a float32 reference.

Each prefix names the files written by ``reference_logits.py`` or
``engine_logits.py``. The report prints, per stage of the block loop, the KL
divergence from the truth, the share of positions far from it and how often
the most likely token agrees.
"""

import argparse
from pathlib import Path

import numpy as np

STAGES = (
    ("prompt_logits", "prefill"),
    ("preview_full_mask_logits", "preview, all-mask block"),
    ("preview_partial_logits", "preview, after first unmask"),
    ("replay_logits", "replay of resolved block"),
    ("next_preview_logits", "preview of next block"),
)


def parse_cases(value):
    """Return the workload ids given as a comma-separated list."""
    try:
        cases = sorted({int(item) for item in value.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Cases must be a comma-separated list of ids."
        ) from error
    if not cases or cases[0] < 0:
        raise argparse.ArgumentTypeError("Cases must be a comma-separated list of ids.")
    return cases


def softmax(logits):
    """Return the softmax of the last axis, evaluated in float64."""
    shifted = logits.astype(np.float64)
    shifted -= shifted.max(-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(-1, keepdims=True)


def kl_divergence(truth, logits):
    """Return the per-position KL divergence `KL(truth || logits)`."""
    expected, actual = softmax(truth), softmax(logits)
    return (expected * (np.log(expected + 1e-300) - np.log(actual + 1e-300))).sum(-1)


def load(prefix, cases):
    """Return the recorded cases of a run, one dictionary per case."""
    missing = [case for case in cases if not Path(f"{prefix}_{case}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"`{prefix}` has no record for case {missing}.")
    return [np.load(f"{prefix}_{case}.npz") for case in cases]


def gather(truth, runs, key, measure):
    """Return `measure` over the positions of a stage, one array per run.

    A case is left out when the truth or any run did not record the stage,
    which happens when a block resolves in its first unmask step. The number
    of cases used is returned as well.
    """
    cases = [
        index
        for index, reference in enumerate(truth)
        if all(key in case for case in (reference, *(run[index] for run in runs)))
    ]
    if not cases:
        # Every case resolved in its first unmask step or recorded a
        # different stage set; the caller treats zero samples as "stage not
        # present" instead of crashing on an empty concatenation.
        return [[] for _ in runs], 0
    measures = [
        np.concatenate([measure(truth[index][key], run[index][key]) for index in cases])
        for run in runs
    ]
    return measures, len(cases)


def print_table(title, stat, table_rows, labels):
    """Print one table with a stage per row and a run per column."""
    width = max(12, *(len(label) for label in labels)) + 2
    margin = max(28, *(len(name) for name, _ in table_rows)) + 2
    print(title)
    print(f"{'':<{margin}}" + "".join(f"{label:>{width}}" for label in labels))
    for name, measures in table_rows:
        print(
            f"{name:<{margin}}"
            + "".join(f"{stat(measure):>{width}.5f}" for measure in measures)
        )
    whole = [
        np.concatenate([measures[index] for _, measures in table_rows])
        for index in range(len(labels))
    ]
    print(
        f"{'all stages':<{margin}}"
        + "".join(f"{stat(measure):>{width}.5f}" for measure in whole)
    )
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truth", required=True, help="Prefix of the float32 reference."
    )
    parser.add_argument(
        "runs", nargs="+", metavar="LABEL=PREFIX", help="Run to compare with the truth."
    )
    parser.add_argument("--cases", type=parse_cases, default=parse_cases("0,1,2,3"))
    args = parser.parse_args()
    if any("=" not in run for run in args.runs):
        parser.error("Runs are written as LABEL=PREFIX.")
    try:
        truth = load(args.truth, args.cases)
        runs = [load(run.split("=", 1)[1], args.cases) for run in args.runs]
    except FileNotFoundError as error:
        parser.error(str(error))

    labels = [run.split("=", 1)[0] for run in args.runs]
    rows = []
    for key, stage in STAGES:
        divergences, count = gather(truth, runs, key, kl_divergence)
        if not count:
            continue
        agreements, _ = gather(
            truth, runs, key, lambda a, b: (a.argmax(-1) == b.argmax(-1)).astype(float)
        )
        name = stage if count == len(args.cases) else f"{stage} [{count} cases]"
        rows.append((name, divergences, agreements))
    positions = sum(len(row[1][0]) for row in rows)

    print_table(
        f"=== Mean KL from the truth over {positions} positions",
        np.mean,
        [(name, measures) for name, measures, _ in rows],
        labels,
    )
    print_table(
        "=== Median KL per position",
        np.median,
        [(name, measures) for name, measures, _ in rows],
        labels,
    )
    print_table(
        "=== Share of positions with KL above 0.1",
        lambda k: float((k > 0.1).mean()),
        [(name, measures) for name, measures, _ in rows],
        labels,
    )
    print_table(
        "=== Share of positions with KL above 0.5",
        lambda k: float((k > 0.5).mean()),
        [(name, measures) for name, measures, _ in rows],
        labels,
    )
    print_table(
        "=== Share of positions whose most likely token agrees",
        np.mean,
        [(name, measures) for name, _, measures in rows],
        labels,
    )

    common = set(truth[0]).intersection(*(set(case) for run in runs for case in run))
    shift_keys = sorted(
        (key for key in common if key.startswith("prompt_logits_shift_")),
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )
    if not shift_keys:
        return
    reference = np.concatenate([case["prompt_logits"] for case in truth])
    width = max(12, *(len(label) for label in labels)) + 2
    print(
        "=== Mean KL from the truth's prefill for prefills with all positions "
        "shifted by a constant, an equivalent input"
    )
    print(f"{'':<28}" + "".join(f"{label:>{width}}" for label in ("truth", *labels)))
    for key in shift_keys:
        cells = [
            kl_divergence(reference, np.concatenate([case[key] for case in run])).mean()
            for run in (truth, *runs)
        ]
        print(
            f"{'shift ' + key.rsplit('_', 1)[1]:<28}"
            + "".join(f"{value:>{width}.5f}" for value in cells)
        )


if __name__ == "__main__":
    main()
