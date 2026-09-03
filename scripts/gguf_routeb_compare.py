#!/usr/bin/env python3
"""Compare deterministic llama.cpp and InfiniLM token results."""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", required=True)
    ap.add_argument("--infinilm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--case-ids",
        help="Optional comma-separated case IDs for focused comparisons",
    )
    args = ap.parse_args()

    with open(args.llama, encoding="utf-8") as f:
        llama = json.load(f)
    with open(args.infinilm, encoding="utf-8") as f:
        infini = json.load(f)
    lmap = {x["id"]: x for x in llama["cases"]}
    imap = {x["id"]: x for x in infini["cases"]}
    if args.case_ids:
        case_ids = [x.strip() for x in args.case_ids.split(",") if x.strip()]
        missing = [x for x in case_ids if x not in lmap or x not in imap]
        if missing:
            raise ValueError("requested case IDs missing from one side: %s" % missing)
        lmap = {x: lmap[x] for x in case_ids}
        imap = {x: imap[x] for x in case_ids}
    elif set(lmap) != set(imap):
        raise ValueError(
            "case sets differ: llama-only=%s infini-only=%s"
            % (sorted(set(lmap) - set(imap)), sorted(set(imap) - set(lmap)))
        )

    cases = []
    exact = 0
    matched = total = 0
    for case_id in lmap:
        left = lmap[case_id]
        right = imap[case_id]
        if left["input_ids"] != right["input_ids"]:
            raise ValueError("input ids differ for %s" % case_id)
        lt = left["runs"][0]["tokens"]
        rt = right["runs"][0]["tokens"]
        first_difference = next(
            (i for i, (a, b) in enumerate(zip(lt, rt)) if a != b), None
        )
        if first_difference is None and len(lt) != len(rt):
            first_difference = min(len(lt), len(rt))
        is_exact = lt == rt
        exact += int(is_exact)
        same = sum(a == b for a, b in zip(lt, rt))
        matched += same
        total += max(len(lt), len(rt))
        cases.append(
            {
                "id": case_id,
                "exact_sequence_match": is_exact,
                "matched_tokens": same,
                "total_tokens": max(len(lt), len(rt)),
                "first_difference": first_difference,
                "llama_tokens": lt,
                "infinilm_tokens": rt,
                "llama_first_top_logprobs": left["runs"][0].get(
                    "first_token_top_logprobs", []
                ),
            }
        )
        print(
            "%-10s exact=%s first_diff=%s llama=%s infini=%s"
            % (case_id, is_exact, first_difference, lt, rt)
        )

    result = {
        "cases": cases,
        "n_cases": len(cases),
        "exact_cases": exact,
        "prompt_exact_rate": exact / len(cases) if cases else 0.0,
        "matched_tokens": matched,
        "total_tokens": total,
        "token_match_rate": matched / total if total else 0.0,
        "all_exact": exact == len(cases),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(
        "RESULT exact=%d/%d token_match=%d/%d all_exact=%s"
        % (exact, len(cases), matched, total, result["all_exact"])
    )
    return 0 if result["all_exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
