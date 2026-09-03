#!/usr/bin/env python3
"""Capture all llama-server token probabilities for divergent Route-B cases."""

import argparse
import json
import os
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:18080")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--n-probs", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.inputs, encoding="utf-8") as f:
        inputs = {x["id"]: x for x in json.load(f)["cases"]}
    with open(args.compare, encoding="utf-8") as f:
        divergent = [
            x for x in json.load(f)["cases"] if x["first_difference"] is not None
        ]
    results = []
    for item in divergent:
        case_id = item["id"]
        body = {
            "prompt": inputs[case_id]["input_ids"],
            "n_predict": args.new_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "repeat_penalty": 1.0,
            "repeat_last_n": 0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "seed": 1,
            "ignore_eos": True,
            "cache_prompt": False,
            "return_tokens": True,
            "n_probs": args.n_probs,
            "stream": False,
            "samplers": ["top_k", "temperature"],
        }
        req = urllib.request.Request(
            args.server.rstrip("/") + "/completion",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as response:
            raw = json.load(response)
        tokens = [int(x) for x in raw.get("tokens", [])]
        expected = [int(x) for x in item["llama_tokens"]]
        if tokens != expected:
            raise RuntimeError(
                "%s rerun changed: %s != %s" % (case_id, tokens, expected)
            )
        results.append(
            {
                "case_id": case_id,
                "tokens": tokens,
                "completion_probabilities": raw.get("completion_probabilities", []),
            }
        )
        print(
            "%-10s tokens=%d probabilities=%d stable=%s"
            % (
                case_id,
                len(tokens),
                len(results[-1]["completion_probabilities"]),
                tokens == expected,
            ),
            flush=True,
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"cases": results}, f, ensure_ascii=False, indent=2)
    print("RESULT cases=%d" % len(results), flush=True)


if __name__ == "__main__":
    main()
