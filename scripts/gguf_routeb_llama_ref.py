#!/usr/bin/env python3
"""Run deterministic raw-token completions through llama-server."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def post_json(url: str, body: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError("HTTP %d: %s" % (exc.code, detail[:2000])) from exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:18080")
    ap.add_argument("--new-tokens", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--n-probs", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.inputs, encoding="utf-8") as f:
        source = json.load(f)
    outputs = []
    all_ok = True
    for case in source["cases"]:
        runs = []
        for repeat in range(args.repeats):
            body = {
                "prompt": case["input_ids"],
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
            started = time.time()
            response = post_json(
                args.server.rstrip("/") + "/completion", body, args.timeout)
            tokens = [int(x) for x in response.get("tokens", [])]
            probabilities = response.get("completion_probabilities", [])
            runs.append({
                "repeat": repeat,
                "tokens": tokens,
                "content": response.get("content", ""),
                "first_token_top_logprobs": (
                    probabilities[0].get("top_logprobs", []) if probabilities else []),
                "elapsed_s": round(time.time() - started, 4),
                "tokens_evaluated": response.get("tokens_evaluated"),
                "tokens_predicted": response.get("tokens_predicted"),
            })
        deterministic = all(x["tokens"] == runs[0]["tokens"] for x in runs[1:])
        exact_length = all(len(x["tokens"]) == args.new_tokens for x in runs)
        ok = deterministic and exact_length
        all_ok &= ok
        outputs.append({
            "id": case["id"],
            "prompt": case["prompt"],
            "input_ids": case["input_ids"],
            "deterministic": deterministic,
            "exact_length": exact_length,
            "runs": runs,
        })
        print("%-10s deterministic=%s length=%s tokens=%s" % (
            case["id"], deterministic, exact_length, runs[0]["tokens"]))

    result = {
        "engine": "llama.cpp",
        "server": args.server,
        "new_tokens": args.new_tokens,
        "repeats": args.repeats,
        "cases": outputs,
        "all_pass": all_ok,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("RESULT cases=%d all_pass=%s" % (len(outputs), all_ok))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
