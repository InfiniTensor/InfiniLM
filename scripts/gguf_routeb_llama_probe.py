#!/usr/bin/env python3
"""Dump llama-server one-token responses at Route-B divergence prefixes."""

import argparse
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--case-ids", nargs="+", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:18080")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.inputs, encoding="utf-8") as f:
        inputs = {x["id"]: x for x in json.load(f)["cases"]}
    with open(args.compare, encoding="utf-8") as f:
        cases = {x["id"]: x for x in json.load(f)["cases"]}
    results = []
    for case_id in args.case_ids:
        item = cases[case_id]
        diff = item["first_difference"]
        prefix = inputs[case_id]["input_ids"] + item["llama_tokens"][:diff]
        body = {
            "prompt": prefix, "n_predict": 1, "temperature": 0.0,
            "top_k": 1, "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0,
            "repeat_penalty": 1.0, "repeat_last_n": 0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "seed": 1, "ignore_eos": True, "cache_prompt": False,
            "return_tokens": True, "n_probs": 100, "stream": False,
            "samplers": ["top_k", "temperature"],
        }
        req = urllib.request.Request(
            args.server.rstrip("/") + "/completion",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=180) as response:
            raw = json.load(response)
        results.append({"case_id": case_id, "diff": diff, "prefix": prefix,
                        "response": raw})
        probs = raw.get("completion_probabilities", [])
        print(case_id, "tokens=", raw.get("tokens"), "prob_entry=", probs[:1])
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"cases": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
