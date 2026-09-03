#!/usr/bin/env python3
"""Build canonical input IDs with llama.cpp and compare the packaged tokenizer."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def load_cases(path: str, selected: set[str]) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if not selected or item["id"] in selected:
                    cases.append(item)
    missing = selected - {x["id"] for x in cases}
    if missing:
        raise ValueError("unknown case ids: %s" % sorted(missing))
    return cases


def post_json(url: str, body: dict, timeout: int = 30) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError("HTTP %d: %s" % (exc.code, detail[:1000])) from exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:18080")
    ap.add_argument("--case-ids", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    selected = {x for x in args.case_ids.split(",") if x}
    cases = load_cases(args.prompts, selected)

    from transformers import AutoTokenizer

    common = {"local_files_only": True, "trust_remote_code": False}
    tok_default = AutoTokenizer.from_pretrained(args.model_path, **common)
    try:
        tok_fixed = AutoTokenizer.from_pretrained(
            args.model_path, fix_mistral_regex=True, **common)
        fixed_error = None
    except Exception as exc:  # compatibility with older transformers
        tok_fixed = None
        fixed_error = "%s: %s" % (type(exc).__name__, exc)

    results = []
    default_ok = fixed_ok = True
    for case in cases:
        llama = post_json(args.server.rstrip("/") + "/tokenize", {
            "content": case["prompt"],
            "add_special": False,
            "parse_special": True,
            "with_pieces": False,
        })["tokens"]
        llama = [int(x) for x in llama]
        local_default = [int(x) for x in tok_default.encode(
            case["prompt"], add_special_tokens=False)]
        local_fixed = None if tok_fixed is None else [int(x) for x in tok_fixed.encode(
            case["prompt"], add_special_tokens=False)]
        match_default = llama == local_default
        match_fixed = local_fixed is not None and llama == local_fixed
        default_ok &= match_default
        fixed_ok &= match_fixed
        results.append({
            **case,
            "input_ids": llama,
            "local_default_ids": local_default,
            "local_fixed_ids": local_fixed,
            "default_match": match_default,
            "fixed_match": match_fixed,
        })
        print("%-10s llama=%3d default=%s fixed=%s" % (
            case["id"], len(llama), match_default,
            "NA" if local_fixed is None else str(match_fixed)))

    if default_ok:
        selected_variant = "default"
    elif fixed_ok:
        selected_variant = "fix_mistral_regex=True"
    else:
        selected_variant = None

    output = {
        "model_path": os.path.abspath(args.model_path),
        "server": args.server,
        "add_special": False,
        "parse_special": True,
        "selected_local_variant": selected_variant,
        "default_all_match": default_ok,
        "fixed_all_match": fixed_ok,
        "fixed_load_error": fixed_error,
        "cases": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("RESULT default_all=%s fixed_all=%s selected=%s cases=%d" % (
        default_ok, fixed_ok, selected_variant, len(results)))
    return 0 if selected_variant else 1


if __name__ == "__main__":
    sys.exit(main())
