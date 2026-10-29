"""Measure serial Clover chat requests and retain every response and failure."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("infinilm", "sglang"), default="infinilm")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path(__file__).parents[1] / "test/diffusion/workload.json",
    )
    parser.add_argument("--variants", default="baseline,frontier,budget,joint")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frontier-floor", type=float, default=0.7)
    args = parser.parse_args()
    variants = args.variants.split(",")
    if not variants or set(variants) - {"baseline", "frontier", "budget", "joint"}:
        parser.error("Unknown variant.")
    if args.backend == "sglang" and variants != ["baseline"]:
        parser.error("The SGLang comparison requires --variants baseline.")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    workload = json.loads(args.workload.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(-1, args.repeats):
        for item in workload:
            order = variants if (repeat + item["id"]) % 2 == 0 else variants[::-1]
            for variant in order:
                request = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": item["prompt"]}],
                    "temperature": 0,
                    "max_tokens": item["max_tokens"],
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "repetition_penalty": 1,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                if args.backend == "infinilm":
                    request.update(
                        diffusion_policy="frontier"
                        if variant in ("frontier", "joint")
                        else "baseline",
                        expert_budget=variant in ("budget", "joint"),
                        frontier_confidence_floor=args.frontier_floor,
                    )
                record = {
                    "repeat": repeat,
                    "prompt_id": item["id"],
                    "variant": variant,
                    "request": request,
                    "success": False,
                }
                start = time.perf_counter()
                try:
                    http_request = urllib.request.Request(
                        args.url.rstrip("/") + "/v1/chat/completions",
                        data=json.dumps(request).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(http_request, timeout=240) as response:
                        result = json.load(response)
                    record["elapsed_s"] = time.perf_counter() - start
                    text = result["choices"][0]["message"]["content"]
                    record.update(
                        success=True,
                        response=result,
                        visible_tokens=len(
                            tokenizer(text, add_special_tokens=False)["input_ids"]
                        ),
                    )
                except (urllib.error.URLError, ValueError, KeyError) as error:
                    record.update(
                        elapsed_s=time.perf_counter() - start,
                        error=str(error),
                        visible_tokens=0,
                    )
                with args.output.open("a") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    variant,
                    repeat,
                    item["id"],
                    record["success"],
                    round(record["elapsed_s"], 4),
                    flush=True,
                )


if __name__ == "__main__":
    main()
