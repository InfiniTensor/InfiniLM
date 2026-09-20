#!/usr/bin/env python3
"""Generate one deterministic request manifest for all KV admission tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .kv_admission_manifest import manifest_payload
except ImportError:
    from kv_admission_manifest import manifest_payload


def rendered_tokens(tokenizer, content: str) -> tuple[str, list[int]]:
    messages = [{"role": "user", "content": content}]
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return rendered, [int(token_id) for token_id in token_ids]


def exact_prompt(
    tokenizer, target_tokens: int, profile: str
) -> tuple[str, str, list[int]]:
    """Find an ASCII prompt whose rendered chat input has an exact length."""
    if target_tokens <= 0:
        raise ValueError("Target token length must be positive.")

    phrases = (
        f"KV admission benchmark {profile}: ",
        f"Fixed request {profile}: ",
        "x ",
        "hello ",
        "word ",
        "a ",
    )
    for phrase in phrases:
        for repetitions in range(1, target_tokens * 4 + 128):
            content = (phrase * repetitions).rstrip()
            rendered, token_ids = rendered_tokens(tokenizer, content)
            if len(token_ids) == target_tokens:
                return content, rendered, token_ids
            if len(token_ids) > target_tokens + 32:
                break
    raise ValueError(f"Could not build a prompt with exactly {target_tokens} tokens.")


def parse_lengths(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Prompt lengths must be integers.") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Prompt lengths must be positive.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=432)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--prompt-lengths",
        type=parse_lengths,
        default=parse_lengths("64,128,192,256,320,512,768,1024"),
    )
    parser.add_argument("--wave1-requests", type=int, default=128)
    parser.add_argument("--wave2-requests", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.wave1_requests <= 0 or args.wave2_requests <= 0:
        raise SystemExit("Wave request counts must be positive.")
    if args.output_tokens <= 0 or args.block_size <= 0 or args.num_blocks <= 0:
        raise SystemExit("Cache and output sizes must be positive.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt_cache: dict[int, tuple[str, str, list[int]]] = {}
    for length in args.prompt_lengths:
        prompt_cache[length] = exact_prompt(tokenizer, length, str(length))

    requests = []
    total = args.wave1_requests + args.wave2_requests
    for ordinal in range(total):
        wave = "wave1" if ordinal < args.wave1_requests else "wave2"
        wave_index = ordinal if wave == "wave1" else ordinal - args.wave1_requests
        target_length = args.prompt_lengths[wave_index % len(args.prompt_lengths)]
        prompt, rendered, token_ids = prompt_cache[target_length]
        requests.append(
            {
                "request_id": f"{wave}-{wave_index:04d}",
                "wave": wave,
                "prompt": prompt,
                "prompt_token_ids": token_ids,
                "prompt_tokens": len(token_ids),
                "max_tokens": args.output_tokens,
                "ordinal": ordinal,
                "rendered_prompt_sha256": hashlib.sha256(
                    rendered.encode("utf-8")
                ).hexdigest(),
            }
        )

    payload = manifest_payload(
        model=args.model,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        seed=args.seed,
        requests=requests,
        metadata={
            "tokenizer_path": str(Path(args.model_path).resolve()),
            "prompt_lengths": args.prompt_lengths,
            "wave1_requests": args.wave1_requests,
            "wave2_requests": args.wave2_requests,
            "output_tokens": args.output_tokens,
            "chat_template": "user message + generation prompt",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"requests={len(requests)} prompt_lengths={args.prompt_lengths}")


if __name__ == "__main__":
    main()
