"""Generate text with the native Clover block diffusion engine on a few prompts."""

import argparse

from infinilm.diffusion.clover import CloverGenerator

PROMPTS = (
    "What is the capital of France?",
    "Write a Python function that reverses a string.",
    "简要说一下光合作用的过程。",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Checkpoint directory.")
    parser.add_argument(
        "--device", required=True, help="InfiniLM device name, for example `hygon`."
    )
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt", action="append", help="Prompt to generate; repeatable."
    )
    args = parser.parse_args()

    generator = CloverGenerator(
        args.model, tp=args.tp, max_context=args.max_context, device=args.device
    )
    for prompt in args.prompt or PROMPTS:
        result = generator.generate(prompt, max_new_tokens=args.max_new_tokens)
        print(f"PROMPT: {prompt}")
        print(f"TEXT: {result.text}")
        print(
            f"STAT: tokens={len(result.token_ids)} iterations={result.iterations} "
            f"elapsed={result.elapsed_s:.2f}s tok/s={result.output_tokens_per_second:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
