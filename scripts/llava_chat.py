import argparse
import sys

from libinfinicore_infer import DeviceType
from llava import LLaVAForCauslLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dev",
        choices=["hygon", "moore"],
        default="hygon",
        help="Device backend (currently only hygon is supported for this script).",
    )
    ap.add_argument("--ndev", type=int, default=1)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", default="Describe this image.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--topp", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--kv-compress", action="store_true", help="Enable in-place KV cache compression after prefill.")
    ap.add_argument("--kv-compress-bin", default="", help="Path to llava_mlp.bin compressor weights.")
    ap.add_argument("--kv-compress-factor", type=int, default=5)
    ap.add_argument("--kv-compress-min-seq-len", type=int, default=2)
    ap.add_argument("--perplexity", action="store_true", help="Collect logits for perplexity calculation")
    args = ap.parse_args()

    if args.dev not in ["hygon", "moore"]:
        raise SystemExit("Only --dev hygon/moore is supported for this script.")
    if args.kv_compress:
        if args.ndev != 1:
            ap.error("--kv-compress currently requires --ndev 1")
        if not args.kv_compress_bin:
            ap.error("--kv-compress requires --kv-compress-bin")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": args.image},
                {"type": "text", "text": args.question},
            ],
        }
    ]

    device_type = (
        DeviceType.DEVICE_TYPE_HYGON
        if args.dev == "hygon"
        else DeviceType.DEVICE_TYPE_MOORE
    )
    model = LLaVAForCauslLM(
        args.model_dir,
        device= device_type,
        ndev=args.ndev,
    )
    text = model.generate(
        messages,
        max_new_tokens=args.max_new_tokens,
        topk_=args.topk,
        topp_=args.topp,
        temperature_=args.temperature,
        verbose=args.verbose,
        kv_compress=bool(args.kv_compress),
        kv_compress_bin=str(args.kv_compress_bin),
        kv_compress_factor=int(args.kv_compress_factor),
        kv_compress_min_seq_len=int(args.kv_compress_min_seq_len),
        perplexity=bool(args.perplexity),
    )
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
