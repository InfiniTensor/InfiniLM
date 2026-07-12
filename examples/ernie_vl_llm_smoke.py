import argparse
import json
import os
import time

from infinilm.llm.llm import LLM
from infinilm.llm.sampling_params import SamplingParams
from PIL import Image

EXPECTED_IDS = {
    "static": [
        3843,
        1510,
        1386,
        5434,
        38225,
        6554,
        93977,
        5119,
        94101,
        6554,
        2293,
        94035,
        93986,
        96101,
        94552,
        94397,
    ],
    "paged": [3843, 1510, 1386, 5434, 38225, 6554, 93977, 5119],
}


def prepare_image(image_path, output_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image.save(output_path)
    return output_path


def build_messages(image_path):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_path}},
                {"type": "text", "text": "请用一句中文简短描述这张图片。"},
            ],
        }
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--cache-type", choices=["static", "paged"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-cache-len", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = os.path.expanduser(args.model)
    image_path = args.image or os.path.join(model_path, "benchmark.jpg")
    resized_image = prepare_image(image_path, "/tmp/ernie_vl_llm_smoke_224.jpg")
    expected_ids = EXPECTED_IDS[args.cache_type]
    max_tokens = args.max_tokens or len(expected_ids)

    run_config = {
        "cache_type": args.cache_type,
        "device": args.device,
        "tp": args.tp,
        "max_cache_len": args.max_cache_len,
        "num_blocks": args.num_blocks,
        "block_size": args.block_size,
        "max_tokens": max_tokens,
        "image": resized_image,
    }
    print(json.dumps({"run_config": run_config}, ensure_ascii=False), flush=True)

    start = time.time()
    llm = LLM(
        model_path=model_path,
        device=args.device,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        cache_type=args.cache_type,
        max_batch_size=1,
        max_tokens=max_tokens,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_cache_len=args.max_cache_len,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        enable_graph=False,
        attn_backend="default",
    )
    init_sec = time.time() - start

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        ignore_eos=True,
    )
    gen_start = time.time()
    outputs = llm.generate(
        messages=build_messages(resized_image),
        sampling_params=params,
        use_tqdm=False,
    )
    gen_sec = time.time() - gen_start
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError(f"unexpected output shape: {outputs!r}")

    request = outputs[0]
    completion = request.outputs[0]
    token_ids = completion.token_ids or []
    result = {
        "ok": token_ids == expected_ids,
        "run_config": run_config,
        "prompt_len": len(request.prompt_token_ids or []),
        "expected_token_ids": expected_ids,
        "new_token_ids": token_ids,
        "output_text": completion.text,
        "finish_reason": str(completion.finish_reason),
        "init_sec": init_sec,
        "elapsed_sec": gen_sec,
    }
    print(
        json.dumps(
            {
                "cache_type": args.cache_type,
                "prompt_len": result["prompt_len"],
                "new_token_ids": result["new_token_ids"],
                "match_expected": result["ok"],
                "output_text": result["output_text"],
                "init_sec": round(init_sec, 3),
                "elapsed_sec": round(gen_sec, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
