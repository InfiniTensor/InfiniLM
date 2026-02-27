import argparse
import itertools
import time
import random

import torch
import json


from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams

# import os
# import logging
# os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
# logging.getLogger("vllm").setLevel(logging.ERROR)


def parse_list(value: str):
    """Parse parse_list argument: can be a single int or a list of ints.

    Examples:
        "1" -> 1
        "[1,2,4]" -> [1, 2, 4]
        "1,2,4" -> [1, 2, 4]
    """
    value = value.strip()
    # Try to parse as JSON list first
    if value.startswith("[") and value.endswith("]"):
        try:
            result = json.loads(value)
            if isinstance(result, list):
                return [int(x) for x in result]
            return int(result)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to parse as comma-separated values
    if "," in value:
        try:
            return [int(x.strip()) for x in value.split(",")]
        except ValueError:
            pass

    # Try to parse as a single integer
    try:
        return [int(value)]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"batch-size must be an int or list[int], got: {value}"
        )


def run_one_case(
    engine: LLMEngine,
    batch_size: int,
    input_len: int,
    output_len: int,
    vocab_size: int,
):
    # ------------------------------------------------------------
    # 1. Random input token IDs
    # ------------------------------------------------------------
    input_ids_list = [
        [random.randint(0, vocab_size - 1) for _ in range(input_len)]
        for _ in range(batch_size)
    ]

    sampling_params = SamplingParams(
        max_tokens=output_len,
        ignore_eos=True,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
    )

    request_ids = []
    for i, input_ids in enumerate(input_ids_list):
        rid = f"req_{i}"
        engine.add_request(
            request_id=rid,
            prompt=TokensPrompt(prompt_token_ids=input_ids),
            params=sampling_params,
        )
        request_ids.append(rid)

    # ------------------------------------------------------------
    # 2. Run until first decode token appears for all requests (prefill timing)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    pre_decode = 0  # some decode tokens can be mixed with prefill batch
    pending = set(f"req_{i}" for i in range(batch_size))
    while pending:
        outputs = engine.step()
        for out in outputs:
            if len(out.outputs[0].token_ids) > 0:
                if out.request_id in pending:
                    pending.remove(out.request_id)
                else:
                    pre_decode += 1
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    prefill_time = t1 - t0
    prefill_tokens = batch_size * input_len

    # ------------------------------------------------------------
    # 3. Decode until all requests finish
    # ------------------------------------------------------------
    decode_start = time.perf_counter()

    while engine.has_unfinished_requests():
        outputs = engine.step()

    torch.cuda.synchronize()
    decode_end = time.perf_counter()

    decode_time = decode_end - decode_start
    decode_tokens = (
        batch_size * (output_len - 1) - pre_decode
    )  # exclude prefill-mixed tokens

    return {
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "prefill_tput": prefill_tokens / prefill_time,
        "decode_tput": decode_tokens / decode_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--batch-size",
        type=parse_list,
        default=[1],
        help=(
            "number of prompts in a batch (int or list, e.g. '1', '1,2,4', '[1,2,4]')"
        ),
    )
    parser.add_argument(
        "--input-len",
        type=parse_list,
        default=[256],
        help="input sequence length(s)",
    )
    parser.add_argument(
        "--output-len",
        type=parse_list,
        default=[256],
        help="output sequence length(s)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        type=int,
        default=1,
        help="total rank for tensor parallel",
    )
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--num-iters-warmup", type=int, default=2)
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Engine init (TP supported here)
    # ------------------------------------------------------------
    engine_args = EngineArgs(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        distributed_executor_backend="mp",
        enable_chunked_prefill=False,
    )

    engine = LLMEngine.from_engine_args(engine_args)
    vocab_size = engine.model_config.get_vocab_size()

    # ------------------------------------------------------------
    # Sweep all combinations
    # ------------------------------------------------------------
    print("\n=== Running benchmark ===")
    results = []

    try:
        for bs, il, ol in itertools.product(
            args.batch_size, args.input_len, args.output_len
        ):
            # Warmup
            for _ in range(args.num_iters_warmup):
                run_one_case(
                    engine,
                    batch_size=bs,
                    input_len=il,
                    output_len=ol,
                    vocab_size=vocab_size,
                )

            res = run_one_case(
                engine,
                batch_size=bs,
                input_len=il,
                output_len=ol,
                vocab_size=vocab_size,
            )
            results.append(res)

            print(
                f"[TP={args.tensor_parallel_size} | "
                f"bs={bs} in={il} out={ol}] "
                f"prefill={res['prefill_tput']} tok/s | "
                f"decode={res['decode_tput']} tok/s"
            )
    except Exception as e:
        print(f"Error Occured: {e}")
    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n=== Summary ===")
    print("bs   in_len   out_len   prefill_tok/s     decode_tok/s")
    for r in results:
        print(
            f"{r['batch_size']:3d}  "
            f"{r['input_len']:7d}  "
            f"{r['output_len']:8d}  "
            f"{r['prefill_tput']:14.2f}  "
            f"{r['decode_tput']:14.2f}"
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    main()
