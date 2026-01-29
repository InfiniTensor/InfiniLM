import infinicore
from transformers import AutoTokenizer
from tokenizers import decoders as _dec
from infinilm.modeling_utils import load_model_state_dict_by_file
from infinilm.distributed import DistConfig
from infinilm.infer_engine import GenerationConfig, InferEngine
import argparse
import sys
import time
import os
import numpy as np
from infinilm.cache import StaticKVCacheConfig, PagedKVCacheConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))


def get_args():
    parser = argparse.ArgumentParser(description="run Llama args")

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run cpu test",
    )
    parser.add_argument(
        "--nvidia",
        action="store_true",
        help="Run nvidia test",
    )
    parser.add_argument(
        "--metax",
        action="store_true",
        help="Run metax test",
    )
    parser.add_argument(
        "--moore",
        action="store_true",
        help="Run moore test",
    )
    parser.add_argument(
        "--iluvatar",
        action="store_true",
        help="Run iluvatar test",
    )
    parser.add_argument(
        "--cambricon",
        action="store_true",
        help="Run cambricon test",
    )
    parser.add_argument(
        "--ali",
        action="store_true",
        help="Run alippu test",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="model_path",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="max_new_tokens",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cpp",
        help="python or cpp model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="number of prompts in a batch",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="How are you",
        help="input prompt",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="total rank for tensor parallel",
    )
    parser.add_argument(
        "--enable-paged-attn",
        action="store_true",
        help="use paged cache",
    )

    return parser.parse_args()


def test(
    prompts: str | list[str],
    model_path,
    max_new_tokens=100,
    infini_device=infinicore.device("cpu", 0),
    tp=1,
    enable_paged_attn=False,
):
    model_path = os.path.expanduser(model_path)
    # ---------------------------------------------------------------------------- #
    #                        Create Model
    # ---------------------------------------------------------------------------- #
    model = InferEngine(
        model_path,
        device=infini_device,
        distributed_config=DistConfig(tp),
    )

    # ---------------------------------------------------------------------------- #
    #                        Load Weights
    # ---------------------------------------------------------------------------- #
    load_model_state_dict_by_file(model, model_path, dtype=model.config.dtype)

    # ---------------------------------------------------------------------------- #
    #                        create tokenizer
    # ---------------------------------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if "llama" == model.config.model_type:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        target = getattr(backend, "_tokenizer", backend)
        norm = getattr(target, "normalizer", None)
        dec = getattr(target, "decoder", None)
        sn = repr(norm)[:800] if norm is not None else ""
        sd = repr(dec)[:800] if dec is not None else ""
        has_prepend = "Prepend" in sn
        has_strip = "Strip" in sd
        if has_prepend and has_strip:
            target.decoder = _dec.Sequence(
                [
                    _dec.Replace("▁", " "),
                    _dec.ByteFallback(),
                    _dec.Fuse(),
                ]
            )

    # ---------------------------------------------------------------------------- #
    #                        tokenize
    # ---------------------------------------------------------------------------- #
    # prompt = "山东最高的山是？"
    if isinstance(prompts, str):
        prompts = [prompts]
    input_contents = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in prompts
    ]

    input_ids_list = tokenizer.batch_encode_plus(input_contents)[
        "input_ids"
    ]  # List: [[1, 1128, 526, 366, 29892]]

    # ---------------------------------------------------------------------------- #
    #                       Create KVCache
    # ---------------------------------------------------------------------------- #
    if enable_paged_attn:
        batch_size = 1 if prompts is str else len(prompts)
        max_total_tokens = max_new_tokens + len(input_ids_list[0])
        cache_config = PagedKVCacheConfig(
            num_blocks=(max_total_tokens // 16 + 1) * batch_size, block_size=16
        )
    else:
        batch_size = 1 if prompts is str else len(prompts)
        initial_capacity = max_new_tokens + len(input_ids_list[0])
        cache_config = StaticKVCacheConfig(
            max_batch_size=batch_size, max_cache_len=initial_capacity
        )

    model.reset_cache(cache_config)

    # ---------------------------------------------------------------------------- #
    #                        Generate
    # ---------------------------------------------------------------------------- #
    print(input_contents[0], end="", flush=True)
    input_ids_infini = infinicore.from_list(input_ids_list)

    t1 = time.time()
    print("=================== start generate ====================")
    output_ids = model.generate(
        input_ids_infini,
        GenerationConfig(
            max_new_tokens=max_new_tokens, temperature=1, top_k=1, top_p=0.8
        ),
        _measure_and_log_time=True,
    )
    t2 = time.time()

    numpy_output_ids = np.array([output_id.to_numpy()[0] for output_id in output_ids])
    print(tokenizer.decode(numpy_output_ids, skip_special_tokens=True))

    print(
        f"total_time: {round((t2 - t1) * 1000, 2)} ms",
    )


if __name__ == "__main__":
    args = get_args()
    print(args)

    # Parse command line arguments
    device_str = "cpu"
    if args.cpu:
        device_str = "cpu"
    elif args.nvidia:
        device_str = "cuda"
    elif args.metax:
        device_str = "cuda"
    elif args.moore:
        device_str = "musa"
    elif args.iluvatar:
        device_str = "cuda"
    elif args.cambricon:
        device_str = "mlu"
    elif args.ali:
        device_str = "cuda"
    else:
        print(
            "Usage:  python examples/jiuge.py [--cpu | --nvidia | --metax | --moore | --iluvatar] --model_path=<path/to/model_dir>\n"
            "such as, python examples/jiuge.py --nvidia --model_path=~/TinyLlama-1.1B-Chat-v1.0"
        )
        sys.exit(1)
    prompts = [args.prompt for _ in range(args.batch_size)]

    model_path = args.model_path
    max_new_tokens = args.max_new_tokens
    backend = args.backend
    tp = args.tp
    enable_paged_attn = args.enable_paged_attn
    if backend != "cpp":
        raise ValueError(f"Unsupported backend: {backend}.")

    infini_device = infinicore.device(device_str, 0)

    test(
        prompts,
        model_path,
        max_new_tokens,
        infini_device=infini_device,
        tp=tp,
        enable_paged_attn=enable_paged_attn,
    )
