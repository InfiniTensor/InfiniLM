import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

import argparse
import infinilm
from infinilm.modeling_utils import get_model_state_dict
from tokenizers import decoders as _dec
from transformers import AutoTokenizer

import infinicore


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
    return parser.parse_args()


def test(model_path, device_str="cuda", max_new_tokens=100):
    # ---------------------------------------------------------------------------- #
    #                        创建模型,
    # ---------------------------------------------------------------------------- #
    infini_device = infinicore.device(device_str, 0)
    infini_dtype = infinicore.bfloat16

    model = infinilm.LlamaForCausalLM.from_pretrained(
        model_path,
        device=infini_device,
        dtype=infini_dtype,
    )

    # ---------------------------------------------------------------------------- #
    #                        加载权重
    # ---------------------------------------------------------------------------- #
    model_param_infini = get_model_state_dict(
        model_path,
        device=infini_device,
        dtype=infini_dtype,
    )

    model.load_state_dict(model_param_infini)

    config = model.config

    # ---------------------------------------------------------------------------- #
    #                        创建 tokenizer
    # ---------------------------------------------------------------------------- #

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if "llama" == config.model_type:
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
    #                        token编码
    # ---------------------------------------------------------------------------- #
    prompt = "山东最高的山是？"
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    print(input_content, end="", flush=True)
    input_ids = tokenizer.encode(input_content)

    # ---------------------------------------------------------------------------- #
    #                        自回归生成
    # ---------------------------------------------------------------------------- #
    input_ids_list = [input_ids]  # List: [[1, 1128, 526, 366, 29892]]
    input_ids_infini = infinicore.from_list(input_ids_list)

    t1 = time.time()
    model.generate(
        input_ids_infini,
        max_new_tokens=max_new_tokens,
        device=infini_device,
        tokenizer=tokenizer,
        config=config,
    )
    t2 = time.time()

    print(
        f"total_time: {round((t2 - t1) * 1000, 2)} ms",
    )


if __name__ == "__main__":
    args = get_args()
    print(args)

    # Parse command line arguments
    device_type = "cpu"
    if args.cpu:
        device_type = "cpu"
    elif args.nvidia:
        device_type = "cuda"
    elif args.metax:
        device_type = "cuda"
    else:
        print(
            "Usage:  python examples/llama.py [--cpu | --nvidia] --model_path=<path/to/model_dir>"
        )
        sys.exit(1)

    model_path = args.model_path
    max_new_tokens = args.max_new_tokens

    test(model_path, device_type, max_new_tokens)
