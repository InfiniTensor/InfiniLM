import infinicore
from transformers import AutoTokenizer
from tokenizers import decoders as _dec
from infinilm.modeling_utils import load_model_state_dict_by_file
import infinilm
from infinilm.distributed import DistConfig
import argparse
import sys
import time
import os

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
        default="python",
        help="python or cpp model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="float32, float16, bfloat16",
    )
    parser.add_argument(
        "--batch_size",
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
        default=None,
        help="total rank for tensor parallel",
    )

    return parser.parse_args()


def test(
    prompts: str | list[str],
    model_path,
    max_new_tokens=100,
    infini_dtype=infinicore.bfloat16,
    infini_device=infinicore.device("cpu", 0),
    backend="python",
):
    model_path = os.path.expanduser(model_path)
    # ---------------------------------------------------------------------------- #
    #                        创建模型,
    # ---------------------------------------------------------------------------- #
    model = infinilm.AutoLlamaModel.from_pretrained(
        model_path,
        device=infini_device,
        dtype=infini_dtype,
        backend=backend,
        distributed_config=DistConfig(args.tp),
    )

    # ---------------------------------------------------------------------------- #
    #                        加载权重
    # ---------------------------------------------------------------------------- #
    load_model_state_dict_by_file(model, model_path, dtype=infini_dtype)

    # ---------------------------------------------------------------------------- #
    #                        创建 tokenizer
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
    #                        token编码
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
    print(input_contents[0], end="", flush=True)
    input_ids_list = tokenizer.batch_encode_plus(input_contents)[
        "input_ids"
    ]  # List: [[1, 1128, 526, 366, 29892]]

    # ---------------------------------------------------------------------------- #
    #                        自回归生成
    # ---------------------------------------------------------------------------- #
    input_ids_infini = infinicore.from_list(input_ids_list)

    t1 = time.time()
    print("=================== start generate ====================")
    model.generate(
        input_ids_infini,
        max_new_tokens=max_new_tokens,
        device=infini_device,
        tokenizer=tokenizer,
    )
    t2 = time.time()

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
    else:
        print(
            "Usage:  python examples/llama.py [--cpu | --nvidia | --metax | --moore | --iluvatar] --model_path=<path/to/model_dir>\n"
            "such as, python examples/llama.py --nvidia --model_path=~/TinyLlama-1.1B-Chat-v1.0"
        )
        sys.exit(1)
    prompts = [args.prompt for _ in range(args.batch_size)]

    model_path = args.model_path
    max_new_tokens = args.max_new_tokens
    backend = args.backend

    infini_device = infinicore.device(device_str, 0)
    if args.dtype == "float32":
        infini_dtype = infinicore.float32
    elif args.dtype == "bfloat16":
        infini_dtype = infinicore.bfloat16
    elif args.dtype == "float16":
        infini_dtype = infinicore.float16
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    test(
        prompts,
        model_path,
        max_new_tokens,
        infini_device=infini_device,
        infini_dtype=infini_dtype,
        backend=backend,
    )
