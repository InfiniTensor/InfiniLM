import infinicore
from transformers import AutoTokenizer
from infinilm.modeling_utils import load_model_state_dict_by_file
from infinilm.distributed import DistConfig
from infinilm.infer_engine import GenerationConfig, InferEngine
from infinilm.cache import StaticKVCacheConfig, PagedKVCacheConfig
import argparse
import sys
import time
import os
import json
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))


DATA_TYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}

# BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128]
# INPUT_LENS = [32, 256, 1024, 4096]
# OUTPUT_LENS = [256, 1024, 4096]


def read_json_file(file_path):
    """Load and return JSON content from file_path."""
    with open(file_path, "r") as file:
        return json.load(file)


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
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"batch-size must be an int or list[int], got: {value}"
        )


def get_test_cases(
    model_path: str,
    batch_size_list: list[int],
    input_len_list: list[int],
    output_len_list: list[int],
):
    model_path = os.path.expanduser(model_path)

    """Generate cases ordered by ascending KV cache memory usage."""
    # Load model config to derive attention dimensions
    config = read_json_file(os.path.join(model_path, "config.json"))
    head_dim = config.get(
        "head_dim", config.get("hidden_size") // config.get("num_attention_heads")
    )
    # KV heads and layers drive cache size
    num_key_value_heads = config.get("num_key_value_heads")
    num_hidden_layers = config.get("num_hidden_layers")

    # Enumerate all batch/input/output combinations and compute KV cache size
    case_list = []
    for batch_size in batch_size_list:
        for input_len in input_len_list:
            for output_len in output_len_list:
                for data_type in ["bfloat16"]:
                    data_type_bytes = DATA_TYPE_BYTES[data_type]

                    total_seq_len = input_len + output_len
                    kvcache_memory_bytes = (
                        data_type_bytes
                        * (batch_size * total_seq_len * num_key_value_heads * head_dim)
                        * num_hidden_layers
                    )
                    kvcache_memory_gb = kvcache_memory_bytes / (1024 * 1024 * 1024)

                    case_list.append(
                        {
                            "idx": len(case_list),
                            "batch_size": batch_size,
                            "input_len": input_len,
                            "output_len": output_len,
                            "data_type": data_type,
                            "kvcache_memory": round(kvcache_memory_gb, 3),
                        }
                    )

    # Sort by KV cache size and wrap in OrderedDict with index keys
    case_dict = OrderedDict(
        (idx, case)
        for idx, case in enumerate(
            sorted(case_list, key=lambda case: case["kvcache_memory"])
        )
    )

    return case_dict


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
        "--qy",
        action="store_true",
        help="Run qy test",
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
        "--model",
        type=str,
        required=True,
        help="model path",
    )
    parser.add_argument(
        "--batch-size",
        type=parse_list,
        default=1,
        help="number of prompts in a batch (can be an int or a list of ints, e.g., '1' or '[1,2,4]' or '1,2,4')",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        type=int,
        default=1,
        help="total rank for tensor parallel",
    )
    parser.add_argument(
        "--input-len",
        type=parse_list,
        default=10,
        help="output tokens",
    )

    parser.add_argument(
        "--output-len",
        type=parse_list,
        default=20,
        help="output tokens",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="skip loading model weights",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="top k sampling",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="top p sampling",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="sampling temperature",
    )
    parser.add_argument(
        "--enable-paged-attn",
        action="store_true",
        help="use paged cache",
    )
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        help="enable graph compiling",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Perform a warmup run before benchmarking/inference.",
    )
    return parser.parse_args()


with open("examples/bench_prompt.md", "r") as f:
    prompt = f.read()


def repeat_prompt(input_ids: list[int], target_length: int):
    num = len(input_ids)
    repeat_times = (target_length + num - 1) // num
    return (input_ids * repeat_times)[:target_length]


class TestModel:
    model: infinicore.nn.Module
    tokenizer: AutoTokenizer
    input_ids_list: list[int]

    def __init__(
        self,
        model_path,
        infini_device=infinicore.device("cpu", 0),
        tp=1,
        skip_load=False,
        cache_config=None,
        enable_graph=False,
    ) -> None:
        model_path = os.path.expanduser(model_path)
        # ---------------------------------------------------------------------------- #
        #                        创建模型,
        # ---------------------------------------------------------------------------- #
        model = InferEngine(
            model_path,
            device=infini_device,
            distributed_config=DistConfig(tp),
            cache_config=cache_config,
            enable_graph_compiling=enable_graph,
        )

        # ---------------------------------------------------------------------------- #
        #                        加载权重
        # ---------------------------------------------------------------------------- #
        if not skip_load:
            load_model_state_dict_by_file(model, model_path, dtype=model.config.dtype)

        # ---------------------------------------------------------------------------- #
        #                        创建 tokenizer
        # ---------------------------------------------------------------------------- #
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # ---------------------------------------------------------------------------- #
        #                        token编码
        # ---------------------------------------------------------------------------- #
        input_content = [
            tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        ]

        # print(input_content, end="", flush=True)
        # Support Transformers >= 5.0 for batch_encode_plus deprecation
        encoding = tokenizer(
            input_content,
            padding=True,
            truncation=True,
            max_length=8192,
        )

        input_ids_list = encoding["input_ids"]

        self.model = model
        self.tokenizer = tokenizer
        self.input_ids_list = input_ids_list

    def run(
        self,
        batch_size: int,
        input_len: int,
        output_len: int,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    ):
        input_ids = repeat_prompt(self.input_ids_list[0], target_length=input_len)
        input_ids_list = [input_ids] * batch_size

        # ---------------------------------------------------------------------------- #
        #                        自回归生成
        # ---------------------------------------------------------------------------- #
        input_ids_infini = infinicore.from_list(input_ids_list)

        t1 = time.time()
        print("=================== start generate ====================")
        output_ids = self.model.generate(
            input_ids_infini,
            GenerationConfig(
                max_new_tokens=output_len,
                eos_token_id=[],
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                stop_on_eos=False,
            ),
            _measure_and_log_time=True,
        )
        t2 = time.time()

        numpy_output_ids = np.array(
            [output_id.to_numpy()[0] for output_id in output_ids]
        )
        print(self.tokenizer.decode(numpy_output_ids, skip_special_tokens=True))

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
    elif args.qy:
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
            "python examples/bench.py --nvidia --model=~/TinyLlama-1.1B-Chat-v1.0/ --batch-size=2 --tp=1 --input-len=50 --output-len=50"
        )
        sys.exit(1)
    # -------------------------------------------------------- #
    #             解析参数
    # -------------------------------------------------------- #
    model_path = args.model

    infini_device = infinicore.device(device_str, 0)

    tp = args.tensor_parallel_size

    skip_load = args.skip_load

    batch_size = args.batch_size
    input_len = args.input_len
    output_len = args.output_len
    enable_paged_attn = args.enable_paged_attn
    enable_graph = args.enable_graph

    if isinstance(batch_size, int):
        batch_size = [batch_size]

    if isinstance(input_len, int):
        input_len = [input_len]

    if isinstance(output_len, int):
        output_len = [output_len]

    cases_dict = get_test_cases(model_path, batch_size, input_len, output_len)
    # -------------------------------------------------------- #
    #             测试
    # -------------------------------------------------------- #
    if enable_paged_attn:
        paged_kv_block_size = 16
        max_num_blocks = max(
            [
                ((c_["input_len"] + c_["output_len"] + 15) // 16) * c_["batch_size"]
                for _, c_ in cases_dict.items()
            ]
        )
        cache_config = PagedKVCacheConfig(max_num_blocks, paged_kv_block_size)
    else:
        cache_config = None

    test = TestModel(
        model_path,
        infini_device=infini_device,
        tp=tp,
        skip_load=skip_load,
        cache_config=cache_config,
        enable_graph=enable_graph,
    )

    # ---------------------------------------------------------------------------- #
    #                                Warmup
    # ---------------------------------------------------------------------------- #
    if args.warmup:
        warmup_steps = 1

        # warmup cache capacity
        warmup_cache_len = 128
        warmup_batch = len(test.input_ids_list)

        test.model.reset_cache(
            StaticKVCacheConfig(
                max_batch_size=warmup_batch,
                max_cache_len=warmup_cache_len,
            )
        )

        avg_prompt_len = min(64, max(len(ids) for ids in test.input_ids_list))

        warmup_ids = [
            ids[:avg_prompt_len] if len(ids) >= avg_prompt_len else ids
            for ids in test.input_ids_list
        ]

        input_ids_infini = infinicore.from_list(warmup_ids)

        print("=================== warmup start ===================")

        for _ in range(warmup_steps):
            _ = test.model.generate(
                input_ids_infini,
                GenerationConfig(
                    max_new_tokens=5,  # decode kernel warmup
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    stop_on_eos=False,
                ),
                _measure_and_log_time=False,
            )

        print("=================== warmup done ====================")

        # reset cache back to benchmark config
        if cache_config is not None:
            test.model.reset_cache(cache_config)

    # ---------------------------------------------------------------------------- #
    #                                Warmup done
    # ---------------------------------------------------------------------------- #

    for idx, case in tqdm(cases_dict.items(), desc="Processing cases"):
        tqdm.write(f"\033[92mProcessing : {case}\033[0m")

        batch_size = case["batch_size"]
        input_len = case["input_len"]
        output_len = case["output_len"]

        if not enable_paged_attn:
            # reset cache if static kvcache is used
            initial_capacity = input_len + output_len
            test.model.reset_cache(
                StaticKVCacheConfig(
                    max_batch_size=batch_size, max_cache_len=initial_capacity
                )
            )

        # run test one case
        test.run(
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
        )
