import json
import os
import sys
import time
from collections import OrderedDict

import infinicore
import numpy as np
from infinilm.base_config import BaseConfig
from infinilm.cache import PagedKVCacheConfig, StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import GenerationConfig, InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file
from infinilm.moe_config import configure_moe_ep_backend
from infinilm.processors import AutoInfinilmProcessor
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))


DATA_TYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}

_PAGED_KV_BLOCK_SIZE = 256

# Maps model_type to its specific config key normalization rules.
# Each rule maps a standard key (e.g., "head_dim") to either:
#   - A string: representing the model-specific key name for direct mapping.
#   - A callable: a function that takes the config dict and computes the derived value.
_CONFIG_KEY_MAP = {
    "chatglm": {
        "num_key_value_heads": "multi_query_group_num",
        "num_hidden_layers": "num_layers",
        "head_dim": "kv_channels",
    },
    "baichuan": {
        "num_key_value_heads": "num_attention_heads",
        "head_dim": lambda cfg: cfg["hidden_size"] // cfg["num_attention_heads"],
    },
}


def _normalize_config(config, model_type):
    """
    Normalize model config to standard keys.

    Applies model-specific key mappings and derived computations defined in
    _CONFIG_KEY_MAP. Standard keys already present in the original config
    will not be overwritten.
    """
    normalized = dict(config)

    key_map = _CONFIG_KEY_MAP.get(model_type)

    if not key_map:
        return normalized

    for std_key, rule in key_map.items():
        # Skip if the standard key already exists in the original config
        if std_key in normalized:
            continue

        # Rule is a string: perform a direct key remapping
        if isinstance(rule, str):
            if rule in normalized:
                normalized[std_key] = normalized[rule]

        # Rule is a callable: compute the derived value dynamically
        elif callable(rule):
            try:
                normalized[std_key] = rule(normalized)
            except (KeyError, ZeroDivisionError, TypeError):
                # Silently skip if dependencies are missing or computation fails
                pass

    return normalized


# BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128]
# INPUT_LENS = [32, 256, 1024, 4096]
# OUTPUT_LENS = [256, 1024, 4096]


def read_json_file(file_path):
    """Load and return JSON content from file_path."""
    with open(file_path, "r") as file:
        return json.load(file)


def get_test_cases(
    model_path: str,
    batch_size_list: list[int],
    input_len_list: list[int],
    output_len_list: list[int],
    use_mla: bool = False,
):
    model_path = os.path.expanduser(model_path)

    """Generate cases ordered by ascending KV cache memory usage."""
    # Load model config to derive attention dimensions
    config = read_json_file(os.path.join(model_path, "config.json"))
    model_type = config.get("model_type", "")
    config = _normalize_config(config, model_type)
    if model_type == "mamba":
        config.setdefault("num_hidden_layers", config.get("n_layer", 1))
        config.setdefault("num_key_value_heads", 1)
        config.setdefault("head_dim", config.get("state_size", 16))
    head_dim = config.get("head_dim")
    if head_dim is None:
        head_dim = config.get("hidden_size") // config.get("num_attention_heads")
    # KV heads and layers drive cache size. DeepSeek MLA stores a single KV head
    # with latent K and V dimensions instead of the regular per-head K/V cache.
    if use_mla and model_type == "deepseek_v2":
        num_key_value_heads = 1
        head_dim = config["kv_lora_rank"] * 2 + config["qk_rope_head_dim"]
    else:
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


prompt_path = (
    "examples/bench_prompt.md"
    if os.path.isfile("examples/bench_prompt.md")
    else "InfiniLM/examples/bench_prompt.md"
)
with open(prompt_path, "r") as f:
    prompt = f.read()


def repeat_prompt(input_ids: list[int], target_length: int):
    num = len(input_ids)
    repeat_times = (target_length + num - 1) // num
    return (input_ids * repeat_times)[:target_length]


class TestModel:
    model: infinicore.nn.Module
    input_ids_list: list[int]

    def __init__(
        self,
        model_path,
        infini_device=infinicore.device("cpu", 0),
        tp=1,
        skip_load=False,
        cache_config=None,
        enable_graph=False,
        attn_backend="default",
        use_mla=False,
        weight_load_mode="async",
        moe_ep_backend="disabled",
        moe_ep_size=1,
    ) -> None:
        model_path = os.path.expanduser(model_path)
        # ---------------------------------------------------------------------------- #
        #                        创建模型,
        # ---------------------------------------------------------------------------- #
        model = InferEngine(
            model_path,
            device=infini_device,
            distributed_config=DistConfig(
                tp,
                moe_ep_backend=moe_ep_backend,
                moe_ep_size=moe_ep_size,
            ),
            cache_config=cache_config,
            enable_graph_compiling=enable_graph,
            attention_backend=attn_backend,
            kv_cache_dtype=cfg.kv_cache_dtype,
            use_mla=use_mla,
            weight_load_mode=weight_load_mode,
        )

        # ---------------------------------------------------------------------------- #
        #                        加载权重
        # ---------------------------------------------------------------------------- #
        if not skip_load:
            load_model_state_dict_by_file(model, model_path, dtype=model.dtype)

        # ---------------------------------------------------------------------------- #
        #                        创建 tokenizer
        # ---------------------------------------------------------------------------- #
        self.processor = AutoInfinilmProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.get_tokenizer()

        # ---------------------------------------------------------------------------- #
        #                        token编码
        # ---------------------------------------------------------------------------- #
        input_content = self.processor.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )

        input_ids_list = [
            self.tokenizer.encode(
                input_content,
            )
        ]

        self.model = model
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
        input_ids_infini = infinicore.from_list(input_ids_list, dtype=infinicore.int64)

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
        if not skip_load:
            print(self.tokenizer.decode(numpy_output_ids, skip_special_tokens=True))

        print(
            f"total_time: {round((t2 - t1) * 1000, 2)} ms",
        )


if __name__ == "__main__":
    cfg = BaseConfig()

    device_str = cfg.get_device_str(cfg.device)

    _PAGED_KV_BLOCK_SIZE = cfg.block_size
    # -------------------------------------------------------- #
    #             解析参数
    # -------------------------------------------------------- #
    model_path = cfg.model

    infini_device = infinicore.device(device_str, 0)

    tp = cfg.tp
    dp = cfg.dp
    moe_ep_backend, ep = configure_moe_ep_backend(
        tp, dp, cfg.ep, cfg.moe_ep_backend, model_path
    )
    print(f"MoE EP backend: {moe_ep_backend}  TP={tp}  DP={dp}  EP={ep}")

    skip_load = cfg.skip_load

    batch_size = cfg.batch_size
    input_len = cfg.input_len
    output_len = cfg.output_len
    enable_paged_attn = cfg.enable_paged_attn
    enable_graph = cfg.enable_graph
    attn_backend = cfg.attn

    if isinstance(batch_size, int):
        batch_size = [batch_size]

    if isinstance(input_len, int):
        input_len = [input_len]

    if isinstance(output_len, int):
        output_len = [output_len]

    cases_dict = get_test_cases(
        model_path, batch_size, input_len, output_len, use_mla=cfg.use_mla
    )
    # -------------------------------------------------------- #
    #             测试
    # -------------------------------------------------------- #
    if enable_paged_attn:
        paged_kv_block_size = _PAGED_KV_BLOCK_SIZE
        max_num_blocks = max(
            [
                (
                    (c_["input_len"] + c_["output_len"] + (paged_kv_block_size - 1))
                    // paged_kv_block_size
                )
                * c_["batch_size"]
                for _, c_ in cases_dict.items()
            ]
        )
        cache_config = PagedKVCacheConfig(max_num_blocks, paged_kv_block_size)
    else:
        cache_config = None

    if enable_paged_attn and attn_backend == "default":
        attn_backend = "paged-attn"

    test = TestModel(
        model_path,
        infini_device=infini_device,
        tp=tp,
        skip_load=skip_load,
        cache_config=cache_config,
        enable_graph=enable_graph,
        attn_backend=attn_backend,
        use_mla=cfg.use_mla,
        weight_load_mode=cfg.weight_load_mode,
        moe_ep_backend=moe_ep_backend,
        moe_ep_size=ep,
    )

    # ---------------------------------------------------------------------------- #
    #                                Warmup
    # ---------------------------------------------------------------------------- #
    if cfg.warmup:
        warmup_steps = 1

        # warmup cache capacity
        warmup_case = next(iter(cases_dict.values()))
        warmup_batch = warmup_case["batch_size"]
        warmup_input_len = warmup_case["input_len"]
        warmup_decode_len = 5

        if enable_paged_attn:
            warmup_num_blocks = (
                (warmup_input_len + warmup_decode_len + paged_kv_block_size - 1)
                // paged_kv_block_size
            ) * warmup_batch
            warmup_cache_config = PagedKVCacheConfig(
                warmup_num_blocks, paged_kv_block_size
            )
        else:
            warmup_cache_config = StaticKVCacheConfig(
                max_batch_size=warmup_batch,
                max_cache_len=warmup_input_len + warmup_decode_len,
            )

        test.model.reset_cache(warmup_cache_config)

        warmup_prompt_ids = repeat_prompt(test.input_ids_list[0], warmup_input_len)
        warmup_ids = [warmup_prompt_ids] * warmup_batch

        input_ids_infini = infinicore.from_list(warmup_ids, dtype=infinicore.int64)

        print(
            f"\033[93m[warmup] batch={warmup_batch}, input_len={warmup_input_len}, "
            f"will prefill + {warmup_decode_len} decode steps\033[0m"
        )
        print("=================== warmup start ===================")

        for _ in range(warmup_steps):
            _ = test.model.generate(
                input_ids_infini,
                GenerationConfig(
                    max_new_tokens=warmup_decode_len,  # decode kernel warmup
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
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
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            temperature=cfg.temperature,
        )
