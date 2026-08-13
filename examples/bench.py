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
from infinilm.llm.llm import LLM
from infinilm.llm.sampling_params import SamplingParams
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

    if "text_config" in normalized:
        normalized = normalized["text_config"]

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


def pair_sequence_lengths(
    input_len_list: list[int], output_len_list: list[int]
) -> list[tuple[int, int]]:
    """Return positional input/output length pairs.

    Lists with equal lengths are paired by position, not expanded as a
    Cartesian product. If either side contains one value, that value is
    broadcast across the other side.
    """
    if not input_len_list or not output_len_list:
        raise ValueError("input_len and output_len must not be empty")
    if any(length <= 0 for length in input_len_list):
        raise ValueError(f"input_len values must be positive: {input_len_list}")
    if any(length <= 0 for length in output_len_list):
        raise ValueError(f"output_len values must be positive: {output_len_list}")

    if len(input_len_list) == len(output_len_list):
        return list(zip(input_len_list, output_len_list))
    if len(input_len_list) == 1:
        return [(input_len_list[0], output_len) for output_len in output_len_list]
    if len(output_len_list) == 1:
        return [(input_len, output_len_list[0]) for input_len in input_len_list]
    raise ValueError(
        "input_len and output_len must have the same number of values, "
        "or one side must contain a single value for broadcasting: "
        f"input_len={input_len_list}, output_len={output_len_list}"
    )


def get_paged_kv_cache_num_blocks(cases, block_size: int) -> int:
    """Return the shared paged-cache capacity required by sequential cases."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive: {block_size}")

    case_list = list(cases)
    if not case_list:
        raise ValueError("at least one benchmark case is required")

    return max(
        (
            (case["input_len"] + case["output_len"] + block_size - 1)
            // block_size
        )
        * case["batch_size"]
        for case in case_list
    )


def get_warmup_shapes(cases) -> OrderedDict:
    """Map each prefill shape to the largest output capacity it needs."""
    warmup_shapes = OrderedDict()
    for case in cases:
        shape = (case["batch_size"], case["input_len"])
        warmup_shapes[shape] = max(
            warmup_shapes.get(shape, 0), case["output_len"]
        )
    return warmup_shapes


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
    """Generate cases from batch sizes and positional length pairs.

    Batch sizes are combined with each input/output pair. The two length lists
    themselves are paired by position (or single-value broadcast), never as a
    Cartesian product. Returned cases are ordered by ascending KV-cache usage.
    """
    model_path = os.path.expanduser(model_path)

    if not batch_size_list or any(
        batch_size <= 0 for batch_size in batch_size_list
    ):
        raise ValueError(f"batch_size values must be positive: {batch_size_list}")

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

    length_pairs = pair_sequence_lengths(input_len_list, output_len_list)

    # Each input/output list position is one case. A one-element list is
    # broadcast so one input length can still be tested with many output lengths.
    case_list = []
    for batch_size in batch_size_list:
        for input_len, output_len in length_pairs:
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


def repeat_tokens(input_ids: list[int], target_length: int):
    num = len(input_ids)
    repeat_times = (target_length + num - 1) // num
    return (input_ids * repeat_times)[:target_length]


def split_chat_prompt_tokens(tokenizer, rendered_prompt: str, user_prompt: str):
    """Split one rendered chat prompt around its user-content token span."""
    full_ids = tokenizer.encode(rendered_prompt)
    content_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
    if not content_ids:
        raise ValueError("bench prompt must contain at least one token")

    last_start = len(full_ids) - len(content_ids)
    for start in range(last_start + 1):
        if full_ids[start : start + len(content_ids)] == content_ids:
            return (
                full_ids[:start],
                content_ids,
                full_ids[start + len(content_ids) :],
            )

    raise ValueError(
        "Could not locate the user prompt inside the rendered chat template"
    )


class TestModel:
    model: infinicore.nn.Module
    input_ids_list: list[int]

    def __init__(
        self,
        model_path,
        draft_model_path=None,
        num_draft_tokens=4,
        infini_device=infinicore.device("cpu", 0),
        tp=1,
        skip_load=False,
        cache_config=None,
        enable_graph=False,
        attn_backend="default",
        use_mla=False,
        weight_load_mode="async",
        pre_transpose=False,
        moe_ep_backend="disabled",
        moe_ep_size=1,
        enable_prefix_caching=False,
        prompt="How are you",
    ) -> None:
        model_path = os.path.expanduser(model_path)
        self.draft_model_path = draft_model_path
        self.num_draft_tokens = num_draft_tokens
        self.model_path = model_path
        self.device_str = infini_device.type
        self.tp = tp
        self.cache_config = cache_config
        self.enable_graph = enable_graph
        self.attn_backend = attn_backend
        self.use_mla = use_mla
        self.weight_load_mode = weight_load_mode
        self.skip_load = skip_load
        self.enable_prefix_caching = enable_prefix_caching

        if draft_model_path is not None:
            self.processor = AutoInfinilmProcessor.from_pretrained(model_path)
            self.tokenizer = self.processor.get_tokenizer()
            input_content = self.processor.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prefix_ids, content_ids, suffix_ids = split_chat_prompt_tokens(
                self.tokenizer, input_content, prompt
            )
            self.prompt_prefix_ids = prefix_ids
            self.prompt_content_ids = content_ids
            self.prompt_suffix_ids = suffix_ids
            self.input_ids_list = [prefix_ids + content_ids + suffix_ids]
            self.model = None
            return

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
            pre_transpose=pre_transpose,
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

        prefix_ids, content_ids, suffix_ids = split_chat_prompt_tokens(
            self.tokenizer, input_content, prompt
        )
        self.prompt_prefix_ids = prefix_ids
        self.prompt_content_ids = content_ids
        self.prompt_suffix_ids = suffix_ids
        self.input_ids_list = [prefix_ids + content_ids + suffix_ids]
        self.model = model

    def build_input_ids(self, target_length: int) -> list[int]:
        template_tokens = len(self.prompt_prefix_ids) + len(self.prompt_suffix_ids)
        if target_length < template_tokens:
            raise ValueError(
                f"input_len={target_length} is shorter than the chat template "
                f"overhead ({template_tokens} tokens)"
            )
        content_length = target_length - template_tokens
        input_ids = (
            self.prompt_prefix_ids
            + repeat_tokens(self.prompt_content_ids, content_length)
            + self.prompt_suffix_ids
        )
        assert len(input_ids) == target_length
        return input_ids

    def run(
        self,
        batch_size: int,
        input_len: int,
        output_len: int,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    ):
        input_ids = self.build_input_ids(input_len)
        input_ids_list = [input_ids] * batch_size

        # ---------------------------------------------------------------------------- #
        #                        自回归生成
        # ---------------------------------------------------------------------------- #
        if self.draft_model_path is not None:
            prompt_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            llm = LLM(
                model_path=self.model_path,
                draft_model_path=self.draft_model_path,
                num_draft_tokens=self.num_draft_tokens,
                device=self.device_str,
                tensor_parallel_size=self.tp,
                cache_type="paged" if self.cache_config is not None else "static",
                max_batch_size=batch_size,
                max_tokens=output_len,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                enable_graph=self.enable_graph,
                attn_backend=self.attn_backend,
                use_mla=self.use_mla,
                weight_load_mode=self.weight_load_mode,
                skip_load=self.skip_load,
                enable_prefix_caching=self.enable_prefix_caching,
            )
            t1 = time.time()
            print("=================== start generate ====================")
            outputs = llm.generate(
                prompts=[prompt_text] * batch_size,
                sampling_params=SamplingParams(max_tokens=output_len, ignore_eos=True),
                use_tqdm=False,
            )
            t2 = time.time()
            if cfg.verbose and not skip_load:
                if output_len <= 256:
                    for output in outputs:
                        print(output.outputs[0].text)
                else:
                    print(
                        f"[bench] output text omitted because output_len={output_len} > 256."
                    )
            print(f"total_time: {round((t2 - t1) * 1000, 2)} ms")
            return

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
        # Cases run sequentially and each generate call rebuilds block tables
        # from block zero, so the shared cache needs the largest case capacity,
        # not the sum of all case capacities.
        max_num_blocks = get_paged_kv_cache_num_blocks(
            cases_dict.values(), paged_kv_block_size
        )
        cache_config = PagedKVCacheConfig(max_num_blocks, paged_kv_block_size)
    else:
        cache_config = None

    if enable_paged_attn and attn_backend == "default":
        attn_backend = "paged-attn"

    test = TestModel(
        model_path,
        draft_model_path=cfg.draft_model,
        num_draft_tokens=cfg.num_draft_tokens,
        infini_device=infini_device,
        tp=tp,
        skip_load=skip_load,
        cache_config=cache_config,
        enable_graph=enable_graph,
        attn_backend=attn_backend,
        use_mla=cfg.use_mla,
        weight_load_mode=cfg.weight_load_mode,
        pre_transpose=cfg.pre_transpose,
        moe_ep_backend=moe_ep_backend,
        moe_ep_size=ep,
        enable_prefix_caching=False,
        prompt=cfg.prompt,
    )

    # ---------------------------------------------------------------------------- #
    #                                Warmup
    # ---------------------------------------------------------------------------- #
    if cfg.warmup:
        warmup_steps = 1

        # Warm every distinct prefill shape once. Repeated benchmark cases keep
        # a single warmup, while mixed input lengths do not include first-use
        # graph/kernel setup in their measured run.
        warmup_shapes = get_warmup_shapes(cases_dict.values())

        for warmup_idx, ((warmup_batch, warmup_input_len), max_output_len) in enumerate(
            warmup_shapes.items(), start=1
        ):
            warmup_decode_len = min(5, max_output_len)
            if not enable_paged_attn:
                # Reserve the largest complete case for this prefill shape,
                # even though warmup itself only runs a few decode steps.
                warmup_cache_config = StaticKVCacheConfig(
                    max_batch_size=warmup_batch,
                    max_cache_len=warmup_input_len + max_output_len,
                )
                test.model.reset_cache(warmup_cache_config)

            warmup_prompt_ids = test.build_input_ids(warmup_input_len)
            warmup_ids = [warmup_prompt_ids] * warmup_batch
            input_ids_infini = infinicore.from_list(
                warmup_ids, dtype=infinicore.int64
            )

            print(
                f"\033[93m[warmup {warmup_idx}/{len(warmup_shapes)}] "
                f"batch={warmup_batch}, input_len={warmup_input_len}, "
                f"will prefill + {warmup_decode_len} decode steps\033[0m"
            )
            print("=================== warmup start ===================")
            for _ in range(warmup_steps):
                _ = test.model.generate(
                    input_ids_infini,
                    GenerationConfig(
                        max_new_tokens=warmup_decode_len,
                        temperature=cfg.temperature,
                        top_k=cfg.top_k,
                        top_p=cfg.top_p,
                        stop_on_eos=False,
                    ),
                    _measure_and_log_time=False,
                )
            print("=================== warmup done ====================")

    # ---------------------------------------------------------------------------- #
    #                                Warmup done
    # ---------------------------------------------------------------------------- #

    for idx, case in tqdm(cases_dict.items(), desc="Processing cases"):
        tqdm.write(f"\033[92mProcessing : {case}\033[0m")

        batch_size = case["batch_size"]
        input_len = case["input_len"]
        output_len = case["output_len"]

        if not enable_paged_attn:
            # Each static-cache case gets its exact full generation capacity.
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
