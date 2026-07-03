import json
import os
import time
from dataclasses import dataclass

import infinicore

from infinilm.cache import PagedKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.lib import _infinilm

from .exception_utils import handle_oom_and_exit
from .modeling_utils import parse_dtype

_MODEL_DEFAULTS = {
    "gpt2": {"torch_dtype": "float32"},
    "mistral": {"torch_dtype": "bfloat16"},
}


def _apply_torch_dtype_defaults(config: dict) -> dict:
    if config.get("torch_dtype") is None:
        config["torch_dtype"] = config.get("dtype") or _MODEL_DEFAULTS.get(
            config.get("model_type"), {}
        ).get("torch_dtype")
    return config


def _normalize_videonsa_config(config_dict):
    model_type = config_dict.get("model_type")

    if model_type == "qwen2_5_vl" and config_dict.get("architectures") == [
        "VideoNSAForConditionalGeneration"
    ]:
        normalized = dict(config_dict)
        normalized["model_type"] = "videonsa"
        normalized["original_model_type"] = model_type
        if "text_config" in normalized:
            text_config = dict(normalized["text_config"])
            text_config["model_type"] = "videonsa"
            text_config.setdefault("torch_dtype", normalized.get("torch_dtype"))
            text_config.setdefault(
                "head_dim",
                text_config["hidden_size"] // text_config["num_attention_heads"],
            )
            text_config.setdefault("attention_bias", True)
            normalized["text_config"] = text_config
        return normalized

    return config_dict


def read_hf_config(model_path):
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    if "model_type" not in config_dict:
        raise ValueError(
            f"`model_type` is not specified in the config file `{config_path}`."
        )

    config_dict = _apply_torch_dtype_defaults(config_dict)
    config_dict = _normalize_videonsa_config(config_dict)

    return config_dict


# config.json (required) defines model architecture, while generation_config.json
# (optional) defines generation behavior. They are kept as separate readers
# because: 1) config.json must exist and requires model_type validation,
# whereas generation_config.json may not exist; 2) keeping them separate
# preserves clear semantics and avoids a one-size-fits-all function with
# multiple conditional parameters.
def read_hf_generation_config(model_path):
    gen_config_path = os.path.join(model_path, "generation_config.json")
    if os.path.exists(gen_config_path):
        with open(gen_config_path, "r") as f:
            return json.load(f)
    return {}


@dataclass
class GenerationConfig:
    max_new_tokens: int | None = None

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0

    eos_token_id: list[int] | None = None
    stop_on_eos: bool = True


class InferEngine(_infinilm.InferEngine):
    def __init__(
        self,
        model_path,
        device=None,
        distributed_config=None,
        cache_config=None,
        enable_graph_compiling=False,
        attention_backend="default",
        kv_cache_dtype=None,
        use_mla=False,
        weight_load_mode="async",
        moe_ep_backend="disabled",
        moe_ep_size=1,
        skip_legacy_moe=False,
    ):
        self.hf_config = read_hf_config(model_path)
        self.hf_generation_config = read_hf_generation_config(model_path)
        self.hf_config["skip_legacy_moe"] = bool(skip_legacy_moe)

        if device is None:
            device = infinicore.device()
        if distributed_config is None:
            distributed_config = DistConfig(1)
        if (
            moe_ep_backend != "disabled"
            or moe_ep_size != 1
            or (
                distributed_config.moe_ep_backend == "disabled"
                and distributed_config.moe_ep_size == 1
            )
        ):
            distributed_config.moe_ep_backend = moe_ep_backend
            distributed_config.moe_ep_size = moe_ep_size

        hf_config_str = json.dumps(self.hf_config)
        super().__init__(
            hf_config_str,
            distributed_config._underlying,
            device._underlying.type,
            cache_config,
            enable_graph_compiling,
            attention_backend,
            (
                parse_dtype(kv_cache_dtype)._underlying
                if kv_cache_dtype is not None
                else None
            ),
            use_mla,
            weight_load_mode,
        )
        self.use_cache = False

        self.enable_paged_attn = isinstance(cache_config, PagedKVCacheConfig)

    @property
    def dtype(self):
        torch_dtype = self.hf_config.get("torch_dtype")
        if torch_dtype is None:
            torch_dtype = self.hf_config.get("dtype")
        return parse_dtype(torch_dtype)

    @property
    def model_type(self):
        return self.hf_config["model_type"]

    @property
    def eos_token_id(self):
        # HuggingFace priority: generation_config.json > config.json
        # HuggingFace's documented loading priority for generation parameters
        # (see transformers/generation/utils.py, GenerationMixin.generate docstring):
        #   1) from the `generation_config.json` model file, if it exists
        #   2) from the model configuration (config.json)
        #
        # config.json may contain incomplete or outdated generation parameters
        # because HuggingFace treats config.json as model architecture config
        # and generation_config.json as generation behavior config. For example,
        # InternLM3's config.json has eos_token_id=2, while
        # generation_config.json has eos_token_id=[2, 128131].
        # Following this priority ensures we always get the authoritative value.
        eos_token_id = (
            self.hf_generation_config.get("eos_token_id")
            or self.hf_config.get("eos_token_id")
            or []
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        return eos_token_id

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids,
        *,
        position_ids=None,
        past_kv_lengths=None,
        total_kv_lengths=None,
        input_offsets=None,
        cu_seqlens=None,
        block_tables=None,
        slot_mapping=None,
        pixel_values=None,
        image_bound=None,
        tgt_sizes=None,
        image_req_ids=None,
        visual_token_ranges=None,
        temperature=None,
        top_k=None,
        top_p=None,
    ):
        try:
            # TODO: Remove `_underlying` and simplify the corresponding code.
            input_ids = input_ids._underlying if input_ids is not None else None
            position_ids = (
                position_ids._underlying if position_ids is not None else None
            )
            past_kv_lengths = (
                past_kv_lengths._underlying if past_kv_lengths is not None else None
            )
            total_kv_lengths = (
                total_kv_lengths._underlying if total_kv_lengths is not None else None
            )
            input_offsets = (
                input_offsets._underlying if input_offsets is not None else None
            )
            block_tables = (
                block_tables._underlying if block_tables is not None else None
            )
            cu_seqlens = cu_seqlens._underlying if cu_seqlens is not None else None
            slot_mapping = (
                slot_mapping._underlying if slot_mapping is not None else None
            )

            def convert_tensor_list(tensor_list_):
                if tensor_list_ is None:
                    return None
                if not isinstance(tensor_list_, list):
                    tensor_list_ = [tensor_list_]
                if len(tensor_list_) == 0:
                    return None
                return [tensor._underlying for tensor in tensor_list_]

            pixel_values = convert_tensor_list(pixel_values)
            image_bound = convert_tensor_list(image_bound)
            tgt_sizes = convert_tensor_list(tgt_sizes)

            return infinicore.Tensor(
                super()
                .forward(
                    super().Input(
                        input_ids,
                        position_ids=position_ids,
                        past_sequence_lengths=past_kv_lengths,
                        total_sequence_lengths=total_kv_lengths,
                        input_offsets=input_offsets,
                        cu_seqlens=cu_seqlens,
                        block_tables=block_tables,
                        slot_mapping=slot_mapping,
                        pixel_values=pixel_values,
                        image_bound=image_bound,
                        tgt_sizes=tgt_sizes,
                        image_req_ids=image_req_ids,
                        visual_token_ranges=visual_token_ranges,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                )
                .output_ids
            )
        except BaseException as e:
            handle_oom_and_exit(e)
            raise

    def generate(
        self,
        input_ids,
        generation_config,
        *,
        pixel_values=None,
        image_bound=None,
        tgt_sizes=None,
        _measure_and_log_time=False,
    ):
        eos_token_id = self.eos_token_id

        past_seq_len = 0
        output_ids = []
        initial_batch_size, initial_seqlen = input_ids.shape[:2]
        seq_len = initial_seqlen
        batch_size = initial_batch_size

        if batch_size != 1 and generation_config.max_new_tokens is None:
            raise ValueError(
                "When `batch_size > 1`, `max_new_tokens` must be specified."
            )

        if _measure_and_log_time:
            time_measurements = []

        block_tables = None
        max_blocks_per_batch = 0
        if self.enable_paged_attn:
            paged_block_size = self.get_cache_config().block_size()
            max_blocks_per_batch = (
                initial_seqlen + generation_config.max_new_tokens + paged_block_size - 1
            ) // paged_block_size

            block_tables_list = [
                list(range(i * max_blocks_per_batch, (i + 1) * max_blocks_per_batch))
                for i in range(batch_size)
            ]

            block_tables = infinicore.from_list(
                block_tables_list,
                dtype=infinicore.int32,
            )

        for iter in range(0, generation_config.max_new_tokens):
            if _measure_and_log_time:
                start_time = time.perf_counter()

            batch_size, seq_len = input_ids.shape[:2]

            if self.enable_paged_attn:
                input_ids = input_ids.view([1, batch_size * seq_len])
                position_ids = infinicore.from_list(
                    list(range(past_seq_len, past_seq_len + seq_len)) * batch_size,
                    dtype=infinicore.int64,
                )

                if iter == 0:
                    slot_mapping_list = []
                    for b in range(batch_size):
                        slot_mapping_list.extend(
                            [
                                b * max_blocks_per_batch * paged_block_size + i
                                for i in range(seq_len)
                            ]
                        )
                else:
                    slot_mapping_list = [
                        i
                        for i in range(
                            past_seq_len,
                            max_blocks_per_batch
                            * paged_block_size
                            * initial_batch_size,
                            max_blocks_per_batch * paged_block_size,
                        )
                    ]

                slot_mapping = infinicore.from_list(
                    slot_mapping_list,
                    dtype=infinicore.int64,
                )
            else:
                position_ids = infinicore.from_list(
                    [
                        list(range(past_seq_len, past_seq_len + seq_len))
                        for _ in range(batch_size)
                    ],
                    dtype=infinicore.int64,
                )

                slot_mapping = None

            past_kv_lengths = infinicore.from_list(
                [past_seq_len] * batch_size, dtype=infinicore.int32
            )
            total_kv_lengths = infinicore.from_list(
                [past_seq_len + seq_len] * batch_size, dtype=infinicore.int32
            )
            cu_seqlens = infinicore.from_list(
                [(past_seq_len + seq_len) * i for i in range(batch_size + 1)],
                dtype=infinicore.int32,
            )
            input_offsets = infinicore.from_list(
                [seq_len * i for i in range(batch_size + 1)], dtype=infinicore.int32
            )

            output_id = self(
                input_ids=input_ids,
                pixel_values=pixel_values if iter == 0 else None,
                position_ids=position_ids,
                past_kv_lengths=past_kv_lengths,
                total_kv_lengths=total_kv_lengths,
                input_offsets=input_offsets,
                cu_seqlens=cu_seqlens,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                image_bound=image_bound if iter == 0 else None,
                tgt_sizes=tgt_sizes if iter == 0 else None,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )

            output_ids.append(output_id)

            if (
                initial_batch_size == 1
                and generation_config.stop_on_eos
                and generation_config.max_new_tokens is not None
                and output_id.to_numpy()[0] in eos_token_id
            ):
                break

            # start_prepare_time = time.perf_counter()
            input_ids = output_id.view([batch_size, 1])

            past_seq_len = past_seq_len + seq_len

            if _measure_and_log_time:
                end_time = time.perf_counter()

                time_measurements.append((end_time - start_time))

        if _measure_and_log_time:
            print(
                f"\n\n\n Generation completed in {round(sum(time_measurements) * 1000, 2)} ms"
            )
            print(
                f" Batchsize={initial_batch_size}  Per_Batch_Input_Len={initial_seqlen}  Per_Batch_New_Tokens={len(time_measurements)}\n"
            )
            print(
                f" Prefill TTFT: {round(time_measurements[0] * 1000, 2)} ms  Throughput: {round((initial_batch_size * initial_seqlen) / time_measurements[0], 2)} tok/s\n",
            )
            if len(time_measurements) > 1:
                print(
                    f" Decode  Avg ITL: {round(sum(time_measurements[1:]) * 1000 / (len(time_measurements) - 1), 2)} ms   Throughput: {round((initial_batch_size * (len(time_measurements) - 1)) / sum(time_measurements[1:]), 2)} tok/s\n",
                )

        return output_ids

    def reset_cache(self, cache_config):
        infinicore.sync_device()
        self.enable_paged_attn = isinstance(cache_config, PagedKVCacheConfig)
        super().reset_cache(cache_config)

    def state_dict_keyname(self):
        return list(super().state_dict_keyname())

    def load_state_dict(self, state_dict, strict=None):
        # MoE/quantized paths may register internal packed tensors that are not
        # present in the HF checkpoint, so callers can request non-strict loads.
        super().load_params(
            {name: param._underlying for name, param in state_dict.items()},
            strict=True if strict is None else strict,
        )

    def process_weights_after_loading(self):
        super().process_weights_after_loading()

    def get_kv_cache(self) -> list[list[infinicore.Tensor]]:
        """
        get per-rank kv cache.
        """
        kv_cache_list = super().get_kv_cache()
        infinicore.sync_device()

        result = []
        for rank_idx, kv_caches_per_rank in enumerate(kv_cache_list):
            result_rank = []
            for layer_idx, layer_kv in enumerate(kv_caches_per_rank):
                result_rank.append(infinicore.Tensor(layer_kv))
            result.append(result_rank)
        return result
