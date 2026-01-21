import time
from dataclasses import dataclass

import infinicore

from infinilm.auto_config import AutoConfig
from infinilm.cache import StaticKVCacheConfig, PagedKVCacheConfig, KVCompressionConfig
from infinilm.distributed import DistConfig
from infinilm.lib import _infinilm


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
        distributed_config=DistConfig(1),
        cache_config=None,
    ):
        self.config = AutoConfig.from_pretrained(model_path)

        if device is None:
            device = infinicore.device()

        super().__init__(
            self.config,
            distributed_config._underlying,
            device._underlying.type,
            cache_config,
        )

        self.use_cache = False

        self.enable_paged_attn = isinstance(cache_config, PagedKVCacheConfig)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids,
        *,
        pixel_values=None,
        position_ids=None,
        past_kv_lengths=None,
        total_kv_lengths=None,
        input_offsets=None,
        block_tables=None,
        slot_mapping=None,
        image_bound=None,
        tgt_sizes=None,
        temperature=None,
        top_k=None,
        top_p=None,
    ):
        # TODO: Remove `_underlying` and simplify the corresponding code.
        input_ids = input_ids._underlying if input_ids is not None else None
        pixel_values = pixel_values._underlying if pixel_values is not None else None
        position_ids = position_ids._underlying if position_ids is not None else None
        past_kv_lengths = (
            past_kv_lengths._underlying if past_kv_lengths is not None else None
        )
        total_kv_lengths = (
            total_kv_lengths._underlying if past_kv_lengths is not None else None
        )
        input_offsets = input_offsets._underlying if input_offsets is not None else None
        block_tables = block_tables._underlying if block_tables is not None else None
        slot_mapping = slot_mapping._underlying if slot_mapping is not None else None
        image_bound = image_bound._underlying if image_bound is not None else None
        tgt_sizes = tgt_sizes._underlying if tgt_sizes is not None else None

        return infinicore.Tensor(
            super()
            .forward(
                super().Input(
                    input_ids,
                    pixel_values=pixel_values,
                    position_ids=position_ids,
                    past_sequence_lengths=past_kv_lengths,
                    total_sequence_lengths=total_kv_lengths,
                    input_offsets=input_offsets,
                    block_tables=block_tables,
                    slot_mapping=slot_mapping,
                    image_bound=image_bound,
                    tgt_sizes=tgt_sizes,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            )
            .output_ids
        )

    def generate(
        self,
        input_ids,
        generation_config,
        *,
        pixel_values=None,
        image_bound=None,
        tgt_sizes=None,
        kv_compression_config=None,
        _measure_and_log_time=False,
        paged_block_size=16,
    ):
        if generation_config.eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        else:
            eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        llava_extra_tokens = 0
        llava_has_image = (
            pixel_values is not None and getattr(self.config, "model_type", "") == "llava"
        )
        if llava_has_image:
            image_token_index = getattr(self.config, "image_token_index", None)
            if image_token_index is None:
                raise ValueError("image_token_index not found in LlavaConfig")
            ids_np = input_ids.to_numpy()
            num_image_tokens = int((ids_np == image_token_index).sum(axis=1)[0])
            patch_size = self.config.vision_config.patch_size
            image_size = self.config.vision_config.image_size
            num_patches = (image_size // patch_size) ** 2
            llava_extra_tokens = num_image_tokens * (num_patches - 1)

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

        if kv_compression_config is not None and self.enable_paged_attn:
            raise ValueError("KV compression is only supported with static KV cache.")

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
                block_tables_list = [
                    [
                        i * batch_size + b
                        for i in range(
                            (past_seq_len + seq_len + paged_block_size - 1)
                            // paged_block_size
                        )
                    ]
                    for b in range(batch_size)
                ]
                slot_mapping_list = [
                    (((past_seq_len + i) // paged_block_size) * batch_size + b)
                    * paged_block_size
                    + (past_seq_len + i) % paged_block_size
                    for b in range(batch_size)
                    for i in range(seq_len)
                ]

                block_tables = infinicore.from_list(
                    block_tables_list,
                    dtype=infinicore.int64,
                )
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

                block_tables = None
                slot_mapping = None

            past_kv_lengths = infinicore.from_list(
                [past_seq_len] * batch_size, dtype=infinicore.int64
            )
            total_kv_lengths = infinicore.from_list(
                [past_seq_len + seq_len] * batch_size, dtype=infinicore.int64
            )

            input_offsets = infinicore.from_list(
                [seq_len * i for i in range(batch_size + 1)], dtype=infinicore.int64
            )

            output_id = self(
                input_ids=input_ids,
                pixel_values=pixel_values if iter == 0 else None,
                position_ids=position_ids,
                past_kv_lengths=past_kv_lengths,
                total_kv_lengths=total_kv_lengths,
                input_offsets=input_offsets,
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

            input_ids = infinicore.from_list(
                [[output_id] for output_id in output_id.to_numpy().tolist()]
            )
            if iter == 0 and llava_has_image:
                past_seq_len = past_seq_len + seq_len + llava_extra_tokens
            else:
                past_seq_len = past_seq_len + seq_len

            if iter == 0 and kv_compression_config is not None:
                if isinstance(kv_compression_config, dict):
                    kv_compression_config = KVCompressionConfig(**kv_compression_config)
                past_seq_len = int(
                    super().compress_kv_cache_inplace(
                        past_seq_len, batch_size, kv_compression_config
                    )
                )

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
                f" Prefill TTFT: {round(time_measurements[0], 2)}ms  Throughput: {round((initial_batch_size * initial_seqlen) / time_measurements[0], 2)}tok/s\n",
            )
            if len(time_measurements) > 1:
                print(
                    f" Decode  Avg ITL: {round(sum(time_measurements[1:]) * 1000 / (len(time_measurements) - 1), 2)}ms   Throughput: {round((initial_batch_size * (len(time_measurements) - 1)) / sum(time_measurements[1:]), 2)}tok/s\n",
                )

        return output_ids

    def reset_cache(self, cache_config):
        infinicore.sync_device()
        self.enable_paged_attn = isinstance(cache_config, PagedKVCacheConfig)
        super().reset_cache(cache_config)

    def compress_kv_cache_inplace(self, seq_len, batch_size, config):
        if isinstance(config, dict):
            config = KVCompressionConfig(**config)
        return super().compress_kv_cache_inplace(seq_len, batch_size, config)

    def state_dict_keyname(self):
        return super().state_dict()[0].keys()

    def load_state_dict(self, state_dict, strict=None):
        for name, param in state_dict.items():
            super().load_param(name, param._underlying)
