import time
from dataclasses import dataclass

import infinicore

from infinilm.auto_config import AutoConfig
from infinilm.cache import StaticKVCacheConfig, PagedKVCacheConfig
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
            # self.config,
            distributed_config._underlying,
            device._underlying.type,
            cache_config,
            model_path,
        )
        self.use_cache = False

        self.enable_paged_attn = isinstance(cache_config, PagedKVCacheConfig)

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
        block_tables=None,
        slot_mapping=None,
        temperature=None,
        top_k=None,
        top_p=None,
    ):
        # TODO: Remove `_underlying` and simplify the corresponding code.
        input_ids = input_ids._underlying if input_ids is not None else None
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

        return infinicore.Tensor(
            super()
            .forward(
                super().Input(
                    input_ids,
                    position_ids=position_ids,
                    past_sequence_lengths=past_kv_lengths,
                    total_sequence_lengths=total_kv_lengths,
                    input_offsets=input_offsets,
                    block_tables=block_tables,
                    slot_mapping=slot_mapping,
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
        _measure_and_log_time=False,
        paged_block_size=16,
    ):
        if generation_config.eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        else:
            eos_token_id = generation_config.eos_token_id

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
                position_ids=position_ids,
                past_kv_lengths=past_kv_lengths,
                total_kv_lengths=total_kv_lengths,
                input_offsets=input_offsets,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
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
        return super().state_dict()[0].keys()

    def load_state_dict(self, state_dict, strict=None):
        for name, param in state_dict.items():
            super().load_param(name, param._underlying)
