from dataclasses import dataclass

import infinicore

from infinilm.auto_config import AutoConfig
from infinilm.cache import StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.lib import _infinilm


@dataclass
class GenerationConfig:
    max_new_tokens: int | None = None

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0

    eos_token_id: list[int] | None = None


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

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids,
        *,
        position_ids=None,
        cache_lengths=None,
        input_lengths=None,
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
        cache_lengths = cache_lengths._underlying if cache_lengths is not None else None
        input_lengths = input_lengths._underlying if input_lengths is not None else None
        input_offsets = input_offsets._underlying if input_offsets is not None else None
        block_tables = block_tables._underlying if block_tables is not None else None
        slot_mapping = slot_mapping._underlying if slot_mapping is not None else None

        return infinicore.Tensor(
            super()
            .forward(
                super().Input(
                    input_ids,
                    position_ids=position_ids,
                    cache_lengths=cache_lengths,
                    input_lengths=input_lengths,
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

    def generate(self, input_ids, generation_config):
        if generation_config.eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        else:
            eos_token_id = generation_config.eos_token_id

        # TODO: Remove the `to_numpy` calls and simplify the corresponding code.
        batch_size, seq_len = input_ids.shape[:2]

        position_ids = infinicore.from_list(
            [list(range(0, seq_len)) for _ in range(batch_size)], dtype=infinicore.int64
        )
        cache_lengths = infinicore.from_list([0], dtype=infinicore.int64)

        output_ids = []

        if batch_size != 1 and generation_config.max_new_tokens is None:
            raise ValueError(
                "When `batch_size > 1`, `max_new_tokens` must be specified."
            )

        for _ in range(0, generation_config.max_new_tokens):
            output_id = self(
                input_ids,
                position_ids=position_ids,
                cache_lengths=cache_lengths,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )

            output_ids.append(output_id)

            if (
                generation_config.max_new_tokens is not None
                and output_id.to_numpy()[0] in eos_token_id
            ):
                break

            seq_len = position_ids.shape[-1]

            input_ids = infinicore.from_list(
                [[output_id] for output_id in output_id.to_numpy().tolist()]
            )
            position_ids = infinicore.from_list(
                [1 for _ in range(batch_size)],
                dtype=position_ids.dtype,
                device=position_ids.device,
            ).view((batch_size, 1)) + position_ids.narrow(1, seq_len - 1, 1)
            cache_lengths += infinicore.from_list(
                [seq_len], dtype=cache_lengths.dtype, device=cache_lengths.device
            )

        return output_ids

    def reset_cache(self, batch_size: int, initial_capacity: int = 1024):
        infinicore.sync_device()

        cache_config = StaticKVCacheConfig(batch_size, initial_capacity)

        super().reset_cache(cache_config)

    def state_dict_keyname(self):
        return super().state_dict()[0].keys()

    def load_state_dict(self, state_dict, strict=None):
        for name, param in state_dict.items():
            super().load_param(name, param._underlying)
