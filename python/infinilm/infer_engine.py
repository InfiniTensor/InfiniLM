from dataclasses import dataclass

import infinicore

from infinilm.auto_config import AutoConfig
from infinilm.cache import StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.lib import _infinilm


@dataclass
class GenerationConfig:
    max_new_tokens: int

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
        self, input_ids, position_ids, cache_positions, *, temperature, top_k, top_p
    ):
        return infinicore.Tensor(
            super()
            .forward(
                super().Input(
                    input_ids._underlying,
                    position_ids._underlying,
                    cache_positions._underlying,
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
        cache_positions = infinicore.from_list([0], dtype=infinicore.int64)

        output_ids = []

        for _ in range(0, generation_config.max_new_tokens):
            output_id = self(
                input_ids,
                position_ids,
                cache_positions,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )

            # TODO: Do not only get the first item here.
            output_id_item = output_id.to_numpy()[0]

            output_ids.append(infinicore.from_list([output_id_item]))

            if output_id_item in eos_token_id:
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
            cache_positions += infinicore.from_list(
                [seq_len], dtype=cache_positions.dtype, device=cache_positions.device
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
