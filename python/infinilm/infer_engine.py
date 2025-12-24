import json
import os

import infinicore

from infinilm.cache import StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.generation.utils import GenerationMixin
from infinilm.lib import _infinilm
from infinilm.models.llama.configuration_llama import LlamaConfig


class InferEngine(_infinilm.InferEngine, GenerationMixin):
    def __init__(
        self,
        model_path,
        device=None,
        distributed_config=DistConfig(1),
        cache_config=None,
    ):
        config_path = os.path.join(model_path, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"`{config_path}` not found")

        with open(config_path) as f:
            config_dict = json.load(f)

        self.config = LlamaConfig(**config_dict)

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

    def forward(self, input_ids, position_ids, cache_positions, *args, **kwargs):
        return infinicore.Tensor(
            super()
            .forward(
                super().Input(
                    input_ids._underlying,
                    position_ids._underlying,
                    cache_positions._underlying,
                )
            )
            .logits
        )

    def reset_cache(self, batch_size: int, initial_capacity: int = 1024):
        infinicore.sync_device()

        cache_config = StaticKVCacheConfig(batch_size, initial_capacity)

        super().reset_cache(cache_config)

    def state_dict_keyname(self):
        return super().state_dict()[0].keys()

    def load_state_dict(self, state_dict, strict=None):
        for name, param in state_dict.items():
            super().load_param(name, param._underlying)
