import infinicore

from infinilm.auto_config import AutoConfig
from infinilm.cache import StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.generation.utils import GenerationMixin
from infinilm.lib import _infinilm


class InferEngine(_infinilm.InferEngine, GenerationMixin):
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
            .output_ids
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
