from ....generation.utils import GenerationMixin
import infinicore
import os
from typing import Optional, Union


class LlamaForCausalLM(GenerationMixin):
    def __init__(self):
        super().__init__()
        self.use_cache = False
        self._model = None
        raise NotImplementedError("NotImplementedError!!")

    def forward(self, input_ids, position_ids, *args, **kwargs):
        kv_caches = None
        return infinicore.Tensor(
            self._model.forward(input_ids, position_ids, kv_caches)
        )

    def __call__(self, input_ids, position_ids, *args, **kwargs):
        return self.forward(input_ids=input_ids, position_ids=position_ids)

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, os.PathLike],
        device: infinicore.device,
        dtype=infinicore.dtype,
    ):
        """
        Load a pretrained LlamaForCausalLM model from a directory.
        Args:
            model_path: Path to the model directory containing config.json
            device: Device instance (defaults to CPU)
        Returns:
            LlamaForCausalLM instance
        """
        raise NotImplementedError("NotImplementedError!!")
