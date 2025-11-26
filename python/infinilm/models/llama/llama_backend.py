import os
from typing import Optional, Union

import infinicore

__all__ = ["AutoLlamaModel"]


class AutoLlamaModel:
    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[Union[str, os.PathLike]],
        device: infinicore.device,
        dtype=infinicore.dtype,
        backend="python",
    ):
        if backend == "python":
            from .modeling_llama import LlamaForCausalLM

            model = LlamaForCausalLM.from_pretrained(
                model_path,
                device=device,
                dtype=dtype,
            )
            return model
        elif backend == "cpp":
            from .llama_cpp import LlamaForCausalLM

            model = LlamaForCausalLM.from_pretrained(
                model_path,
                device=device,
            )
            return model

        raise KeyError("无效的 backend")
