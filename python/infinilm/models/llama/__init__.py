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
            from . import modeling_llama

            return modeling_llama.LlamaForCausalLM.from_pretrained(
                model_path,
                device=device,
                dtype=dtype,
            )

        elif backend == "cpp":
            from .backends import cpp

            return cpp.LlamaForCausalLM.from_pretrained(
                model_path,
                device=device,
                dtype=dtype,
            )

        raise KeyError("invalid backend")
