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
        **kwargs,
    ):
        if backend == "python":
            from . import modeling_llama

            print("\n***************************************************************")
            print("\t\t Loading Llama Model with Python Backend")
            print(f"\t\t Device: {device}, DType: {dtype}")
            print("***************************************************************\n")
            return modeling_llama.LlamaForCausalLM.from_pretrained(
                model_path,
                device=device,
                dtype=dtype,
                **kwargs,
            )

        elif backend == "cpp":
            from .backends import cpp

            print("\n***************************************************************")
            print("\t\tLoading Llama Model with C++ Backend")
            print(f"\t\tDevice: {device}, DType: {dtype}")
            print("***************************************************************\n")
            return cpp.LlamaForCausalLM.from_pretrained(
                model_path,
                device=device,
                dtype=dtype,
                **kwargs,
            )

        raise KeyError("invalid backend")
