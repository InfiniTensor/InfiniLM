import os
from typing import Optional, Union
import infinicore
import time
from . import modeling_llama

__all__ = ["AutoLlamaModel"]


class AutoLlamaModel:
    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[Union[str, os.PathLike]],
        device: infinicore.device,
        dtype=infinicore.dtype,
        **kwargs,
    ):
        t1 = time.time()

        print("\n***************************************************************")
        print("\t Loading Llama Model")
        print(f"\t Device: {device}, DType: {dtype}")
        print("***************************************************************\n")
        print(" create model ......")

        instance = modeling_llama.LlamaForCausalLM.from_pretrained(
            model_path,
            device=device,
            **kwargs,
        )

        t2 = time.time()
        print(f" create model over! {(t2 - t1) * 1000} ms \n")

        return instance
