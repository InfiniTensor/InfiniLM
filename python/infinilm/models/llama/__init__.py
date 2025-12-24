import os
from typing import Optional, Union
import infinicore
import time

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
        t1 = time.time()

        if backend == "python":
            from . import modeling_llama

            print("\n***************************************************************")
            print("\t Loading Llama Model with Python Backend")
            print(f"\t Device: {device}, DType: {dtype}")
            print("***************************************************************\n")
            print(" create model ......")

            instance = modeling_llama.LlamaForCausalLM.from_pretrained(
                model_path,
                device=device,
                **kwargs,
            )

        elif backend == "cpp":
            from infinilm.infer_engine import InferEngine

            print("\n***************************************************************")
            print("\t Loading Llama Model with C++ Backend")
            print(f"\t Device: {device}, DType: {dtype}")
            print("***************************************************************\n")
            print(" create model ......")
            instance = InferEngine(model_path, device=device, **kwargs)
        else:
            raise KeyError("invalid backend")

        t2 = time.time()
        print(f" create model over! {(t2 - t1) * 1000} ms \n")

        return instance
