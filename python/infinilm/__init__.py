# Hygon builds resolve flash-attn kernels via dlsym from `flash_attn_2_cuda*.so`
# (the DTK wheel). That .so must be in the process with RTLD_GLOBAL so kernel
# symbols are visible to dlsym(RTLD_DEFAULT, ...). Gate on the presence of
# libflash_attn_hygon_dlsym.so next to this package; on NVIDIA/Iluvatar/etc.
# builds we skip the preload to avoid touching their flash_attn install.
def _preload_flash_attn_global():
    try:
        import ctypes, importlib.util, os
        marker = os.path.join(os.path.dirname(__file__), "lib", "libflash_attn_hygon_dlsym.so")
        if not os.path.exists(marker):
            return
        # libinfinicore + flash_attn_hygon_dlsym both link libtorch; import it
        # first so libtorch_*.so are RTLD_GLOBAL-loaded before our dlopens.
        # Replaces the manual LD_LIBRARY_PATH=$(...torch/lib) workaround.
        import torch  # noqa: F401
        spec = importlib.util.find_spec("flash_attn_2_cuda")
        if spec is None or not spec.origin:
            return
        flag = getattr(os, "RTLD_GLOBAL", 0x100) | getattr(os, "RTLD_LAZY", 0x1)
        ctypes.CDLL(spec.origin, mode=flag)
    except Exception:
        pass

_preload_flash_attn_global()

from .models import AutoLlamaModel
from . import distributed
from . import cache
from . import llm
from . import base_config

from .llm import (
    LLM,
    AsyncLLMEngine,
    SamplingParams,
    RequestOutput,
    TokenOutput,
)

__all__ = [
    "AutoLlamaModel",
    "distributed",
    "cache",
    "llm",
    "base_config",
    # LLM classes
    "LLM",
    "AsyncLLMEngine",
    "SamplingParams",
    "RequestOutput",
    "TokenOutput",
]
