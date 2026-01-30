from .models import AutoLlamaModel
from . import distributed
from . import cache
from . import llm

# Fusion support (optional)
try:
    from . import fusion_utils
    from . import fused_infer_engine
    _fusion_available = True
except ImportError:
    _fusion_available = False

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
    # LLM classes
    "LLM",
    "AsyncLLMEngine",
    "SamplingParams",
    "RequestOutput",
    "TokenOutput",
]

# Conditionally add fusion exports
if _fusion_available:
    __all__.extend(["fusion_utils", "fused_infer_engine"])
