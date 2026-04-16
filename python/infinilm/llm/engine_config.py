"""
Engine configuration — shared by LLMEngine, Worker, ModelRunner.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EngineConfig:
    """Configuration for LLM Engine.

    Attributes:
        model_path: Path to the model directory.
        device: Device type string ('cpu', 'cuda', 'mlu', etc.).
        dtype: Data type string ('float16', 'bfloat16', 'float32').
        tensor_parallel_size: Number of devices for tensor parallelism.
        cache_type: Cache type ('paged' or 'static').
        max_batch_size: Maximum batch size for inference (only for paged cache).
        max_tokens: Default maximum tokens to generate.
        num_blocks: Number of KV cache blocks (only for paged cache).
        block_size: Size of each KV cache block (only for paged cache).
        max_cache_len: Maximum sequence length (only for static cache).
        temperature: Default sampling temperature.
        top_p: Default top-p sampling parameter.
        top_k: Default top-k sampling parameter.
        enable_graph: Whether to enable graph compiling.
        attn_backend: Attention backend to use ('default', 'flash-attn').
        kv_connector_type: KV connector type for PD separation ('null', etc.).
        kv_connector_role: KV connector role ('none', 'sender', 'receiver', 'both').
        kv_connector_kwargs: Extra keyword arguments for the KV connector.
    """

    model_path: str
    device: str = "cuda"
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    cache_type: str = "paged"  # "paged" or "static"
    max_batch_size: int = 16
    max_tokens: int = 4096
    num_blocks: int = 512
    block_size: int = 256
    max_cache_len: int = 4096
    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 1
    enable_graph: bool = False
    attn_backend: str = "default"
    # ---- PD separation ----
    kv_connector_type: str = "null"
    kv_connector_role: str = "none"
    kv_connector_kwargs: Optional[dict] = field(default=None)

    def __post_init__(self):
        if self.kv_connector_kwargs is None:
            self.kv_connector_kwargs = {}
