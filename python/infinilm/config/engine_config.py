from dataclasses import dataclass
from typing import Optional

from infinilm.config.kv_transfer import KVTransferConfig


@dataclass
class EngineConfig:
    """Configuration for LLM Engine.

    Attributes:
        model_path: Path to the model directory.
        draft_model_path: Optional Eagle/MTP draft model directory.
        num_draft_tokens: Number of Eagle draft tokens to verify per step.
        device: Device type string ('cpu', 'cuda', 'mlu', etc.).
        dtype: Data type string ('float16', 'bfloat16', 'float32').
        tensor_parallel_size: Number of devices for tensor parallelism.
        pipeline_parallel_size: Number of pipeline stages.
        moe_ep_backend: MoE expert-parallel backend.
        moe_ep_size: MoE expert-parallel size.
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
        use_mla: Whether to use DeepSeek V2 MLA attention when supported.
        weight_load_mode: Weight loading mode across tensor-parallel workers.
        skip_load: Whether to skip loading model weights (for testing).
        skip_legacy_moe: Whether to use the new fused MoE implementation for Qwen3 MoE.
    """

    model_path: str
    draft_model_path: Optional[str] = None
    num_draft_tokens: int = 4
    device: str = "cuda"
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    moe_ep_backend: str = "disabled"
    moe_ep_size: int = 1
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
    use_mla: bool = False
    weight_load_mode: str = "async"
    skip_load: bool = False
    skip_legacy_moe: bool = False
    kv_transfer_config: Optional[KVTransferConfig] = None
    pipeline_parallel_size: int = 1

    def __post_init__(self) -> None:
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")

        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be >= 1")

        if self.num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be >= 1")

        if self.weight_load_mode not in {"async", "sync"}:
            raise ValueError("weight_load_mode must be either 'async' or 'sync'")

        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.kv_connector
            and self.cache_type != "paged"
        ):
            raise ValueError("kv_transfer_config requires cache_type='paged'")

        if self.pipeline_parallel_size > 1:
            if self.tensor_parallel_size != 1:
                raise ValueError(
                    "Pipeline parallel MVP requires tensor_parallel_size == 1; "
                    f"got tp={self.tensor_parallel_size}, "
                    f"pp={self.pipeline_parallel_size}"
                )
            if self.enable_graph:
                raise ValueError(
                    "Pipeline parallel MVP does not support graph compiling"
                )
            if self.draft_model_path is not None:
                raise ValueError(
                    "Pipeline parallel MVP does not support speculative decoding"
                )
            if self.kv_transfer_config is not None:
                raise ValueError(
                    "Pipeline parallel MVP does not support KV transfer connectors"
                )
            if self.moe_ep_backend != "disabled" or self.moe_ep_size != 1:
                raise ValueError(
                    "Pipeline parallel MVP does not support expert parallelism"
                )
            if self.cache_type != "paged":
                raise ValueError(
                    "Pipeline parallel MVP currently requires cache_type='paged'"
                )
            if self.use_mla:
                raise ValueError(
                    "Pipeline parallel MVP does not support MLA"
                )
