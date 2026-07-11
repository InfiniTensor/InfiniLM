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
        moe_ep_backend: MoE expert-parallel backend.
        moe_ep_size: MoE expert-parallel size.
        allreduce_backend: Tensor-parallel AllReduce backend selector.
        cache_type: Cache type ('paged' or 'static').
        max_batch_size: Maximum batch size for inference (only for paged cache).
        max_tokens: Default maximum tokens to generate.
        num_blocks: Number of KV cache blocks (only for paged cache).
        block_size: Size of each KV cache block (only for paged cache).
        max_num_batched_tokens: Token budget per paged-cache scheduler step.
        enable_chunked_prefill: Whether to split long prefills across scheduler steps.
        prefill_chunk_size: Maximum tokens per chunked prefill step.
        decode_priority: Whether decode requests are scheduled before prefill chunks.
        max_num_partial_prefills: Maximum partial prefill chunks in one scheduler step.
        max_long_partial_prefills: Maximum long partial prefill chunks in one scheduler step.
        long_prefill_token_threshold: Prompt length threshold for long partial prefills.
        min_prefill_chunk_size: Minimum non-final prefill chunk when decode is batched.
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
    allreduce_backend: str = "nccl"
    cache_type: str = "paged"  # "paged" or "static"
    max_batch_size: int = 16
    max_tokens: int = 4096
    num_blocks: int = 512
    block_size: int = 256
    max_num_batched_tokens: Optional[int] = None
    enable_chunked_prefill: bool = False
    prefill_chunk_size: Optional[int] = None
    decode_priority: Optional[bool] = None
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    long_prefill_token_threshold: Optional[int] = None
    min_prefill_chunk_size: Optional[int] = None
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

    def __post_init__(self) -> None:
        if self.num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be >= 1")

        if self.weight_load_mode not in {"async", "sync"}:
            raise ValueError("weight_load_mode must be either 'async' or 'sync'")
        self.allreduce_backend = self.allreduce_backend.lower().replace("-", "_")
        if self.allreduce_backend not in {"nccl", "auto", "custom"}:
            raise ValueError(
                "allreduce_backend must be one of 'nccl', 'auto', or 'custom'"
            )
        if self.max_num_partial_prefills < 1:
            raise ValueError("max_num_partial_prefills must be >= 1")
        if self.max_long_partial_prefills < 1:
            raise ValueError("max_long_partial_prefills must be >= 1")
        if self.max_long_partial_prefills > self.max_num_partial_prefills:
            raise ValueError(
                "max_long_partial_prefills must be <= max_num_partial_prefills"
            )
        if (
            self.long_prefill_token_threshold is not None
            and self.long_prefill_token_threshold < 0
        ):
            raise ValueError("long_prefill_token_threshold must be >= 0")
        if self.min_prefill_chunk_size is not None and self.min_prefill_chunk_size < 1:
            raise ValueError("min_prefill_chunk_size must be >= 1")

        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.kv_connector
            and self.cache_type != "paged"
        ):
            raise ValueError("kv_transfer_config requires cache_type='paged'")
