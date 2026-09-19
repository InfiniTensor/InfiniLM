from dataclasses import dataclass
from typing import Optional

from infinilm.config.kv_transfer import KVTransferConfig


@dataclass
class EngineConfig:
    """Configuration for LLM Engine.

    Attributes:
        model_path: Path to the model directory.
        draft_model_path: Optional external Eagle draft model directory.
        num_draft_tokens: Number of draft tokens to verify per step.
        enable_mtp: Use the Qwen checkpoint's built-in MTP head.
        num_state_rows: Hybrid state rows, including the reserved zero row.
            Zero selects an automatic capacity.
        mtp_prefix_cache_bytes: Device storage budget for exact-prompt MTP snapshots.
        device: Device type string ('cpu', 'cuda', 'mlu', etc.).
        dtype: Data type string ('float16', 'bfloat16', 'float32').
        tensor_parallel_size: Number of devices for tensor parallelism.
        pipeline_parallel_size: Number of pipeline stages.
        pipeline_parallel_stage: Pipeline stage index for this engine.
        master_addr: Address used to bootstrap distributed communication.
        master_port: TCP port used to bootstrap distributed communication.
        moe_ep_backend: MoE expert-parallel backend.
        moe_ep_size: MoE expert-parallel size.
        cache_type: Cache type ('paged' or 'static').
        max_batch_size: Maximum batch size for inference (only for paged cache).
        max_tokens: Default maximum tokens to generate.
        num_blocks: Number of KV cache blocks (only for paged cache).
        block_size: Size of each KV cache block (only for paged cache).
        max_cache_len: Maximum sequence length (only for static cache).
        enable_prefix_caching: Whether to reuse KV cache across requests.
        temperature: Default sampling temperature.
        top_p: Default top-p sampling parameter.
        top_k: Default top-k sampling parameter.
        enable_graph: Whether to enable graph compiling.
        attn_backend: Attention backend to use ('default', 'flash-attn').
        use_mla: Whether to use DeepSeek V2 MLA attention when supported.
        weight_load_mode: Weight loading mode across tensor-parallel workers.
        skip_load: Whether to skip loading model weights (for testing).
        use_legacy_moe: Whether to use the legacy Qwen3 MoE implementation.
    """

    model_path: str
    draft_model_path: Optional[str] = None
    num_draft_tokens: int = 4
    device: str = "cuda"
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    pipeline_parallel_stage: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
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
    pre_transpose: bool = False
    weight_load_mode: str = "async"
    skip_load: bool = False
    use_legacy_moe: bool = False
    kv_transfer_config: Optional[KVTransferConfig] = None
    enable_prefix_caching: bool = True
    enable_mtp: bool = False
    num_state_rows: int = 0
    mtp_prefix_cache_bytes: int = 0

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("`max_batch_size` must be >= 1.")
        if self.num_state_rows != 0 and self.num_state_rows < 2:
            raise ValueError("`num_state_rows` must be zero (automatic) or >= 2.")
        if self.mtp_prefix_cache_bytes < 0:
            raise ValueError("`mtp_prefix_cache_bytes` must be non-negative.")
        if self.mtp_prefix_cache_bytes and not self.enable_mtp:
            raise ValueError("`mtp_prefix_cache_bytes` requires `enable_mtp`.")
        if self.mtp_prefix_cache_bytes and (
            not self.enable_prefix_caching or self.tensor_parallel_size != 1
        ):
            raise ValueError(
                "MTP prefix snapshots require prefix caching and `tensor_parallel_size=1`."
            )
        if self.enable_mtp:
            if self.draft_model_path is not None:
                raise ValueError(
                    "Built-in MTP cannot be combined with `draft_model_path`."
                )
            if not 1 <= self.num_draft_tokens <= 4:
                raise ValueError("Qwen MTP requires `1 <= num_draft_tokens <= 4`.")
            if self.enable_graph and (
                self.num_draft_tokens != 1 or self.max_batch_size != 1
            ):
                raise ValueError(
                    "Batched or multi-candidate Qwen MTP currently requires eager mode."
                )
            if self.cache_type != "paged" or self.pipeline_parallel_size != 1:
                raise ValueError(
                    "Qwen MTP requires paged caching and `pipeline_parallel_size=1`."
                )
            if self.enable_prefix_caching and not self.mtp_prefix_cache_bytes:
                raise ValueError(
                    "Disable prefix caching or set `mtp_prefix_cache_bytes` "
                    "for exact-prompt MTP snapshots."
                )
            if (
                self.kv_transfer_config is not None
                and self.kv_transfer_config.kv_connector
            ):
                raise ValueError("Qwen MTP does not support remote KV/state transfer.")
            if self.top_k != 1:
                raise ValueError(
                    "Qwen MTP currently supports greedy sampling (`top_k=1`)."
                )
            if self.attn_backend not in ("default", "paged-attn"):
                raise ValueError("Qwen MTP requires the `paged-attn` backend.")
            self.attn_backend = "paged-attn"
            if not self.num_state_rows:
                self.num_state_rows = 1 + self.max_batch_size * (
                    self.num_draft_tokens + 2
                )
        if self.num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be >= 1")
        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be >= 1")
        if not 0 <= self.pipeline_parallel_stage < self.pipeline_parallel_size:
            raise ValueError(
                "pipeline_parallel_stage must be in [0, pipeline_parallel_size)"
            )
        if not 1 <= self.master_port <= 65535:
            raise ValueError("master_port must be in [1, 65535]")

        if self.weight_load_mode not in {"async", "sync"}:
            raise ValueError("weight_load_mode must be either 'async' or 'sync'")

        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.kv_connector
            and self.cache_type != "paged"
        ):
            raise ValueError("kv_transfer_config requires cache_type='paged'")
