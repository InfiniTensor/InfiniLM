from dataclasses import dataclass
from typing import Optional

from infinilm.config.kv_transfer import KVTransferConfig


@dataclass
class EngineConfig:
    """Configuration for LLM Engine.

    Attributes:
        model_path: Path to the model directory.
        draft_model_path: Optional Eagle/MTP draft model directory.
        speculative_method: Speculative decoding method. None disables speculation
            unless draft_model_path is set (which implies "eagle"). "prompt_lookup"
            needs no draft model: draft tokens come from n-gram suffix matching
            against the request's own prompt+output.
        num_draft_tokens: Number of draft tokens to verify per step.
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
        attn_backend: Attention backend to use ('default', 'paged-attn', 'flash-attn', 'hybrid').
        use_mla: Whether to use DeepSeek V2 MLA attention when supported.
        weight_load_mode: Weight loading mode across tensor-parallel workers.
        skip_load: Whether to skip loading model weights (for testing).
        use_legacy_moe: Whether to use the legacy Qwen3 MoE implementation.
    """

    model_path: str
    draft_model_path: Optional[str] = None
    speculative_method: Optional[str] = None  # None / "eagle" / "prompt_lookup"
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

    def __post_init__(self) -> None:
        if self.num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be >= 1")
        # 归一化投机方法：给了 draft_model_path 而未指定方法时默认 eagle，
        # 保持旧调用方行为不变。
        if self.speculative_method is None and self.draft_model_path is not None:
            self.speculative_method = "eagle"
        if self.speculative_method is not None:
            if self.speculative_method not in ("eagle", "prompt_lookup"):
                raise ValueError(
                    "speculative_method must be one of: eagle, prompt_lookup"
                )
            if self.speculative_method == "eagle" and self.draft_model_path is None:
                raise ValueError("speculative_method='eagle' requires draft_model_path")
            if (
                self.speculative_method == "prompt_lookup"
                and self.draft_model_path is not None
            ):
                raise ValueError(
                    "speculative_method='prompt_lookup' takes no draft model; "
                    "leave draft_model_path unset"
                )
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
