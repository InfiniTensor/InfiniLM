import json
from dataclasses import dataclass, field
from pathlib import Path
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
        cache_type: Cache type ('paged' or 'static').
        max_batch_size: Maximum batch size for inference (only for paged cache).
        max_num_batched_tokens: Maximum tokens scheduled in one model step.
        max_num_mixed_prefill_tokens: Maximum prefill tokens beside active decodes.
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
    pipeline_parallel_size: int = 1
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
    max_num_batched_tokens: Optional[int] = None
    max_num_mixed_prefill_tokens: Optional[int] = None
    kernel_block_size: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be positive")
        if self.num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be >= 1")
        if self.weight_load_mode not in {"async", "sync"}:
            raise ValueError("weight_load_mode must be either 'async' or 'sync'")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")

        # The scheduler block can be a multiple of the attention-kernel block.
        # This mirrors vLLM's virtual block splitting for kernels such as DSA.
        self.kernel_block_size = self.block_size

        config_path = Path(self.model_path) / "config.json"
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                model_config = json.load(config_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read model config: {config_path}") from exc
        text_config = model_config.get("text_config", model_config)
        model_type = text_config.get("model_type", model_config.get("model_type"))
        if self.pipeline_parallel_size > 1 and model_type != "glm_moe_dsa":
            raise ValueError(
                "pipeline parallelism is currently supported only for glm_moe_dsa"
            )

        if self.max_num_batched_tokens is None:
            self.max_num_batched_tokens = 4096 if model_type == "glm_moe_dsa" else 1024
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_num_mixed_prefill_tokens is None:
            self.max_num_mixed_prefill_tokens = (
                256 if model_type == "glm_moe_dsa" else self.max_num_batched_tokens
            )
        if self.max_num_mixed_prefill_tokens < 1:
            raise ValueError("max_num_mixed_prefill_tokens must be positive")

        # DeepSeek V2 exposes only the verified MLA path. Keep use_mla as a
        # compatibility input, but derive the effective capability from the model.
        self.use_mla = self.use_mla or model_type in {"deepseek_v2", "glm_moe_dsa"}
        if model_type == "deepseek_v2":
            if self.cache_type != "paged":
                raise ValueError(
                    "DeepSeek V2 MLA requires paged cache; pass --enable-paged-attn"
                )
            self.block_size = 16
            self.kernel_block_size = 16
            self.attn_backend = "flash-attn"
        elif model_type == "glm_moe_dsa":
            if self.cache_type != "paged":
                raise ValueError(
                    "GLM-5.2 DSA requires paged cache; pass --enable-paged-attn"
                )
            self.kernel_block_size = 64
            if self.block_size % self.kernel_block_size != 0:
                raise ValueError(
                    "GLM-5.2 scheduler block_size must be a multiple of 64"
                )

        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.kv_connector
            and self.cache_type != "paged"
        ):
            raise ValueError("kv_transfer_config requires cache_type='paged'")
        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.kv_connector
            and self.cache_block_size_factor != 1
        ):
            raise ValueError(
                "KV transfer does not yet support scheduler/kernel block splitting"
            )

    @property
    def cache_block_size_factor(self) -> int:
        """Number of kernel blocks contained in one scheduler block."""
        if self.cache_type != "paged":
            return 1
        return self.block_size // self.kernel_block_size

    @property
    def num_kernel_blocks(self) -> int:
        """Physical kernel pages allocated by the model worker."""
        if self.cache_type != "paged":
            return 0
        return self.num_blocks * self.cache_block_size_factor

    @property
    def max_cache_tokens(self) -> int:
        """Total token slots shared by all active paged-cache requests."""
        if self.cache_type != "paged":
            return self.max_cache_len
        return self.num_blocks * self.block_size
