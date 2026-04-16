"""
Worker - Orchestrates model execution for inference.

Aligned with vLLM v1, the Worker:
1. Creates and owns the ModelRunner
2. Manages staged initialisation (device → model → cache)
3. Delegates model execution to ModelRunner

Architecture:

    LLMEngine
    └── Worker
        └── ModelRunner
            ├── InferEngine (model_engine)
            └── KVConnector

In a PD-separated deployment, different Worker subclasses can be used
for the prefill node and the decode node.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import infinicore

from infinilm.llm.engine_config import EngineConfig
from infinilm.llm.worker.model_runner import ModelRunner
from infinilm.llm.kv_connector import KVConnectorRole
from infinilm.cache.cache import PagedKVCacheConfig, StaticKVCacheConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class WorkerBase(ABC):
    """Abstract base class for inference workers.

    Workers are responsible for device management, model lifecycle, and
    executing model inference given a scheduler output.
    """

    @abstractmethod
    def init_device(self) -> None:
        """Initialize the compute device. (Stage 1)"""
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and initialise KV connector. (Stage 2)"""
        raise NotImplementedError

    @abstractmethod
    def execute_model(self, scheduler_output: Any) -> Optional[List[int]]:
        """Execute model inference for the given scheduler output.

        Args:
            scheduler_output: Output from the scheduler containing the
                batch of requests and their metadata.

        Returns:
            List of sampled token IDs, or ``None`` if no work was done.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release resources (KV connector, etc.)."""
        pass


# ---------------------------------------------------------------------------
# Default (standalone) worker
# ---------------------------------------------------------------------------


class Worker(WorkerBase):
    """Default worker for single-device / standalone inference.

    Aligned with vLLM v1's ``GPUWorker``: the Worker creates the
    ``ModelRunner`` in ``__init__`` and provides staged initialisation
    methods for device, model, and cache setup.

    Args:
        config: The ``EngineConfig`` for this worker.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.device: Any = None
        self.dtype: Any = None

        # Create ModelRunner (stores config only; model not loaded yet)
        # Aligned with vLLM v1: Worker.__init__ creates ModelRunner
        self.model_runner = ModelRunner(config)

    # ------------------------------------------------------------------
    # Staged initialisation (aligned with vLLM v1)
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        """Initialize infinicore device and dtype."""
        supported_devices = ["cpu", "cuda", "mlu", "musa"]
        device_str = self.config.device
        if device_str not in supported_devices:
            raise ValueError(
                f"Unsupported device: '{device_str}'. "
                f"Supported devices: {supported_devices}"
            )
        self.device = infinicore.device(device_str, 0)

        dtype_map = {
            "float32": infinicore.float32,
            "float16": infinicore.float16,
            "bfloat16": infinicore.bfloat16,
        }

        if self.config.dtype not in dtype_map:
            raise ValueError(
                f"Unsupported dtype: '{self.config.dtype}'. "
                f"Supported dtypes: {list(dtype_map.keys())}"
            )
        self.dtype = dtype_map[self.config.dtype]

        logger.info(
            f"Worker: device initialized — {device_str}, dtype={self.config.dtype}"
        )

    def load_model(self) -> None:
        """Load the model via ModelRunner. (Stage 2)

        Corresponds to ``GPUWorker.load_model()`` in vLLM v1.
        """
        if self.device is None:
            raise RuntimeError(
                "Worker.init_device() must be called before load_model()"
            )
        self.model_runner.load_model(self.device)
        logger.info("Worker: model loaded")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_model(self, scheduler_output: Any) -> Optional[List[int]]:
        """Execute model inference using the model runner.

        Args:
            scheduler_output: Output from the scheduler.

        Returns:
            List of sampled token IDs.
        """
        return self.model_runner.execute_model(scheduler_output)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def model_config(self) -> Any:
        """Return the model configuration from ModelRunner → InferEngine."""
        return self.model_runner.model_config

    def get_kv_connector_handshake_metadata(self):
        """Get KV connector metadata from this worker if available."""

        # TODO: Mooncake: 握手数据
        return self.model_runner.kv_connector.get_kv_connector_handshake_metadata()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources held by the KV connector."""
        # TODO: Mooncake: shutdown mooncake
        if self.model_runner is not None and self.model_runner.kv_connector is not None:
            self.model_runner.kv_connector.close()
            logger.info("Worker: KV connector closed")


def create_worker(config: EngineConfig) -> Worker:
    """创建 Worker。
    不再根据 role 选择不同子类——
    PD 差异由 KVConnector role 处理。
    """
    worker = Worker(config)
    logger.info(f"Created Worker (kv_connector_role={config.kv_connector_role})")
    return worker
