import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import infinicore
from infinilm.cache.cache import PagedKVCacheConfig, StaticKVCacheConfig
from infinilm.config.engine_config import EngineConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.kv_connector import (
    KVConnectorFactory,
    KVConnectorRole,
)
from infinilm.modeling_utils import load_model_state_dict_by_file
from infinilm.processors import AutoInfinilmProcessor

logger = logging.getLogger(__name__)


@dataclass
class KVConnectorOutput:
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None

    # consumer failed to recv
    failed_recving: set[str] | None = None

    # IDs of externally computed KV blocks that failed to load.
    # Requests referencing these blocks should be rescheduled to recompute them
    invalid_block_ids: set[int] = field(default_factory=set)  # not used
    kv_connector_stats = None  # not used


@dataclass
class ModelRunnerOutput:
    # [num_reqs]
    req_ids: list[str] = field(default_factory=list)
    sampled_token_ids: list[int] = field(default_factory=list)
    kv_connector_output: KVConnectorOutput | None = None


class ModelRunner:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.kv_transfer_config = config.kv_transfer_config
        logger.info(f"kv_transfer_config: {self.kv_transfer_config}")

        self._init_device()

        # Initialize KV cache based on cache type
        if config.cache_type == "static":
            cache_config = StaticKVCacheConfig(
                max_batch_size=1, max_cache_len=config.max_cache_len
            )
            logger.info(
                f"Using Static KV Cache with max_cache_len={config.max_cache_len}"
            )
        elif config.cache_type == "paged":
            cache_config = PagedKVCacheConfig(
                num_blocks=config.num_blocks, block_size=config.block_size
            )
            logger.info(f"Using Paged KV Cache with num_blocks={config.num_blocks}")
        else:
            raise ValueError(f"Unsupported cache_type: {config.cache_type}")

        # Initialize model engine
        self.model_engine = InferEngine(
            model_path=config.model_path,
            device=self.device,
            distributed_config=DistConfig(
                config.tensor_parallel_size,
                moe_ep_backend=config.moe_ep_backend,
                moe_ep_size=config.moe_ep_size,
            ),
            cache_config=cache_config,
            enable_graph_compiling=config.enable_graph,
            attention_backend=config.attn_backend,
            use_mla=config.use_mla,
            weight_load_mode=config.weight_load_mode,
            skip_legacy_moe=config.skip_legacy_moe,
        )

        # Load model weights
        if not self.config.skip_load:
            load_model_state_dict_by_file(
                self.model_engine, config.model_path, dtype=self.model_engine.dtype
            )

        # Initialize processor
        self.processor = AutoInfinilmProcessor.from_pretrained(config.model_path)

        # Initialize KV connector
        self.kv_connector = None
        if self.kv_transfer_config is not None and self.kv_transfer_config.kv_connector:
            connector_name = self.kv_transfer_config.kv_connector
            self.kv_connector = KVConnectorFactory.create_connector(
                connector_name=connector_name,
                role=KVConnectorRole.WORKER,
                kv_transfer_config=self.kv_transfer_config,
            )

            kv_cache_list = self.model_engine.get_kv_cache()
            assert len(kv_cache_list) == self.config.tensor_parallel_size

            kv_caches = {}
            for rank_idx, kv_cache_vec in enumerate(kv_cache_list):
                for layer_idx, layer_kv_cache in enumerate(kv_cache_vec):
                    # print(layer_kv.shape)  # shape：[2, 8, 8, 256, 128]
                    key_name = (
                        f"rank.{rank_idx}.model.layers.{layer_idx}.self_attn.attn"
                    )
                    kv_caches[key_name] = layer_kv_cache

            self.kv_connector.register_kv_caches(kv_caches)

    @property
    def model_type(self):
        return self.model_engine.model_type

    @property
    def eos_token_id(self):
        return self.model_engine.eos_token_id

    def _init_device(self):
        """Initialize infinicore device and dtype."""
        supported_devices = ["cpu", "cuda", "mlu", "musa", "npu"]
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

    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        sampled_tokens_list = []
        kv_connector_output = None

        if self.kv_connector is None:
            sampled_tokens_list = self._model_forward(scheduler_output)
        else:
            with self.maybe_get_kv_connector_output(
                scheduler_output,
            ) as kv_connector_output:
                if scheduler_output.num_requests > 0:
                    sampled_tokens_list = self._model_forward(scheduler_output)

        #  model_runner_output
        req_ids = []
        for i in range(scheduler_output.num_requests):
            req_ids.append(scheduler_output.scheduled_requests[i].request_id)

        return ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=sampled_tokens_list,
            kv_connector_output=kv_connector_output,
        )

    def _model_forward(self, scheduler_output):
        # Build model inputs
        model_input = self.processor.build_model_inputs(
            scheduler_output,
            self.config.temperature,
            self.config.top_p,
            self.config.top_k,
        )

        # Run inference
        sampled_tokens = self.model_engine.forward(**model_input)
        sampled_tokens_list = sampled_tokens.to_numpy().tolist()

        return sampled_tokens_list

    @contextmanager
    def maybe_get_kv_connector_output(
        self, scheduler_output: Any
    ) -> Generator[KVConnectorOutput, None, None]:
        """Context manager for KV connector operations around model forward."""

        output = KVConnectorOutput()
        assert scheduler_output.kv_connector_metadata is not None

        self.kv_connector.bind_connector_metadata(
            scheduler_output.kv_connector_metadata
        )

        self.kv_connector.start_load_kv()

        try:
            yield output
        finally:
            output.finished_sending, output.failed_recving, output.finished_recving = (
                self.kv_connector.get_finished("finished_req_ids")
            )
            output.invalid_block_ids = (
                self.kv_connector.get_block_ids_with_load_errors()
            )
            output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()

    def close(self) -> None:
        """Release resources held by the KV connector."""
        if self.kv_connector is not None:
            self.kv_connector.shutdown()
