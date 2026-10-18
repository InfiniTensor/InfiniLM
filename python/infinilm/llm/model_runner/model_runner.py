import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from ... import _core as infinicore

from infinilm.cache.cache import PagedKVCacheConfig, StaticKVCacheConfig
from infinilm.config.engine_config import EngineConfig
from infinilm.distributed import DistConfig
from infinilm.distributed.pipeline_transport import PipelineControlServer
from infinilm.infer_engine import InferEngine, read_hf_config
from infinilm.kv_connector import (
    KVConnectorFactory,
    KVConnectorRole,
)
from infinilm.llm.model_runner.speculative_runner import SpeculativeRunner
from infinilm.llm.nsa_paged_owner import freeze_model_input
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
    sampled_token_ids: list[int | list[int]] = field(default_factory=list)
    kv_connector_output: KVConnectorOutput | None = None


class ModelRunner:
    def __init__(self, config: EngineConfig, initialize_processor: bool = True):
        self.config = config
        self._closed = False
        self._nsa_scheduler = None
        self.kv_transfer_config = config.kv_transfer_config
        logger.info(f"kv_transfer_config: {self.kv_transfer_config}")

        self._nsa_paged_owner = (config.cache_type == "paged"
                                 and read_hf_config(config.model_path).get("model_type") == "videonsa")
        if self._nsa_paged_owner and (
            config.draft_model_path is not None or config.kv_transfer_config is not None
        ):
            raise ValueError("NSA request ownership does not support speculative decoding or KV transfer")
        self._init_device()

        # Initialize KV cache based on cache type
        if config.cache_type == "static":
            cache_config = StaticKVCacheConfig(
                max_batch_size=config.static_cache_capacity,
                max_cache_len=config.max_cache_len,
            )
            logger.info(
                f"Using Static KV Cache with max_cache_len={config.max_cache_len}"
            )
        elif config.cache_type == "paged":
            cache_config = PagedKVCacheConfig(
                num_blocks=config.num_blocks,
                block_size=config.block_size,
                max_batch_size=config.max_batch_size,
            )
            logger.info(f"Using Paged KV Cache with num_blocks={config.num_blocks}")
        else:
            raise ValueError(f"Unsupported cache_type: {config.cache_type}")

        # InferEngine creates the per-node TP communicator first. For PP it then
        # uses the short-lived C++ TCP rendezvous to bootstrap one global
        # InfiniCCL communicator spanning every (PP stage, TP rank) pair.
        self.model_engine = InferEngine(
            model_path=config.model_path,
            device=self.device,
            distributed_config=DistConfig(
                config.tensor_parallel_size,
                moe_ep_backend=config.moe_ep_backend,
                moe_ep_size=config.moe_ep_size,
                pp_size=config.pipeline_parallel_size,
                pp_stage=config.pipeline_parallel_stage,
                master_addr=config.master_addr,
                master_port=config.master_port,
            ),
            cache_config=cache_config,
            enable_graph_compiling=config.enable_graph,
            attention_backend=config.attn_backend,
            use_mla=config.use_mla,
            weight_load_mode=config.weight_load_mode,
            pre_transpose=config.pre_transpose,
        )

        if self.model_engine.model_type == "minicpm_eagle":
            raise RuntimeError(
                "MiniCPM4 Eagle-vLLM is a speculative draft head, not a standalone "
                "causal LM. Use the MiniCPM4-8B base model as --model and pass "
                "this checkpoint through --draft-model for Eagle speculative decoding."
            )

        # Load model weights
        if not self.config.skip_load:
            load_model_state_dict_by_file(
                self.model_engine, config.model_path, dtype=self.model_engine.dtype
            )

        self.speculative_runner = None
        if config.draft_model_path is not None:
            self.speculative_runner = SpeculativeRunner(
                config, self.model_engine, self.device
            )

        self.processor = (
            AutoInfinilmProcessor.from_pretrained(config.model_path)
            if initialize_processor
            else None
        )

        self.pipeline_control = None
        if config.pipeline_parallel_size > 1 and config.pipeline_parallel_stage == 0:
            # The bootstrap listener has closed by this point. Stage 0 now
            # reuses master_port for the persistent Python control plane; tensor
            # activations continue to use the global InfiniCCL communicator.
            self.pipeline_control = PipelineControlServer(
                config.pipeline_parallel_size,
                config.master_port,
            )

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

    def bind_nsa_scheduler(self, scheduler) -> None:
        """Bind the actual LLM-owned producer for the managed NSA route only."""
        if not self._nsa_paged_owner or self._nsa_scheduler is not None:
            raise RuntimeError("NSA Scheduler binding requires one fresh NSA runner")
        from infinilm.llm.scheduler import Scheduler

        if not isinstance(scheduler, Scheduler) or scheduler._nsa_owner_clock is None:
            raise TypeError("NSA runner requires the actual paged Scheduler")
        manager = scheduler.cache_manager
        if (manager._nsa_owner_clock is not scheduler._nsa_owner_clock
                or scheduler._nsa_owner_clock.pool_epoch != self.model_engine.get_nsa_pool_epoch()
                or manager.num_blocks != self.config.num_blocks
                or manager.block_size != self.config.block_size):
            raise ValueError("NSA Scheduler/engine pool or geometry binding differs")
        self._nsa_scheduler = scheduler

    @property
    def model_type(self):
        return self.model_engine.model_type

    @property
    def eos_token_id(self):
        return self.model_engine.eos_token_id

    def _init_device(self):
        """Initialize infinicore device and dtype."""
        supported_devices = [
            "cpu",
            "cuda",
            "hygon",
            "iluvatar",
            "metax",
            "mlu",
            "musa",
            "npu",
        ]
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
        requests = tuple(scheduler_output.scheduled_requests)
        req_ids = [req.request_id for req in requests]
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

        return ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=sampled_tokens_list,
            kv_connector_output=kv_connector_output,
        )

    def _model_forward(self, scheduler_output):
        nsa_owner = None
        if self._nsa_paged_owner:
            if self._nsa_scheduler is None:
                raise RuntimeError("NSA managed forward requires the actual bound Scheduler")
            nsa_owner = self._nsa_scheduler.take_nsa_owner_batch(scheduler_output)
        # Build model inputs
        requests = scheduler_output.scheduled_requests
        first_params = requests[0].sampling_params if requests else self.config
        temperature, top_p, top_k = (
            first_params.temperature, first_params.top_p, first_params.top_k
        )
        model_input = self.processor.build_model_inputs(
            scheduler_output,
            temperature,
            top_p,
            top_k,
        )

        if self._nsa_paged_owner:
            freeze_model_input(self.model_engine, nsa_owner, model_input, infinicore)

        if self.speculative_runner is not None and all(
            request.sampling_params.top_k == 1
            or request.sampling_params.top_p == 0.0
            or request.sampling_params.temperature == 0.0
            for request in requests
        ):
            return self._model_forward_with_speculative(scheduler_output, model_input)

        model_input["sampling_rows"] = [
            (request.sampling_params.temperature,
             request.sampling_params.top_p,
             request.sampling_params.top_k)
            for request in requests
        ]
        model_input["random_values"] = [
            request.next_sampling_random_value()
            for request in requests
        ]

        # Wake every stage before stage 0 enters forward. Each worker receives
        # the same metadata and then blocks in its model on the activation from
        # the preceding stage. Stage 0 waits for all acknowledgements afterward.
        try:
            if self.pipeline_control is not None:
                self.pipeline_control.dispatch_forward(model_input)
            sampled_tokens = self.model_engine.forward(**model_input)
        except BaseException:
            if self.pipeline_control is not None:
                # A downstream stage may already be blocked waiting for an
                # activation that this stage failed to produce. Waiting for its
                # acknowledgement here would deadlock the coordinator.
                self.pipeline_control.abort()
            raise
        if self.pipeline_control is not None:
            self.pipeline_control.wait_forward()
        sampled_tokens_list = sampled_tokens.to_numpy().tolist()

        return sampled_tokens_list

    def _model_forward_with_speculative(self, scheduler_output, model_input):
        return self.speculative_runner.forward(scheduler_output, model_input)

    @contextmanager
    def maybe_get_kv_connector_output(
        self, scheduler_output: Any
    ) -> Generator[KVConnectorOutput, None, None]:
        """Context manager for KV connector operations around model forward."""

        output = KVConnectorOutput()
        if scheduler_output.kv_connector_metadata is None:
            raise ValueError("KV connector step requires scheduler metadata")

        connector = self.kv_connector
        load_started = False
        first_error = None
        try:
            connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)
            connector.start_load_kv()
            load_started = True
            yield output
        except BaseException as error:
            first_error = error
            raise
        finally:
            try:
                if load_started:
                    (output.finished_sending, output.failed_recving,
                     output.finished_recving) = connector.get_finished()
                    output.invalid_block_ids = connector.get_block_ids_with_load_errors()
                    output.kv_connector_stats = connector.get_kv_connector_stats()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                    raise
                logger.exception("KV completion polling failed after an earlier step error")
            finally:
                try:
                    connector.clear_connector_metadata()
                except BaseException:
                    if first_error is None:
                        raise
                    logger.exception("KV metadata cleanup failed after an earlier step error")

    def close(self) -> None:
        """Stop external resources and deterministically release native engines."""
        if self._closed:
            return
        self._closed = True

        first_error: BaseException | None = None
        if self.pipeline_control is not None:
            try:
                self.pipeline_control.close()
            except BaseException as error:
                first_error = error
        if self.kv_connector is not None:
            try:
                self.kv_connector.shutdown()
            except BaseException as error:
                if first_error is None:
                    first_error = error

        # SpeculativeRunner retains the target engine as well as its draft
        # engine, so release it first.  Do not defer native teardown to Python
        # interpreter shutdown, where module and CUDA destruction order is no
        # longer deterministic.
        self.speculative_runner = None
        self.model_engine = None
        self._nsa_scheduler = None

        if first_error is not None:
            raise first_error
