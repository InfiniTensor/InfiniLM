import logging
import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import infinicore
from infinilm.cache.cache import PagedKVCacheConfig, StaticKVCacheConfig
from infinilm.config.engine_config import EngineConfig
from infinilm.distributed import DistConfig
from infinilm.distributed.pipeline_transport import PipelineControlServer
from infinilm.infer_engine import InferEngine
from infinilm.kv_connector import (
    KVConnectorFactory,
    KVConnectorRole,
)
from infinilm.llm.model_runner.speculative_runner import SpeculativeRunner
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


@dataclass
class PendingModelOutput:
    """GPU output whose host token has not been retired yet."""

    scheduler_output: Any
    sampled_tokens: Any = None
    relay_tokens: Any = None
    host_tokens: Any = None
    host_ready: Any = None
    ready: threading.Event = field(default_factory=threading.Event)
    exception: BaseException | None = None


class ModelRunner:
    def __init__(self, config: EngineConfig, initialize_processor: bool = True):
        self.config = config
        self._closed = False
        self.kv_transfer_config = config.kv_transfer_config
        logger.info(f"kv_transfer_config: {self.kv_transfer_config}")

        self._init_device()

        self._closed = False
        self._relay_pools = {}
        self._relay_buffer_indices = {}
        self._forward_queue = None
        self._forward_thread = None
        self._async_token_handoff_enabled = False

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

        dist_config_kwargs = {
            "moe_ep_backend": config.moe_ep_backend,
            "moe_ep_size": config.moe_ep_size,
            "pp_size": config.pipeline_parallel_size,
            "pp_stage": config.pipeline_parallel_stage,
            "master_addr": config.master_addr,
            "master_port": config.master_port,
        }
        if self.tp_device_ids is not None:
            distributed_config = DistConfig(
                tp_device_ids=self.tp_device_ids,
                **dist_config_kwargs,
            )
            logger.info("Using explicit TP device ids: %s", self.tp_device_ids)
        else:
            distributed_config = DistConfig(
                config.tensor_parallel_size,
                **dist_config_kwargs,
            )

        # InferEngine creates the per-node TP communicator first. For PP it then
        # uses the short-lived C++ TCP rendezvous to bootstrap one global
        # InfiniCCL communicator spanning every (PP stage, TP rank) pair.
        self.model_engine = InferEngine(
            model_path=config.model_path,
            device=self.device,
            distributed_config=distributed_config,
            cache_config=cache_config,
            enable_graph_compiling=config.enable_graph,
            attention_backend=config.attn_backend,
            use_mla=config.use_mla,
            weight_load_mode=config.weight_load_mode,
            use_legacy_moe=config.use_legacy_moe,
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

        self._configure_async_token_handoff()

    def _async_token_handoff_unsupported_reasons(self) -> list[str]:
        reasons = []
        if self.config.pipeline_parallel_size != 1:
            reasons.append("pipeline parallelism is not supported")
        if self.config.cache_type != "paged":
            reasons.append("paged KV cache is required")
        if self.config.device != "cuda":
            reasons.append("only the CUDA backend is currently validated")
        if not self.config.enable_graph:
            reasons.append("CUDA graph compilation is required")
        if not getattr(self.processor, "supports_async_token_handoff", False):
            reasons.append(
                f"processor {type(self.processor).__name__} does not support GPU decode inputs"
            )
        if self.kv_connector is not None:
            reasons.append("KV transfer connectors are not supported")
        if self.speculative_runner is not None:
            reasons.append("draft-model speculation is not supported")
        if getattr(self.model_engine, "has_mamba_cache", False):
            reasons.append("Mamba state caches are not supported")
        return reasons

    def _configure_async_token_handoff(self) -> None:
        preference = getattr(self.config, "enable_async_token_handoff", None)
        if preference is False:
            logger.info("Async GPU token handoff disabled by configuration")
            return

        reasons = self._async_token_handoff_unsupported_reasons()
        if reasons:
            message = "; ".join(reasons)
            if preference is True:
                self.close()
                raise ValueError(
                    "Async GPU token handoff was explicitly enabled but is unavailable: "
                    + message
                )
            logger.info("Async GPU token handoff auto-disabled: %s", message)
            return

        self._forward_queue = queue.Queue()
        self._forward_thread = threading.Thread(
            target=self._forward_submission_loop,
            daemon=True,
            name="InfiniLMForwardSubmit",
        )
        self._forward_thread.start()
        self._async_token_handoff_enabled = True
        logger.info("Async GPU token handoff enabled")

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

        self.tp_device_ids = self.config.tp_device_ids
        device_index = self.tp_device_ids[0] if self.tp_device_ids else 0

        self.device = infinicore.device(device_str, device_index)

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

    def _build_model_input(self, scheduler_output, decode_input_ids=None):
        return self.processor.build_model_inputs(
            scheduler_output,
            self.config.temperature,
            self.config.top_p,
            self.config.top_k,
            decode_input_ids=decode_input_ids,
        )

    def can_async_token_handoff(self, scheduler_output) -> bool:
        """Return whether this step may use the stable paged-batch relay path."""
        return bool(
            self._async_token_handoff_enabled
            and scheduler_output.num_requests > 0
            and all(
                not req.has_multimodal_inputs
                and not req.sampling_params.stop
                and not req.sampling_params.stop_token_ids
                for req in scheduler_output.scheduled_requests
            )
        )

    def _acquire_relay_buffer(self, num_requests):
        pool = self._relay_pools.get(num_requests)
        if pool is None:
            pool = []
            for _ in range(2):
                relay = infinicore.empty(
                    [num_requests],
                    dtype=infinicore.int64,
                    device=self.device,
                )
                host = infinicore.empty(
                    [num_requests],
                    dtype=infinicore.int64,
                    device=infinicore.device("cpu", 0),
                    pin_memory=True,
                )
                if not host.is_pinned():
                    raise RuntimeError(
                        "async token handoff requires pinned host output memory"
                    )
                pool.append((relay, host, infinicore.DeviceEvent(self.device)))
            self._relay_pools[num_requests] = pool
            self._relay_buffer_indices[num_requests] = 0

        index = self._relay_buffer_indices[num_requests]
        relay, host, host_ready = pool[index]
        self._relay_buffer_indices[num_requests] = (index + 1) % len(pool)
        return relay, host, host_ready

    def _forward_submission_loop(self):
        """Run pre-queued forwards back-to-back with minimal handoff latency."""
        while True:
            job = self._forward_queue.get()
            if job is None:
                self._forward_queue.task_done()
                return

            pending, model_input, predecessor, num_requests = job
            try:
                if predecessor is not None:
                    predecessor.ready.wait()
                    if predecessor.exception is not None:
                        raise RuntimeError(
                            "predecessor async forward failed"
                        ) from predecessor.exception
                    model_input["input_ids"] = predecessor.sampled_tokens.view(
                        [1, num_requests]
                    )

                pending.sampled_tokens = self.model_engine.forward(**model_input)
                # Queue the stable copy before this thread can start a newer
                # forward and overwrite InferEngine.last_output_ids_.
                self.model_engine.copy_last_output_to(pending.relay_tokens)
                pending.host_tokens.copy_async_(pending.relay_tokens)
                pending.host_ready.record()
            except BaseException as exc:
                pending.exception = exc
            finally:
                pending.ready.set()
                self._forward_queue.task_done()

    def launch_async_token_handoff(
        self,
        scheduler_output,
        model_input=None,
        predecessor=None,
    ) -> PendingModelOutput:
        """Queue a forward on the dedicated submission thread."""
        if not self.can_async_token_handoff(scheduler_output):
            raise RuntimeError("async token handoff is not supported for this step")

        if model_input is None:
            model_input = self._build_model_input(scheduler_output)

        relay_tokens, host_tokens, host_ready = self._acquire_relay_buffer(
            scheduler_output.num_requests
        )
        pending = PendingModelOutput(
            scheduler_output=scheduler_output,
            relay_tokens=relay_tokens,
            host_tokens=host_tokens,
            host_ready=host_ready,
        )
        self._forward_queue.put(
            (
                pending,
                model_input,
                predecessor,
                scheduler_output.num_requests,
            )
        )
        return pending

    def prepare_decode_lookahead_input(
        self,
        lookahead_output,
    ):
        """Build the next decode metadata while the current GPU step is running."""
        decode_input_ids = infinicore.from_list(
            [[0] * lookahead_output.num_requests],
            dtype=infinicore.int64,
        )
        return self._build_model_input(
            lookahead_output,
            decode_input_ids=decode_input_ids,
        )

    def finish_async_token_handoff(
        self, pending: PendingModelOutput
    ) -> ModelRunnerOutput:
        """Retire one relay output on the host after its D2D copy is ordered."""
        if pending.relay_tokens is None:
            raise RuntimeError(
                "async token handoff relay must be queued before a newer forward"
            )

        # The task returns only after the sampled tensor exists and its stable
        # relay copy has been queued before any newer forward submission.
        pending.ready.wait()
        if pending.exception is not None:
            raise pending.exception

        # D2H was queued by the submission thread before the next graph. Wait
        # only for this output event; DeviceEvent.synchronize releases the GIL.
        pending.host_ready.synchronize()

        sampled_tokens_list = pending.host_tokens.to_numpy().tolist()
        return ModelRunnerOutput(
            req_ids=[
                req.request_id for req in pending.scheduler_output.scheduled_requests
            ],
            sampled_token_ids=sampled_tokens_list,
            kv_connector_output=None,
        )

    def reset_async_token_handoff_state(self) -> None:
        """Clear engine-owned output events after a speculative step is discarded."""
        self.model_engine.reset_request_state()

    def _model_forward(self, scheduler_output):
        # Build model inputs
        model_input = self._build_model_input(scheduler_output)

        if self.speculative_runner is not None:
            return self._model_forward_with_speculative(scheduler_output, model_input)

        # Wake every stage before stage 0 enters forward. Each worker receives
        # the same metadata and then blocks in its model on the activation from
        # the preceding stage. Stage 0 waits for all acknowledgements afterward.
        if self.pipeline_control is not None:
            self.pipeline_control.dispatch_forward(model_input)
        try:
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
        """Drain the submission thread and release native engine resources."""
        if self._closed:
            return
        self._closed = True
        if self._forward_queue is not None:
            self._forward_queue.join()
            self._forward_queue.put(None)
            self._forward_queue.join()
        if self._forward_thread is not None:
            self._forward_thread.join()
        if getattr(self, "pipeline_control", None) is not None:
            self.pipeline_control.close()
        if getattr(self, "kv_connector", None) is not None:
            self.kv_connector.shutdown()
        if getattr(self, "model_engine", None) is not None:
            self.model_engine.close()
