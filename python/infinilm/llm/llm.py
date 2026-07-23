"""
LLM Engine - Main interface for LLM inference.

This module provides:
- LLM class for batch generation (offline use)
- AsyncLLM class for asynchronous streaming (server use)
"""

import asyncio
import json
import os
import queue
import time
import uuid
import logging
import threading
from typing import List, Optional, Union, AsyncIterator
from dataclasses import dataclass

import infinicore

# Side-effect: patches infinicore.Tensor.to_numpy (required by LLMEngine.step).
from infinilm.generation import utils as _generation_utils  # noqa: F401

from infinilm.llm.request import (
    InferenceRequest,
    RequestOutput,
    TokenOutput,
    FinishReason,
)
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.context_limit import (
    cap_max_tokens,
    effective_max_model_len,
    truncate_prompt_token_ids,
    validate_prompt_length,
)
from infinilm.llm.scheduler import Scheduler, SchedulerOutput
from infinilm.llm.static_scheduler import StaticScheduler
from infinilm.processors import AutoInfinilmProcessor
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.cache.cache import PagedKVCacheConfig, StaticKVCacheConfig
from infinilm.modeling_utils import load_model_state_dict_by_file
from infinilm.multimodal.multimodal import resolve_multimodal_inputs

logger = logging.getLogger(__name__)



def _scheduler_queue_stats(engine) -> dict:
    sched = engine.scheduler
    cache = sched.get_cache_stats()
    return {
        "waiting": sched.waiting_queue.sync_q.qsize(),
        "running": sched.running_queue.sync_q.qsize(),
        "chunking": sched.chunking_queue.sync_q.qsize(),
        "free_blocks": cache.get("num_free_blocks"),
        "used_blocks": cache.get("num_used_blocks"),
    }


def _step_profile_enabled() -> bool:
    return os.environ.get("INFINI_STEP_PROFILE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _hang_trace_enabled() -> bool:
    return os.environ.get("INFINI_HANG_TRACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


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
        skip_load: Whether to skip loading model weights (for testing).
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
    chunk_size: int = 0
    attn_backend: str = "default"
    skip_load: bool = False


class LLMEngine:
    """Low-level LLM engine that handles inference execution."""

    def __init__(self, config: EngineConfig):
        self.config = config

        # Initialize device and dtype
        self._init_device()

        # Initialize model engine
        self.model_engine = InferEngine(
            model_path=config.model_path,
            device=self.device,
            distributed_config=DistConfig(config.tensor_parallel_size),
            enable_graph_compiling=config.enable_graph,
            attention_backend=config.attn_backend,
        )

        # Load model weights
        if not self.config.skip_load:
            load_model_state_dict_by_file(
                self.model_engine, config.model_path, dtype=self.model_engine.dtype
            )

        # Initialize processor/tokenizer
        self.processor = AutoInfinilmProcessor.from_pretrained(config.model_path)
        self.tokenizer = self.processor.get_tokenizer()

        self.max_model_len = effective_max_model_len(self.model_engine.hf_config)
        logger.info("Using max_model_len=%s", self.max_model_len)

        # Initialize KV cache based on cache type
        if config.cache_type == "static":
            static_cache_len = min(config.max_cache_len, self.max_model_len)
            cache_config = StaticKVCacheConfig(
                max_batch_size=1, max_cache_len=static_cache_len
            )
            self.scheduler = StaticScheduler(max_cache_len=static_cache_len)
            logger.info(
                f"Using Static KV Cache with max_cache_len={static_cache_len}"
            )
        elif config.cache_type == "paged":
            cache_config = PagedKVCacheConfig(
                num_blocks=config.num_blocks,
                block_size=config.block_size,
                max_batch_size=config.max_batch_size,
            )
            disable_prefix_cache = os.environ.get(
                "INFINI_PREFILL_DISABLE_PREFIX_CACHE", "0"
            ) == "1"
            self.scheduler = Scheduler(
                max_batch_size=config.max_batch_size,
                num_blocks=config.num_blocks,
                block_size=config.block_size,
                max_prefill_batch_size=config.max_batch_size,
                enable_prefix_cache=not disable_prefix_cache,
                max_model_len=self.max_model_len,
            )
            logger.info(f"Using Paged KV Cache with num_blocks={config.num_blocks}")
            if disable_prefix_cache:
                logger.info(
                    "Prefix cache disabled (INFINI_PREFILL_DISABLE_PREFIX_CACHE=1)"
                )
        else:
            raise ValueError(f"Unsupported cache_type: {config.cache_type}")

        self.model_engine.reset_cache(cache_config)
        self.cache_type = config.cache_type

        try:
            from infinilm.compile.env import prefill_chunked_enabled, prefill_chunk_size

            if config.chunk_size == 0 and prefill_chunked_enabled():
                config.chunk_size = prefill_chunk_size()
                logger.info(
                    "chunked prefill enabled chunk_size=%s (INFINI_PREFILL_CHUNKED=1)",
                    config.chunk_size,
                )
        except ImportError:
            pass

        if config.enable_graph:
            try:
                from infinilm.compile.env import prefill_native_cg_enabled

                if prefill_native_cg_enabled():
                    logger.info(
                        "native piecewise CG enabled (cudagraph_policy / legacy "
                        "INFINI_PREFILL_NATIVE_CG); prefill graphs captured in C++ at init"
                    )
            except ImportError:
                pass

        # Get EOS token IDs from model config
        self.eos_token_ids = self.model_engine.eos_token_id or []
        if isinstance(self.eos_token_ids, int):
            self.eos_token_ids = [self.eos_token_ids]

        logger.info(
            f"LLMEngine initialized with model at {config.model_path} "
            f"on device {config.device}"
            f"enable_graph={config.enable_graph}"
        )

    def _init_device(self):
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

    def _admit_request(self, request: InferenceRequest) -> None:
        """Validate prompt length and cap output budget (vLLM-style admission)."""
        truncate_prompt_tokens = None
        if request.request_data:
            truncate_prompt_tokens = request.request_data.get(
                "truncate_prompt_tokens"
            )

        prompt_token_ids = truncate_prompt_token_ids(
            request.prompt_token_ids,
            truncate_prompt_tokens=truncate_prompt_tokens,
            max_model_len=self.max_model_len,
        )
        if prompt_token_ids is not request.prompt_token_ids:
            request.prompt_token_ids = prompt_token_ids
            request.prompt_length = len(prompt_token_ids)
            if request.prompt:
                request.prompt = self.detokenize(prompt_token_ids)

        validate_prompt_length(request.prompt_length, self.max_model_len)

        capped = cap_max_tokens(
            request.sampling_params.max_tokens,
            request.prompt_length,
            self.max_model_len,
            default_max_tokens=self.config.max_tokens,
        )
        if capped != request.sampling_params.max_tokens:
            request.sampling_params = request.sampling_params.clone()
            request.sampling_params.max_tokens = capped

    def add_request(self, request: InferenceRequest):
        """Add a request to the scheduler."""
        if request.prompt_token_ids:
            self._admit_request(request)
        if self.cache_type == "paged" and self.config.chunk_size > 0:
            request.chunk_size = self.config.chunk_size
        self.scheduler.add_request(request)

    def step(self) -> tuple[list[InferenceRequest], list[tuple]]:
        """Run one inference step.

        Returns:
            A tuple of:
            - scheduled_requests: Requests that were scheduled and processed in this step.
            - pending: Pending streaming outputs as (async_queue, TokenOutput) pairs.
        """
        profile = _step_profile_enabled()
        hang_trace = _hang_trace_enabled()
        t_step0 = time.perf_counter() if profile else 0.0
        # Schedule requests
        scheduler_output = self.scheduler.schedule()
        if scheduler_output is None or not scheduler_output.scheduled_requests:
            return [], []
        if profile:
            rows = getattr(scheduler_output, "rows", None) or []
            logger.info(
                "step_profile: scheduled n_rows=%d total_tokens=%d mode=%s debt_ms=%.3f",
                len(rows),
                getattr(scheduler_output, "total_scheduled_tokens", 0),
                getattr(scheduler_output, "scheduling_mode", "static"),
                (time.perf_counter() - t_step0) * 1000.0,
            )

        # Build model inputs
        model_input = self.processor.build_model_inputs(
            scheduler_output,
            self.config.temperature,
            self.config.top_p,
            self.config.top_k,
        )

        cpp_model_input = {
            k: v for k, v in model_input.items() if k != "input_ids_torch"
        }

        t_fwd0 = time.perf_counter() if (profile or hang_trace) else 0.0
        if profile:
            logger.info("step_profile: forward begin")
        if hang_trace:
            logger.info(
                "hang_trace: forward_begin mode=%s n_req=%d",
                scheduler_output.scheduling_mode
                if hasattr(scheduler_output, "scheduling_mode")
                else ("prefill" if scheduler_output.is_prefill else "decode"),
                len(scheduler_output.scheduled_requests),
            )
        sampled_tokens = self.model_engine.forward(**cpp_model_input)
        if profile:
            logger.info(
                "step_profile: forward end ms=%.3f",
                (time.perf_counter() - t_fwd0) * 1000.0,
            )
        if hang_trace:
            logger.info(
                "hang_trace: forward_end ms=%.3f",
                (time.perf_counter() - t_fwd0) * 1000.0,
            )
        sampled_tokens_list = sampled_tokens.to_numpy().tolist()

        t_up0 = time.perf_counter() if profile else 0.0
        pending = self._update_requests(scheduler_output, sampled_tokens_list)
        if profile:
            logger.info(
                "step_profile: update_requests end pending=%d ms=%.3f total_step_ms=%.3f",
                len(pending),
                (time.perf_counter() - t_up0) * 1000.0,
                (time.perf_counter() - t_step0) * 1000.0,
            )

        return scheduler_output.scheduled_requests, pending

    def _update_requests(
        self,
        scheduler_output: SchedulerOutput,
        sampled_tokens: List[int],
    ) -> List[tuple]:
        """Update request status after inference step."""
        rows = getattr(scheduler_output, "rows", None)
        if rows:
            return self._update_requests_from_rows(scheduler_output, sampled_tokens)
        return self._update_requests_legacy_phase(
            scheduler_output.is_prefill,
            scheduler_output.scheduled_requests,
            sampled_tokens,
        )

    def _update_requests_from_rows(
        self,
        scheduler_output: SchedulerOutput,
        sampled_tokens: List[int],
    ) -> List[tuple]:
        rows = scheduler_output.rows
        token_iter = iter(sampled_tokens)
        pending: List[tuple] = []

        all_mid_chunk = bool(rows) and all(
            r.is_prefill_row and not r.is_final_prefill_chunk for r in rows
        )
        if all_mid_chunk:
            for row in rows:
                req = row.request
                req.chunk_prefill_offset += row.num_scheduled_tokens
                if req.is_aborted():
                    logger.info(
                        f"Request {req.request_id} aborted by client during chunked-prefill"
                    )
                    continue
            self.scheduler.complete_requests(scheduler_output.scheduled_requests)
            return []

        has_final_prefill_complete = any(
            r.is_prefill_row and r.is_final_prefill_chunk for r in rows
        )
        if has_final_prefill_complete:
            match self.cache_type:
                case "paged":
                    self.scheduler.cache_manager.reset_req_blocks()
                case "static":
                    self.scheduler.update_cache()
                case _:
                    raise ValueError(f"Unsupported cache_type: {self.cache_type}")

        rows_needing_sample = [
            row
            for row in rows
            if not (row.is_prefill_row and not row.is_final_prefill_chunk)
        ]
        expected_tokens = len(rows_needing_sample)
        if len(sampled_tokens) != expected_tokens:
            req_ids = [row.request.request_id for row in rows_needing_sample]
            raise RuntimeError(
                f"sampled token count mismatch: got {len(sampled_tokens)} "
                f"expected {expected_tokens} for request_ids={req_ids}"
            )

        for row in rows:
            req = row.request
            if row.is_prefill_row and not row.is_final_prefill_chunk:
                req.chunk_prefill_offset += row.num_scheduled_tokens
                if req.is_aborted():
                    logger.info(
                        f"Request {req.request_id} aborted by client during chunked-prefill"
                    )
                    continue
                continue

            if req.is_aborted():
                next(token_iter, None)
                logger.info(
                    f"Request {req.request_id} aborted by client, skipping update"
                )
                continue

            try:
                token_id = next(token_iter)
            except StopIteration:
                raise RuntimeError(
                    f"Missing sampled token for request_id={req.request_id} "
                    f"prefill={row.is_prefill_row} final={row.is_final_prefill_chunk}"
                ) from None

            if row.is_prefill_row:
                req.chunk_prefill_offset = 0
                req.chunk_size = 0
                req.is_prefill = False

            req.generated_token_ids.append(token_id)
            holds_back = self._update_generated_text_from_tokens(req)

            is_finished = self._check_request_finished(req, token_id)

            if req._output_queue is None:
                if is_finished:
                    req.mark_finished(req.finish_reason)
            else:
                if holds_back and not is_finished:
                    token_text = ""
                else:
                    if is_finished and req.finish_reason in (
                        FinishReason.EOS_TOKEN,
                        FinishReason.LENGTH,
                        FinishReason.STOP_STRING,
                    ):
                        token_text = ""
                    else:
                        token_text = req.generated_text[
                            req._stream_last_yielded_length :
                        ]
                        if token_text:
                            req._stream_last_yielded_length = len(req.generated_text)

                    if is_finished:
                        req.mark_finished(req.finish_reason)

                output = TokenOutput(
                    request_id=req.request_id,
                    token_id=token_id,
                    token_text=token_text,
                    finished=is_finished,
                    finish_reason=req.finish_reason if is_finished else None,
                    generated_text=req.generated_text,
                )
                if req.is_aborted():
                    logger.info(
                        f"Request {req.request_id} aborted before putting token"
                    )
                    continue
                pending.append((req.output_queue.async_q, output))

        self.scheduler.complete_requests(scheduler_output.scheduled_requests)
        return pending

    def _update_requests_legacy_phase(
        self,
        is_prefill: bool,
        requests: List[InferenceRequest],
        sampled_tokens: List[int],
    ) -> List[tuple]:
        """Legacy global-phase request update (INFINI_V1_SCHEDULER=0)."""
        chunk_mid_step = (
            is_prefill
            and len(requests) > 0
            and all(r.is_chunking() and not r.chunk_is_last() for r in requests)
        )

        has_final_prefill_complete = any(
            r.is_prefill and (not r.is_chunking() or r.chunk_is_last()) for r in requests
        )

        if is_prefill and not chunk_mid_step and has_final_prefill_complete:
            match self.cache_type:
                case "paged":
                    self.scheduler.cache_manager.reset_req_blocks()
                case "static":
                    self.scheduler.update_cache()
                case _:
                    raise ValueError(f"Unsupported cache_type: {self.cache_type}")

        if chunk_mid_step:
            for req in requests:
                chunk_len = min(
                    req.chunk_size, req.prompt_length - req.chunk_prefill_offset
                )
                req.chunk_prefill_offset += chunk_len
                if req.is_aborted():
                    logger.info(
                        f"Request {req.request_id} aborted by client during chunked-prefill"
                    )
                    continue
                self.scheduler.requeue_chunking(req)
            return []

        prefill_final_indices = (
            [
                i
                for i, r in enumerate(requests)
                if r.is_prefill and (not r.is_chunking() or r.chunk_is_last())
            ]
            if is_prefill
            else list(range(len(requests)))
        )

        if is_prefill and prefill_final_indices:
            pending: List[tuple] = []
            if len(sampled_tokens) != len(prefill_final_indices):
                req_ids = [requests[i].request_id for i in prefill_final_indices]
                raise RuntimeError(
                    f"prefill token/sample mismatch: n_tokens={len(sampled_tokens)} "
                    f"n_final={len(prefill_final_indices)} request_ids={req_ids}"
                )
            token_by_idx = dict(zip(prefill_final_indices, sampled_tokens))
            prefill_final_set = set(prefill_final_indices)
            requests_for_complete: List[InferenceRequest] = []
            for i, req in enumerate(requests):
                if i in prefill_final_set:
                    if i not in token_by_idx:
                        if req.is_chunking():
                            self.scheduler.requeue_chunking(req)
                        else:
                            self.scheduler.waiting_queue.sync_q.put(req)
                        continue
                    token_id = token_by_idx[i]
                    pending.extend(
                        self._apply_sampled_token(req, token_id)
                    )
                    requests_for_complete.append(req)
                elif req.is_prefill:
                    if req.is_chunking():
                        self.scheduler.requeue_chunking(req)
                    else:
                        self.scheduler.waiting_queue.sync_q.put(req)
                else:
                    requests_for_complete.append(req)
            self.scheduler.complete_requests(requests_for_complete)
            return pending

        pending = []
        if len(sampled_tokens) != len(requests):
            req_ids = [req.request_id for req in requests]
            raise RuntimeError(
                f"decode token/sample mismatch: n_tokens={len(sampled_tokens)} "
                f"n_req={len(requests)} request_ids={req_ids}"
            )
        for req, token_id in zip(requests, sampled_tokens):
            pending.extend(self._apply_sampled_token(req, token_id))

        self.scheduler.complete_requests(requests)
        return pending

    def _update_generated_text_from_tokens(self, req: InferenceRequest) -> bool:
        """Decode all generated tokens; return True if held back (incomplete UTF-8)."""
        full_text = self.tokenizer.decode(req.generated_token_ids)
        holds_back = bool(full_text) and full_text.endswith("\ufffd")
        if not holds_back:
            req.generated_text = full_text
        return holds_back

    def _apply_sampled_token(
        self, req: InferenceRequest, token_id: int
    ) -> List[tuple]:
        """Apply one sampled token to a request; return pending stream outputs."""
        pending: List[tuple] = []
        if req.is_aborted():
            logger.info(
                f"Request {req.request_id} aborted by client, skipping update"
            )
            return pending

        if req.is_prefill:
            req.chunk_prefill_offset = 0
            req.chunk_size = 0
            req.is_prefill = False

        req.generated_token_ids.append(token_id)
        holds_back = self._update_generated_text_from_tokens(req)

        is_finished = self._check_request_finished(req, token_id)

        # vLLM-style replacement character handling is primarily relevant for streaming.
        # For offline generation (no output queue), keep the fast incremental path.
        if req._output_queue is None:
            if is_finished:
                req.mark_finished(req.finish_reason)

        else:
            if holds_back and not is_finished:
                token_text = ""
            else:
                if is_finished and req.finish_reason in (
                    FinishReason.EOS_TOKEN,
                    FinishReason.LENGTH,
                    FinishReason.STOP_STRING,
                ):
                    token_text = ""
                else:
                    token_text = req.generated_text[
                        req._stream_last_yielded_length :
                    ]
                    if token_text:
                        req._stream_last_yielded_length = len(req.generated_text)

                if is_finished:
                    req.mark_finished(req.finish_reason)

            output = TokenOutput(
                request_id=req.request_id,
                token_id=token_id,
                token_text=token_text,
                finished=is_finished,
                finish_reason=req.finish_reason if is_finished else None,
                generated_text=req.generated_text,
            )
            if req.is_aborted():
                logger.info(
                    f"Request {req.request_id} aborted before putting token"
                )
                return pending
            pending.append((req.output_queue.async_q, output))
        return pending

    def _check_request_finished(self, req: InferenceRequest, token_id: int) -> bool:
        """Check if request generation is finished."""
        if req.get_total_length() >= self.max_model_len:
            req.finish_reason = FinishReason.LENGTH
            return True

        max_tokens = req.sampling_params.max_tokens
        if max_tokens and req.get_num_generated_tokens() >= max_tokens:
            req.finish_reason = FinishReason.LENGTH
            return True

        if not req.sampling_params.ignore_eos:
            # Check EOS token - only stop if ignore_eos is False
            eos_ids = req.eos_token_ids or self.eos_token_ids
            if eos_ids and token_id in eos_ids:
                req.finish_reason = FinishReason.EOS_TOKEN
                return True

            # While ignoring EOS, stop strings are also ignored to avoid requiring additional arguments for benchmarking.
            # Check stop strings
            # Remove stop string from generated_text if STOP_STRING is the finishing reason
            stop_strings = req.sampling_params.stop or []
            for stop_str in stop_strings:
                if req.generated_text.endswith(stop_str):
                    req.generated_text = req.generated_text[: -len(stop_str)]
                    req.finish_reason = FinishReason.STOP_STRING
                    return True

        return False

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs (matches ``processor`` / server HTTP path)."""
        enc = self.processor.tokenizer(text, add_special_tokens=False)
        return list(enc["input_ids"])

    def detokenize(self, token_ids: List[int]) -> str:
        """Detokenize token IDs to text."""
        return self.tokenizer.decode(token_ids)

    def process(self, prompt, images, videos, audios, **kwargs) -> dict:
        """Process the input prompt and media into final model inputs."""
        return self.processor(
            prompt, images=images, videos=videos, audios=audios, **kwargs
        )

    def apply_chat_template(
        self,
        messages: List[dict],
        add_generation_prompt: bool = True,
        chat_template_kwargs: Optional[dict] = None,
    ) -> str:
        """Apply chat template to messages."""
        chat_template_kwargs = chat_template_kwargs or {}
        return self.processor.apply_chat_template(
            conversation=messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **chat_template_kwargs,
        )


class LLM:
    """High-level LLM interface for batch generation."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        cache_type: str = "paged",
        max_batch_size: int = 16,
        max_tokens: int = 4096,
        num_blocks: int = 512,
        block_size: int = 256,
        max_cache_len: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        enable_graph: bool = False,
        attn_backend: str = "default",
        skip_load: bool = False,
    ):
        """Initialize LLM.

        Args:
            model_path: Path to the model directory.
            device: Device type ('cpu', 'cuda', 'mlu', 'moore').
            dtype: Data type ('float16', 'bfloat16', 'float32').
            tensor_parallel_size: Number of devices for tensor parallelism.
            cache_type: Cache type ('paged' or 'static').
            max_batch_size: Maximum batch size (only for paged cache).
            max_tokens: Default maximum tokens to generate.
            num_blocks: Number of KV cache blocks (only for paged cache).
            block_size: Size of each KV cache block (only for paged cache).
            max_cache_len: Maximum sequence length (only for static cache).
            temperature: Default sampling temperature.
            top_p: Default top-p sampling parameter.
            top_k: Default top-k sampling parameter.
            enable_graph: Whether to enable graph compiling.
            attn_backend: Attention backend to use ('default', 'flash-attn').
        """
        config = EngineConfig(
            model_path=model_path,
            device=device,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            cache_type=cache_type,
            max_batch_size=max_batch_size,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            block_size=block_size,
            max_cache_len=max_cache_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_graph=enable_graph,
            attn_backend=attn_backend,
            skip_load=skip_load,
        )
        self.engine = LLMEngine(config)
        self.config = config

    def generate(
        self,
        prompts: Union[str, List[str]] = None,
        messages: Union[List[dict], List[List[dict]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generate completions for the given prompts.

        Args:
            prompts: A single prompt string or list of prompt strings.
            sampling_params: Sampling parameters for generation.
            use_tqdm: Whether to show progress bar.

        Returns:
            List of RequestOutput objects containing generated text.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(messages, list) and isinstance(messages[0], dict):
            messages = [messages]

        contents = prompts
        apply_chat_template = False
        if messages:
            contents = messages
            apply_chat_template = True

        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=self.config.max_tokens)
        elif sampling_params.max_tokens is None:
            sampling_params = sampling_params.clone()
            sampling_params.max_tokens = self.config.max_tokens

        requests = []
        for content in contents:
            request_id = f"cmpl-{uuid.uuid4().hex}"
            processed_inputs = None
            if apply_chat_template:
                prompt = self.engine.apply_chat_template(
                    content, add_generation_prompt=True
                )

                images, videos, audios = resolve_multimodal_inputs(content)
                processed_inputs = self.engine.process(
                    prompt, images, videos, audios, return_tensors="pt"
                )

                prompt_token_ids = processed_inputs.get("input_ids").flatten().tolist()
            else:
                prompt = content
                prompt_token_ids = self.engine.tokenize(prompt)

            req = InferenceRequest(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                processed_inputs=processed_inputs,
                sampling_params=sampling_params,
                eos_token_ids=self.engine.eos_token_ids,
            )
            requests.append(req)
            self.engine.add_request(req)

        # Run inference until all requests are finished
        if use_tqdm:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=len(requests), desc="Generating")
            except ImportError:
                pbar = None
                use_tqdm = False
        else:
            pbar = None

        finished_count = 0
        while finished_count < len(requests):
            self.engine.step()

            new_finished = sum(1 for req in requests if req.is_finished())
            if use_tqdm and pbar and new_finished > finished_count:
                pbar.update(new_finished - finished_count)
            finished_count = new_finished

        if pbar:
            pbar.close()

        outputs = [req.to_request_output() for req in requests]
        return outputs

    def chat(
        self,
        messages: Union[List[dict], List[List[dict]]],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generate chat completions for the given messages.

        Args:
            messages: A single conversation (list of message dicts) or
                     a list of conversations.
            sampling_params: Sampling parameters for generation.
            use_tqdm: Whether to show progress bar.

        Returns:
            List of RequestOutput objects containing generated responses.
        """
        if messages and isinstance(messages[0], dict):
            messages = [messages]

        return self.generate(
            messages=messages, sampling_params=sampling_params, use_tqdm=use_tqdm
        )


def _serving_warmup_http_fidelity() -> bool:
    """Use AutoTokenizer + list content + defer tokenize (matches HTTP openai-chat path)."""
    explicit = os.environ.get("INFINI_PREFILL_SERVING_WARMUP_HTTP_FIDELITY")
    if explicit is not None:
        return explicit.strip() == "1"
    return False


def _should_defer_tokenize_to_step_thread() -> bool:
    """Defer message normalize + tokenize to the step thread for compiled CUDAGraph prefill."""
    return False


def normalize_chat_messages(messages: list) -> list:
    """Normalize OpenAI/vLLM-bench messages (list multimodal content → string)."""
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            normalized.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                elif isinstance(part, str):
                    text_parts.append(part)
            normalized_msg = msg.copy()
            normalized_msg["content"] = "".join(text_parts) if text_parts else ""
            normalized.append(normalized_msg)
        else:
            normalized.append(msg)
    return normalized


class AsyncLLMEngine:
    """Asynchronous LLM engine for server use with streaming support."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        cache_type: str = "paged",
        max_batch_size: int = 16,
        max_tokens: int = 512,
        num_blocks: int = 512,
        block_size: int = 256,
        max_cache_len: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        enable_graph: bool = False,
        attn_backend: str = "default",
    ):
        """Initialize AsyncLLMEngine.

        Args:
            model_path: Path to the model directory.
            device: Device type ('cpu', 'cuda', 'mlu', 'moore').
            dtype: Data type ('float16', 'bfloat16', 'float32').
            tensor_parallel_size: Number of devices for tensor parallelism.
            cache_type: Cache type ('paged' or 'static').
            max_batch_size: Maximum batch size (only for paged cache).
            max_tokens: Default maximum tokens to generate.
            num_blocks: Number of KV cache blocks (only for paged cache).
            block_size: Size of each KV cache block (only for paged cache).
            max_cache_len: Maximum sequence length (only for static cache).
            temperature: Default sampling temperature.
            top_p: Default top-p sampling parameter.
            top_k: Default top-k sampling parameter.
            enable_graph: Whether to enable graph compiling.
            attn_backend: Attention backend to use ('default', 'flash-attn').
        """
        config = EngineConfig(
            model_path=model_path,
            device=device,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            cache_type=cache_type,
            max_batch_size=max_batch_size,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            block_size=block_size,
            max_cache_len=max_cache_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_graph=enable_graph,
            attn_backend=attn_backend,
        )
        self.engine = LLMEngine(config)
        self.config = config

        self._running = False
        self._step_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._healthy = True
        self._serving_capture_ready = threading.Event()
        self._serving_capture_error: Optional[BaseException] = None
        self._deferred_tokenize_queue: queue.Queue[InferenceRequest] = queue.Queue()
        self._hang_trace_forward_in_progress = False
        self._hang_trace_last_step_end = time.monotonic()
        self._hang_trace_last_idle_log = 0.0

    def is_healthy(self) -> bool:
        return bool(self._healthy)

    def wait_for_serving_capture(self, timeout: Optional[float] = None) -> bool:
        """Block until piecewise CUDAGraph capture finishes on the step thread."""
        if not self._serving_capture_ready.wait(timeout=timeout):
            return False
        if self._serving_capture_error is not None:
            raise RuntimeError(
                "CUDAGraph capture failed on serving thread"
            ) from self._serving_capture_error
        return True

    def start(self):
        """Start the background inference loop."""
        if self._running:
            logger.warning("AsyncLLMEngine is already running")
            return

        self._loop = asyncio.get_running_loop()
        self._running = True
        self._serving_capture_ready.clear()
        self._serving_capture_error = None
        self._step_thread = threading.Thread(
            target=self._step_loop, daemon=True, name="AsyncLLMEngineStepThread"
        )
        self._step_thread.start()
        logger.info("AsyncLLMEngine started")

    def stop(self):
        """Stop the background inference loop."""
        if not self._running:
            logger.warning("AsyncLLMEngine is not running")
            return

        self._running = False
        if self._step_thread:
            self._step_thread.join(timeout=5)
        logger.info("AsyncLLMEngine stopped")

    def _warmup_serving_bench_prefill(self) -> None:
        """No-op: PRD-02 vLLM serving warmup removed (native CG only)."""
        return

    def _step_loop(self):
        """Background loop that runs inference steps."""
        self._serving_capture_ready.set()
        while self._running:
            try:
                self._flush_deferred_tokenize()
                profile = _step_profile_enabled()
                t_iter0 = time.perf_counter() if profile else 0.0
                if _hang_trace_enabled():
                    now = time.monotonic()
                    if now - self._hang_trace_last_idle_log >= 30.0:
                        sched = self.engine.scheduler
                        waiting = sched.waiting_queue.sync_q.qsize()
                        running = sched.running_queue.sync_q.qsize()
                        chunking = sched.chunking_queue.sync_q.qsize()
                        cache_stats = sched.get_cache_stats()
                        idle_sec = now - self._hang_trace_last_step_end
                        if (
                            waiting > 0
                            or running > 0
                            or chunking > 0
                            or self._hang_trace_forward_in_progress
                        ):
                            logger.info(
                                "hang_trace: idle waiting=%d running=%d chunking=%d "
                                "free_blocks=%d forward_in_progress=%s idle_sec=%.1f",
                                waiting,
                                running,
                                chunking,
                                cache_stats["num_free_blocks"],
                                self._hang_trace_forward_in_progress,
                                idle_sec,
                            )
                            self._hang_trace_last_idle_log = now
                self._hang_trace_forward_in_progress = True
                requests, pending = self.engine.step()
                self._hang_trace_forward_in_progress = False
                self._hang_trace_last_step_end = time.monotonic()
                forward_ms = (
                    (time.perf_counter() - t_iter0) * 1000.0 if profile and requests else 0.0
                )
                if not requests:
                    idle_sleep_ms = 10.0
                    time.sleep(idle_sleep_ms / 1000.0)
                    if profile:
                        total_iter_ms = (time.perf_counter() - t_iter0) * 1000.0
                        logger.info(
                            "step_profile: iteration schedule_empty=1 forward_ms=0.000 "
                            "idle_sleep_ms=%.3f total_iter_ms=%.3f",
                            idle_sleep_ms,
                            total_iter_ms,
                        )
                else:
                    if pending:
                        self._loop.call_soon_threadsafe(self._batch_put, pending)
                    idle_sleep_ms = 0.5
                    time.sleep(idle_sleep_ms / 1000.0)
                    if profile:
                        total_iter_ms = (time.perf_counter() - t_iter0) * 1000.0
                        logger.info(
                            "step_profile: iteration schedule_empty=0 forward_ms=%.3f "
                            "idle_sleep_ms=%.3f total_iter_ms=%.3f n_req=%d",
                            forward_ms,
                            idle_sleep_ms,
                            total_iter_ms,
                            len(requests),
                        )
            except Exception as e:
                self._hang_trace_forward_in_progress = False
                logger.error(f"Error in step loop: {e}", exc_info=True)
                self._healthy = False
                self._running = False
                break

    @staticmethod
    def _batch_put(pending):
        for async_q, output in pending:
            try:
                async_q.put_nowait(output)
            except Exception as e:
                logger.warning(
                    f"Failed to put token for request {output.request_id}: {e}. "
                    f"Likely due to client disconnecting or request cancelation."
                )

    def _flush_deferred_tokenize(self) -> None:
        """Tokenize pending HTTP/async requests on the serving step thread."""
        while True:
            try:
                request = self._deferred_tokenize_queue.get_nowait()
            except queue.Empty:
                break
            self._finalize_deferred_tokenize(request)
            self.engine.add_request(request)

    def _finalize_deferred_tokenize(self, request: InferenceRequest) -> None:
        spec = getattr(request, "_deferred_tokenize", None)
        if not spec:
            return

        messages = spec.get("messages")
        if messages is not None:
            messages = normalize_chat_messages(messages)
        prompt = spec.get("prompt")
        apply_chat_template = bool(spec.get("apply_chat_template", True))
        add_generation_prompt = bool(spec.get("add_generation_prompt", True))
        chat_template_kwargs = spec.get("chat_template_kwargs") or {}

        processed_inputs = None
        if prompt is not None:
            prompt_token_ids = self.engine.tokenize(prompt)
        else:
            assert messages is not None, (
                "deferred tokenize requires messages or prompt"
            )
            assert apply_chat_template, (
                "apply_chat_template needs to be true for multi-role conversation"
            )
            prompt = self.engine.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                chat_template_kwargs=chat_template_kwargs,
            )
            images, videos, audios = resolve_multimodal_inputs(messages)
            if images or videos or audios:
                processed_inputs = self.engine.process(
                    prompt, images, videos, audios, return_tensors="pt"
                )
                prompt_token_ids = processed_inputs.get("input_ids").flatten().tolist()
            else:
                prompt_token_ids = self.engine.tokenize(prompt)

        request.prompt = prompt
        request.prompt_token_ids = prompt_token_ids
        request.prompt_length = len(prompt_token_ids)
        request.processed_inputs = processed_inputs
        request._deferred_tokenize = None
        logger.info(
            "compiled prefill: deferred tokenize done request=%s prompt_len=%s",
            request.request_id,
            len(prompt_token_ids),
        )

    def add_request(
        self,
        messages: Optional[List[dict]],
        apply_chat_template: bool = True,
        add_generation_prompt: bool = True,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        # For server use
        request_data: Optional[dict] = None,
        http_request: Optional[any] = None,
    ) -> InferenceRequest:
        """Add a request to the engine.

        Args:
            messages: List of message dicts (chat conversation). Following this format:
                    [
                        {
                            "role": "user",
                            "content": [
                            {
                                "type": "text",
                                "text": "xxxxxxxxx"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": "xxx.jpg"
                                }
                            },
                            ]
                        },
                    ]
            apply_chat_template: Whether to apply the chat template.
            add_generation_prompt: Whether to add a generation prompt.
            prompt: Text prompt for generation. If provided, it will be used directly after encoded by tokenizer, ignoring messages.
            prompt_token_ids: Pre-tokenized prompt. If provided, it will be used directly as input.
            sampling_params: Sampling parameters.
            request_id: Optional request ID.
            request_data: Optional request data dict (for server use).
            http_request: Optional HTTP request object (for server use).

        Returns:
            The created InferenceRequest object.
        """
        if request_id is None:
            request_id = f"cmpl-{uuid.uuid4().hex}"

        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=self.config.max_tokens)
        elif sampling_params.max_tokens is None:
            sampling_params = sampling_params.clone()
            sampling_params.max_tokens = self.config.max_tokens

        needs_tokenize = prompt_token_ids is None and (
            messages is not None or prompt is not None
        )
        if needs_tokenize and _should_defer_tokenize_to_step_thread():
            request = InferenceRequest(
                request_id=request_id,
                prompt=None,
                prompt_token_ids=[],
                processed_inputs=None,
                sampling_params=sampling_params,
                eos_token_ids=self.engine.eos_token_ids,
                request_data=request_data,
                http_request=http_request,
            )
            request._deferred_tokenize = {
                "messages": messages,
                "prompt": prompt,
                "apply_chat_template": apply_chat_template,
                "add_generation_prompt": add_generation_prompt,
                "chat_template_kwargs": chat_template_kwargs or {},
            }
            _ = request.output_queue
            self._deferred_tokenize_queue.put(request)
            logger.info(
                "compiled prefill: deferred request %s to serving thread (partial PIECEWISE)",
                request_id,
            )
            return request

        images, videos, audios = None, None, None
        processed_inputs = None

        if prompt_token_ids is not None:
            prompt = self.engine.detokenize(prompt_token_ids)
        elif prompt is not None:
            prompt_token_ids = self.engine.tokenize(prompt)
        else:
            assert messages is not None, (
                "Either messages or prompt/prompt_token_ids must be provided"
            )
            messages = normalize_chat_messages(messages)

            assert apply_chat_template, (
                "apply_chat_template needs to be true for multi-role conversation"
            )

            prompt = self.engine.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                chat_template_kwargs=chat_template_kwargs or {},
            )

            images, videos, audios = resolve_multimodal_inputs(messages)
            if images or videos or audios:
                processed_inputs = self.engine.process(
                    prompt, images, videos, audios, return_tensors="pt"
                )
                prompt_token_ids = processed_inputs.get("input_ids").flatten().tolist()
            else:
                # CPU-only token ids: ``return_tensors=\"pt\"`` on the asyncio /
                # HTTP thread leaves torch state that ATU-faults partial PIECEWISE
                # CUDAGraph replay on the AsyncLLMEngine step thread (MetaX).
                prompt_token_ids = self.engine.tokenize(prompt)

        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            processed_inputs=processed_inputs,
            sampling_params=sampling_params,
            eos_token_ids=self.engine.eos_token_ids,
            request_data=request_data,
            http_request=http_request,
        )

        # Initialize output queue for streaming
        _ = request.output_queue

        self.engine.add_request(request)
        return request

    def add_chat_request(
        self,
        messages: List[dict],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        request_data: Optional[dict] = None,
        http_request: Optional[any] = None,
        add_generation_prompt: bool = True,
        chat_template_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> InferenceRequest:
        """Add a chat request to the engine.

        Args:
            messages: List of message dicts (chat conversation).
            sampling_params: Sampling parameters.
            request_id: Optional request ID.
            request_data: Optional request data dict.
            http_request: Optional HTTP request object.

        Returns:
            The created InferenceRequest object.
        """

        return self.add_request(
            messages=messages,
            apply_chat_template=True,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
            sampling_params=sampling_params,
            request_id=request_id,
            request_data=request_data,
            http_request=http_request,
        )

    async def stream_request(
        self,
        request: InferenceRequest,
        timeout: float = 100.0,
        request_timeout: Optional[float] = None,
    ) -> AsyncIterator[TokenOutput]:
        """Stream tokens from a request.

        Args:
            request: The inference request to stream from.
            timeout: Timeout for waiting on each token.

        Yields:
            TokenOutput objects for each generated token.
        """
        import asyncio

        start = time.time()
        while True:
            try:
                if request_timeout and time.time() - start > float(request_timeout):
                    request.mark_timeout()
                    yield TokenOutput(
                        request_id=request.request_id,
                        token_id=-1,
                        token_text="",
                        finished=True,
                        finish_reason=FinishReason.TIMEOUT,
                        generated_text=request.generated_text,
                    )
                    break

                token_output = await asyncio.wait_for(
                    request.output_queue.async_q.get(), timeout=timeout
                )

                request.output_queue.async_q.task_done()

                yield token_output

                if token_output.finished:
                    break
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout while waiting for token from request {request.request_id}"
                )
                if request.is_aborted():
                    while not request.output_queue.async_q.empty():
                        try:
                            token_output = request.output_queue.async_q.get_nowait()
                            request.output_queue.async_q.task_done()
                            yield token_output
                        except asyncio.QueueEmpty:
                            break

                    yield TokenOutput(
                        request_id=request.request_id,
                        token_id=-1,
                        token_text="",
                        finished=True,
                        finish_reason=request.finish_reason,
                        generated_text=request.generated_text,
                    )
                    break
                continue
            except Exception as e:
                logger.error(f"Error while streaming request {request.request_id}: {e}")
                break
