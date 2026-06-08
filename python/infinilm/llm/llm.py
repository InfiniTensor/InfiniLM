"""
LLM Engine - Main interface for LLM inference.

This module provides:
- LLM class for batch generation (offline use)
- AsyncLLM class for asynchronous streaming (server use)
"""

import asyncio
import os
import queue
import time
import uuid
import logging
import threading
from typing import List, Optional, Union, AsyncIterator
from dataclasses import dataclass

import infinicore

from infinilm.llm.request import (
    InferenceRequest,
    RequestOutput,
    TokenOutput,
    FinishReason,
)
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler
from infinilm.llm.static_scheduler import StaticScheduler
from infinilm.processors import AutoInfinilmProcessor
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.cache.cache import PagedKVCacheConfig, StaticKVCacheConfig
from infinilm.modeling_utils import load_model_state_dict_by_file
from infinilm.multimodal.multimodal import resolve_multimodal_inputs

logger = logging.getLogger(__name__)


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

        # Initialize KV cache based on cache type
        if config.cache_type == "static":
            cache_config = StaticKVCacheConfig(
                max_batch_size=1, max_cache_len=config.max_cache_len
            )
            self.scheduler = StaticScheduler(max_cache_len=config.max_cache_len)
            logger.info(
                f"Using Static KV Cache with max_cache_len={config.max_cache_len}"
            )
        elif config.cache_type == "paged":
            cache_config = PagedKVCacheConfig(
                num_blocks=config.num_blocks, block_size=config.block_size
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
                        "native piecewise CG enabled (INFINI_PREFILL_NATIVE_CG=1); "
                        "prefill graphs captured in C++ at init"
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

    def add_request(self, request: InferenceRequest):
        """Add a request to the scheduler."""
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
        from infinilm.torch_llama.kv_paged import ensure_hybrid_prefill_gpu_context

        ensure_hybrid_prefill_gpu_context()
        # Schedule requests
        scheduler_output = self.scheduler.schedule()
        if scheduler_output is None or not scheduler_output.scheduled_requests:
            return [], []

        # Build model inputs
        model_input = self.processor.build_model_inputs(
            scheduler_output,
            self.config.temperature,
            self.config.top_p,
            self.config.top_k,
        )

        # Run inference (hybrid compiled prefill for single-request prefill steps).
        cpp_model_input = {
            k: v for k, v in model_input.items() if k != "input_ids_torch"
        }
        if scheduler_output.is_prefill:
            skip_hybrid = False
            compute_len = 0
            for req in scheduler_output.scheduled_requests:
                if req.is_chunking():
                    start = req.chunk_prefill_offset
                    end = min(start + req.chunk_size, len(req.get_input_tokens()))
                    req_compute = end - start
                    slot_start = start - req.num_cached_tokens
                    slot_end = end - req.num_cached_tokens
                    non_cached_slots = req.slot_mapping[slot_start:slot_end]
                else:
                    req_compute = len(req.get_input_tokens()) - req.num_cached_tokens
                    non_cached_slots = req.slot_mapping
                compute_len += req_compute
                if req_compute == 0 or len(non_cached_slots) == 0:
                    skip_hybrid = True
            hybrid_tokens = (
                None
                if skip_hybrid
                else self.model_engine.try_hybrid_prefill_forward(**model_input)
            )
            if hybrid_tokens is not None:
                sampled_tokens_list = hybrid_tokens
            else:
                sampled_tokens = self.model_engine.forward(**cpp_model_input)
                sampled_tokens_list = sampled_tokens.to_numpy().tolist()
        else:
            sampled_tokens = self.model_engine.forward(**cpp_model_input)
            sampled_tokens_list = sampled_tokens.to_numpy().tolist()

        # Update request status
        pending = self._update_requests(
            scheduler_output.is_prefill,
            scheduler_output.scheduled_requests,
            sampled_tokens_list,
        )

        return scheduler_output.scheduled_requests, pending

    def _update_requests(
        self,
        is_prefill: bool,
        requests: List[InferenceRequest],
        sampled_tokens: List[int],
    ) -> List[tuple]:
        """Update request status after inference step."""
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

        prefill_final_requests = (
            [
                r
                for r in requests
                if r.is_prefill and (not r.is_chunking() or r.chunk_is_last())
            ]
            if is_prefill
            else requests
        )

        pending = []
        for req, token_id in zip(prefill_final_requests, sampled_tokens):
            if req.is_aborted():
                logger.info(
                    f"Request {req.request_id} aborted by client, skipping update"
                )
                continue

            if req.is_prefill:
                req.chunk_prefill_offset = 0
                req.chunk_size = 0
                req.is_prefill = False

            req.generated_token_ids.append(token_id)
            pending_tokens = req.generated_token_ids[req._pending_token_offset :]
            delta = self.tokenizer.decode(pending_tokens)
            holds_back = bool(delta) and delta.endswith("\ufffd")

            last_committed_text = req.generated_text

            if not holds_back:
                req.generated_text = last_committed_text + delta
                req._pending_token_offset = len(req.generated_token_ids)

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
                    continue
                pending.append((req.output_queue.async_q, output))

        self.scheduler.complete_requests(requests)
        return pending

    def _check_request_finished(self, req: InferenceRequest, token_id: int) -> bool:
        """Check if request generation is finished."""
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
    try:
        from infinilm.compile.env import prefill_compile_enabled, prefill_cudagraph_enabled

        return prefill_compile_enabled() and prefill_cudagraph_enabled()
    except ImportError:
        return False


def _should_defer_tokenize_to_step_thread() -> bool:
    """Defer message normalize + tokenize to the step thread for compiled CUDAGraph prefill."""
    try:
        from infinilm.compile.env import prefill_compile_enabled, prefill_cudagraph_enabled

        return prefill_compile_enabled() and prefill_cudagraph_enabled()
    except ImportError:
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
        """Run one vLLM-bench-shaped partial prefill on the step thread before HTTP."""
        from infinilm.compile.env import prefill_compile_enabled, prefill_cudagraph_enabled

        if not prefill_compile_enabled() or not prefill_cudagraph_enabled():
            return
        skip_warmup = os.environ.get("INFINI_PREFILL_SKIP_SERVING_WARMUP", "1").strip() or "1"
        if skip_warmup == "1":
            return
        try:
            from vllm.benchmarks.datasets import RandomDataset
        except ImportError:
            logger.warning(
                "compiled prefill: skip serving warmup (vllm benchmarks unavailable)"
            )
            return

        seed_raw = os.environ.get("INFINI_PREFILL_SERVING_WARMUP_SEED", "0").strip()
        input_len_raw = os.environ.get(
            "INFINI_PREFILL_SERVING_WARMUP_INPUT_LEN", "512"
        ).strip()
        seed = int(seed_raw or "0")
        input_len = int(input_len_raw or "512")
        http_fidelity = _serving_warmup_http_fidelity()
        ds = RandomDataset(random_seed=seed)
        sample_kwargs = dict(
            input_len=input_len,
            output_len=1,
            prefix_len=0,
        )
        if not http_fidelity:
            sample_kwargs["range_ratio"] = 0.0
        bench_tokenizer = self.engine.tokenizer
        if http_fidelity:
            from transformers import AutoTokenizer

            bench_tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, trust_remote_code=True
            )
        prompt = ds.sample(bench_tokenizer, 1, **sample_kwargs)[0].prompt
        messages = (
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            if http_fidelity
            else normalize_chat_messages(
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            )
        )
        sampling_params = SamplingParams(
            max_tokens=1, temperature=1.0, top_p=1.0, top_k=1
        )
        if http_fidelity and _should_defer_tokenize_to_step_thread():
            request = InferenceRequest(
                request_id="__serving_warmup__",
                prompt=None,
                prompt_token_ids=[],
                processed_inputs=None,
                sampling_params=sampling_params,
                eos_token_ids=self.engine.eos_token_ids,
            )
            request._deferred_tokenize = {
                "messages": messages,
                "prompt": None,
                "apply_chat_template": True,
                "add_generation_prompt": True,
                "chat_template_kwargs": {},
            }
            self._finalize_deferred_tokenize(request)
            self.engine.add_request(request)
            prompt_len = len(request.prompt_token_ids)
        else:
            prompt_text = self.engine.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            token_ids = self.engine.tokenize(prompt_text)
            self.engine.add_request(
                InferenceRequest(
                    request_id="__serving_warmup__",
                    prompt=prompt_text,
                    prompt_token_ids=token_ids,
                    sampling_params=sampling_params,
                )
            )
            prompt_len = len(token_ids)
        scheduled, _pending = self.engine.step()
        if not scheduled:
            raise RuntimeError("serving warmup produced no scheduled requests")
        logger.info(
            "compiled prefill: serving-thread bench warmup OK (prompt_len=%s)",
            prompt_len,
        )

    def _step_loop(self):
        """Background loop that runs inference steps."""
        from infinilm.compile.hybrid_prefill import ensure_serving_thread_cudagraph_capture

        try:
            ensure_serving_thread_cudagraph_capture(self.engine.model_engine)
            import torch
            from infinilm.torch_llama.kv_paged import ensure_hybrid_prefill_gpu_context

            ensure_hybrid_prefill_gpu_context()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._warmup_serving_bench_prefill()
        except Exception as e:
            logger.error(
                "Failed to prepare CUDAGraph capture on serving thread: %s",
                e,
                exc_info=True,
            )
            self._serving_capture_error = e
            self._healthy = False
            self._running = False
            self._serving_capture_ready.set()
            return
        self._serving_capture_ready.set()
        while self._running:
            try:
                from infinilm.torch_llama.kv_paged import ensure_hybrid_prefill_gpu_context

                ensure_hybrid_prefill_gpu_context()
                self._flush_deferred_tokenize()
                requests, pending = self.engine.step()
                if not requests:
                    time.sleep(0.01)
                else:
                    if pending:
                        self._loop.call_soon_threadsafe(self._batch_put, pending)
                    time.sleep(0.0005)
            except Exception as e:
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
        from infinilm.torch_llama.kv_paged import ensure_hybrid_prefill_gpu_context

        ensure_hybrid_prefill_gpu_context()
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
                messages, add_generation_prompt=add_generation_prompt
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
