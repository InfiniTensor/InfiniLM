"""
LLM Engine - Main interface for LLM inference.

This module provides:
- LLM class for batch generation (offline use)
- AsyncLLM class for asynchronous streaming (server use)
"""

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

from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.cache.cache import PagedKVCacheConfig, StaticKVCacheConfig
from infinilm.modeling_utils import load_model_state_dict_by_file
from transformers import AutoTokenizer
from tokenizers import decoders as _dec

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
    """

    model_path: str
    device: str = "cuda"
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    cache_type: str = "paged"  # "paged" or "static"
    max_batch_size: int = 16
    max_tokens: int = 4096
    num_blocks: int = 8 * 1024
    block_size: int = 16
    max_cache_len: int = 4096
    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 1
    enable_graph: bool = False


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
        )

        # Load model weights
        load_model_state_dict_by_file(
            self.model_engine, config.model_path, dtype=self.model_engine.config.dtype
        )

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        self._fix_tokenizer_decoder()

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
            self.scheduler = Scheduler(
                max_batch_size=config.max_batch_size,
                num_blocks=config.num_blocks,
                block_size=config.block_size,
            )
            logger.info(f"Using Paged KV Cache with num_blocks={config.num_blocks}")
        else:
            raise ValueError(f"Unsupported cache_type: {config.cache_type}")

        self.model_engine.reset_cache(cache_config)
        self.cache_type = config.cache_type

        # Get EOS token IDs from model config
        self.eos_token_ids = self.model_engine.config.eos_token_id or []
        if isinstance(self.eos_token_ids, int):
            self.eos_token_ids = [self.eos_token_ids]

        logger.info(
            f"LLMEngine initialized with model at {config.model_path} "
            f"on device {config.device}"
            f"enable_graph={config.enable_graph}"
        )

    def _init_device(self):
        """Initialize infinicore device and dtype."""
        supported_devices = ["cpu", "cuda", "mlu", "moore"]
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

    def _fix_tokenizer_decoder(self):
        """Fix tokenizer decoder for llama models."""
        if "llama" in self.model_engine.config.model_type.lower():
            backend = getattr(self.tokenizer, "backend_tokenizer", None)
            target = getattr(backend, "_tokenizer", backend)
            norm = getattr(target, "normalizer", None)
            dec = getattr(target, "decoder", None)
            sn = repr(norm)[:800] if norm is not None else ""
            sd = repr(dec)[:800] if dec is not None else ""
            has_prepend = "Prepend" in sn
            has_strip = "Strip" in sd
            if has_prepend and has_strip:
                target.decoder = _dec.Sequence(
                    [
                        _dec.Replace("▁", " "),
                        _dec.ByteFallback(),
                        _dec.Fuse(),
                    ]
                )

    def add_request(self, request: InferenceRequest):
        """Add a request to the scheduler."""
        self.scheduler.add_request(request)

    def step(self) -> List[InferenceRequest]:
        """Run one inference step.

        Returns:
            List of requests that were processed in this step.
        """
        # Schedule requests
        scheduler_output = self.scheduler.schedule()
        if scheduler_output is None or not scheduler_output.scheduled_requests:
            return []

        # Build model inputs
        model_input_dict = scheduler_output.build_model_inputs(
            self.config.temperature, self.config.top_p, self.config.top_k
        )
        model_input = self._prepare_model_input(model_input_dict)

        # Run inference
        sampled_tokens = self.model_engine.forward(**model_input)
        sampled_tokens_list = sampled_tokens.to_numpy().tolist()

        # Update request status
        self._update_requests(
            scheduler_output.is_prefill,
            scheduler_output.scheduled_requests,
            sampled_tokens_list,
        )

        return scheduler_output.scheduled_requests

    def _prepare_model_input(self, model_input_dict: dict) -> dict:
        """Convert model input dict to infinicore tensors."""
        model_input = {}
        for key, value in model_input_dict.items():
            if value is None:
                # Skip None values (block_tables/slot_mapping for static cache)
                model_input[key] = None
            elif key in [
                "input_ids",
                "position_ids",
                "past_kv_lengths",
                "total_kv_lengths",
                "input_offsets",
                "slot_mapping",
                "block_tables",
            ]:
                model_input[key] = infinicore.from_list(value, dtype=infinicore.int64)
            else:
                # temperature, top_k, top_p, etc.
                model_input[key] = value
        return model_input

    def _update_requests(
        self,
        is_prefill: bool,
        requests: List[InferenceRequest],
        sampled_tokens: List[int],
    ):
        """Update request status after inference step."""
        # Only reset req blocks for paged cache
        if is_prefill and self.cache_type == "paged":
            self.scheduler.cache_manager.reset_req_blocks()

        for req, token_id in zip(requests, sampled_tokens):
            req.generated_token_ids.append(token_id)
            if req.is_prefill:
                req.is_prefill = False
            # vLLM-style replacement character handling is primarily relevant for streaming.
            # For offline generation (no output queue), keep the fast incremental path.
            if req._output_queue is None:
                token_text = self.detokenize([token_id])
                req.generated_text += token_text
            else:
                # Streaming path: compute delta from a full decode so we can hold back
                # trailing '\ufffd' (likely an incomplete UTF-8 sequence).
                decoded_text = self.detokenize(req.generated_token_ids)

                finished_now = False
                # Update generated_text to the latest decode (used for stop-string checks and debugging)
                req.generated_text = decoded_text

                if self._check_request_finished(req, token_id):
                    req.mark_finished(req.finish_reason)
                    finished_now = True

                    # Remove stop string from generated_text if STOP_STRING finish reason
                    if req.finish_reason == FinishReason.STOP_STRING:
                        stop_strings = req.sampling_params.stop or []
                        for stop_str in stop_strings:
                            if decoded_text.endswith(stop_str):
                                # Remove the stop string from the end
                                decoded_text = decoded_text[: -len(stop_str)]
                                req.generated_text = decoded_text
                                break

                holds_back_incomplete_utf8 = bool(
                    decoded_text
                ) and decoded_text.endswith("\ufffd")

                # vLLM-style: hold back only if we are not on the final chunk.
                # Suppress output when finish reason is LENGTH or STOP_STRING.
                # Root cause fix: When STOP_STRING is detected, we suppress output for the token
                # that completes the stop string, preventing additional tokens from being output.
                if (holds_back_incomplete_utf8 and not finished_now) or (
                    finished_now
                    and req.finish_reason
                    in (FinishReason.LENGTH, FinishReason.STOP_STRING)
                ):
                    token_text = ""
                else:
                    last_len = getattr(req, "_stream_last_yielded_length", 0)
                    token_text = decoded_text[last_len:]
                    if token_text:
                        req._stream_last_yielded_length = len(decoded_text)

            # For non-streaming, finish checks happen here.
            if req._output_queue is None and self._check_request_finished(
                req, token_id
            ):
                req.mark_finished(req.finish_reason)
                # Remove stop string from generated_text if STOP_STRING finish reason
                if req.finish_reason == FinishReason.STOP_STRING:
                    stop_strings = req.sampling_params.stop or []
                    for stop_str in stop_strings:
                        if req.generated_text.endswith(stop_str):
                            # Remove the stop string from the end
                            req.generated_text = req.generated_text[: -len(stop_str)]
                            break
            # Put output in queue if it exists (for async streaming)
            if req._output_queue is not None:
                output = TokenOutput(
                    request_id=req.request_id,
                    token_id=token_id,
                    token_text=token_text,
                    finished=req.is_finished(),
                    finish_reason=req.finish_reason,
                    generated_text=req.generated_text,
                )
                req.output_queue.sync_q.put(output)

        self.scheduler.complete_requests(requests)

    def _check_request_finished(self, req: InferenceRequest, token_id: int) -> bool:
        """Check if request generation is finished."""
        max_tokens = req.sampling_params.max_tokens
        if max_tokens and req.get_num_generated_tokens() >= max_tokens:
            req.finish_reason = FinishReason.LENGTH
            return True

        # Check EOS token
        eos_ids = req.eos_token_ids or self.eos_token_ids
        if eos_ids and token_id in eos_ids:
            req.finish_reason = FinishReason.EOS_TOKEN
            return True

        # Check stop strings
        stop_strings = req.sampling_params.stop or []
        for stop_str in stop_strings:
            if req.generated_text.endswith(stop_str):
                req.finish_reason = FinishReason.STOP_STRING
                return True

        return False

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs."""
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        """Detokenize token IDs to text."""
        return self.tokenizer.decode(token_ids)

    def apply_chat_template(
        self,
        messages: List[dict],
        add_generation_prompt: bool = True,
        chat_template_kwargs: Optional[dict] = None,
    ) -> str:
        """Apply chat template to messages."""
        chat_template_kwargs = chat_template_kwargs or {}
        return self.tokenizer.apply_chat_template(
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
        num_blocks: int = 8 * 1024,
        block_size: int = 16,
        max_cache_len: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        enable_graph: bool = False,
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
        )
        self.engine = LLMEngine(config)
        self.config = config

    def generate(
        self,
        prompts: Union[str, List[str]],
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

        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=self.config.max_tokens)
        elif sampling_params.max_tokens is None:
            sampling_params = sampling_params.clone()
            sampling_params.max_tokens = self.config.max_tokens

        requests = []
        for prompt in prompts:
            request_id = f"cmpl-{uuid.uuid4().hex}"
            token_ids = self.engine.tokenize(prompt)
            req = InferenceRequest(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=token_ids,
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

        prompts = []
        for conversation in messages:
            prompt = self.engine.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            prompts.append(prompt)

        return self.generate(prompts, sampling_params, use_tqdm)


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
        num_blocks: int = 8 * 1024,
        block_size: int = 16,
        max_cache_len: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        enable_graph: bool = False,
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
        )
        self.engine = LLMEngine(config)
        self.config = config

        self._running = False
        self._step_thread: Optional[threading.Thread] = None
        self._healthy = True

    def is_healthy(self) -> bool:
        return bool(self._healthy)

    def start(self):
        """Start the background inference loop."""
        if self._running:
            logger.warning("AsyncLLMEngine is already running")
            return

        self._running = True
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

    def _step_loop(self):
        """Background loop that runs inference steps."""
        while self._running:
            try:
                requests = self.engine.step()
                if not requests:
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in step loop: {e}", exc_info=True)
                self._healthy = False
                self._running = False
                break

    def add_request(
        self,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        # For server use
        request_data: Optional[dict] = None,
        http_request: Optional[any] = None,
    ) -> InferenceRequest:
        """Add a request to the engine.

        Args:
            prompt: Text prompt for generation.
            prompt_token_ids: Pre-tokenized prompt.
            sampling_params: Sampling parameters.
            request_id: Optional request ID.
            request_data: Optional request data dict (for server use).
            http_request: Optional HTTP request object (for server use).

        Returns:
            The created InferenceRequest object.
        """
        if request_id is None:
            request_id = f"cmpl-{uuid.uuid4().hex}"

        if prompt_token_ids is None and prompt is not None:
            prompt_token_ids = self.engine.tokenize(prompt)

        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=self.config.max_tokens)
        elif sampling_params.max_tokens is None:
            sampling_params = sampling_params.clone()
            sampling_params.max_tokens = self.config.max_tokens

        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
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
        prompt = self.engine.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )
        return self.add_request(
            prompt=prompt,
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
            if request.is_finished() and request.output_queue.async_q.empty():
                break

            try:
                token_output = await asyncio.wait_for(
                    request.output_queue.async_q.get(), timeout=timeout
                )

                request.output_queue.async_q.task_done()

                yield token_output

                if token_output.finished:
                    break
            except asyncio.TimeoutError:
                # Enforce request-level timeout even if no tokens are produced.
                if request_timeout is not None:
                    now = time.time()
                    if now - start > float(request_timeout):
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
                if request.is_finished():
                    break
                continue
            except asyncio.CancelledError:
                request.mark_canceled()
                break
            except Exception as e:
                logger.error(f"Error streaming request {request.request_id}: {e}")
                await asyncio.sleep(0.01)
