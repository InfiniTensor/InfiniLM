"""
LLM Engine - Main interface for LLM inference.

This module provides:
- LLM class for batch generation (offline use)
- AsyncLLM class for asynchronous streaming (server use)
"""

import asyncio
import logging
import os
import threading
import time
import uuid
from typing import AsyncIterator, List, Optional, Union

import janus

from infinilm.config.engine_config import EngineConfig
from infinilm.config.kv_transfer import KVTransferConfig
from infinilm.kv_connector import KVConnectorFactory, KVConnectorRole
from infinilm.llm.model_runner.model_runner import ModelRunner
from infinilm.llm.request import (
    FinishReason,
    InferenceRequest,
    RequestOutput,
    TokenOutput,
)
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler
from infinilm.llm.static_scheduler import StaticScheduler
from infinilm.multimodal.multimodal import resolve_multimodal_inputs

logger = logging.getLogger(__name__)


class LLMEngine:
    """Low-level LLM engine that handles inference execution."""

    def __init__(self, config: EngineConfig):
        self.config = config

        self.model_runner = ModelRunner(config)

        self.device = self.model_runner.device
        self.dtype = self.model_runner.dtype

        # Initialize processor
        self.processor = self.model_runner.processor
        self.tokenizer = self.processor.get_tokenizer()

        # Initialize KV cache based on cache type
        if config.cache_type == "static":
            self.scheduler = StaticScheduler(max_cache_len=config.max_cache_len)
            logger.info(
                f"Using Static KV Cache with max_cache_len={config.max_cache_len}"
            )
        elif config.cache_type == "paged":
            connector = None
            if config.kv_transfer_config and config.kv_transfer_config.kv_connector:
                connector = KVConnectorFactory.create_connector(
                    connector_name=config.kv_transfer_config.kv_connector,
                    role=KVConnectorRole.SCHEDULER,
                    kv_transfer_config=config.kv_transfer_config,
                )
                logger.info(
                    f"KV Connector created: {config.kv_transfer_config.kv_connector} "
                    f"(role={config.kv_transfer_config.kv_role})"
                )

            max_position_embeddings = self.model_runner.model_engine.hf_config.get(
                "max_position_embeddings", config.max_cache_len
            )
            max_num_batched_tokens = int(
                os.getenv("INFINILM_MAX_NUM_BATCHED_TOKENS", max_position_embeddings)
            )
            assert 1024 <= max_num_batched_tokens <= max_position_embeddings

            self.scheduler = Scheduler(
                max_batch_size=config.max_batch_size,
                num_blocks=config.num_blocks,
                block_size=config.block_size,
                max_num_batched_tokens=max_num_batched_tokens,
                connector=connector,
            )
            logger.info(f"Using Paged KV Cache with num_blocks={config.num_blocks}")
        else:
            raise ValueError(f"Unsupported cache_type: {config.cache_type}")

        self.cache_type = config.cache_type

        # Get EOS token IDs from model config
        self.eos_token_ids = self.model_runner.eos_token_id or []
        if isinstance(self.eos_token_ids, int):
            self.eos_token_ids = [self.eos_token_ids]

        logger.info(
            f"LLMEngine initialized with model at {config.model_path} "
            f"on device {config.device}, "
            f"enable_graph={config.enable_graph}"
        )

    def add_request(self, request: InferenceRequest):
        """Add a request to the scheduler."""
        self.scheduler.add_request(request)

    def step(self) -> tuple[bool, list[tuple]]:
        """Run one inference step.

        Returns:
            A tuple of:
            - did_work
            - pending: Pending streaming outputs as (async_queue, TokenOutput) pairs.
        """
        # Schedule the next unit of work, which may be model execution,
        # connector control metadata, or both.
        scheduler_output = self.scheduler.schedule()
        if scheduler_output is None:
            return False, []

        # Execute model
        runner_output = self.model_runner.execute_model(scheduler_output)
        sampled_tokens_list = runner_output.sampled_token_ids
        self.scheduler.update_from_output(runner_output)

        # Update request status
        pending = self._update_requests(
            scheduler_output.is_prefill,
            scheduler_output.scheduled_requests,
            sampled_tokens_list,
        )

        # Return False (no immediate work) only when no requests were scheduled
        # and no KV transfers completed in this step.
        if not scheduler_output.scheduled_requests:
            if not runner_output.kv_connector_output or (
                not getattr(runner_output.kv_connector_output, "finished_sending", None)
                and not getattr(
                    runner_output.kv_connector_output, "finished_recving", None
                )
            ):
                return False, pending

        return True, pending

    def _update_requests(
        self,
        is_prefill: bool,
        requests: List[InferenceRequest],
        sampled_tokens: List[int],
    ) -> List[tuple]:
        """Update request status after inference step."""
        if is_prefill:
            match self.cache_type:
                case "paged":
                    pass
                case "static":
                    self.scheduler.update_cache()
                case _:
                    raise ValueError(f"Unsupported cache_type: {self.cache_type}")
        pending = []
        for req, token_id in zip(requests, sampled_tokens):
            if req.is_aborted():
                logger.info(
                    f"Request {req.request_id} aborted by client, skipping update"
                )
                # close() may have set _aborted=True without setting a terminal status
                # (status still RUNNING).
                if not req.is_finished():
                    req.mark_canceled()
                continue

            req.generated_token_ids.append(token_id)
            pending_tokens = req.generated_token_ids[req._token_decode_offset :]
            delta = self.tokenizer.decode(pending_tokens)
            holds_back = bool(delta) and delta.endswith("\ufffd")

            last_committed_text = req.generated_text

            if not holds_back:
                req.generated_text = last_committed_text + delta
                req._token_decode_offset = len(req.generated_token_ids)

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
                        token_text = req.generated_text[req._text_output_offset :]
                        if token_text:
                            req._text_output_offset = len(req.generated_text)

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
        """Tokenize text to token IDs."""
        return self.tokenizer.encode(text)

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
        moe_ep_backend: str = "disabled",
        moe_ep_size: int = 1,
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
        use_mla: bool = False,
        weight_load_mode: str = "async",
        skip_load: bool = False,
        skip_legacy_moe: bool = False,
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
            use_mla: Whether to use DeepSeek V2 MLA attention when supported.
            weight_load_mode: Weight loading mode across tensor-parallel workers.
        """
        config = EngineConfig(
            model_path=model_path,
            device=device,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            moe_ep_backend=moe_ep_backend,
            moe_ep_size=moe_ep_size,
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
            use_mla=use_mla,
            weight_load_mode=weight_load_mode,
            skip_load=skip_load,
            skip_legacy_moe=skip_legacy_moe,
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
            mm_index_mappings = None
            if apply_chat_template:
                prompt = self.engine.apply_chat_template(
                    content, add_generation_prompt=True
                )

                mm_inputs = resolve_multimodal_inputs(content)

                processed_inputs = self.engine.process(
                    prompt,
                    mm_inputs["images"],
                    mm_inputs["videos"],
                    mm_inputs["audios"],
                    return_tensors="pt",
                )

                prompt_token_ids = processed_inputs.get("input_ids").flatten().tolist()
                mm_index_mappings = self.engine.processor.get_mm_token_index_list(
                    prompt_token_ids,
                    image_ids=mm_inputs["image_urls"],
                    video_ids=mm_inputs["video_urls"],
                    audio_ids=mm_inputs["audio_urls"],
                )
            else:
                prompt = content
                prompt_token_ids = self.engine.tokenize(prompt)

            req = InferenceRequest(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                processed_inputs=processed_inputs,
                mm_token_index_mappings=mm_index_mappings,
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


class AsyncLLMEngine:
    """Asynchronous LLM engine for server use with streaming support."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        moe_ep_backend: str = "disabled",
        moe_ep_size: int = 1,
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
        kv_transfer_config: Optional[KVTransferConfig] = None,
        use_mla: bool = False,
        weight_load_mode: str = "async",
        skip_legacy_moe: bool = False,
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
            kv_connector: KV connector type ('MooncakeConnector').
            kv_role: Role in KV connector ('kv_producer' or 'kv_consumer').
            kv_connector_extra_config: Extra config dict for KV connector.
            use_mla: Whether to use DeepSeek V2 MLA attention when supported.
            weight_load_mode: Weight loading mode across tensor-parallel workers.
        """
        config = EngineConfig(
            model_path=model_path,
            device=device,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            moe_ep_backend=moe_ep_backend,
            moe_ep_size=moe_ep_size,
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
            kv_transfer_config=kv_transfer_config,
            use_mla=use_mla,
            weight_load_mode=weight_load_mode,
            skip_legacy_moe=skip_legacy_moe,
        )
        self.engine = LLMEngine(config)
        self.config = config

        self._running = False
        self._step_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._healthy = True
        self._abort_queue: Optional[janus.Queue] = None

    def is_healthy(self) -> bool:
        return bool(self._healthy)

    def start(self):
        """Start the background inference loop."""
        if self._running:
            logger.warning("AsyncLLMEngine is already running")
            return

        self._loop = asyncio.get_running_loop()
        self._abort_queue = janus.Queue()
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

    def add_aborted_req(
        self,
        req: InferenceRequest,
        reason: FinishReason = FinishReason.CANCELED,
    ):
        """Submit an abort request from async side to the step thread.

        The step thread processes this in _drain_abort_queue() before each schedule().
        """
        if self._abort_queue is not None:
            self._abort_queue.sync_q.put((req, reason))

    def _drain_abort_queue(self):
        """Process all pending abort requests before each schedule() call.

        Runs in the step thread (sync context). Guarantees mark_*() is called
        before schedule() so is_finished() checks in waiting/running queue loops
        work correctly. Puts a final TokenOutput into the output queue to unblock
        stream_request when _stream_chat is still alive after abort.
        """
        if self._abort_queue is None:
            return
        while True:
            try:
                req, reason = self._abort_queue.sync_q.get_nowait()
            except Exception:
                break

            if req.is_finished():
                continue

            if reason == FinishReason.CANCELED:
                req.mark_canceled()
            elif reason == FinishReason.TIMEOUT:
                req.mark_timeout()
            else:
                req.mark_failed(reason)

            # Put a final token to unblock stream_request.
            # If Starlette already cancelled _stream_chat, aclose() may have closed
            # the queue; put_nowait will raise and we silently ignore it.
            if req._output_queue is not None:
                final = TokenOutput(
                    request_id=req.request_id,
                    token_id=-1,
                    token_text="",
                    finished=True,
                    finish_reason=req.finish_reason,
                    generated_text=req.generated_text,
                )
                try:
                    req.output_queue.sync_q.put_nowait(final)
                except Exception:
                    pass

    def _step_loop(self):
        """Background loop that runs inference steps."""
        while self._running:
            try:
                self._drain_abort_queue()
                did_work, pending = self.engine.step()
                if not did_work:
                    time.sleep(0.003)
                elif pending:
                    self._loop.call_soon_threadsafe(self._batch_put, pending)
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

    def add_request(
        self,
        messages: Optional[List[dict]],
        apply_chat_template: bool = True,
        add_generation_prompt: bool = True,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        # For server use
        request_data: Optional[dict] = None,
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

        Returns:
            The created InferenceRequest object.
        """
        if request_id is None:
            request_id = f"cmpl-{uuid.uuid4().hex}"

        mm_index_mappings = None
        processed_inputs = None

        if prompt_token_ids is not None:
            prompt = self.engine.detokenize(prompt_token_ids)
        elif prompt is not None:
            prompt_token_ids = self.engine.tokenize(prompt)
        else:
            assert messages is not None, (
                "Either messages or prompt/prompt_token_ids must be provided"
            )

            assert apply_chat_template, (
                "apply_chat_template needs to be true for multi-role conversation"
            )

            prompt = self.engine.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt
            )

            mm_inputs = resolve_multimodal_inputs(messages)

            processed_inputs = self.engine.process(
                prompt,
                mm_inputs["images"],
                mm_inputs["videos"],
                mm_inputs["audios"],
                return_tensors="pt",
            )

            prompt_token_ids = processed_inputs.get("input_ids").flatten().tolist()
            mm_index_mappings = self.engine.processor.get_mm_token_index_list(
                prompt_token_ids,
                image_ids=mm_inputs["image_urls"],
                video_ids=mm_inputs["video_urls"],
                audio_ids=mm_inputs["audio_urls"],
            )

        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=self.config.max_tokens)
        elif sampling_params.max_tokens is None:
            sampling_params = sampling_params.clone()
            sampling_params.max_tokens = self.config.max_tokens

        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            processed_inputs=processed_inputs,
            mm_token_index_mappings=mm_index_mappings,
            sampling_params=sampling_params,
            eos_token_ids=self.engine.eos_token_ids,
            request_data=request_data,
        )

        if request_data and "kv_transfer_params" in request_data:
            kv_params = request_data["kv_transfer_params"]
            request.kv_transfer_params = kv_params

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
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> InferenceRequest:
        """Add a chat request to the engine.

        Args:
            messages: List of message dicts (chat conversation).
            sampling_params: Sampling parameters.
            request_id: Optional request ID.
            request_data: Optional request data dict.

        Returns:
            The created InferenceRequest object.
        """

        return self.add_request(
            messages=messages,
            apply_chat_template=True,
            add_generation_prompt=add_generation_prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            request_data=request_data,
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
        try:
            while True:
                try:
                    if request_timeout and time.time() - start > float(request_timeout):
                        logger.warning(
                            f"Request {request.request_id} exceeded request timeout of {request_timeout} seconds"
                        )
                        self.add_aborted_req(request, FinishReason.TIMEOUT)

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
                    logger.error(
                        f"Error while streaming request {request.request_id}: {e}"
                    )
                    break
        finally:
            # Unified cleanup point: runs whether the loop exits normally,
            # via exception, or via aclose() (GeneratorExit from Starlette).
            await request.close()
