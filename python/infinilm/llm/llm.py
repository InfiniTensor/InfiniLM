"""
LLM Engine - Main interface for LLM inference.

This module provides:
- LLM class for batch generation (offline use)
- AsyncLLM class for asynchronous streaming (server use)
"""

import asyncio
import logging
import os
import queue
import threading
import time
import uuid
from typing import AsyncIterator, List, Optional, Union

import janus

from infinilm.config.engine_config import EngineConfig
from infinilm.config.kv_transfer import KVTransferConfig
from infinilm.infer_engine import model_uses_mamba_cache, read_hf_config
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
from infinilm.processors.basic_llm_processor import BasicLLMProcessor

logger = logging.getLogger(__name__)


def _chat_tensor_mode(processor, has_multimodal_inputs: bool) -> str:
    """Use NumPy only for the built-in text processor's media-free inputs."""
    if (
        not has_multimodal_inputs
        and isinstance(processor, BasicLLMProcessor)
        and type(processor).__call__ is BasicLLMProcessor.__call__
    ):
        return "np"
    return "pt"


def _resolve_sampling_params(sampling_params, config):
    if sampling_params is None:
        return SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
        )
    if sampling_params.max_tokens is None:
        sampling_params = sampling_params.clone()
        sampling_params.max_tokens = config.max_tokens
    return sampling_params


def _finalize_aborted_request(
    req: InferenceRequest, reason: FinishReason, notification_errors=None
) -> bool:
    """Commit an abort reason and its final notification exactly once."""
    with req._terminal_lock:
        if req.is_finished():
            return False
        if reason == FinishReason.CANCELED:
            req.mark_canceled()
        elif reason == FinishReason.TIMEOUT:
            req.mark_timeout()
        else:
            req.mark_failed(reason)
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
            except Exception as error:
                if notification_errors is not None:
                    if len(notification_errors) < 4:
                        notification_errors.append((req.request_id, error))
                else:
                    logger.warning(
                        "Final abort output could not be delivered for %s: %s",
                        req.request_id,
                        error,
                    )
    return True


class LLMEngine:
    """Low-level LLM engine that handles inference execution."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._async_abort_owner = False
        hf_config = read_hf_config(config.model_path)
        has_mamba_cache = model_uses_mamba_cache(hf_config)
        if has_mamba_cache and config.enable_prefix_caching:
            model_type = hf_config["model_type"]
            raise RuntimeError(
                "Prefix caching is not supported for Mamba-cache model "
                f"{model_type!r} yet. Restart with "
                "--disable-prefix-caching."
            )

        if config.pipeline_parallel_size > 1 and config.pipeline_parallel_stage != 0:
            raise ValueError(
                "LLMEngine can only run pipeline stage 0; launch non-host nodes "
                "with inference_server.py --node-rank=N"
            )

        self.model_runner = ModelRunner(config)

        self.device = self.model_runner.device
        self.dtype = self.model_runner.dtype

        # Initialize processor
        self.processor = self.model_runner.processor
        self.tokenizer = self.processor.get_tokenizer()

        # Initialize KV cache based on cache type
        if config.cache_type == "static":
            self.scheduler = StaticScheduler(
                max_cache_len=config.max_cache_len,
                enable_prefix_caching=config.enable_prefix_caching,
                cache_capacity=config.static_cache_capacity,
                on_aborted_request=self._on_static_scheduler_abort,
            )
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
            model_engine = self.model_runner.model_engine
            llm_config = model_engine.hf_config
            if "text_config" in llm_config:
                llm_config = llm_config["text_config"]

            max_position_embeddings = llm_config.get(
                "max_position_embeddings", config.max_cache_len
            )
            num_mamba_cache_blocks = max(2, config.num_blocks // 4)

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
                has_mamba_cache=has_mamba_cache,
                num_mamba_cache_blocks=num_mamba_cache_blocks,
                enable_prefix_caching=config.enable_prefix_caching,
                nsa_owner_pool_epoch=(
                    model_engine.get_nsa_pool_epoch()
                    if model_engine.model_type == "videonsa"
                    else None
                ),
            )
            if model_engine.model_type == "videonsa":
                self.model_runner.bind_nsa_scheduler(self.scheduler)
            logger.info(f"Using Paged KV Cache with num_blocks={config.num_blocks}")
            if has_mamba_cache:
                logger.info(
                    "Using Mamba cache with num_blocks=%s, zero_state_index=0",
                    num_mamba_cache_blocks,
                )
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

    def close(self):
        self.model_runner.close()

    def _on_static_scheduler_abort(self, req: InferenceRequest) -> None:
        # The async queue owns both the terminal reason and final output.
        # A standalone synchronous engine has no such queue.
        if not self._async_abort_owner:
            _finalize_aborted_request(req, FinishReason.CANCELED)

    def step(self) -> bool:
        """Run one inference step.

        Returns:
            Whether work was scheduled or a KV transfer completed. Token
            outputs are published to request queues before step() returns.
        """
        # Schedule the next unit of work, which may be model execution,
        # connector control metadata, or both.
        scheduler_output = self.scheduler.schedule()
        if scheduler_output is None:
            return False

        requests = tuple(scheduler_output.scheduled_requests)
        request_ids = tuple(req.request_id for req in requests)
        computed_lengths = tuple(req.get_total_length() for req in requests)
        generations = tuple(req._static_cache_lease_generation for req in requests)
        physical_rows = (
            tuple(scheduler_output.physical_rows)
            if hasattr(scheduler_output, "physical_rows")
            else None
        )
        if (
            len({id(req) for req in requests}) != len(requests)
            or len(set(request_ids)) != len(request_ids)
            or scheduler_output.num_requests != len(requests)
        ):
            raise RuntimeError("scheduled batch has conflicting request identity")
        runner_output = self.model_runner.execute_model(scheduler_output)
        # Validate the entire returned batch before connector, KV, token or queue
        # consumption. Native writes/SPEC reservation commits already happened.
        if (
            tuple(runner_output.req_ids) != request_ids
            or (
                tuple(scheduler_output.physical_rows)
                if hasattr(scheduler_output, "physical_rows")
                else None
            )
            != physical_rows
            or len(scheduler_output.scheduled_requests) != len(requests)
            or any(
                actual is not expected
                for actual, expected in zip(
                    scheduler_output.scheduled_requests, requests
                )
            )
            or any(
                req.request_id != rid or req.get_total_length() != length
                for req, rid, length in zip(requests, request_ids, computed_lengths)
            )
        ):
            raise RuntimeError(
                "model output does not match its scheduled request snapshot"
            )
        sampled_token_ids = self._validate_sampled_token_ids(
            requests, runner_output.sampled_token_ids
        )
        self.scheduler.accept_model_output(requests, generations, physical_rows)
        self.scheduler.update_from_output(runner_output)
        self._update_requests(requests, sampled_token_ids)

        # Return False (no immediate work) only when no requests were scheduled
        # and no KV transfers completed in this step.
        if not scheduler_output.scheduled_requests:
            if not runner_output.kv_connector_output or (
                not getattr(runner_output.kv_connector_output, "finished_sending", None)
                and not getattr(
                    runner_output.kv_connector_output, "finished_recving", None
                )
            ):
                return False

        return True

    @staticmethod
    def _validate_sampled_token_ids(requests, sampled_token_ids):
        """Freeze all rows before any returned-output consumer mutation."""
        if not isinstance(sampled_token_ids, (list, tuple)):
            raise RuntimeError("model output requires a token row sequence")
        if len(requests) != len(sampled_token_ids):
            raise RuntimeError(
                "model output count does not match the scheduled request count: "
                f"requests={len(requests)}, outputs={len(sampled_token_ids)}"
            )
        rows = []
        for row in sampled_token_ids:
            if isinstance(row, (list, tuple)):
                # Empty rows are legal on the existing speculative remaining path.
                frozen = tuple(row)
                if any(type(token) is not int or token < 0 for token in frozen):
                    raise RuntimeError(
                        "model output requires non-negative integer token IDs"
                    )
                rows.append(frozen)
            elif type(row) is int and row >= 0:
                rows.append(row)
            else:
                raise RuntimeError(
                    "model output requires non-negative integer token IDs"
                )
        return tuple(rows)

    def _update_requests(
        self,
        requests: List[InferenceRequest],
        sampled_token_ids: list[int | list[int]],
    ) -> None:
        """Apply sampled tokens and publish their target-model KV boundary."""
        sampled_token_ids = self._validate_sampled_token_ids(
            requests, sampled_token_ids
        )
        for req, token_ids in zip(requests, sampled_token_ids):
            # The model successfully consumed the request's current logical tokens.
            # Commit this boundary before observing a concurrent client abort.
            pre_output_computed_tokens = req.get_total_length()
            self.scheduler.commit_computed_tokens(req, pre_output_computed_tokens)

            if req.is_aborted():
                logger.info(
                    f"Request {req.request_id} aborted by client, skipping update"
                )
                if not self._async_abort_owner:
                    _finalize_aborted_request(req, FinishReason.CANCELED)
                continue

            if isinstance(token_ids, (list, tuple)):
                num_computed_output_tokens = max(len(token_ids) - 1, 0)
            else:
                num_computed_output_tokens = 0
                token_ids = [token_ids]

            num_appended_tokens = 0
            for token_id in token_ids:
                # Decode a prospective token without changing request state.
                # Abort finalization may run while tokenizer.decode executes.
                with req._terminal_lock:
                    if req.is_finished() or req.is_aborted():
                        break
                    pending_tokens = list(
                        req.generated_token_ids[req._token_decode_offset :]
                    ) + [token_id]
                delta = self.tokenizer.decode(pending_tokens)
                holds_back = bool(delta) and delta.endswith("\ufffd")

                # Token, text, finish reason, and output form one commit.
                # A concurrent abort wins if registered before this lock;
                # otherwise the normal final output wins.
                with req._terminal_lock:
                    if req.is_finished() or req.is_aborted():
                        logger.info(
                            "Request %s aborted before publishing token", req.request_id
                        )
                        break
                    req.append_generated_token_id(token_id)
                    num_appended_tokens += 1
                    last_committed_text = req.generated_text
                    if not holds_back:
                        req.generated_text = last_committed_text + delta
                        req._token_decode_offset = len(req.generated_token_ids)
                    is_finished = self._check_request_finished(req, token_id)
                    if is_finished and req.finish_reason == FinishReason.EOS_TOKEN:
                        req.generated_text = last_committed_text
                    # vLLM-style replacement character handling is primarily
                    # relevant for streaming. Offline requests have no queue.
                    if req._output_queue is None:
                        if is_finished:
                            req.mark_finished(req.finish_reason)
                    else:
                        if holds_back and not is_finished:
                            token_text = ""
                        else:
                            token_text = req.generated_text[req._text_output_offset :]
                            if token_text:
                                req._text_output_offset = len(req.generated_text)

                        output = TokenOutput(
                            request_id=req.request_id,
                            token_id=token_id,
                            token_text=token_text,
                            finished=is_finished,
                            finish_reason=req.finish_reason if is_finished else None,
                            generated_text=req.generated_text,
                        )
                        # Queue delivery and terminal state share one commit
                        # boundary. A failed put must stop this step, not leave
                        # a FINISHED request without its final notification.
                        req.output_queue.sync_q.put_nowait(output)
                        if is_finished:
                            req.mark_finished(req.finish_reason)

            post_output_computed_tokens = pre_output_computed_tokens + min(
                num_appended_tokens,
                num_computed_output_tokens,
            )
            if post_output_computed_tokens > pre_output_computed_tokens:
                self.scheduler.commit_computed_tokens(req, post_output_computed_tokens)

        self.scheduler.complete_requests(requests)

    def _check_request_finished(self, req: InferenceRequest, token_id: int) -> bool:
        """Check if request generation is finished."""
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

        max_tokens = req.sampling_params.max_tokens
        if max_tokens and req.get_num_generated_tokens() >= max_tokens:
            req.finish_reason = FinishReason.LENGTH
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
        draft_model_path: Optional[str] = None,
        num_draft_tokens: int = 4,
        device: str = "cuda",
        dtype: Optional[str] = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        pipeline_parallel_stage: int = 0,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
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
        pre_transpose: bool = False,
        weight_load_mode: str = "async",
        skip_load: bool = False,
        enable_prefix_caching: bool = True,
        static_cache_capacity: int = 1,
    ):
        """Initialize LLM.

        Args:
            model_path: Path to the model directory.
            device: Device type ('cpu', 'cuda', 'mlu', 'moore').
            dtype: Data type ('float16', 'bfloat16', 'float32').
            tensor_parallel_size: Number of devices for tensor parallelism.
            cache_type: Cache type ('paged' or 'static').
            max_batch_size: Maximum batch size (only for paged cache).
            static_cache_capacity: Static KV row capacity; separate prefills and batched decode.
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
            draft_model_path=draft_model_path,
            num_draft_tokens=num_draft_tokens,
            device=device,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            pipeline_parallel_stage=pipeline_parallel_stage,
            master_addr=master_addr,
            master_port=master_port,
            moe_ep_backend=moe_ep_backend,
            moe_ep_size=moe_ep_size,
            cache_type=cache_type,
            max_batch_size=max_batch_size,
            static_cache_capacity=static_cache_capacity,
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
            pre_transpose=pre_transpose,
            weight_load_mode=weight_load_mode,
            skip_load=skip_load,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.engine = LLMEngine(config)
        self.config = config

    def close(self):
        self.engine.close()

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

        sampling_params = _resolve_sampling_params(sampling_params, self.config)

        requests = []
        for content in contents:
            request_id = f"cmpl-{uuid.uuid4().hex}"
            processed_inputs = None
            mm_index_mappings = None
            has_multimodal_inputs = False
            if apply_chat_template:
                prompt = self.engine.apply_chat_template(
                    content, add_generation_prompt=True
                )

                mm_inputs = resolve_multimodal_inputs(content)

                has_multimodal_inputs = any(
                    mm_inputs[key] for key in ("images", "videos", "audios")
                )
                processed_inputs = self.engine.process(
                    prompt,
                    mm_inputs["images"],
                    mm_inputs["videos"],
                    mm_inputs["audios"],
                    return_tensors=_chat_tensor_mode(
                        self.engine.processor, has_multimodal_inputs
                    ),
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
                has_multimodal_inputs=has_multimodal_inputs,
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
        draft_model_path: Optional[str] = None,
        num_draft_tokens: int = 4,
        device: str = "cuda",
        dtype: Optional[str] = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        pipeline_parallel_stage: int = 0,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
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
        pre_transpose: bool = False,
        skip_load: bool = False,
        weight_load_mode: str = "async",
        enable_prefix_caching: bool = True,
        static_cache_capacity: int = 1,
    ):
        """Initialize AsyncLLMEngine.

        Args:
            model_path: Path to the model directory.
            device: Device type ('cpu', 'cuda', 'mlu', 'moore').
            dtype: Data type ('float16', 'bfloat16', 'float32').
            tensor_parallel_size: Number of devices for tensor parallelism.
            cache_type: Cache type ('paged' or 'static').
            max_batch_size: Maximum batch size (only for paged cache).
            static_cache_capacity: Static KV row capacity; separate prefills and batched decode.
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
            skip_load: Whether to skip loading model weights.
            weight_load_mode: Weight loading mode across tensor-parallel workers.
        """
        config = EngineConfig(
            model_path=model_path,
            draft_model_path=draft_model_path,
            num_draft_tokens=num_draft_tokens,
            device=device,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            pipeline_parallel_stage=pipeline_parallel_stage,
            master_addr=master_addr,
            master_port=master_port,
            moe_ep_backend=moe_ep_backend,
            moe_ep_size=moe_ep_size,
            cache_type=cache_type,
            max_batch_size=max_batch_size,
            static_cache_capacity=static_cache_capacity,
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
            pre_transpose=pre_transpose,
            skip_load=skip_load,
            weight_load_mode=weight_load_mode,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.engine = LLMEngine(config)
        self.engine._async_abort_owner = True
        self.config = config

        self._running = False
        self._step_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._healthy = True
        self._abort_queue: Optional[janus.Queue] = None
        self._stopped = False
        self._stop_lock = threading.Lock()
        self._stop_requested = False
        self._stop_drain_complete = False
        self._stop_drain_error: Optional[str] = None

    def is_healthy(self) -> bool:
        return bool(self._healthy)

    def start(self):
        """Start the background inference loop."""
        with self._stop_lock:
            if self._stop_requested or self._stopped:
                raise RuntimeError(
                    "AsyncLLMEngine cannot restart after stop or during a pending stop"
                )
            if self._running:
                logger.warning("AsyncLLMEngine is already running")
                return
            if self._step_thread is not None:
                raise RuntimeError(
                    "AsyncLLMEngine cannot restart after stop or during a pending stop"
                )

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
        with self._stop_lock:
            if self._stopped:
                return
            if not self._running and self._step_thread is None:
                # Before start there cannot be accepted requests. Closing is
                # safe without a consumer thread and must still release native.
                self._stop_requested = True
                self.engine.close()
                self._stopped = True
                logger.info("AsyncLLMEngine stopped before start")
                return
            # The step thread owns scheduler resources. Keep it running until
            # it drains requests; stop() only requests that transition.
            self._stop_requested = True
            step_thread = self._step_thread

        if step_thread:
            if step_thread is threading.current_thread():
                raise RuntimeError(
                    "AsyncLLMEngine stop cannot join its own step thread; "
                    "native engine remains owned"
                )
            step_thread.join(timeout=5)
            if step_thread.is_alive():
                raise RuntimeError(
                    "AsyncLLMEngine step thread is still running; native engine "
                    "remains owned and stop() may be retried"
                )
        with self._stop_lock:
            if self._stopped:
                return
            if not self._stop_drain_complete:
                raise RuntimeError(
                    "AsyncLLMEngine shutdown drain did not complete; native engine "
                    f"remains owned: {self._stop_drain_error or 'step thread exited early'}"
                )
            self.engine.close()
            self._step_thread = None
            self._stopped = True
        logger.info("AsyncLLMEngine stopped")

    def _submit_request(self, request: InferenceRequest) -> None:
        """Linearize queue submission with stop, including tokenizer races."""
        with self._stop_lock:
            if (
                not self._running
                or self._step_thread is None
                or self._stop_requested
                or self._stopped
                or not self._healthy
            ):
                raise RuntimeError(
                    "AsyncLLMEngine is not running or stopping; request not submitted"
                )
            self.engine.add_request(request)

    def add_aborted_req(
        self,
        req: InferenceRequest,
        reason: FinishReason = FinishReason.CANCELED,
    ):
        """Submit an abort request from async side to the step thread.

        The step thread processes this in _drain_abort_queue() before each schedule().
        """
        with self._stop_lock:
            if self._abort_queue is None:
                return
            with req._terminal_lock:
                if req.is_finished() or getattr(req, "_abort_enqueued", False):
                    return
                if not self._healthy:
                    # No consumer remains after failure. Notification does not
                    # retire any scheduler/native resource owned by this request.
                    _finalize_aborted_request(req, FinishReason.ERROR)
                    return
                # close() may have set _aborted already. Only a successful queue
                # insertion owns the terminal reason; the first insertion wins.
                req.abort()
                self._abort_queue.sync_q.put_nowait((req, reason))
                req._abort_enqueued = True

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

            _finalize_aborted_request(req, reason)

    def _step_loop(self):
        """Background loop that runs inference steps."""
        while self._running:
            try:
                self._drain_abort_queue()
                if self._stop_requested:
                    self.engine.scheduler.shutdown_drain(
                        lambda req: _finalize_aborted_request(
                            req, FinishReason.CANCELED
                        )
                    )
                    self._stop_drain_complete = True
                    self._running = False
                    break
                did_work = self.engine.step()
                if not did_work:
                    time.sleep(0.003)
            except Exception as e:
                notification_errors = []
                with self._stop_lock:
                    self._healthy = False
                    self._running = False
                    if self._stop_drain_error is None:
                        self._stop_drain_error = repr(e)
                    seen = set()

                    def notify(req, reason=FinishReason.ERROR):
                        if id(req) in seen:
                            return
                        seen.add(id(req))
                        try:
                            _finalize_aborted_request(
                                req, reason, notification_errors=notification_errors
                            )
                        except Exception as notification_error:
                            if len(notification_errors) < 4:
                                notification_errors.append(
                                    (req.request_id, notification_error)
                                )

                    # An abort accepted before failure owns its reason. Hold
                    # stop -> terminal locks so late aborts cannot race this.
                    try:
                        if self._abort_queue is not None:
                            while True:
                                try:
                                    req, reason = self._abort_queue.sync_q.get_nowait()
                                except queue.Empty:
                                    break
                                notify(req, reason)
                    except Exception as notification_error:
                        if len(notification_errors) < 4:
                            notification_errors.append(
                                ("abort queue", notification_error)
                            )
                    try:
                        self.engine.scheduler.notify_failed_requests(notify)
                    except Exception as notification_error:
                        if len(notification_errors) < 4:
                            notification_errors.append(
                                ("scheduler enumeration", notification_error)
                            )
                    # Unknown native completion remains owned: no drain, retry,
                    # cache retirement, close, or successful-drain declaration.
                try:
                    logger.error(
                        "Error in step loop: %s; notification errors (first four): %s",
                        e,
                        notification_errors,
                        exc_info=True,
                    )
                except Exception:
                    # Logging cannot undo the fault latch or request notifications.
                    pass
                break

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
        chat_template_kwargs: Optional[dict] = None,
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
        # Reject the common pre-start/stopping case before constructing an
        # output queue. _submit_request repeats this under the same lock after
        # tokenization, closing a concurrent stop race.
        with self._stop_lock:
            if (
                not self._running
                or self._step_thread is None
                or self._stop_requested
                or self._stopped
                or not self._healthy
            ):
                raise RuntimeError(
                    "AsyncLLMEngine is not running or stopping; request not submitted"
                )
        if request_id is None:
            request_id = f"cmpl-{uuid.uuid4().hex}"

        mm_index_mappings = None
        processed_inputs = None
        has_multimodal_inputs = False

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
                messages,
                add_generation_prompt=add_generation_prompt,
                chat_template_kwargs=chat_template_kwargs,
            )

            mm_inputs = resolve_multimodal_inputs(messages)

            has_multimodal_inputs = any(
                mm_inputs[key] for key in ("images", "videos", "audios")
            )
            processed_inputs = self.engine.process(
                prompt,
                mm_inputs["images"],
                mm_inputs["videos"],
                mm_inputs["audios"],
                return_tensors=_chat_tensor_mode(
                    self.engine.processor, has_multimodal_inputs
                ),
            )

            prompt_token_ids = processed_inputs.get("input_ids").flatten().tolist()
            mm_index_mappings = self.engine.processor.get_mm_token_index_list(
                prompt_token_ids,
                image_ids=mm_inputs["image_urls"],
                video_ids=mm_inputs["video_urls"],
                audio_ids=mm_inputs["audio_urls"],
            )

        sampling_params = _resolve_sampling_params(sampling_params, self.config)

        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            processed_inputs=processed_inputs,
            mm_token_index_mappings=mm_index_mappings,
            has_multimodal_inputs=has_multimodal_inputs,
            sampling_params=sampling_params,
            eos_token_ids=self.engine.eos_token_ids,
            request_data=request_data,
        )

        if request_data and "kv_transfer_params" in request_data:
            kv_params = request_data["kv_transfer_params"]
            request.kv_transfer_params = kv_params

        # Initialize output queue for streaming
        _ = request.output_queue

        try:
            self._submit_request(request)
        except BaseException:
            # A stop that wins during tokenization must not leave this newly
            # constructed, never-submitted janus queue alive.
            request.output_queue.close()
            raise
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
            chat_template_kwargs=kwargs.get("chat_template_kwargs"),
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
        timeout_submitted = False
        exit_reason = FinishReason.CANCELED
        try:
            while True:
                try:
                    if (
                        request_timeout
                        and not timeout_submitted
                        and time.time() - start > float(request_timeout)
                    ):
                        logger.warning(
                            f"Request {request.request_id} exceeded request timeout of {request_timeout} seconds"
                        )
                        self.add_aborted_req(request, FinishReason.TIMEOUT)
                        timeout_submitted = True

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
                    if (
                        request_timeout
                        and not timeout_submitted
                        and time.time() - start > float(request_timeout)
                    ):
                        self.add_aborted_req(request, FinishReason.TIMEOUT)
                        timeout_submitted = True
                    if not self._healthy or not self._running:
                        # The step thread may have committed a final item to
                        # Janus immediately before this timed get was canceled.
                        # Consume that item directly before treating its exit
                        # as a missing finalization; close() would discard it.
                        try:
                            pending = request.output_queue.async_q.get_nowait()
                        except asyncio.QueueEmpty:
                            _finalize_aborted_request(
                                request,
                                FinishReason.TIMEOUT
                                if timeout_submitted
                                else FinishReason.ERROR,
                            )
                            raise RuntimeError(
                                "abort finalization unavailable: step thread stopped"
                            )
                        request.output_queue.async_q.task_done()
                        yield pending
                        if pending.finished:
                            break
                        continue
                    if request.is_aborted():
                        # The step thread still owns the terminal reason and
                        # final output. It may be inside a long native forward;
                        # do not invent a final token with finish_reason=None.
                        continue
                    continue
                except Exception as e:
                    logger.error(
                        f"Error while streaming request {request.request_id}: {e}"
                    )
                    raise
        except (GeneratorExit, asyncio.CancelledError):
            raise
        except BaseException:
            exit_reason = FinishReason.ERROR
            raise
        finally:
            # Unified cleanup point: runs whether the loop exits normally,
            # via exception, or via aclose() (GeneratorExit from Starlette).
            # A direct API iterator has no HTTP wrapper to submit its abort.
            if not request.is_finished() and not timeout_submitted:
                self.add_aborted_req(request, exit_reason)
            await request.close()
