"""
Inference Server - HTTP API server for LLM inference.
"""

from contextlib import asynccontextmanager
import sys
import time
import json
import uuid
import argparse
import signal
import uvicorn
import logging
import os
import asyncio
from infinilm.base_config import BaseConfig
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from infinilm.llm import AsyncLLMEngine, SamplingParams, FinishReason
from infinilm.llm.llm import normalize_chat_messages, _should_defer_tokenize_to_step_thread

logger = logging.getLogger(__name__)

DEFAULT_STREAM_TIMEOUT = 100.0
DEFAULT_REQUEST_TIMEOUT = 1000.0


def chunk_json(
    id_, content=None, role=None, finish_reason=None, model: str = "unknown"
):
    """Generate JSON chunk for streaming response."""
    delta = {}
    if content:
        delta["content"] = content
    if role:
        delta["role"] = role
    return {
        "id": id_,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "text": content,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


def completion_json(
    id_,
    content,
    role="assistant",
    finish_reason="stop",
    model: str = "unknown",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
):
    """Generate JSON response for non-streaming completion."""
    return {
        "id": id_,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": role,
                    "content": content,
                },
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


class InferenceServer:
    """HTTP server for LLM inference."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        cache_type: str = "paged",
        max_tokens: int = 4096,
        max_batch_size: int = 16,
        num_blocks: int = 512,
        block_size: int = 256,
        max_cache_len: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        host: str = "0.0.0.0",
        port: int = 8000,
        enable_graph: bool = False,
        attn_backend: str = "default",
        ignore_eos: bool = False,
    ):
        """Initialize inference server.

        Args:
            model_path: Path to the model directory.
            device: Device type ('cpu', 'cuda', 'mlu', 'moore').
            dtype: Data type ('float16', 'bfloat16', 'float32').
            tensor_parallel_size: Number of devices for tensor parallelism.
            cache_type: Cache type ('paged' or 'static').
            max_tokens: Default maximum tokens to generate.
            max_batch_size: Maximum batch size for inference (only for paged cache).
            num_blocks: Number of KV cache blocks (only for paged cache).
            block_size: Size of each KV cache block (only for paged cache).
            max_cache_len: Maximum sequence length (only for static cache).
            temperature: Default sampling temperature.
            top_p: Default top-p sampling parameter.
            top_k: Default top-k sampling parameter.
            host: Server host address.
            port: Server port number.
            enable_graph: Whether to enable graph compiling.
            attn_backend: Attention backend to use ('default', 'flash-attn').
        """
        self.model_path = model_path
        # vLLM-like served model id: directory name of model_path
        self.model_id = os.path.basename(os.path.normpath(model_path)) or "model"
        self.device = device
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.cache_type = cache_type
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_cache_len = max_cache_len
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.host = host
        self.port = port
        self.enable_graph = enable_graph
        self.attn_backend = attn_backend
        self.ignore_eos = ignore_eos

        self.engine: AsyncLLMEngine = None

    @staticmethod
    def _preflight_vllm_platform() -> None:
        """Touch vLLM platform/datasets before engine init (matches ASGI smoke order)."""
        try:
            from infinilm.compile.env import (
                prefill_compile_enabled,
                prefill_cudagraph_enabled,
                prefill_native_cg_enabled,
            )

            if prefill_native_cg_enabled():
                return
            if not (prefill_compile_enabled() and prefill_cudagraph_enabled()):
                return
        except ImportError:
            return
        try:
            from vllm.benchmarks.datasets import RandomDataset  # noqa: F401

            from vllm.platforms import current_platform

            logger.debug(
                "compiled prefill: vLLM preflight platform=%s",
                type(current_platform).__name__,
            )
        except ImportError as exc:
            logger.warning("compiled prefill: vLLM preflight skipped: %s", exc)

    async def _bootstrap_engine(self) -> None:
        """Initialize engine on the asyncio loop that will serve HTTP requests."""
        self._preflight_vllm_platform()
        self.engine = AsyncLLMEngine(
            model_path=self.model_path,
            device=self.device,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            cache_type=self.cache_type,
            max_batch_size=self.max_batch_size,
            max_tokens=self.max_tokens,
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            max_cache_len=self.max_cache_len,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            enable_graph=self.enable_graph,
            attn_backend=self.attn_backend,
        )
        me = self.engine.engine.model_engine
        if getattr(me, "_compiled_prefill_supported", lambda: False)():
            deadline = time.monotonic() + float(
                os.environ.get("INFINI_PREFILL_CAPTURE_WAIT_S", "900")
            )
            while not me._hybrid_prefill_ready():
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Compiled prefill did not finish before server ready deadline"
                    )
                await asyncio.sleep(2)
            self.engine.start()
            if not self.engine.wait_for_serving_capture(
                timeout=max(0.0, deadline - time.monotonic())
            ):
                raise RuntimeError(
                    "CUDAGraph capture did not finish on serving thread "
                    "before server ready deadline"
                )
            logger.info(
                "compiled prefill: server ready "
                "(serving-thread CUDAGraph capture done)"
            )
        else:
            self.engine.start()
        logger.info(f"Engine initialized with model at {self.model_path}")
        logger.info(f"  enable_graph: {self.enable_graph}")

    async def _serve(self) -> None:
        """Run engine + uvicorn on one asyncio loop (matches embedded/runtime smokes)."""
        await self._bootstrap_engine()
        app = self._create_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            loop="asyncio",
            log_level=os.environ.get("UVICORN_LOG_LEVEL", "info"),
        )
        server = uvicorn.Server(config)
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _request_shutdown() -> None:
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_shutdown)
            except NotImplementedError:
                pass

        serve_task = asyncio.create_task(server.serve())
        try:
            await stop_event.wait()
        finally:
            server.should_exit = True
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass

    def start(self):
        """Start the HTTP server."""
        logger.info(f"Starting API Server at {self.host}:{self.port}...")
        asyncio.run(self._serve())
        logger.info("Inference Server stopped")

    def _create_app(self):
        """Create FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Engine bootstrapped in ``_serve`` before uvicorn binds the port.
            yield
            if self.engine is not None:
                self.engine.stop()

        app = FastAPI(lifespan=lifespan)
        self._register_routes(app)
        return app

    def _register_routes(self, app: FastAPI):
        """Register API routes."""

        # OpenAI-compatible chat completions endpoint.
        # Support both legacy path and OpenAI-style /v1 prefix for proxy/router compatibility.
        @app.post("/chat/completions")
        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            try:
                data = await request.json()
                logger.debug(f"Received request data: {data}")
            except Exception as e:
                logger.error(f"Failed to parse request JSON: {e}")
                return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

            if not data.get("messages"):
                if not data.get("prompt"):
                    return JSONResponse(
                        content={"error": "No message provided"}, status_code=400
                    )
                else:
                    data["messages"] = [{"role": "user", "content": data.get("prompt")}]

            # Normalize on the asyncio thread only when not deferring to step thread.
            # vLLM bench openai-chat sends list-shaped ``content``; step-thread
            # normalize+tokenize avoids MetaX ATU on partial PIECEWISE replay.
            if not _should_defer_tokenize_to_step_thread():
                data["messages"] = normalize_chat_messages(data.get("messages", []))
            else:
                data["messages"] = data.get("messages", [])

            stream = data.get("stream", False)
            request_id = f"cmpl-{uuid.uuid4().hex}"

            from infinilm.torch_llama.kv_paged import ensure_hybrid_prefill_gpu_context

            ensure_hybrid_prefill_gpu_context()
            if stream:
                return StreamingResponse(
                    self._stream_chat(request_id, data, request),
                    media_type="text/event-stream",
                )
            else:
                response = await self._chat(request_id, data, request)
                if isinstance(response, JSONResponse):
                    return response
                return JSONResponse(content=response)

        @app.get("/health")
        async def health():
            # Expose engine health so babysitter/registry can treat backend as unhealthy.
            if (
                self.engine is not None
                and hasattr(self.engine, "is_healthy")
                and not self.engine.is_healthy()
            ):
                return JSONResponse(content={"status": "unhealthy"}, status_code=503)
            return {"status": "healthy"}

        def _models_payload():
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "infinilm",
                    }
                ],
            }

        # Support both /v1/models (OpenAI) and /models (common legacy) for compatibility.
        @app.get("/v1/models")
        async def list_models():
            return _models_payload()

        @app.get("/models")
        async def list_models_legacy():
            return _models_payload()

    def _normalize_messages(self, messages: list) -> list:
        """Normalize messages to handle multimodal content (list format)."""
        return normalize_chat_messages(messages)

    def _build_sampling_params(self, data: dict) -> SamplingParams:
        """Build SamplingParams from request data."""
        # Support both:
        # - top-level OpenAI-ish fields: temperature/top_p/top_k/max_tokens/stop
        # - nested dict: sampling_params: { ... }
        sp = data.get("sampling_params") or {}
        if not isinstance(sp, dict):
            sp = {}

        def pick(key: str, default):
            # Priority: explicit top-level field > nested sampling_params > server default
            if key in data and data.get(key) is not None:
                return data.get(key)
            if key in sp and sp.get(key) is not None:
                return sp.get(key)
            return default

        # Accept OpenAI / vLLM bench aliases (bench sends max_completion_tokens).
        max_tokens = pick("max_tokens", None)
        if max_tokens is None:
            max_tokens = pick("max_completion_tokens", None)
        if max_tokens is None:
            max_tokens = pick("max_new_tokens", None)
        if max_tokens is None:
            max_tokens = self.max_tokens

        stop = pick("stop", None)
        if isinstance(stop, str):
            stop = [stop]

        return SamplingParams(
            temperature=float(pick("temperature", self.temperature)),
            top_p=float(pick("top_p", self.top_p)),
            top_k=int(pick("top_k", self.top_k)),
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            stop=stop,
            ignore_eos=self.ignore_eos,
        )

    async def _stream_chat(self, request_id: str, data: dict, http_request: Request):
        """Handle streaming chat request."""
        req = None

        try:
            messages = data.get("messages", [])
            sampling_params = self._build_sampling_params(data)

            req = self.engine.add_chat_request(
                messages=messages,
                sampling_params=sampling_params,
                request_id=request_id,
                request_data=data,
                http_request=http_request,
                add_generation_prompt=bool(data.get("add_generation_prompt", True)),
                chat_template_kwargs=data.get("chat_template_kwargs") or {},
            )

            async for token_output in self.engine.stream_request(
                req,
                timeout=DEFAULT_STREAM_TIMEOUT,
                request_timeout=DEFAULT_REQUEST_TIMEOUT,
            ):
                # Check client disconnect
                if await http_request.is_disconnected():
                    logger.info(f"Client disconnected for request {request_id}")
                    req.mark_canceled()
                    break

                # If stream_request enforces timeout, we can just surface the state to the client.
                if token_output.finish_reason == FinishReason.TIMEOUT:
                    logger.warning(
                        f"Request {request_id} timed out after {DEFAULT_REQUEST_TIMEOUT}s"
                    )
                    error_chunk = json.dumps(
                        chunk_json(
                            request_id,
                            content="[Request timeout]",
                            finish_reason="timeout",
                            model=self.model_id,
                        ),
                        ensure_ascii=False,
                    )
                    yield f"data: {error_chunk}\n\n"
                    break

                # Skip EOS token text for OpenAI API compatibility
                # Check if this token is an EOS token by comparing token_id with eos_token_ids
                eos_token_ids = self.engine.engine.eos_token_ids
                is_eos_token = (
                    not sampling_params.ignore_eos
                    and eos_token_ids
                    and token_output.token_id in eos_token_ids
                )

                if not is_eos_token and token_output.token_text:
                    # Send token
                    chunk = json.dumps(
                        chunk_json(
                            request_id,
                            content=token_output.token_text,
                            model=self.model_id,
                        ),
                        ensure_ascii=False,
                    )
                    yield f"data: {chunk}\n\n"

                if token_output.finished:
                    finish_reason = self._convert_finish_reason(
                        token_output.finish_reason
                    )
                    chunk = json.dumps(
                        chunk_json(
                            request_id, finish_reason=finish_reason, model=self.model_id
                        ),
                        ensure_ascii=False,
                    )
                    yield f"data: {chunk}\n\n"
                    break

        except asyncio.CancelledError:
            logger.info(f"Request {request_id} was cancelled")
            if req:
                req.mark_canceled()
            raise

        except Exception as e:
            logger.error(f"Stream error for {request_id}: {e}", exc_info=True)
            if req:
                req.mark_failed()
            error_chunk = json.dumps(
                chunk_json(
                    request_id,
                    content=f"[Error: {str(e)}]",
                    finish_reason="error",
                    model=self.model_id,
                ),
                ensure_ascii=False,
            )
            yield f"data: {error_chunk}\n\n"

        finally:
            if req and not req.is_finished():
                req.mark_canceled()
            if req:
                await req.close()
            yield "data: [DONE]\n\n"

    async def _chat(self, request_id: str, data: dict, http_request: Request):
        """Handle non-streaming chat request."""
        req = None

        try:
            messages = data.get("messages", [])
            sampling_params = self._build_sampling_params(data)

            req = self.engine.add_chat_request(
                messages=messages,
                sampling_params=sampling_params,
                request_id=request_id,
                request_data=data,
                http_request=http_request,
                add_generation_prompt=bool(data.get("add_generation_prompt", True)),
                chat_template_kwargs=data.get("chat_template_kwargs") or {},
            )

            # Collect all generated tokens
            output_text = ""
            async for token_output in self.engine.stream_request(
                req,
                timeout=DEFAULT_STREAM_TIMEOUT,
                request_timeout=DEFAULT_REQUEST_TIMEOUT,
            ):
                # Check client disconnect
                if await http_request.is_disconnected():
                    logger.info(f"Client disconnected for request {request_id}")
                    req.mark_canceled()
                    break

                # Request-level timeout is handled inside stream_request.
                if token_output.finish_reason == FinishReason.TIMEOUT:
                    logger.warning(f"Request {request_id} timed out")
                    break

                # Skip EOS token text for OpenAI API compatibility
                # Check if this token is an EOS token by comparing token_id with eos_token_ids
                eos_token_ids = self.engine.engine.eos_token_ids
                is_eos_token = eos_token_ids and token_output.token_id in eos_token_ids

                if not is_eos_token and token_output.token_text:
                    output_text += token_output.token_text

                if token_output.finished:
                    break

            output_text = output_text.strip()
            finish_reason = self._convert_finish_reason(req.finish_reason)

            response = completion_json(
                request_id,
                content=output_text,
                role="assistant",
                finish_reason=finish_reason or "stop",
                model=self.model_id,
                prompt_tokens=req.get_prompt_length(),
                completion_tokens=req.get_num_generated_tokens(),
                total_tokens=req.get_total_length(),
            )
            return response

        except asyncio.CancelledError:
            logger.info(f"Request {request_id} was cancelled")
            if req:
                req.mark_canceled()
            raise

        except Exception as e:
            logger.error(f"Chat error for {request_id}: {e}", exc_info=True)
            if req:
                req.mark_failed()
            return JSONResponse(content={"error": str(e)}, status_code=500)

        finally:
            if req and not req.is_finished():
                req.mark_canceled()
            if req:
                await req.close()

    def _convert_finish_reason(self, reason: FinishReason) -> str:
        """Convert FinishReason enum to string."""
        if reason is None:
            return None
        if reason in (FinishReason.EOS_TOKEN, FinishReason.STOP_STRING):
            return "stop"

        return reason.value


def setup_logging(log_level: str = "INFO"):
    """Configure logging system with proper formatting and handlers."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def main():
    cfg = BaseConfig()
    setup_logging(cfg.log_level)
    device = cfg.get_device_str(cfg.device)

    server = InferenceServer(
        model_path=cfg.model,
        device=device,
        dtype=cfg.dtype,
        tensor_parallel_size=cfg.tp,
        cache_type="paged" if cfg.enable_paged_attn else "static",
        max_tokens=cfg.max_new_tokens,
        max_batch_size=cfg.max_batch_size,
        num_blocks=cfg.num_blocks,
        block_size=cfg.block_size,
        max_cache_len=cfg.max_cache_len,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        host=cfg.host,
        port=cfg.port,
        enable_graph=cfg.enable_graph,
        attn_backend=cfg.attn,
        ignore_eos=cfg.ignore_eos,
    )
    server.start()


if __name__ == "__main__":
    main()
