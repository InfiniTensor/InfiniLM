"""
Inference Server - HTTP API server for LLM inference.
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from infinilm.base_config import BaseConfig
from infinilm.config import KVTransferConfig
from infinilm.llm import AsyncLLMEngine, FinishReason, SamplingParams
from infinilm.moe_config import configure_moe_ep_backend

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
        moe_ep_backend: str = "disabled",
        moe_ep_size: int = 1,
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
        use_mla: bool = False,
        weight_load_mode: str = "async",
        ignore_eos: bool = False,
        kv_transfer_config: Optional[KVTransferConfig] = None,
    ):
        """Initialize inference server.

        Args:
            model_path: Path to the model directory.
            device: Device type ('cpu', 'cuda', 'mlu', 'moore').
            dtype: Data type ('float16', 'bfloat16', 'float32').
            tensor_parallel_size: Number of devices for tensor parallelism.
            moe_ep_backend: MoE expert-parallel backend.
            moe_ep_size: MoE expert-parallel size.
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
            use_mla: Whether to use DeepSeek V2 MLA attention when supported.
            weight_load_mode: Weight loading mode across tensor-parallel workers.
            ignore_eos: Whether to ignore EOS tokens during generation.
            kv_transfer_config: Optional configuration for the KV transfer mechanism.
        """
        self.model_path = model_path
        # vLLM-like served model id: directory name of model_path
        self.model_id = os.path.basename(os.path.normpath(model_path)) or "model"
        self.device = device
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.moe_ep_backend = moe_ep_backend
        self.moe_ep_size = moe_ep_size
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
        self.use_mla = use_mla
        self.weight_load_mode = weight_load_mode
        self.ignore_eos = ignore_eos
        self.kv_transfer_config = kv_transfer_config

        self.engine: AsyncLLMEngine = None

    def start(self):
        """Start the HTTP server."""
        app = self._create_app()
        logger.info(f"Starting API Server at {self.host}:{self.port}...")
        uvicorn.run(app, host=self.host, port=self.port)
        logger.info("Inference Server stopped")

    def _create_app(self):
        """Create FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self.engine = AsyncLLMEngine(
                model_path=self.model_path,
                device=self.device,
                dtype=self.dtype,
                tensor_parallel_size=self.tensor_parallel_size,
                moe_ep_backend=self.moe_ep_backend,
                moe_ep_size=self.moe_ep_size,
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
                use_mla=self.use_mla,
                weight_load_mode=self.weight_load_mode,
                kv_transfer_config=self.kv_transfer_config,
            )
            self.engine.start()
            logger.info(f"Engine initialized with model at {self.model_path}")
            logger.info(f"  enable_graph: {self.enable_graph}")
            yield
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
                # logger.debug(f"Received request data: {data}")
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

            # Normalize messages to handle multimodal content (list format)
            data["messages"] = data.get("messages", [])

            stream = data.get("stream", False)
            request_id = f"cmpl-{uuid.uuid4().hex}"

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
        """Normalize messages to handle multimodal content (list format).

        Converts content from list format [{"type": "text", "text": "..."}]
        to string format for chat template compatibility.
        """
        normalized = []
        for msg in messages:
            if not isinstance(msg, dict):
                normalized.append(msg)
                continue

            content = msg.get("content")
            if isinstance(content, list):
                # Extract text from multimodal content list
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text" and "text" in part:
                            text_parts.append(part["text"])
                        elif isinstance(part, str):
                            text_parts.append(part)
                    elif isinstance(part, str):
                        text_parts.append(part)
                # Join all text parts
                normalized_msg = msg.copy()
                normalized_msg["content"] = "".join(text_parts) if text_parts else ""
                normalized.append(normalized_msg)
            else:
                normalized.append(msg)

        return normalized

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

        # Accept common alias
        max_tokens = pick("max_tokens", self.max_tokens)
        if max_tokens is None:
            # Some clients use max_new_tokens
            max_tokens = pick("max_new_tokens", self.max_tokens)

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
        _abort_reason = FinishReason.CANCELED

        try:
            messages = data.get("messages", [])
            sampling_params = self._build_sampling_params(data)

            req = self.engine.add_chat_request(
                messages=messages,
                sampling_params=sampling_params,
                request_id=request_id,
                request_data=data,
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
                    break

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
            # Starlette cancelled us (client disconnected); stream_request will be
            # aclose()'d automatically via the async-for destructor.
            logger.info(f"Request {request_id} was cancelled")
            raise

        except Exception as e:
            logger.error(f"Stream error for {request_id}: {e}", exc_info=True)
            _abort_reason = FinishReason.ERROR
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
            # Unified abort: reason is ERROR if we got here via Exception, else CANCELED.
            # req.close() is handled by stream_request.finally.
            if req and not req.is_finished():
                self.engine.add_aborted_req(req, _abort_reason)
        yield "data: [DONE]\n\n"

    async def _chat(self, request_id: str, data: dict, http_request: Request):
        """Handle non-streaming chat request."""
        req = None
        _abort_reason = FinishReason.CANCELED

        try:
            messages = data.get("messages", [])
            sampling_params = self._build_sampling_params(data)

            req = self.engine.add_chat_request(
                messages=messages,
                sampling_params=sampling_params,
                request_id=request_id,
                request_data=data,
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
            raise

        except Exception as e:
            logger.error(f"Chat error for {request_id}: {e}", exc_info=True)
            _abort_reason = FinishReason.ERROR
            return JSONResponse(content={"error": str(e)}, status_code=500)

        finally:
            # Unified abort: reason is ERROR if we got here via Exception, else CANCELED.
            # req.close() is handled by stream_request.finally.
            if req and not req.is_finished():
                self.engine.add_aborted_req(req, _abort_reason)

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


def parse_kv_transfer_config(kv_transfer_config_str: str) -> KVTransferConfig:
    """Parse JSON string into KVTransferConfig."""
    kv_dict = json.loads(kv_transfer_config_str)
    if not isinstance(kv_dict, dict):
        raise ValueError("--kv-transfer-config must be a JSON object")

    return KVTransferConfig(
        kv_connector=kv_dict.get("kv_connector", None),
        engine_id=kv_dict.get("engine_id", None),
        kv_role=kv_dict.get("kv_role", None),
        kv_connector_extra_config=kv_dict.get("kv_connector_extra_config", None),
    )


def main():
    cfg = BaseConfig()
    setup_logging(cfg.log_level)
    device = cfg.get_device_str(cfg.device)

    kv_transfer_config = None
    if cfg.kv_transfer_config:
        kv_transfer_config = parse_kv_transfer_config(cfg.kv_transfer_config)

    moe_ep_backend, ep = configure_moe_ep_backend(
        cfg.tp, cfg.dp, cfg.ep, cfg.moe_ep_backend, cfg.model
    )
    logger.info(
        "MoE EP backend: %s  TP=%s  DP=%s  EP=%s",
        moe_ep_backend,
        cfg.tp,
        cfg.dp,
        ep,
    )

    server = InferenceServer(
        model_path=cfg.model,
        device=device,
        dtype=cfg.dtype,
        tensor_parallel_size=cfg.tp,
        moe_ep_backend=moe_ep_backend,
        moe_ep_size=ep,
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
        use_mla=cfg.use_mla,
        weight_load_mode=cfg.weight_load_mode,
        ignore_eos=cfg.ignore_eos,
        kv_transfer_config=kv_transfer_config,
    )
    server.start()


if __name__ == "__main__":
    main()
