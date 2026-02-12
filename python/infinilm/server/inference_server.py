"""
Inference Server - HTTP API server for LLM inference.
"""

from contextlib import asynccontextmanager
import sys
import time
import json
import uuid
import argparse
import uvicorn
import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from infinilm.llm import AsyncLLMEngine, SamplingParams, FinishReason

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
        num_blocks: int = 8 * 1024,
        block_size: int = 16,
        max_cache_len: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        host: str = "0.0.0.0",
        port: int = 8000,
        enable_graph: bool = False,
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

            # Normalize messages to handle multimodal content (list format)
            data["messages"] = self._normalize_messages(data.get("messages", []))

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

                # Check client disconnect
                if await http_request.is_disconnected():
                    logger.info(f"Client disconnected for request {request_id}")
                    req.mark_canceled()
                    break

                # Skip EOS token text for OpenAI API compatibility
                # Check if this token is an EOS token by comparing token_id with eos_token_ids
                eos_token_ids = self.engine.engine.eos_token_ids
                is_eos_token = eos_token_ids and token_output.token_id in eos_token_ids

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
                # Request-level timeout is handled inside stream_request.
                if token_output.finish_reason == FinishReason.TIMEOUT:
                    logger.warning(f"Request {request_id} timed out")
                    break

                # Check client disconnect
                if await http_request.is_disconnected():
                    logger.info(f"Client disconnected for request {request_id}")
                    req.mark_canceled()
                    break

                # Skip EOS token text for OpenAI API compatibility
                # Check if this token is an EOS token by comparing token_id with eos_token_ids
                eos_token_ids = self.engine.engine.eos_token_ids
                is_eos_token = eos_token_ids and token_output.token_id in eos_token_ids

                if not is_eos_token:
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="InfiniLM Inference Server")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model directory"
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    parser.add_argument(
        "--cache_type",
        type=str,
        default="paged",
        choices=["paged", "static"],
        help="Cache type: paged or static",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="Maximum batch size (paged cache only)",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=8 * 1024,
        help="Number of blocks for KV cache (paged cache only)",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Block size for KV cache (paged cache only)",
    )
    parser.add_argument(
        "--max_cache_len",
        type=int,
        default=4096,
        help="Maximum sequence length (static cache only)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.8, help="Top-p sampling parameter"
    )
    parser.add_argument("--top_k", type=int, default=1, help="Top-k sampling parameter")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--cpu", action="store_true", help="Use CPU")
    parser.add_argument("--nvidia", action="store_true", help="Use NVIDIA GPU")
    parser.add_argument("--qy", action="store_true", help="Use QY GPU")
    parser.add_argument("--metax", action="store_true", help="Use MetaX device")
    parser.add_argument("--moore", action="store_true", help="Use Moore device")
    parser.add_argument("--iluvatar", action="store_true", help="Use Iluvatar device")
    parser.add_argument("--cambricon", action="store_true", help="Use Cambricon device")
    parser.add_argument("--ali", action="store_true", help="Use Ali PPU device")
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        help="Enable graph compiling",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    setup_logging(args.log_level)

    if args.cpu:
        device = "cpu"
    elif args.nvidia:
        device = "cuda"
    elif args.qy:
        device = "cuda"
    elif args.metax:
        device = "cuda"
    elif args.moore:
        device = "musa"
    elif args.iluvatar:
        device = "cuda"
    elif args.cambricon:
        device = "mlu"
    elif args.ali:
        device = "cuda"
    else:
        print(
            "Usage: python infinilm.server.inference_server [--cpu | --nvidia | --qy | --metax | --moore | --iluvatar | --cambricon | --ali] "
            "--model_path=<path/to/model_dir> --max_tokens=MAX_TOKENS --max_batch_size=MAX_BATCH_SIZE"
            "\n"
            "Example: python infinilm.server.inference_server --nvidia --model_path=/data/shared/models/9G7B_MHA/ "
            "--max_tokens=100 --max_batch_size=32 --tp=1 --temperature=1.0 --top_p=0.8 --top_k=1"
            "\n"
            "Optional: --enable-paged-attn --enable-graph"
        )
        sys.exit(1)

    server = InferenceServer(
        model_path=args.model_path,
        device=device,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        cache_type=args.cache_type,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_cache_len=args.max_cache_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        host=args.host,
        port=args.port,
        enable_graph=args.enable_graph,
    )
    server.start()


if __name__ == "__main__":
    main()
