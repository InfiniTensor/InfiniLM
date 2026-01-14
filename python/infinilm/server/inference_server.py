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

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from infinilm.llm import AsyncLLMEngine, SamplingParams, FinishReason

logger = logging.getLogger(__name__)

DEFAULT_STREAM_TIMEOUT = 100.0
DEFAULT_REQUEST_TIMEOUT = 1000.0


def chunk_json(id_, content=None, role=None, finish_reason=None):
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
        "model": "jiuge",
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


class InferenceServer:
    """HTTP server for LLM inference."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        max_tokens: int = 4096,
        max_batch_size: int = 16,
        num_blocks: int = 8 * 1024,
        block_size: int = 16,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        """Initialize inference server.

        Args:
            model_path: Path to the model directory.
            device: Device type ('cpu', 'cuda', 'mlu', 'moore').
            dtype: Data type ('float16', 'bfloat16', 'float32').
            tensor_parallel_size: Number of devices for tensor parallelism.
            max_tokens: Default maximum tokens to generate.
            max_batch_size: Maximum batch size for inference.
            num_blocks: Number of KV cache blocks.
            block_size: Size of each KV cache block.
            temperature: Default sampling temperature.
            top_p: Default top-p sampling parameter.
            top_k: Default top-k sampling parameter.
            host: Server host address.
            port: Server port number.
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.host = host
        self.port = port

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
                max_batch_size=self.max_batch_size,
                max_tokens=self.max_tokens,
                num_blocks=self.num_blocks,
                block_size=self.block_size,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            self.engine.start()
            logger.info(f"Engine initialized with model at {self.model_path}")
            yield
            self.engine.stop()

        app = FastAPI(lifespan=lifespan)
        self._register_routes(app)
        return app

    def _register_routes(self, app: FastAPI):
        """Register API routes."""

        @app.post("/chat/completions")
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
            return {"status": "healthy"}

        @app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": "jiuge",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "infinilm",
                    }
                ],
            }

    def _build_sampling_params(self, data: dict) -> SamplingParams:
        """Build SamplingParams from request data."""
        return SamplingParams(
            temperature=data.get("temperature", self.temperature),
            top_p=data.get("top_p", self.top_p),
            top_k=data.get("top_k", self.top_k),
            max_tokens=data.get("max_tokens", self.max_tokens),
            stop=data.get("stop"),
        )

    async def _stream_chat(self, request_id: str, data: dict, http_request: Request):
        """Handle streaming chat request."""
        req = None
        start_time = time.time()

        try:
            messages = data.get("messages", [])
            sampling_params = self._build_sampling_params(data)

            req = self.engine.add_chat_request(
                messages=messages,
                sampling_params=sampling_params,
                request_id=request_id,
                request_data=data,
                http_request=http_request,
            )

            async for token_output in self.engine.stream_request(
                req, timeout=DEFAULT_STREAM_TIMEOUT
            ):
                # Check timeout
                if time.time() - start_time > DEFAULT_REQUEST_TIMEOUT:
                    logger.warning(
                        f"Request {request_id} timed out after {DEFAULT_REQUEST_TIMEOUT}s"
                    )
                    req.mark_timeout()
                    error_chunk = json.dumps(
                        chunk_json(
                            request_id,
                            content="[Request timeout]",
                            finish_reason="timeout",
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

                # Send token
                chunk = json.dumps(
                    chunk_json(request_id, content=token_output.token_text),
                    ensure_ascii=False,
                )
                yield f"data: {chunk}\n\n"

                if token_output.finished:
                    finish_reason = self._convert_finish_reason(
                        token_output.finish_reason
                    )
                    chunk = json.dumps(
                        chunk_json(request_id, finish_reason=finish_reason),
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
                    request_id, content=f"[Error: {str(e)}]", finish_reason="error"
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
        start_time = time.time()

        try:
            messages = data.get("messages", [])
            sampling_params = self._build_sampling_params(data)

            req = self.engine.add_chat_request(
                messages=messages,
                sampling_params=sampling_params,
                request_id=request_id,
                request_data=data,
                http_request=http_request,
            )

            # Collect all generated tokens
            output_text = ""
            async for token_output in self.engine.stream_request(
                req, timeout=DEFAULT_STREAM_TIMEOUT
            ):
                # Check timeout
                if time.time() - start_time > DEFAULT_REQUEST_TIMEOUT:
                    logger.warning(f"Request {request_id} timed out")
                    req.mark_timeout()
                    break

                # Check client disconnect
                if await http_request.is_disconnected():
                    logger.info(f"Client disconnected for request {request_id}")
                    req.mark_canceled()
                    break

                output_text += token_output.token_text

                if token_output.finished:
                    break

            output_text = output_text.strip()
            finish_reason = self._convert_finish_reason(req.finish_reason)

            response = chunk_json(
                request_id,
                content=output_text,
                role="assistant",
                finish_reason=finish_reason or "stop",
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
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=8, help="Maximum batch size"
    )
    parser.add_argument(
        "--num_blocks", type=int, default=8 * 1024, help="Number of blocks for KV cache"
    )
    parser.add_argument(
        "--block_size", type=int, default=16, help="Block size for KV cache"
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
    parser.add_argument("--metax", action="store_true", help="Use MetaX device")
    parser.add_argument("--moore", action="store_true", help="Use Moore device")
    parser.add_argument("--iluvatar", action="store_true", help="Use Iluvatar device")
    parser.add_argument("--cambricon", action="store_true", help="Use Cambricon device")
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
    elif args.metax:
        device = "cuda"
    elif args.moore:
        device = "moore"
    elif args.iluvatar:
        device = "cuda"
    elif args.cambricon:
        device = "mlu"
    else:
        print(
            "Usage: python infinilm.server.inference_server [--cpu | --nvidia | --metax | --moore | --iluvatar | --cambricon] "
            "--model_path=<path/to/model_dir> --max_tokens=MAX_TOKENS --max_batch_size=MAX_BATCH_SIZE"
            "\n"
            "Example: python infinilm.server.inference_server --nvidia --model_path=/data/shared/models/9G7B_MHA/ "
            "--max_tokens=100 --max_batch_size=32 --tp=1 --temperature=1.0 --top_p=0.8 --top_k=1"
        )
        sys.exit(1)

    server = InferenceServer(
        model_path=args.model_path,
        device=device,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        host=args.host,
        port=args.port,
    )
    server.start()


if __name__ == "__main__":
    main()
