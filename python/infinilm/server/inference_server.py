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
from fastapi.responses import JSONResponse, StreamingResponse, Response

from infinilm.llm import AsyncLLMEngine, SamplingParams, FinishReason

logger = logging.getLogger(__name__)

DEFAULT_STREAM_TIMEOUT = 100.0
DEFAULT_REQUEST_TIMEOUT = 1000.0


def chunk_json(id_, content=None, role=None, finish_reason=None, model: str = "unknown"):
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
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


def completion_json(
    id_: str,
    content: str,
    role: str,
    finish_reason: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict:
    """Generate JSON response for non-streaming chat completion (OpenAI-compatible format)."""
    response = {
        "id": id_,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": role,
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
    }

    # Add usage field if token counts are available
    if prompt_tokens > 0 or completion_tokens > 0:
        response["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    return response


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
        max_context_length: int = None,
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
            max_context_length: Maximum context length for input prompts. If None,
                uses the model's max_position_embeddings. If set, must be <= max_position_embeddings.
        """
        self.model_path = model_path
        # vLLM-like served model id: directory name of model_path
        self.model_id = os.path.basename(os.path.normpath(model_path)) or "model"
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
        self.max_context_length = max_context_length

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

    def _truncate_messages_if_needed(
        self, messages: list, max_tokens: int = None, reserve_tokens: int = 256
    ) -> list:
        """Truncate messages if they exceed the model's max context length.

        Args:
            messages: List of message dicts.
            max_tokens: Maximum tokens to generate (used to reserve space).
            reserve_tokens: Additional tokens to reserve for generation and safety margin.

        Returns:
            Truncated messages list if needed, original list otherwise.
        """
        if not self.engine or not self.engine.engine:
            return messages

        try:
            # Get max context length from model config
            model_config = self.engine.engine.model_engine.config
            model_max_position_embeddings = getattr(
                model_config, "max_position_embeddings", None
            )
            if model_max_position_embeddings is None:
                # Try to get from tokenizer config as fallback
                tokenizer = self.engine.engine.tokenizer
                if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length:
                    model_max_position_embeddings = tokenizer.model_max_length
                else:
                    # Default fallback
                    logger.warning(
                        "Could not determine max context length, using default 2048"
                    )
                    model_max_position_embeddings = 2048

            # Use server-level max_context_length if set, otherwise use model's max_position_embeddings
            if self.max_context_length is not None:
                if self.max_context_length > model_max_position_embeddings:
                    logger.warning(
                        f"Server max_context_length ({self.max_context_length}) exceeds "
                        f"model's max_position_embeddings ({model_max_position_embeddings}). "
                        f"Using model's max_position_embeddings instead."
                    )
                    max_context_len = model_max_position_embeddings
                else:
                    max_context_len = self.max_context_length
                    logger.debug(
                        f"Using server-level max_context_length: {max_context_len} "
                        f"(model max_position_embeddings: {model_max_position_embeddings})"
                    )
            else:
                max_context_len = model_max_position_embeddings

            # Calculate available length for prompt
            # Reserve space for generation tokens and safety margin
            if max_tokens is None:
                max_tokens = self.max_tokens
            available_len = max_context_len - max_tokens - reserve_tokens
            if available_len <= 0:
                available_len = max_context_len - reserve_tokens

            # Apply chat template to get prompt
            try:
                prompt = self.engine.engine.apply_chat_template(
                    messages, add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Failed to apply chat template for length check: {e}")
                return messages

            # Tokenize to check length
            tokenizer = self.engine.engine.tokenizer
            encoded = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(encoded)

            if prompt_len <= available_len:
                return messages

            # Prompt is too long, need to truncate
            logger.warning(
                f"Prompt length ({prompt_len}) exceeds available context length "
                f"({available_len}). Truncating messages..."
            )

            # Try to truncate by removing oldest non-system messages
            # Keep system message if present
            system_messages = [msg for msg in messages if msg.get("role") == "system"]
            non_system_messages = [msg for msg in messages if msg.get("role") != "system"]

            # Binary search for the right number of messages to keep
            # Start with keeping all system messages and last user/assistant pair
            truncated = system_messages.copy()
            if non_system_messages:
                # Keep the last message (usually user's current question)
                truncated.append(non_system_messages[-1])

                # Try adding more messages from the end until we hit the limit
                for i in range(len(non_system_messages) - 2, -1, -1):
                    test_messages = system_messages + non_system_messages[i:]
                    try:
                        test_prompt = self.engine.engine.apply_chat_template(
                            test_messages, add_generation_prompt=True
                        )
                        test_encoded = tokenizer.encode(
                            test_prompt, add_special_tokens=False
                        )
                        if len(test_encoded) <= available_len:
                            truncated = test_messages
                        else:
                            break
                    except Exception:
                        break

            # Final check - if still too long, remove more messages
            try:
                final_prompt = self.engine.engine.apply_chat_template(
                    truncated, add_generation_prompt=True
                )
                final_encoded = tokenizer.encode(final_prompt, add_special_tokens=False)
                final_len = len(final_encoded)
                if final_len > available_len:
                    # Still too long, keep only system message and last user message
                    if system_messages and non_system_messages:
                        truncated = system_messages + [non_system_messages[-1]]
                    elif non_system_messages:
                        truncated = [non_system_messages[-1]]
                    else:
                        truncated = system_messages

                    # Check again
                    final_prompt = self.engine.engine.apply_chat_template(
                        truncated, add_generation_prompt=True
                    )
                    final_encoded = tokenizer.encode(final_prompt, add_special_tokens=False)
                    final_len = len(final_encoded)
                    if final_len > available_len:
                        logger.warning(
                            f"Even minimal messages result in prompt length {final_len}, "
                            f"which exceeds available {available_len}. "
                            f"Model may crash or produce errors."
                        )
            except Exception as e:
                logger.warning(f"Error in final truncation check: {e}")

            removed_count = len(messages) - len(truncated)
            if removed_count > 0:
                # Get final length for logging
                try:
                    final_prompt_check = self.engine.engine.apply_chat_template(
                        truncated, add_generation_prompt=True
                    )
                    final_encoded_check = tokenizer.encode(
                        final_prompt_check, add_special_tokens=False
                    )
                    final_len_check = len(final_encoded_check)
                except Exception:
                    final_len_check = "unknown"
                logger.warning(
                    f"Removed {removed_count} message(s) to fit within context limit "
                    f"(original: {prompt_len} tokens, truncated: {final_len_check} tokens)"
                )

            return truncated

        except Exception as e:
            logger.error(f"Error during message truncation: {e}", exc_info=True)
            # Return original messages on error to avoid breaking requests
            return messages

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

            # Truncate messages if they exceed model's max context length
            max_tokens = data.get("max_tokens") or data.get("max_new_tokens") or self.max_tokens
            data["messages"] = self._truncate_messages_if_needed(
                data["messages"], max_tokens=max_tokens
            )

            stream = data.get("stream", False)
            request_id = f"cmpl-{uuid.uuid4().hex}"
            messages_info = data.get("messages", [])
            if messages_info and len(messages_info) > 0:
                first_msg_content = messages_info[0].get("content", "") if isinstance(messages_info[0], dict) else str(messages_info[0])
                logger.info(f"Received request {request_id} with {len(messages_info)} message(s), first message length={len(first_msg_content)}")
            else:
                logger.info(f"Received request {request_id} with empty messages")

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

        @app.get("/metrics")
        async def metrics():
            """Prometheus-compatible metrics endpoint."""
            # Return empty metrics for now to avoid 404
            # Can be extended with actual metrics later
            return Response(
                content="# InfiniLM Metrics\n# No metrics collected yet\n",
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )

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

        stop_token_ids = pick("stop_token_ids", None)
        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]

        return SamplingParams(
            temperature=float(pick("temperature", self.temperature)),
            top_p=float(pick("top_p", self.top_p)),
            top_k=int(pick("top_k", self.top_k)),
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            stop=stop,
            stop_token_ids=stop_token_ids,
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

                # Send token
                chunk = json.dumps(
                    chunk_json(
                        request_id, content=token_output.token_text, model=self.model_id
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
                    elapsed_time = time.time() - start_time
                    generated_tokens = req.get_num_generated_tokens() if req else 0
                    logger.info(
                        f"Stream completed for {request_id}: "
                        f"finish_reason={finish_reason}, "
                        f"tokens={generated_tokens}, "
                        f"duration={elapsed_time:.2f}s"
                    )
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
            elapsed_time = time.time() - start_time
            logger.info(
                f"Stream ended for {request_id}: "
                f"total_duration={elapsed_time:.2f}s"
            )
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

                output_text += token_output.token_text

                if token_output.finished:
                    break

            output_text = output_text.strip()
            finish_reason = self._convert_finish_reason(req.finish_reason)

            # Get token counts from request
            prompt_tokens = req.get_prompt_length() if req else 0
            completion_tokens = req.get_num_generated_tokens() if req else 0

            elapsed_time = time.time() - start_time
            logger.info(
                f"Non-streaming request completed for {request_id}: "
                f"finish_reason={finish_reason or 'stop'}, "
                f"prompt_tokens={prompt_tokens}, "
                f"completion_tokens={completion_tokens}, "
                f"duration={elapsed_time:.2f}s"
            )

            response = completion_json(
                id_=request_id,
                content=output_text,
                role="assistant",
                finish_reason=finish_reason or "stop",
                model=self.model_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
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
            elapsed_time = time.time() - start_time
            logger.info(
                f"Non-streaming request ended for {request_id}: "
                f"total_duration={elapsed_time:.2f}s"
            )

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
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=None,
        help="Maximum context length for input prompts. If not set, uses model's max_position_embeddings.",
    )
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
        max_context_length=args.max_context_length,
    )
    server.start()


if __name__ == "__main__":
    main()
