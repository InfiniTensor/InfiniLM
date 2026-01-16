"""
Inference Server - Main server controlling all components and the inference loop.
"""

from contextlib import asynccontextmanager
import sys
import time
import json
import uuid
import threading
import argparse
import uvicorn
import asyncio
import logging
from infinilm.core.scheduler import Scheduler
from infinilm.core.request import RequestStatus, RequestOutput, InferenceRequest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from transformers import AutoTokenizer
from tokenizers import decoders as _dec
from infinilm.cache.cache import PagedKVCacheConfig, StaticKVCacheConfig
from infinilm.modeling_utils import load_model_state_dict_by_file
import infinicore

logger = logging.getLogger(__name__)

DEFAULT_STREAM_TIMEOUT = 10.0
DEFAULT_REQUEST_TIMEOUT = 100.0


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
    def __init__(
        self,
        model_path,
        device,
        dtype,
        ndev,
        max_tokens,
        max_batch_size,
        backend,
        num_blocks: int = 8 * 1024,
        block_size: int = 16,
        temperature=1.0,
        top_p=0.8,
        top_k=1,
        cache_type: str = "paged",
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.ndev = ndev
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.backend = backend

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.cache_type = cache_type
        self.current_static_cache_batch_size = None  # Track current batch size for static cache

        self.running = False
        self.step_thread = None

    def start(self):
        app = self._create_app()
        logger.info("Starting API Server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
        logger.info("Inference Server stopped")

    def _create_app(self):
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Initialize resources on startup
            self.engine = InferEngine(
                model_path=self.model_path,
                device=self.device,
                distributed_config=DistConfig(self.ndev),
            )
            load_model_state_dict_by_file(
                self.engine, self.model_path, dtype=self.engine.config.dtype
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            logger.info("Built engine and tokenizer")
            if "llama" in self.engine.config.model_type:
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

            # Create cache config based on cache_type
            if self.cache_type == "paged":
                cache_config = PagedKVCacheConfig(
                    num_blocks=self.num_blocks, block_size=self.block_size
                )
            elif self.cache_type == "static":
                # For static cache, initialize with batch_size=1
                # The cache will be reset dynamically if batch size changes
                # max_cache_len should accommodate max_tokens
                max_cache_len = self.max_tokens
                cache_config = StaticKVCacheConfig(
                    max_batch_size=1, max_cache_len=max_cache_len
                )
            else:
                raise ValueError(f"Unknown cache_type: {self.cache_type}. Must be 'paged' or 'static'")

            self.engine.reset_cache(cache_config)

            self.scheduler = Scheduler(
                max_batch_size=self.max_batch_size,
                num_blocks=self.num_blocks,
                block_size=self.block_size,
            )

            logger.info(
                f"Starting Inference Loop with model at {self.model_path} on device {self.device} with dtype {self.dtype}"
            )
            self.running = True
            self.step_thread = threading.Thread(
                target=self._step_loop, daemon=True, name="InferenceServerStepThread"
            )
            self.step_thread.start()
            yield
            # Cleanup on shutdown
            self.stop()

        app = FastAPI(lifespan=lifespan)
        self._register_chat_routes(app)
        return app

    def _register_chat_routes(self, app: FastAPI):
        @app.post("/chat/completions")
        async def chat_completions(request: Request):
            try:
                data = await request.json()
                logger.debug(f"Received request data: {data}")
            except Exception as e:
                logger.error(f"Failed to parse request JSON: {e}")
                return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

            logger.debug(f"Received request: {data}")

            if not data.get("messages"):
                if not data.get("prompt"):
                    return JSONResponse(
                        content={"error": "No message provided"}, status_code=400
                    )
                else:
                    data["messages"] = [{"role": "user", "content": data.get("prompt")}]

            stream = data.get("stream", False)
            id_ = f"cmpl-{uuid.uuid4().hex}"
            if stream:
                return StreamingResponse(
                    self._chat_stream(id_, data, request),
                    media_type="text/event-stream",
                )
            else:
                response = await self._chat(id_, data, request)
                if isinstance(response, JSONResponse):
                    return response
                return JSONResponse(content=response)

    async def _chat_stream(self, id_, request_data, request):
        """Handle streaming chat request."""
        req = None
        start_time = time.time()

        try:
            messages = request_data.get("messages", [])
            input_content = self.tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            input_tokens = self.tokenizer.encode(input_content)

            req = InferenceRequest(
                id_,
                request_data,
                request,
                input_tokens,
                request_data.get("max_tokens", self.max_tokens),
                request_data.get("temperature", self.temperature),
                request_data.get("top_k", self.top_k),
                request_data.get("top_p", self.top_p),
                self.engine.config.eos_token_id,
            )
            self.scheduler.add_request(req)

            while True:
                if time.time() - start_time > DEFAULT_REQUEST_TIMEOUT:
                    logger.warning(
                        f"Request {id_} timed out after {DEFAULT_REQUEST_TIMEOUT}s"
                    )
                    req.status = RequestStatus.Timeout
                    req.finish_reason = "timeout"
                    error_chunk = json.dumps(
                        chunk_json(
                            id_, content="[Request timeout]", finish_reason="timeout"
                        ),
                        ensure_ascii=False,
                    )
                    yield f"data: {error_chunk}\n\n"
                    break

                if await request.is_disconnected():
                    logger.info(f"Client disconnected for request {id_}")
                    req.status = RequestStatus.Canceled
                    req.finish_reason = "client_disconnected"
                    break

                # Check if request is finished and queue is empty
                if req.finish_reason is not None and req.output_queue.async_q.empty():
                    chunk = json.dumps(
                        chunk_json(id_, finish_reason=req.finish_reason),
                        ensure_ascii=False,
                    )
                    yield f"data: {chunk}\n\n"
                    break

                # Get token from queue
                # Use shorter timeout if request is already finished (to drain remaining tokens quickly)
                timeout = 0.1 if req.finish_reason is not None else DEFAULT_STREAM_TIMEOUT
                try:
                    token_output = await asyncio.wait_for(
                        req.output_queue.async_q.get(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    # If request is finished and we're timing out, no more tokens - send final chunk and break
                    if req.finish_reason is not None:
                        chunk = json.dumps(
                            chunk_json(id_, finish_reason=req.finish_reason),
                            ensure_ascii=False,
                        )
                        yield f"data: {chunk}\n\n"
                        break
                    # Request not finished, continue waiting
                    continue
                except asyncio.CancelledError:
                    logger.info(f"Request {id_} was cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error getting token for {id_}: {e}")
                    # If request is finished, send final chunk and break
                    if req.finish_reason is not None:
                        chunk = json.dumps(
                            chunk_json(id_, finish_reason=req.finish_reason),
                            ensure_ascii=False,
                        )
                        yield f"data: {chunk}\n\n"
                        break
                    await asyncio.sleep(0.01)
                    continue

                if isinstance(token_output, RequestOutput):
                    content = token_output.token_text
                    finish_reason = token_output.finish_reason
                    chunk = json.dumps(
                        chunk_json(id_, content=content, finish_reason=finish_reason),
                        ensure_ascii=False,
                    )
                    yield f"data: {chunk}\n\n"

                    # Mark task as done to allow queue.join() to complete
                    req.output_queue.async_q.task_done()

                    # If this token has a finish_reason, the request is done
                    if finish_reason is not None:
                        break
                else:
                    logger.warning(
                        f"Unexpected token output type: {type(token_output)}"
                    )
                    # Mark task as done even for unexpected types
                    req.output_queue.async_q.task_done()

        except Exception as e:
            logger.error(f"Stream error for {id_}: {e}", exc_info=True)
            if req:
                req.status = RequestStatus.Failed
                req.finish_reason = "error"
            error_chunk = json.dumps(
                chunk_json(id_, content=f"[Error: {str(e)}]", finish_reason="error"),
                ensure_ascii=False,
            )
            yield f"data: {error_chunk}\n\n"

        finally:
            if req and req.finish_reason is None:
                req.status = RequestStatus.Canceled
                req.finish_reason = "cancel"

            # Drain remaining tokens from queue and mark them as done to avoid hanging on join()
            if req:
                while not req.output_queue.async_q.empty():
                    try:
                        # Get and mark as done immediately (don't process, just drain)
                        req.output_queue.async_q.get_nowait()
                        req.output_queue.async_q.task_done()
                    except Exception:
                        # Queue might be closed or empty, break
                        break

            await req.close() if req else None

            yield "data: [DONE]\n\n"

    async def _chat(self, id_, data, request):
        """Handle non-streaming chat request."""
        req = None
        start_time = time.time()

        try:
            messages = data.get("messages", [])
            input_content = self.tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            input_tokens = self.tokenizer.encode(input_content)

            req = InferenceRequest(
                id_,
                data,
                request,
                input_tokens,
                data.get("max_tokens", self.max_tokens),
                data.get("temperature", 1.0),
                data.get("top_k", 1),
                data.get("top_p", 0.8),
                self.engine.config.eos_token_id,
            )

            self.scheduler.add_request(req)

            # Collect all generated tokens
            output = []
            while True:
                if time.time() - start_time > DEFAULT_REQUEST_TIMEOUT:
                    logger.warning(f"Request {id_} timed out")
                    req.status = RequestStatus.Timeout
                    req.finish_reason = "timeout"
                    break

                if await request.is_disconnected():
                    logger.info(f"Client disconnected for request {id_}")
                    req.status = RequestStatus.Canceled
                    req.finish_reason = "client_disconnected"
                    break

                if req.finish_reason is not None and req.output_queue.async_q.empty():
                    break

                try:
                    token_output = await asyncio.wait_for(
                        req.output_queue.async_q.get(), timeout=DEFAULT_STREAM_TIMEOUT
                    )
                    if isinstance(token_output, RequestOutput):
                        output.append(token_output.token_text)
                        # Mark task as done to allow queue.join() to complete
                        req.output_queue.async_q.task_done()
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error getting token for {id_}: {e}")
                    await asyncio.sleep(0.01)
                    continue

            output_text = "".join(output).strip()
            response = chunk_json(
                id_,
                content=output_text,
                role="assistant",
                finish_reason=req.finish_reason or "stop",
            )

            return response

        except Exception as e:
            logger.error(f"Chat error for {id_}: {e}", exc_info=True)
            if req:
                req.status = RequestStatus.Failed
                req.finish_reason = "error"
            return JSONResponse(content={"error": str(e)}, status_code=500)
        finally:
            if req and req.finish_reason is None:
                req.status = RequestStatus.Canceled
                req.finish_reason = "cancel"

            # Drain remaining tokens from queue and mark them as done to avoid hanging on join()
            if req:
                while not req.output_queue.async_q.empty():
                    try:
                        # Get and mark as done immediately (don't process, just drain)
                        req.output_queue.async_q.get_nowait()
                        req.output_queue.async_q.task_done()
                    except Exception:
                        # Queue might be closed or empty, break
                        break

            await req.close() if req else None

    def stop(self):
        """Stop server loop and cleanup resources."""
        if not self.running:
            logger.warning("Inference Server is not running")
            return
        logger.info("Stopping Inference Server...")
        self.running = False
        self.step_thread.join(timeout=5)

    def _step_loop(self):
        """Main server loop: schedule, execute, sample, and output."""
        while self.running:
            # Schedule requests
            scheduler_output = self.scheduler.schedule()
            if scheduler_output is None:
                time.sleep(0.01)
                continue
            if not scheduler_output.scheduled_requests:
                logger.warning(
                    "SchedulerOutput has empty scheduled_requests, skipping this step"
                )
                time.sleep(0.01)
                continue

            # Build model inputs
            model_input_dict = scheduler_output.build_model_inputs(
                self.temperature, self.top_p, self.top_k
            )
            model_input = {}
            for key, value in model_input_dict.items():
                if key == "input_ids":
                    model_input[key] = infinicore.from_list(
                        [value], dtype=infinicore.int64
                    )
                elif key in [
                    "position_ids",
                    "past_kv_lengths",
                    "total_kv_lengths",
                    "input_offsets",
                    "slot_mapping",
                ]:
                    model_input[key] = infinicore.from_list(
                        value, dtype=infinicore.int64
                    )
                elif key == "block_tables":
                    model_input[key] = infinicore.from_list(
                        value, dtype=infinicore.int64
                    )
                else:
                    model_input[key] = value

            # Run inference
            sampled_tokens = self.engine.forward(**model_input)

            # Convert to Python list
            sampled_tokens_list = sampled_tokens.to_numpy().tolist()

            # Update request status and handle output
            self._update_requests_status(
                scheduler_output.is_prefill,
                scheduler_output.scheduled_requests,
                sampled_tokens_list,
            )

    def _update_requests_status(self, is_prefill, requests, sampled_tokens):
        """Update request status and process output."""
        if is_prefill:
            self.scheduler.cache_manager.reset_req_blocks()

        for req, token_id in zip(requests, sampled_tokens):
            req.generated_token_ids.append(token_id)
            if req.is_prefill:
                req.is_prefill = False

            token_text = self.tokenizer.decode(token_id)
            req.generated_text += token_text

            # Check if generation finished
            if self._check_request_finished(req, token_id):
                req.status = RequestStatus.Finished
                req.finished_time = time.time()

            # Build output and put in queue
            output = RequestOutput(
                request_id=req.request_id,
                token_id=token_id,
                token_text=token_text,
                status=req.status,
                finish_reason=(
                    req.finish_reason if req.status == RequestStatus.Finished else None
                ),
                generated_text=req.generated_text,
            )
            req.output_queue.sync_q.put(output)
        self.scheduler.complete_requests(requests)

    def _check_request_finished(self, req, token_id):
        """Check if request generation is finished."""
        # Check max length
        if req.get_num_generated_tokens() >= req.max_new_tokens:
            req.finish_reason = "length"
            return True
        # Check EOS token
        if req.eos_token_id and token_id in req.eos_token_id:
            req.finish_reason = "eos_token"
            return True
        # Check end strings
        for end_str in req.end_strings:
            if req.generated_text.endswith(end_str):
                req.finish_reason = "end_string"
                return True
        return False


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


# Parse command line arguments
def parse_args():
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
        "--backend",
        type=str,
        default="cpp",
        choices=["cpp", "python"],
        help="Backend to use",
    )
    parser.add_argument(
        "--num_blocks", type=int, default=8 * 1024, help="Number of blocks for KV cache"
    )
    parser.add_argument(
        "--block_size", type=int, default=16, help="Block size for KV cache"
    )
    parser.add_argument(
        "--cache_type",
        type=str,
        default="paged",
        choices=["paged", "static"],
        help="KV cache type: 'paged' (default, requires device support) or 'static' (fallback for unsupported devices)",
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

    model_path = args.model_path
    ndev = args.tp
    max_tokens = args.max_tokens
    max_batch_size = args.max_batch_size
    backend = args.backend
    num_blocks = args.num_blocks
    block_size = args.block_size
    cache_type = args.cache_type
    device_str = "cpu"
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k

    if args.cpu:
        device_str = "cpu"
    elif args.nvidia:
        device_str = "cuda"
    elif args.metax:
        device_str = "cuda"
    elif args.moore:
        device_str = "moore"
    elif args.iluvatar:
        device_str = "cuda"
    elif args.cambricon:
        device_str = "mlu"
    else:
        print(
            "Usage: python python/infinilm/server/inference_server.py [--cpu | --nvidia | --metax | --moore | --iluvatar] "
            "--model_path=<path/to/model_dir> --backend=[cpp/python] --max_tokens=MAX_TOKENS --max_batch_size=MAX_BATCH_SIZE"
            "\n"
            "Example: python python/infinilm/server/inference_server.py --nvidia --model_path=/data/shared/models/9G7B_MHA/ "
            "--backend=cpp --max_tokens=100 --max_batch_size=32 --tp=1 --temperature=1.0 --top_p=0.8 --top_k=1"
        )
        sys.exit(1)

    infini_device = infinicore.device(device_str, 0)
    if args.dtype == "float32":
        infini_dtype = infinicore.float32
    elif args.dtype == "float16":
        infini_dtype = infinicore.float16
    elif args.dtype == "bfloat16":
        infini_dtype = infinicore.bfloat16
    else:
        raise ValueError("Only float32, float16 and bfloat16 are supported")

    server = InferenceServer(
        model_path,
        infini_device,
        infini_dtype,
        ndev,
        max_tokens,
        max_batch_size,
        backend,
        num_blocks,
        block_size,
        temperature,
        top_p,
        top_k,
        cache_type,
    )
    server.start()


if __name__ == "__main__":
    main()
