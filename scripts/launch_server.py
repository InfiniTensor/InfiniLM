from jiuge import JiugeForCauslLM
from jiuge_awq import JiugeAWQForCausalLM
from libinfinicore_infer import DeviceType
from infer_task import InferTask
from kvcache_pool import KVCachePool
import torch

import argparse
import queue
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import contextlib
import uvicorn
import time
import uuid
import json
import threading
import janus
import os
import signal


DEVICE_TYPE_MAP = {
    "cpu": DeviceType.DEVICE_TYPE_CPU,
    "nvidia": DeviceType.DEVICE_TYPE_NVIDIA,
    "cambricon": DeviceType.DEVICE_TYPE_CAMBRICON,
    "ascend": DeviceType.DEVICE_TYPE_ASCEND,
    "metax": DeviceType.DEVICE_TYPE_METAX,
    "moore": DeviceType.DEVICE_TYPE_MOORE,
    "iluvatar": DeviceType.DEVICE_TYPE_ILUVATAR,
    "kunlun": DeviceType.DEVICE_TYPE_KUNLUN,
    "hygon": DeviceType.DEVICE_TYPE_HYGON,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the LLM inference server.")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--dev",
        type=str,
        choices=DEVICE_TYPE_MAP.keys(),
        default="cpu",
        help="Device type to run the model on (default: cpu)",
    )
    parser.add_argument(
        "--ndev",
        type=int,
        default=1,
        help="Number of devices to use (default: 1)",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=3,
        help="Maximum number of requests that can be batched together (default: 3)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=False,
        default=None,
        help="Max token sequence length that model will handle (follows model config if not provided)",
    )
    parser.add_argument(
        "--awq",
        action="store_true",
        help="Whether to use AWQ quantized model (default: False)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=30,
        help="Request timeout in seconds. Process will exit if a request hangs longer than this (default: 30)",
    )
    return parser.parse_args()


args = parse_args()
device_type = DEVICE_TYPE_MAP[args.dev]
model_path = args.model_path
ndev = args.ndev
max_tokens = args.max_tokens
USE_AWQ = args.awq
MAX_BATCH = args.max_batch
SERVER_PORT = args.port
SERVER_HOST = args.host
REQUEST_TIMEOUT = args.request_timeout
print(
    f"Using MAX_BATCH={MAX_BATCH}. Try reduce this value if out of memory error occurs."
)
print(
    f"Request timeout: {REQUEST_TIMEOUT}s. Process will exit if a request hangs longer than this."
)


def chunk_json(id_, content=None, role=None, finish_reason=None):
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


# A wrapper for InferTask that supports async output queue
class AsyncInferTask(InferTask):
    def __init__(self, id, tokens, max_tokens, temperature, topk, topp, end_tokens, repetition_penalty=1.0, test_hang_seconds=0):
        super().__init__(id, tokens, max_tokens, temperature, topk, topp, end_tokens, repetition_penalty)
        self.output_queue = janus.Queue()
        self.last_activity_time = time.time()  # Track when task was last active
        self.test_hang_seconds = test_hang_seconds  # Test parameter: sleep for this many seconds to simulate hang (set to 0 after first use)
        self.timed_out = False  # Flag to mark if task has timed out
        print(f"[INFO] Create InferTask {self.id}" + (f" (TEST: will hang for {test_hang_seconds}s once)" if test_hang_seconds > 0 else ""))

    def output(self, out_token):
        self.next(out_token)
        self.last_activity_time = time.time()  # Update activity time when output is generated
        self.output_queue.sync_q.put(out_token)

    def signal_timeout(self):
        """Signal that this task has timed out"""
        self.timed_out = True
        self.finish_reason = "timeout"

    def signal_internal_error(self):
        """Signal that an internal error occurred (process will be killed)"""
        self.timed_out = True  # Reuse timed_out flag to trigger error response
        self.finish_reason = "internal_error"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if USE_AWQ:
        app.state.model = JiugeAWQForCausalLM(
            model_path, device_type, ndev, max_tokens=max_tokens
        )
    else:
        app.state.model = JiugeForCauslLM(
            model_path, device_type, ndev, max_tokens=max_tokens
        )
    app.state.kv_cache_pool = KVCachePool(app.state.model, MAX_BATCH)
    app.state.request_queue = janus.Queue()
    app.state.active_tasks = {}  # Track active tasks: task_id -> task object
    app.state.task_lock = threading.Lock()  # Lock for accessing active_tasks
    worker_thread = threading.Thread(target=worker_loop, args=(app,), daemon=True)
    worker_thread.start()

    # Start timeout checker thread
    timeout_checker_thread = threading.Thread(target=timeout_checker_loop, args=(app,), daemon=True)
    timeout_checker_thread.start()

    try:
        yield  # The app runs here
    finally:
        # Shutdown
        app.state.request_queue.sync_q.put(None)
        worker_thread.join()
        app.state.request_queue.shutdown()

        app.state.kv_cache_pool.finalize()
        app.state.model.destroy_model_instance()


App = FastAPI(lifespan=lifespan)


# Timeout checker: monitors active tasks and kills process if any task hangs
def timeout_checker_loop(app):
    """Monitor active tasks and kill the process if any task hangs beyond timeout"""
    while True:
        try:
            time.sleep(5)  # Check every 5 seconds

            current_time = time.time()
            hung_tasks = []

            with app.state.task_lock:
                # Check all active tasks for timeout
                for task_id, task in list(app.state.active_tasks.items()):
                    time_since_activity = current_time - task.last_activity_time
                    if time_since_activity > REQUEST_TIMEOUT:
                        hung_tasks.append((task_id, time_since_activity))

            # If we found hung tasks, signal all active tasks and then kill the process
            if hung_tasks:
                print(f"[ERROR] Detected {len(hung_tasks)} hung task(s) exceeding timeout of {REQUEST_TIMEOUT}s:")
                for task_id, hang_time in hung_tasks:
                    print(f"  - Task {task_id}: hung for {hang_time:.1f}s")

                # Signal all active tasks (not just hung ones) to send error responses to clients
                # This ensures all processing requests get error responses before process is killed
                with app.state.task_lock:
                    all_active_tasks = list(app.state.active_tasks.items())
                    print(f"[ERROR] Signaling {len(all_active_tasks)} active task(s) to send error responses...")
                    for task_id, task in all_active_tasks:
                        if task_id in [tid for tid, _ in hung_tasks]:
                            # Hung tasks get timeout error
                            task.signal_timeout()
                            print(f"[ERROR] Signaled timeout to hung task {task_id}")
                        else:
                            # Other active tasks get internal error (process will be killed)
                            task.signal_internal_error()
                            print(f"[ERROR] Signaled internal error to active task {task_id}")

                # Give a short time for error responses to be sent to clients
                print(f"[ERROR] Waiting 2 seconds for error responses to be sent to clients...")
                time.sleep(2)

                print(f"[ERROR] Killing process to trigger recovery mechanism...")
                # Kill the process - this will be detected by the babysitter and trigger restart
                os.kill(os.getpid(), signal.SIGTERM)
                # If SIGTERM doesn't work, use SIGKILL as fallback after a delay
                time.sleep(2)
                os.kill(os.getpid(), signal.SIGKILL)
                break

        except Exception as e:
            print(f"[ERROR] Exception in timeout checker: {e}")
            time.sleep(5)


# App loop: take requests from the queue, do inference, and put unfinished requests back into the queue.
def worker_loop(app):
    while True:
        try:
            task = app.state.request_queue.sync_q.get(timeout=0.01)
        except queue.Empty:
            continue

        if task is None:
            return

        # Register task as active
        with app.state.task_lock:
            app.state.active_tasks[task.id] = task
            task.last_activity_time = time.time()

        batch = [task]
        while len(batch) < MAX_BATCH:
            try:
                req = app.state.request_queue.sync_q.get_nowait()
                if req is not None:
                    batch.append(req)
                    # Register additional tasks as active
                    with app.state.task_lock:
                        app.state.active_tasks[req.id] = req
                        req.last_activity_time = time.time()
            except queue.Empty:
                break

        # Update activity time before inference
        batch_start_time = time.time()
        with app.state.task_lock:
            for t in batch:
                t.last_activity_time = batch_start_time

        # Test hang simulation: if any task has test_hang_seconds > 0, sleep to simulate hang
        # Only apply once per task by setting test_hang_seconds to 0 after use
        tasks_needing_hang = [t for t in batch if t.test_hang_seconds > 0]
        if tasks_needing_hang:
            max_hang_time = max(t.test_hang_seconds for t in tasks_needing_hang)
            print(f"[TEST] Simulating hang for {max_hang_time}s (task will exceed timeout if timeout < {max_hang_time}s)")
            time.sleep(max_hang_time)
            print(f"[TEST] Hang simulation complete, continuing with inference...")
            # Reset test_hang_seconds to 0 for all tasks that used it (so it won't hang again)
            for t in tasks_needing_hang:
                t.test_hang_seconds = 0

        output_tokens = app.state.model.batch_infer_one_round(batch)

        # Update activity time after inference (critical: if batch_infer_one_round hangs,
        # this won't execute, and timeout checker will detect it)
        batch_end_time = time.time()
        with app.state.task_lock:
            for task, token in zip(batch, output_tokens):
                task.last_activity_time = batch_end_time
                task.output(token)
                if task.finish_reason is None:
                    # Task continues, keep it tracked but update activity time
                    # It will be put back in queue and processed again
                    pass
                else:
                    print(f"[INFO] Task {task.id} finished infer.")
                    app.state.kv_cache_pool.release_sync(task)
                    # Remove task from active tracking when finished
                    app.state.active_tasks.pop(task.id, None)

        # Put unfinished tasks back in queue (outside lock to avoid deadlock)
        for task, token in zip(batch, output_tokens):
            if task.finish_reason is None:
                app.state.request_queue.sync_q.put(task)


def build_task(id_, request_data, request: Request):
    # Handle both chat and completion formats
    if "messages" in request_data:
        # Chat format
        messages = request_data.get("messages", [])
        # Get chat_template_kwargs from request, default to empty dict
        chat_template_kwargs = request_data.get("chat_template_kwargs", {})
        # Merge with default parameters, allowing chat_template_kwargs to override
        template_params = {
            "conversation": messages,
            "add_generation_prompt": True,
            "tokenize": False,
            **chat_template_kwargs  # Allow override of defaults
        }
        input_content = request.app.state.model.tokenizer.apply_chat_template(**template_params)
        tokens = request.app.state.model.tokenizer.encode(input_content)
        max_tokens = request_data.get("max_tokens", request.app.state.model.max_context_len())
    else:
        # Completion format
        prompt = request_data.get("prompt", "")
        tokens = request.app.state.model.tokenizer.encode(prompt)
        max_tokens = request_data.get("max_tokens", 0)

    # Test parameter: test_hang_seconds - sleep for this many seconds to simulate hang
    # This is useful for testing the timeout checker mechanism.
    # Example: Set "test_hang_seconds": 350 in request to test timeout (if timeout is 300s)
    # The sleep happens in the worker loop before batch_infer_one_round, simulating a hang
    test_hang_seconds = request_data.get("test_hang_seconds", 0)

    return AsyncInferTask(
        id_,
        tokens,
        max_tokens,
        request_data.get("temperature", 1.0),
        request_data.get("top_k", 0),  # Default to 0 (disabled) to consider all tokens, matching vLLM behavior
        request_data.get("top_p", 1.0),
        request.app.state.model.eos_token_id,
        request_data.get("repetition_penalty", 1.0),
        test_hang_seconds=test_hang_seconds,
    )


async def chat_stream(id_, request_data, request: Request):
    try:
        infer_task = build_task(id_, request_data, request)
        # Track task from creation
        with request.app.state.task_lock:
            request.app.state.active_tasks[infer_task.id] = infer_task
            infer_task.last_activity_time = time.time()

        await request.app.state.kv_cache_pool.acquire(infer_task)

        # Check if task already timed out before starting stream
        if infer_task.timed_out:
            raise HTTPException(
                status_code=504,
                detail={
                    "message": f"Request timeout: task exceeded {REQUEST_TIMEOUT}s timeout",
                    "type": "timeout_error",
                    "code": "timeout"
                }
            )

        # Initial empty content
        chunk = json.dumps(
            chunk_json(id_, content="", role="assistant"), ensure_ascii=False
        )
        yield f"data: {chunk}\n\n"

        request.app.state.request_queue.sync_q.put(infer_task)

        while True:
            if await request.is_disconnected():
                print("Client disconnected. Aborting stream.")
                break
            if (
                infer_task.finish_reason is not None
                and infer_task.output_queue.async_q.empty()
            ):
                # Check if timed out or internal error - yield error chunk instead of raising HTTPException
                # (can't raise HTTPException after streaming has started)
                if infer_task.timed_out:
                    # Both timeout and internal_error result in internal error response
                    # because the process will be killed and restarted
                    # Yield error chunk in SSE format
                    error_chunk = {
                        "id": id_,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "unknown",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": None
                        }],
                        "error": {
                            "message": "Internal server error: process will be restarted",
                            "type": "internal_error",
                            "code": "internal_error"
                        }
                    }
                    chunk = json.dumps(error_chunk, ensure_ascii=False)
                    yield f"data: {chunk}\n\n"
                    yield "data: [DONE]\n\n"
                else:
                    chunk = json.dumps(
                        chunk_json(id_, finish_reason=infer_task.finish_reason),
                        ensure_ascii=False,
                    )
                    yield f"data: {chunk}\n\n"
                break

            # Check for timeout or internal error before getting next token
            if infer_task.timed_out:
                # Both timeout and internal_error result in internal error response
                # because the process will be killed and restarted
                # Yield error chunk in SSE format (can't raise HTTPException after streaming started)
                error_chunk = {
                    "id": id_,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "unknown",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": None
                    }],
                    "error": {
                        "message": "Internal server error: process will be restarted",
                        "type": "internal_error",
                        "code": "internal_error"
                    }
                }
                chunk = json.dumps(error_chunk, ensure_ascii=False)
                yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
                break

            token = await infer_task.output_queue.async_q.get()
            content = request.app.state.model.tokenizer.decode(token)

            chunk = json.dumps(chunk_json(id_, content=content), ensure_ascii=False)
            yield f"data: {chunk}\n\n"

    except HTTPException:
        # Re-raise HTTPException to propagate error status code
        raise
    except Exception as e:
        print(f"[Error] ID : {id_} Exception: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(e),
                "type": "internal_error",
                "code": "internal_error"
            }
        )
    finally:
        if infer_task:
            if infer_task.finish_reason is None:
                infer_task.finish_reason = "cancel"
            # Clean up task from active tracking
            with request.app.state.task_lock:
                request.app.state.active_tasks.pop(infer_task.id, None)


async def chat(id_, request_data, request: Request):
    try:
        infer_task = build_task(id_, request_data, request)
        # Track task from creation
        with request.app.state.task_lock:
            request.app.state.active_tasks[infer_task.id] = infer_task
            infer_task.last_activity_time = time.time()

        await request.app.state.kv_cache_pool.acquire(infer_task)
        request.app.state.request_queue.sync_q.put(infer_task)
        output = []
        while True:
            if (
                infer_task.finish_reason is not None
                and infer_task.output_queue.async_q.empty()
            ):
                break

            # Check for timeout or internal error before getting next token
            if infer_task.timed_out:
                # Both timeout and internal_error result in internal error response
                # because the process will be killed and restarted
                return JSONResponse(
                    content={
                        "error": {
                            "message": "Internal server error: process will be restarted",
                            "type": "internal_error",
                            "code": "internal_error"
                        }
                    },
                    status_code=500  # Internal Server Error
                )

            token = await infer_task.output_queue.async_q.get()
            content = request.app.state.model.tokenizer.decode(token)
            output.append(content)

        # Check if timed out or internal error before returning response
        if infer_task.timed_out:
            # Both timeout and internal_error result in internal error response
            # because the process will be killed and restarted
            return JSONResponse(
                content={
                    "error": {
                        "message": "Internal server error: process will be restarted",
                        "type": "internal_error",
                        "code": "internal_error"
                    }
                },
                status_code=500  # Internal Server Error
            )

        output_text = "".join(output).strip()
        response = chunk_json(
            id_,
            content=output_text,
            role="assistant",
            finish_reason=infer_task.finish_reason or "stop",
        )
        return response

    except Exception as e:
        print(f"[Error] ID: {id_} Exception: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if infer_task.finish_reason is None:
            infer_task.finish_reason = "cancel"
        # Clean up task from active tracking
        with request.app.state.task_lock:
            request.app.state.active_tasks.pop(infer_task.id, None)


@App.post("/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    print('-----------------------------------------')
    print(data)
    print('-----------------------------------------')

    if not data.get("messages"):
        if not data.get("prompt"):
            return JSONResponse(content={"error": "No message provided"}, status_code=400)
        else:
            data['messages'] = [{"role": "user", "content": data.get("prompt")}]

    stream = data.get("stream", False)
    id_ = f"cmpl-{uuid.uuid4().hex}"
    if stream:
        # FastAPI's exception handler will catch HTTPException raised from the generator
        return StreamingResponse(
            chat_stream(id_, data, request), media_type="text/event-stream"
        )
    else:
        response = await chat(id_, data, request)
        # If response is already a JSONResponse (error case), return it directly
        if isinstance(response, JSONResponse):
            return response
        return JSONResponse(content=response)





async def completion(id_, request_data, request: Request):
    infer_task = None  # Initialize to None to avoid UnboundLocalError
    try:
        # Check if max_tokens > 0 is requested
        max_tokens = request_data.get("max_tokens", 0)
        if max_tokens > 0:
            return JSONResponse(
                content={"error": "max_tokens > 0 is not supported yet. Please use max_tokens=0 for logprobs calculation."},
                status_code=400
            )

        infer_task = build_task(id_, request_data, request)
        await request.app.state.kv_cache_pool.acquire(infer_task)

        output = []
        logprobs = []

        # Handle echo and logprobs calculation
        echo = request_data.get("echo", False)
        if echo:
            # Add input tokens to output
            input_tokens = infer_task.tokens
            for token in input_tokens:
                content = (
                    request.app.state.model.tokenizer._tokenizer.id_to_token(token)
                    .replace("▁", " ")
                    .replace("<0x0A>", "\n")
                )
                output.append(content)

            # Calculate logprobs for input tokens
            from jiuge import JiugeBatchedTask
            batch_inputs = JiugeBatchedTask([infer_task])
            log_probs = torch.zeros(
                (batch_inputs.ntok, request.app.state.model.meta.dvoc),
                dtype=request.app.state.model.meta.torch_dtype_logits
            )
            request.app.state.model.jiuge_model.forward_batch(
                request.app.state.model.model_instance,
                batch_inputs.tokens,
                batch_inputs.ntok,
                batch_inputs.req_lens,
                batch_inputs.nreq,
                batch_inputs.req_pos,
                batch_inputs.kv_caches,
                log_probs.data_ptr(),
            )

            log_probs = log_probs.float()

            # Calculate correct logprobs for input tokens
            token_logprobs = []
            for i in range(len(infer_task.tokens) - 1):  # Only up to second-to-last token
                next_token = infer_task.tokens[i+1]      # Next token to predict
                logprob = log_probs[i, next_token].item() # Use position i logits to predict position i+1 token
                token_logprobs.append(logprob)

            # First token has no context, so logprob is None
            logprobs = [None] + token_logprobs
        else:
            # echo=false: don't calculate logprobs since user can't see input text
            logprobs = []

        # For max_tokens=0, we need to manually release the KV cache since we don't go through worker
        await request.app.state.kv_cache_pool.release(infer_task)
        print(f"[DEBUG] {id_} Released KV cache for max_tokens=0")

        output_text = "".join(output).strip()

        # Prepare tokens list for logprobs
        tokens_list = []
        text_offset_list = []
        current_offset = 0

        # Build tokens list and text offsets
        for i, content in enumerate(output):
            tokens_list.append(content)
            text_offset_list.append(current_offset)
            current_offset += len(content)

        # Build response according to DeepSeek API completion format
        response = {
            "id": id_,
            "object": "text_completion",
            "created": int(time.time()),
            "model": "jiuge",
            "choices": [
                {
                    "text": output_text,
                    "index": 0,
                    "logprobs": {
                        "token_logprobs": logprobs,
                        "tokens": tokens_list,
                        "text_offset": text_offset_list,
                        "top_logprobs": []
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(infer_task.tokens),
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": len(infer_task.tokens),
                "completion_tokens": 0,
                "total_tokens": len(infer_task.tokens),
                "completion_tokens_details": {
                    "reasoning_tokens": 0
                }
            }
        }
        return response

    except Exception as e:
        print(f"[Error] ID: {id_} Exception: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if infer_task and infer_task.finish_reason is None:
            infer_task.finish_reason = "cancel"


@App.post("/completions")
async def completions(request: Request):
    data = await request.json()

    if not data.get("prompt"):
        return JSONResponse(content={"error": "No prompt provided"}, status_code=400)

    id_ = f"cmpl-{uuid.uuid4().hex}"
    response = await completion(id_, data, request)

    # Check if response is already a JSONResponse (error case)
    if isinstance(response, JSONResponse):
        return response
    else:
        return JSONResponse(content=response)


@App.get("/models")
async def list_models(request: Request):
    """
    OpenAI-compatible /models endpoint.
    Returns a list of available models.
    """
    try:
        # Get model information from app state
        model = request.app.state.model
        model_id = "jiuge"  # Default model ID

        # Try to get model name from config if available
        if hasattr(model, 'config') and model.config:
            # Try model_type first
            model_id = model.config.get("model_type", "jiuge")
            # If model_type is not informative, try architectures
            if model_id == "jiuge" and "architectures" in model.config:
                architectures = model.config.get("architectures", [])
                if architectures:
                    model_id = architectures[0].lower()

        return JSONResponse(content={
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "infini",
                    "permission": [],
                    "root": model_id,
                    "parent": None
                }
            ]
        })
    except Exception as e:
        print(f"[Error] Exception in /models: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(App, host=SERVER_HOST, port=SERVER_PORT)

"""
curl -N -H "Content-Type: application/json" \
     -X POST http://127.0.0.1:8000/chat/completions \
     -d '{
       "model": "jiuge",
       "messages": [
         {"role": "user", "content": "介绍你自己"}
       ],
       "temperature": 0.7,
       "top_p": 0.7,
       "repetition_penalty": 1.02,
       "stream": false,
       "chat_template_kwargs": {"enable_thinking": false}
     }'


curl -N -H "Content-Type: application/json" \
     -X POST http://127.0.0.1:8000/chat/completions \
     -d '{
       "model": "jiuge",
       "messages": [
         {"role": "system", "content": "你是一个由启元实验室开发的九格助手，你擅长中英文对话，能够理解并处理各种问题，提供安全、有帮助、准确的回答。当前时间：2025-12-24#注意：回复之前注意结合上下文和工具返回内容进行回复"},
         {"role": "user", "content": "怎么看待台海局势"}
       ],
       "temperature": 0.7,
       "top_p": 0.7,
       "max_tokens": 512,
       "repetition_penalty": 1.1,
       "stream": false,
        "chat_template_kwargs": {"enable_thinking": false}
     }'

# Test timeout checker: simulate a hang that exceeds the timeout
# This will cause the process to be killed by the timeout checker
# (assuming --request-timeout is set to a value less than test_hang_seconds)
# Example: if --request-timeout=300, use test_hang_seconds=350 to trigger timeout
curl -N -H "Content-Type: application/json" \
     -X POST http://127.0.0.1:8000/chat/completions \
     -d '{
       "model": "jiuge",
       "messages": [
         {"role": "user", "content": "Hello"}
       ],
       "temperature": 0.7,
       "test_hang_seconds": 350,
       "stream": false
     }'


"""
