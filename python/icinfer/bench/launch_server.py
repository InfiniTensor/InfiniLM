from icinfer.models.jiuge import JiugeForCausalLM
from icinfer.engine.libinfinicore_infer import DeviceType
from icinfer.engine.infer_task import InferTask
from icinfer.engine.kvcache_pool import KVCachePool

import argparse
import queue
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import contextlib
import uvicorn
import time
import uuid
import json
import threading
import janus
import traceback

from icinfer.engine.llm_engine_async import InfiniEngineAsync
from icinfer.sampling_params import SamplingParams


DEVICE_TYPE_MAP = {
    "cpu": DeviceType.DEVICE_TYPE_CPU,
    "nvidia": DeviceType.DEVICE_TYPE_NVIDIA,
    "cambricon": DeviceType.DEVICE_TYPE_CAMBRICON,
    "ascend": DeviceType.DEVICE_TYPE_ASCEND,
    "metax": DeviceType.DEVICE_TYPE_METAX,
    "moore": DeviceType.DEVICE_TYPE_MOORE,
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

    parser.add_argument("--max-kvcache-tokens", type=int, default=4096)
    parser.add_argument("--enable-paged-attn", action="store_true")

    return parser.parse_args()


args = parse_args()
device_type = DEVICE_TYPE_MAP[args.dev]
model_path = args.model_path
ndev = args.ndev
max_kvcache_tokens = args.max_kvcache_tokens
enable_paged_attn = args.enable_paged_attn


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
    def __init__(self, id, tokens, max_tokens, temperature, topk, topp, end_tokens):
        super().__init__(id, tokens, max_tokens, temperature, topk, topp, end_tokens)
        self.output_queue = janus.Queue()
        print(f"[INFO] Create InferTask {self.id}")

    def output(self, out_token):
        self.next(out_token)
        self.output_queue.sync_q.put(out_token)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.model = InfiniEngineAsync(
        model_path,
        device=device_type,
        ndev=ndev,
        enforce_eager=True,
        tensor_parallel_size=ndev,
        trust_remote_code=True,
        attention_bias=True,
        enable_paged_attn=enable_paged_attn,
        max_kvcache_tokens=max_kvcache_tokens,
    )

    engine_thread = threading.Thread(target=app.state.model.engine_loop, daemon=True)
    engine_thread.start()

    try:
        yield  # The app runs here
    finally:
        pass


App = FastAPI(lifespan=lifespan)


async def chat_stream(id_, request_data, request: Request):
    try:
        messages = request_data.get("messages", [])
        input_content = request.app.state.model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        max_tokens = request_data.get("max_tokens", 512)
        temperature = request_data.get("temperature", 1.0)
        top_k = request_data.get("top_k", 1)
        top_p = request_data.get("top_p", 1.0)

        sampling_params = SamplingParams(
            temperature=temperature, topk=top_k, topp=top_p, max_tokens=max_tokens
        )

        # 1. 提交请求到引擎，并获取结果队列
        result_queue = await request.app.state.model.add_request(
            input_content, sampling_params, id_
        )

        # 2. 初始响应块
        yield f"data: {json.dumps(chunk_json(id_, content='', role='assistant'), ensure_ascii=False)}\n\n"

        # 3. 从结果队列中异步读取 token 并流式返回
        while True:
            token = await result_queue.get()

            if token is None:  # 结束信号
                yield f"data: {json.dumps(chunk_json(id_, finish_reason='stop'), ensure_ascii=False)}\n\n"
                break

            content = (
                request.app.state.model.tokenizer._tokenizer.id_to_token(token)
                .replace(" ", " ")
                .replace("<0x0A>", "\n")
            )
            yield f"data: {json.dumps(chunk_json(id_, content=content), ensure_ascii=False)}\n\n"

    except Exception as e:
        error_details = traceback.format_exc()
        print(
            f"[Error] ID : {id_} Exception: {e}\n--- TRACEBACK ---\n{error_details}--- END TRACEBACK ---"
        )


@App.post("/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    print("-----------------------------------------")
    print(data)
    print("-----------------------------------------")

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
            chat_stream(id_, data, request), media_type="text/event-stream"
        )
    else:
        messages = data.get("messages", [])
        input_content = request.app.state.model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        max_tokens = data.get("max_tokens", request.app.state.model.max_context_len())
        temperature = data.get("temperature", 1.0)
        top_k = data.get("top_k", 1)
        top_p = data.get("top_p", 1.0)
        sampling_params = SamplingParams(
            temperature=temperature, topk=top_k, topp=top_p, max_tokens=max_tokens
        )
        result_queue = await request.app.state.model.add_request(
            input_content, sampling_params, id_
        )

        output_tokens = []
        while True:
            token = await result_queue.get()
            if token is None:
                break
            output_tokens.append(token)

        output_text = request.app.state.model.tokenizer.decode(output_tokens).strip()
        response = chunk_json(
            id_, content=output_text, role="assistant", finish_reason="stop"
        )
        return JSONResponse(content=response)


if __name__ == "__main__":
    uvicorn.run(App, host="0.0.0.0", port=8000)

"""
curl -N -H "Content-Type: application/json" \
     -X POST http://127.0.0.1:8000/chat/completions \
     -d '{
       "model": "jiuge",
       "messages": [
         {"role": "user", "content": "山东最高的山是？"}
       ],
       "temperature": 1.0,
       "top_k": 50,
       "top_p": 0.8,
       "max_tokens": 512,
       "stream": true
     }'
"""
