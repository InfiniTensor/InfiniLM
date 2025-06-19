from jiuge import JiugeForCauslLM
from libinfinicore_infer import DeviceType, KVCache
from ctypes import POINTER

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import uvicorn
import time
import uuid
import sys
import signal
import json
from collections import deque
from typing import Dict, List, Tuple, Optional
import threading
import transformers

if len(sys.argv) < 3:
    print(
        "Usage: python launch_server.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
    )
    sys.exit(1)
model_path = sys.argv[2]
device_type = DeviceType.DEVICE_TYPE_CPU
if sys.argv[1] == "--cpu":
    device_type = DeviceType.DEVICE_TYPE_CPU
elif sys.argv[1] == "--nvidia":
    device_type = DeviceType.DEVICE_TYPE_NVIDIA
elif sys.argv[1] == "--cambricon":
    device_type = DeviceType.DEVICE_TYPE_CAMBRICON
elif sys.argv[1] == "--ascend":
    device_type = DeviceType.DEVICE_TYPE_ASCEND
elif sys.argv[1] == "--metax":
    device_type = DeviceType.DEVICE_TYPE_METAX
elif sys.argv[1] == "--moore":
    device_type = DeviceType.DEVICE_TYPE_MOORE
else:
    print(
        "Usage: python launch_server.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
    )
    sys.exit(1)
ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1

model = JiugeForCauslLM(model_path, device_type, ndev)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# kv_cache = model.create_kv_cache()


def signal_handler(sig, frame):
    print(f"Received signal {sig}, cleaning up...")
    model.drop_kv_cache(kv_cache)
    model.destroy_model_instance()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Handle docker stop / system shutdown

app = FastAPI()

class RandSampleArgs:
    def __init__(self, max_tokens, temperature, topk, topp):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.topk = topk
        self.topp = topp

class RequestMeta:
    def __init__(self, id, tokens, args: RandSampleArgs, kv_cache: POINTER(KVCache), pos):
        self.id_ = id
        self.tokens_ = tokens
        self.args_ = args
        self.kv_cache_ = kv_cache
        self.pos_ = pos

class KVCachePool:
    def __init__(self, model, max_caches: int = 8):
        self.model = model
        self.max_caches = max_caches
        self.caches: Dict[str, Tuple[int, List[int], POINTER(KVCache)]] = {}
        self.in_use_caches: Dict[str, Tuple[List[int], POINTER(KVCache)]] = {}
        self.lock = threading.Lock()
    
    def find_most_matching_cache(self, tokens: List[int], caches: Dict[str, Tuple[List[int], POINTER(KVCache)]]):
        max_match = 0
        result_pointer = None
        result_tokens = None
    
        set_tokens = set(tokens)
    
        for key, (list_cache_tokens, pointer) in caches.items():
            
            common_elements = len(set_tokens & set(list_cache_tokens))
            if common_elements > max_match:
                max_match = common_elements
                result_tokens = list_cache_tokens
                result_pointer = pointer
    
        return (max_match, result_tokens, result_pointer)
        
    def get_cache(self, tokens:List[int], id: str) -> Optional[Tuple[int, POINTER(KVCache)]]:
        with self.lock: 
            (pos, pre_tokens, kv_cache) = self.find_most_matching_cache(tokens[:-1], self.caches)
            if kv_cache:
                self.in_use_caches[id] = (pos, pre_tokens, kv_cache)
                self.caches.pop(id)
                return (pos, kv_cache)
            elif (len(self.in_use_caches) + len(self.caches)) < self.max_caches:
                new_kv_cache = self.model.create_kv_cache()
                self.in_use_caches[id] = (0, tokens, new_kv_cache)
                return (0, new_kv_cache)
            else:
                return None
            
    def release_cache(self, request: RequestMeta):
        with self.lock:
            self.in_use_caches.pop(request.id_)
            self.caches[request.id_] = (request.tokens_, request.kv_cache_)

kv_cache_pool = KVCachePool(model)

class BatchProcessor:
    def __init__(self, model, kv_cache_pool: KVCachePool):
        self.model = model
        self.kv_cache_pool = kv_cache_pool 
        self.request_queue = deque()
        self.response_queues: Dict[str, deque] = {}
        self.lock = threading.Lock()
        self.max_batch_size = 8  # Adjust based on your hardware capabilities
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    async def process_batches(self):
        while not self.stop_event.is_set():
            with self.lock:
                if len(self.request_queue) >= 1: 
                    current_batch = []
                    batch_ids = []
                    response_queues = []
                    batch_caches = []

                    while len(current_batch) < self.max_batch_size and self.request_queue:
                        request_meta, response_queue = self.request_queue.popleft()
                        kvcache = self.kv_cache_pool.get_cache(request_meta)
                        if kvcache is None:
                            self.request_queue.appendleft((request_meta, response_queue))
                            break
                        current_batch.append(request_meta.data_)
                        batch_ids.append(request_meta.id_)
                        response_queues.append(response_queue)
                        batch_caches.append(kvcache)

                    if current_batch:
                        try:
                            async for batch_output in self.model.chat_stream_async():
                                for i, (output, response_queue) in enumerate(zip(batch_output, response_queues)):
                                    if output:
                                        response_queue.append(output)
                            for response_queue in response_queues:
                                response_queue.append(None)
                                
                        except Exception as e:
                            print(f"Error processing batch: {e}")
                            for response_queue in response_queues:
                                response_queue.append(None)
            await asyncio.sleep(0.01)  # Small sleep to prevent busy waiting
    
    def add_request(self, request_meta: RequestMeta) -> deque:
        response_queue = deque()
        with self.lock:
            self.request_queue.append((request_meta, response_queue))
            self.response_queues[request_meta.id_] = response_queue
        return response_queue

    def cleanup_request(self, request_meta: RequestMeta):
        with self.lock:
            self.kv_cache_pool.release_cache(request_meta)
            if request_id in self.response_queues:
                del self.response_queues[request_meta.id_]

    def start(self):
        self.thread = threading.Thread(target=lambda: asyncio.run(self.process_batches()))
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.stop_event.set()
        self.thread.join()

batch_processor = BatchProcessor(model, kv_cache_pool)
batch_processor.start()

@app.on_event("shutdown")
def shutdown_event():
    batch_processor.stop()

async def chat_stream(request_meta: RequestMeta):
    response_queue = batch_processor.add_request(request_meta)
    try:
        chunk = json.dumps(
            {
                "id": request_meta.id_,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "jiuge",
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            },
            ensure_ascii=False,
        )
        yield f"{chunk}\n\n"

        while True:
            if await request_meta.request_.is_disconnected():
                print(f"Client disconnected for request {request_meta.id_}")
                break
            try:
                token = response_queue.popleft() if response_queue else None
            except IndexError:
                await asyncio.sleep(0.01)
                continue
                
            if token is None:  # End of stream
                break

            chunk = json.dumps(
                {
                    "id": request_meta.id_,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "jiuge",
                    "system_fingerprint": None,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                },
                ensure_ascii=False,
            )
            yield f"{chunk}\n\n"
    finally:
        chunk = json.dumps(
            {
                "id": request_meta.id_,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "jiuge",
                "system_fingerprint": None,
                "choices": [
                    {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}
                ],
            },
            ensure_ascii=False,
        )
        yield f"{chunk}\n\n"
        batch_processor.cleanup_request(request_meta)


def chat(id_, request_data):
    output_text = model.chat(
        request_data,
        kv_cache,
    )

    response = {
        "id": id_,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "jiuge",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text.strip()},
                "finish_reason": "stop",
            }
        ],
    }
    return JSONResponse(response)


@app.post("/jiuge/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()

    if not data.get("messages"):
        return JSONResponse(content={"error": "No message provided"}, status_code=400)
    messages = data.get("messages")
    input_content = self.tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    tokens = self.tokenizer.encode(input_content)
    args = RandSampleArgs(
        data.get("max_tokens", 512), 
        data.get("temperature", 1.0), 
        data.get("top_k", 1), 
        data.get("top_p", 1.0))
    stream = data.get("stream", False)
    id = f"cmpl-{uuid.uuid4().hex}"
    kv_cache = kv_cache_pool.get_cache(tokens, id)
    if kv_cache is None:
        return JSONResponse(content={"error": "No enough memory for KVCache."}, status_code=429)
    request_meta = RequestMeta(id, tokens, args, kv_cache)
    if stream:
        return StreamingResponse(
            chat_stream(request_meta), media_type="text/event-stream"
        )
    else:
        return chat(id, data)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
