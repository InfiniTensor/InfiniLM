from jiuge import JiugeForCauslLM
from libinfinicore_infer import DeviceType
from kv_cache_pool import KVCachePool, RequestMeta, RandSampleArgs

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import uvicorn
import time
import uuid
import sys
import signal
import json

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
kv_cache_pool = KVCachePool(model)

def signal_handler(sig, frame):
    print(f"Received signal {sig}, cleaning up...")
    model.drop_kv_cache(kv_cache)
    model.destroy_model_instance()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Handle docker stop / system shutdown

app = FastAPI()

async def chat_stream(request_meta: RequestMeta):
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

        async for token in model.chat_stream_async(request_data, kv_cache):
            if await request_meta.request_.is_disconnected():
                print("Client disconnected. Aborting stream.")
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
   (pos, kv_cache) = kv_cache_pool.get_cache(tokens, id)
   if kv_cache is None:
       return JSONResponse(content={"error": "No enough memory for KVCache."}, status_code=429)
   request_meta = RequestMeta(id, tokens, args, request, kv_cache, pos)
   if stream:
       return StreamingResponse(
           chat_stream(request_meta), media_type="text/event-stream"
       )
   else:
       return chat(id, data)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
curl -N -H "Content-Type: application/json" \
     -X POST http://127.0.0.1:8000/jiuge/chat/completions \
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