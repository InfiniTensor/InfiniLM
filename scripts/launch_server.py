from jiuge import JiugeForCauslLM
from libinfinicore_infer import DeviceType

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import anyio
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
kv_cache = model.create_kv_cache()


def signal_handler(sig, frame):
    print(f"Received signal {sig}, cleaning up...")
    model.drop_kv_cache(kv_cache)
    model.destroy_model_instance()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Handle docker stop / system shutdown

app = FastAPI()

# TO REMOVE: Global lock to ensure only one request is handled at a time
# Remove this after multiple requests handling is implemented
request_lock = anyio.Lock()


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
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


async def chat_stream(id_, request_data, request: Request):
    try:
        await request_lock.acquire()
        chunk = json.dumps(
            chunk_json(id_, content="", role="assistant"),
            ensure_ascii=False,
        )
        yield f"{chunk}\n\n"

        async for token in model.chat_stream_async(request_data, kv_cache):
            if await request.is_disconnected():
                print("Client disconnected. Aborting stream.")
                break
            chunk = json.dumps(
                chunk_json(id_, content=token),
                ensure_ascii=False,
            )
            yield f"{chunk}\n\n"
    finally:
        if request_lock.locked():
            request_lock.release()
        chunk = json.dumps(
            chunk_json(id_, finish_reason="stop"),
            ensure_ascii=False,
        )
        yield f"{chunk}\n\n"


def chat(id_, request_data):
    output_text = model.chat(
        request_data,
        kv_cache,
    )
    response = chunk_json(
        id_, content=output_text.strip(), role="assistant", finish_reason="stop"
    )
    return JSONResponse(response)


@app.post("/jiuge/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()

    if not data.get("messages"):
        return JSONResponse(content={"error": "No message provided"}, status_code=400)

    stream = data.get("stream", False)
    id_ = f"cmpl-{uuid.uuid4().hex}"
    if stream:
        return StreamingResponse(
            chat_stream(id_, data, request), media_type="text/event-stream"
        )
    else:
        return chat(id_, data)


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
