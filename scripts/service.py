import asyncio
import uuid
import time
import json
import signal
import sys
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from libinfinicore_infer import DeviceType
from jiuge import JiugeForCauslLM
from runner import ModelRunner

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
runner = ModelRunner(model)  # 使用 ModelRunner 替代单实例


# 注册信号处理
async def shutdown():
    print("Shutting down...")
    model.destroy_model_instance()
    sys.exit(0)


def signal_handler(sig, frame):
    asyncio.create_task(shutdown())


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

app = FastAPI()


# 启动后台服务任务
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(runner.service())  # 后台 batch loop


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


@app.post("/jiuge/chat/completions")
async def chat_completions(request: Request):
    request_data = await request.json()
    if not request_data.get("messages"):
        return JSONResponse(content={"error": "No message provided"}, status_code=400)

    stream = request_data.get("stream", False)
    request_id = f"cmpl-{uuid.uuid4().hex}"

    if stream:
        async def stream_result():
            yield json.dumps(
                chunk_json(request_id, content="", role="assistant"), ensure_ascii=False
            ) + "\n\n"
            queue = asyncio.Queue()

            def callback(tokens):
                asyncio.create_task(queue.put(tokens))

            await runner.request(request_data, request_id=request_id, callback=callback)

            while True:
                tokens = await queue.get()
                if not tokens:
                    break
                content = runner.model.tokenizer.decode(tokens)
                yield json.dumps(
                    chunk_json(request_id, content=content), ensure_ascii=False
                ) + "\n\n"

            yield json.dumps(
                chunk_json(request_id, finish_reason="stop"), ensure_ascii=False
            ) + "\n\n"

        return StreamingResponse(stream_result(), media_type="text/event-stream")
    else:
        result = []

        def callback(tokens):
            result.extend(tokens)

        await runner.request(request_data, request_id=request_id, callback=callback)
        await asyncio.sleep(1.0)
        output = runner.model.tokenizer.decode(result)
        # return JSONResponse(
        #     chunk_json(request_id, content=output, finish_reason="stop")
        # )
        return JSONResponse(
            content=json.loads(
                json.dumps(
                    chunk_json(request_id, content=output, finish_reason="stop"),
                    ensure_ascii=False,
                )
            )
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
