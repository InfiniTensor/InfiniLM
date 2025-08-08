import sys
from vllm.entrypoints.openai.api_server import app
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import uvicorn
import argparse

# ==== CONFIGURATION ====
DEVICE_LIST = [
    "nvidia",
    "cambricon",
    "ascend",
    "metax",
    "moore",
    "iluvatar",
    "kunlun",
]


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
        choices=DEVICE_LIST,
        default="nvidia",
        help="Device type to run the model on (default: nvidia)",
    )
    parser.add_argument(
        "--ndev",
        type=int,
        default=1,
        help="Number of devices to use (default: 1)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )

    return parser.parse_args()


def start_server(model_path, dev, ndev, HOST="0.0.0.0", PORT=8000):
    # Build the engine args
    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=ndev,
        trust_remote_code=True,
    )

    # Create the engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Inject the engine into the FastAPI app
    app.state.engine = engine

    # Launch server with Uvicorn
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    args = parse_args()
    start_server(args.model_path, args.dev, args.ndev, args.host, args.port)
    