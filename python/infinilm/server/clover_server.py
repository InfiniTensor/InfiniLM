"""OpenAI-compatible serial serving for the native Clover diffusion engine."""

import argparse
import asyncio
import time
import uuid
from functools import partial

import anyio
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from infinilm.diffusion.clover import CloverGenerator


class ChatRequest(BaseModel):
    model: str = "clover"
    messages: list[dict]
    max_tokens: int = Field(default=128, ge=0)
    temperature: float = 0
    stream: bool = False
    diffusion_policy: str | None = None
    expert_budget: bool | None = None
    frontier_confidence_floor: float | None = Field(default=None, ge=0, le=1)
    presence_penalty: float = 0
    frequency_penalty: float = 0
    repetition_penalty: float = 1
    chat_template_kwargs: dict = Field(default_factory=dict)

    # OpenAI clients send `max_completion_tokens`; a server that accepts the
    # field but ignores it would silently generate a different length than
    # requested.
    max_completion_tokens: int | None = Field(default=None, ge=0)
    # Accepted-with-ignored fields the endpoint cannot serve yet: rejecting
    # them up front is honest, silent truncation is not.
    stop: list[str] | None = None
    n: int | None = Field(default=None, ge=1)


async def _generate_when_connected(request, slot, generate):
    """Wait for the one model slot while the client is connected."""
    disconnected = False
    owns_slot = False

    async def watch_disconnect():
        nonlocal disconnected
        # FastAPI has already consumed the chat request body.
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                disconnected = True
                watchers.cancel_scope.cancel()
                return

    def generation_finished(future):
        slot.release()
        if not future.cancelled():
            # The request may have left before the worker raised.
            future.exception()

    try:
        async with anyio.create_task_group() as watchers:
            watchers.start_soon(watch_disconnect)
            await slot.acquire()
            owns_slot = True
            watchers.cancel_scope.cancel()

        # The watcher has exited. Check once more for a disconnect delivered
        # as the previous generation handed off the slot.
        if disconnected or await request.is_disconnected():
            raise HTTPException(499, "Client disconnected.")

        future = asyncio.get_running_loop().run_in_executor(None, generate)
        future.add_done_callback(generation_finished)
        owns_slot = False  # The worker's completion now releases the slot.
        return await asyncio.shield(future)
    finally:
        if owns_slot:
            slot.release()


def create_app(
    generator, *, policy="baseline", expert_budget=False, frontier_floor=0.9
):
    """Serialize requests because each worker owns one recurrent state slot."""
    slot = asyncio.Lock()
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "clover", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest, raw: Request):
        if (
            request.presence_penalty != 0
            or request.frequency_penalty != 0
            or request.repetition_penalty != 1
            or request.chat_template_kwargs.get("enable_thinking", False)
        ):
            raise HTTPException(
                400, "This endpoint requires neutral penalties and thinking disabled."
            )
        if request.stream or request.temperature != 0:
            raise HTTPException(
                400, "Clover currently supports non-streaming greedy requests."
            )
        if request.stop:
            raise HTTPException(
                400, "Stop sequences are not supported by the Clover endpoint."
            )
        if request.n is not None and request.n != 1:
            raise HTTPException(
                400, "Only a single candidate per request is supported."
            )
        if any(
            message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str)
            for message in request.messages
        ):
            raise HTTPException(
                400, "Messages require a supported role and text content."
            )
        if not request.messages:
            raise HTTPException(400, "At least one message is required.")
        # One effective cap, computed once: the generation and the response
        # finish_reason both use it, so an OpenAI `max_completion_tokens`
        # request reports `length` at exactly the cap that was honored.
        max_tokens = (
            request.max_completion_tokens
            if request.max_completion_tokens is not None
            else request.max_tokens
        )
        generate = partial(
            generator.generate,
            request.messages,
            max_new_tokens=max_tokens,
            policy=request.diffusion_policy or policy,
            frontier_confidence_floor=frontier_floor
            if request.frontier_confidence_floor is None
            else request.frontier_confidence_floor,
            expert_budget=expert_budget
            if request.expert_budget is None
            else request.expert_budget,
        )
        try:
            result = await _generate_when_connected(raw, slot, generate)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        prompt = generator.tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_count = len(
            generator.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        output_count = len(result.token_ids)
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": "length"
                    if max_tokens == 0 or output_count >= max_tokens
                    else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_count,
                "completion_tokens": output_count,
                "total_tokens": prompt_count + output_count,
            },
        }

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "device platform: auto, cpu, nvidia/cuda, hygon, metax, moore, "
            "iluvatar, cambricon, ascend"
        ),
    )
    parser.add_argument(
        "--policy", choices=("baseline", "frontier"), default="baseline"
    )
    parser.add_argument("--expert-budget", action="store_true")
    parser.add_argument("--frontier-floor", type=float, default=0.9)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    from infinilm.base_config import BaseConfig

    # Reuse the framework's device resolution (auto-detection + alias
    # normalization) without parsing the server's argv a second time.
    config = BaseConfig.__new__(BaseConfig)
    config.backend = "cpp"  # native backend: probe via infinicore, not vendor CLIs
    config._auto_detected_device = None
    device = config.get_device_str(args.device)
    generator = CloverGenerator(
        args.model, tp=args.tp, max_context=args.max_context, device=device
    )
    app = create_app(
        generator,
        policy=args.policy,
        expert_budget=args.expert_budget,
        frontier_floor=args.frontier_floor,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
