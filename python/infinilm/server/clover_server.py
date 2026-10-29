"""OpenAI-compatible serial serving for the native Clover diffusion engine."""

import argparse
import asyncio
import time
import uuid

from fastapi import FastAPI, HTTPException
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


def create_app(
    generator, *, policy="baseline", expert_budget=False, frontier_floor=0.9
):
    """Serialize requests because each worker owns one recurrent state slot."""
    lock = asyncio.Lock()

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "clover", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest):
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
        async with lock:
            try:
                result = await asyncio.to_thread(
                    generator.generate,
                    request.messages,
                    max_new_tokens=request.max_tokens,
                    policy=request.diffusion_policy or policy,
                    frontier_confidence_floor=frontier_floor
                    if request.frontier_confidence_floor is None
                    else request.frontier_confidence_floor,
                    expert_budget=expert_budget
                    if request.expert_budget is None
                    else request.expert_budget,
                )
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
                    if output_count >= request.max_tokens
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
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument(
        "--policy", choices=("baseline", "frontier"), default="baseline"
    )
    parser.add_argument("--expert-budget", action="store_true")
    parser.add_argument("--frontier-floor", type=float, default=0.9)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    generator = CloverGenerator(args.model, tp=args.tp, max_context=args.max_context)
    app = create_app(
        generator,
        policy=args.policy,
        expert_budget=args.expert_budget,
        frontier_floor=args.frontier_floor,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
