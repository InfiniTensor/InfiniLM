"""Opt-in, single-owner bounded-hash gateway for full-model replicas."""

import argparse
import asyncio
import contextlib
import fcntl
import json
import logging
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from infinilm.routing.bounded_hash.cache_router_v3 import Router
from infinilm.server.bounded_hash_config import GatewayConfig

logger = logging.getLogger(__name__)


def encode_event(data):
    """Encode one SSE data event."""
    return "data: " + (data if isinstance(data, str) else json.dumps(data)) + "\n\n"


async def sse_events(response):
    """Parse SSE data fields without treating HTTP chunks as tokens."""
    fields = []
    async for line in response.aiter_lines():
        if not line:
            if fields:
                yield "\n".join(fields)
                fields = []
        elif line.startswith("data:"):
            value = line[5:]
            fields.append(value[1:] if value.startswith(" ") else value)
    if fields:
        raise ValueError("Truncated backend SSE event.")


def validate_request(body):
    """Accept the bounded text chat surface before any generation POST."""
    allowed = {
        "messages", "model", "stream", "stream_options", "max_tokens",
        "temperature", "top_p", "top_k",
        "add_generation_prompt", "chat_template_kwargs",
    }
    if not isinstance(body, dict) or set(body) - allowed:
        raise ValueError("Only the documented text chat fields are supported.")
    if type(body.get("max_tokens")) is not int or body["max_tokens"] <= 0:
        raise ValueError("Supply a positive integer `max_tokens`.")
    if not isinstance(body.get("stream", False), bool):
        raise ValueError("`stream` must be boolean.")
    options = body.get("stream_options") or {}
    if not isinstance(options, dict) or set(options) - {"include_usage"} or not isinstance(options.get("include_usage", False), bool):
        raise ValueError("Only boolean `stream_options.include_usage` is supported.")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Supply a nonempty text `messages` array.")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str):
            raise ValueError("Only system, user, and assistant text messages are supported.")
    return dict(body, stream=True, stream_options={"include_usage": True})


class Gateway:
    """Own all admission state in one process and one event loop."""

    def __init__(self, config):
        self.config = config
        self.router = Router(
            config.urls, phase_model=config.phase_model,
            max_inflight=config.max_inflight, kv_tokens=config.kv_tokens,
            page_size=config.page_size, skew=config.skew,
            virtual_nodes=config.virtual_nodes, cache_ttl_s=config.cache_ttl_s,
            queue_timeout_s=config.queue_timeout_s, warm_wait_ms=config.warm_wait_ms,
        )
        self.client = None
        self.quarantined = {}

    async def prepare(self, body):
        """Obtain token contracts from the replicas' existing tokenizers."""
        async def contract(worker):
            response = await self.client.post(
                self.config.urls[worker] + "/internal/bounded-hash/contract",
                headers={"Authorization": "Bearer " + self.config.keys[worker]},
                json={"request": body, "prefix_tokens": self.config.prefix_tokens},
            )
            response.raise_for_status()
            result = response.json()
            self.config.validate_contract(result)
            return result

        contracts = await asyncio.gather(*(contract(i) for i in range(len(self.config.urls))))
        reference = contracts[0]
        equal_fields = ("identity", "token_digest", "prompt_tokens", "output_tokens", "shared_tokens", "prefix_digest")
        for worker, current in enumerate(contracts):
            if any(current.get(field) != reference.get(field) for field in equal_fields):
                raise ValueError("Replica tokenization or model identities disagree.")
            instance = current["backend_instance_id"]
            self.router.observe_backend(worker, instance)
            if worker in self.quarantined and self.quarantined[worker] != instance:
                del self.quarantined[worker]
            self.router.workers[worker].healthy = worker not in self.quarantined
        item = {
            "id": uuid.uuid4().hex,
            "key": reference.get("prefix_digest") or reference["token_digest"],
            **{field: reference[field] for field in ("prompt_tokens", "output_tokens", "shared_tokens", "prefix_digest")},
        }
        if item["output_tokens"] != body["max_tokens"]:
            raise ValueError("Backend output reservation differs from `max_tokens`.")
        if (item["prompt_tokens"] + item["output_tokens"] + self.config.page_size - 1) // self.config.page_size * self.config.page_size > self.config.kv_tokens:
            raise ValueError("The complete input and maximum output exceed replica KV capacity.")
        return item, contracts

    async def generate(self, body, item, contracts):
        """Attempt generation once and release its lease on every exit."""
        lease = None
        success = False
        attempted = False
        usage = None
        done = False
        finish = None
        try:
            lease = await self.router.acquire(item)
            backend = contracts[lease.worker]
            if lease.backend_instance_id != backend["backend_instance_id"]:
                raise ValueError("Backend instance changed while awaiting admission.")
            headers = {
                "Authorization": "Bearer " + self.config.keys[lease.worker],
                "X-InfiniLM-Backend-Instance": backend["backend_instance_id"],
                "X-InfiniLM-Token-Digest": backend["token_digest"],
                "X-InfiniLM-Prefix-Tokens": str(backend["shared_tokens"]),
            }
            attempted = True
            async with self.client.stream(
                "POST", self.config.urls[lease.worker] + "/v1/chat/completions",
                headers=headers, json=body,
            ) as response:
                response.raise_for_status()
                if response.headers.get("X-InfiniLM-Backend-Instance") != backend["backend_instance_id"]:
                    raise ValueError("The generation response has a stale backend identity.")
                async for data in sse_events(response):
                    if done:
                        raise ValueError("Backend sent data after `[DONE]`.")
                    if data == "[DONE]":
                        done = True
                        continue
                    event = json.loads(data)
                    if event.get("error"):
                        raise ValueError("Backend returned an error event.")
                    info = event.get("x_infinilm", {})
                    if info and info.get("backend_instance_id") != backend["backend_instance_id"]:
                        raise ValueError("Missing or stale backend identity in feedback.")
                    self.router.progress(lease, event)
                    if event.get("usage"):
                        usage = dict(event["usage"], x_infinilm=info)
                    for choice in event.get("choices", []):
                        reason = choice.get("finish_reason")
                        if reason is not None:
                            if reason not in {"stop", "length"}:
                                raise ValueError("Backend generation did not finish normally.")
                            finish = reason
                    yield event
            if not done or finish not in {"stop", "length"} or not usage:
                raise ValueError("Backend stream is incomplete.")
            generated = usage.get("completion_tokens")
            if type(generated) is not int or not 0 <= generated <= item["output_tokens"] or usage.get("prompt_tokens") != item["prompt_tokens"]:
                raise ValueError("Backend usage violates the reserved token contract.")
            success = True
        finally:
            if lease is not None:
                if attempted and not success:
                    # An interrupted POST may still execute. Restore admission only
                    # after a new backend instance proves the old work has gone.
                    self.quarantined[lease.worker] = contracts[lease.worker]["backend_instance_id"]
                    self.router.workers[lease.worker].healthy = False
                self.router.release(lease, success, usage)


def create_app(config):
    """Build an application whose CLI owns one exclusive admission state."""
    gateway = Gateway(config)

    @asynccontextmanager
    async def lifespan(app):
        with config.state_lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            transport = httpx.AsyncHTTPTransport(retries=0)
            async with httpx.AsyncClient(
                transport=transport, timeout=config.request_timeout_s,
                follow_redirects=False, trust_env=False,
            ) as client:
                gateway.client = client
                try:
                    yield
                finally:
                    gateway.router.close()

    app = FastAPI(lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        try:
            original = await request.json()
            body = validate_request(original)
            item, contracts = await gateway.prepare(body)
        except (ValueError, KeyError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {400, 422}:
                return JSONResponse({"error": "The request is unsupported by the replica pool."}, status_code=400)
            return JSONResponse({"error": "Backend token contract is unavailable."}, status_code=503)
        except httpx.HTTPError:
            return JSONResponse({"error": "Backend token contract is unavailable."}, status_code=503)

        async def streaming():
            try:
                async with contextlib.aclosing(gateway.generate(body, item, contracts)) as events:
                    async for event in events:
                        event.pop("x_infinilm", None)
                        if not event.get("choices") and not event.get("usage"):
                            continue
                        if event.get("usage") and not (original.get("stream_options") or {}).get("include_usage", False):
                            if not event.get("choices"):
                                continue
                            event.pop("usage", None)
                        yield encode_event(event)
                yield encode_event("[DONE]")
            except (ValueError, RuntimeError, TimeoutError, httpx.HTTPError):
                logger.exception("Bounded-hash generation failed.")
                yield encode_event({"error": {"message": "Backend generation failed.", "type": "server_error"}})
                yield encode_event("[DONE]")

        if original.get("stream", False):
            return StreamingResponse(streaming(), media_type="text/event-stream")

        async def aggregate():
            content = []
            result = None
            usage = None
            finish = None
            async with contextlib.aclosing(gateway.generate(body, item, contracts)) as events:
                async for event in events:
                    result = event
                    usage = event.get("usage") or usage
                    for choice in event.get("choices", []):
                        content.append(choice.get("delta", {}).get("content") or "")
                        finish = choice.get("finish_reason") or finish
            return {
                "id": result["id"], "object": "chat.completion",
                "created": result["created"], "model": result["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(content)}, "finish_reason": finish, "logprobs": None}],
                "usage": usage,
            }

        async def disconnected():
            while not await request.is_disconnected():
                await asyncio.sleep(0.05)

        work = asyncio.create_task(aggregate())
        monitor = asyncio.create_task(disconnected())
        try:
            finished, _ = await asyncio.wait((work, monitor), return_when=asyncio.FIRST_COMPLETED)
            if work in finished:
                return JSONResponse(await work)
            return JSONResponse({"error": "Client disconnected."}, status_code=499)
        except (ValueError, RuntimeError, TimeoutError, httpx.HTTPError):
            logger.exception("Bounded-hash generation failed.")
            return JSONResponse({"error": "Backend generation failed."}, status_code=502)
        finally:
            for task in (work, monitor):
                task.cancel()
            await asyncio.gather(work, monitor, return_exceptions=True)

    return app


def main():
    """Run exactly one gateway process; never load an inference model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = GatewayConfig(args.config)
    uvicorn.run(create_app(config), host=config.host, port=config.port, workers=1)


if __name__ == "__main__":
    main()
