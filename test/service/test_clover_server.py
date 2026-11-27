"""CPU tests for the Clover chat endpoint and its serial generation slot."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from jinja2 import TemplateError
from test_agent_protocol import SERVER, load_module

frontier = load_module(
    "clover_frontier_under_test", SERVER.parent / "diffusion/frontier.py"
)
server = load_module(
    "clover_server_under_test",
    SERVER / "clover_server.py",
    {"infinilm.diffusion.frontier": frontier},
)
_generate_when_connected = server._generate_when_connected


def test_chat_template_error_releases_slot_and_valid_request_reports_usage():
    generator = Mock()
    generator.generate.side_effect = [
        TemplateError("No user query found in messages."),
        SimpleNamespace(text="4", token_ids=[19], prompt_token_count=7),
    ]
    with TestClient(server.create_app(generator)) as client:
        rejected = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "system", "content": "System only."}]},
        )
        assert rejected.status_code == 400
        assert rejected.json()["detail"] == "No user query found in messages."
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "2+2?"}]},
        )
    assert response.status_code == 200
    assert response.json()["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 1,
        "total_tokens": 8,
    }
    assert generator.generate.call_count == 2


def test_frontier_floor_above_policy_threshold_is_rejected_before_generation():
    generator = Mock()
    with TestClient(server.create_app(generator)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "2+2?"}],
                "frontier_confidence_floor": 0.96,
            },
        )
    assert response.status_code == 422
    generator.generate.assert_not_called()


@pytest.fixture
def anyio_backend():
    return "asyncio"


def client():
    messages = asyncio.Queue()
    listening = asyncio.Event()

    async def receive():
        listening.set()
        return await messages.get()

    return Request({"type": "http"}, receive), messages, listening


@pytest.mark.anyio
@pytest.mark.parametrize("release_at_disconnect", [False, True])
async def test_disconnected_waiter_never_generates(release_at_disconnect):
    slot = asyncio.Lock()
    await slot.acquire()
    request, messages, listening = client()
    calls = []
    waiter = asyncio.create_task(
        _generate_when_connected(request, slot, lambda: calls.append("disconnected"))
    )
    await asyncio.wait_for(listening.wait(), 2)

    if release_at_disconnect:
        # Make both the slot and the disconnect available in one event-loop turn.
        slot.release()
    messages.put_nowait({"type": "http.disconnect"})
    with pytest.raises(HTTPException) as exc:
        await asyncio.wait_for(waiter, 2)
    assert exc.value.status_code == 499
    assert calls == []
    assert slot.locked() == (not release_at_disconnect)
    if slot.locked():
        slot.release()

    request, _, _ = client()
    assert (
        await asyncio.wait_for(
            _generate_when_connected(request, slot, lambda: "next"), 2
        )
        == "next"
    )


@pytest.mark.anyio
async def test_disconnect_already_buffered_on_admission():
    slot = asyncio.Lock()
    request, messages, _ = client()
    messages.put_nowait({"type": "http.disconnect"})
    with pytest.raises(HTTPException) as exc:
        await _generate_when_connected(request, slot, lambda: pytest.fail("dispatched"))
    assert exc.value.status_code == 499
    assert not slot.locked()


@pytest.mark.anyio
async def test_running_worker_keeps_slot_after_request_cancellation():
    loop = asyncio.get_running_loop()
    slot = asyncio.Lock()
    started = asyncio.Event()
    finish = threading.Event()
    calls = []

    def first():
        calls.append("first")
        loop.call_soon_threadsafe(started.set)
        assert finish.wait(5)
        calls.append("finished")

    request, _, _ = client()
    active = asyncio.create_task(_generate_when_connected(request, slot, first))
    try:
        await asyncio.wait_for(started.wait(), 2)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active

        request, _, listening = client()
        queued = asyncio.create_task(
            _generate_when_connected(request, slot, lambda: calls.append("cancelled"))
        )
        await asyncio.wait_for(listening.wait(), 2)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert slot.locked()

        request, _, listening = client()
        live = asyncio.create_task(
            _generate_when_connected(request, slot, lambda: calls.append("live"))
        )
        await asyncio.wait_for(listening.wait(), 2)
    finally:
        finish.set()
    await asyncio.wait_for(live, 2)
    assert calls == ["first", "finished", "live"]
