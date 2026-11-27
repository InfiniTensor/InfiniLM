"""Regression tests for cancellation at the single Clover generation slot."""

import asyncio
import threading

import pytest
from fastapi import HTTPException, Request
from infinilm.server.clover_server import _generate_when_connected


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
