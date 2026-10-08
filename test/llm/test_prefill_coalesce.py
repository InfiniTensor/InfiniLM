from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

from infinilm.llm.llm import AsyncLLMEngine


def make_engine(*, waiting=False, running=False, delay=0.02, batch_size=4):
    waiting_queue = Queue()
    running_queue = Queue()
    if waiting:
        waiting_queue.put(object())
    if running:
        running_queue.put(object())
    engine = object.__new__(AsyncLLMEngine)
    engine.config = SimpleNamespace(cache_type="paged", max_batch_size=batch_size)
    engine.engine = SimpleNamespace(
        scheduler=SimpleNamespace(
            waiting_queue=SimpleNamespace(sync_q=waiting_queue),
            running_queue=SimpleNamespace(sync_q=running_queue),
        )
    )
    engine._prefill_coalesce_s = delay
    engine._abort_queue = None
    return engine


def test_coalesces_waiting_prefill_without_running_requests():
    engine = make_engine(waiting=True)
    with patch("infinilm.llm.llm.time.sleep") as sleep:
        engine._coalesce_prefill_if_idle()
    sleep.assert_called_once_with(0.02)


def test_does_not_delay_active_decode():
    engine = make_engine(waiting=True, running=True)
    with patch("infinilm.llm.llm.time.sleep") as sleep:
        engine._coalesce_prefill_if_idle()
    sleep.assert_not_called()


def test_no_delay_without_waiting_prefill():
    engine = make_engine()
    with patch("infinilm.llm.llm.time.sleep") as sleep:
        engine._coalesce_prefill_if_idle()
    sleep.assert_not_called()


def test_default_disabled_and_batch_one_skip_delay():
    for delay, batch_size in ((0, 4), (0.02, 1)):
        engine = make_engine(waiting=True, delay=delay, batch_size=batch_size)
        with patch("infinilm.llm.llm.time.sleep") as sleep:
            engine._coalesce_prefill_if_idle()
        sleep.assert_not_called()
