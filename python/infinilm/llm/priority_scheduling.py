"""Priority ordering helpers for inference request admission."""

import queue
import time
from collections import Counter
from threading import Lock
from typing import Callable

from infinilm.llm.request import InferenceRequest


class PrioritySchedulingStats:
    """Track cumulative priority admission statistics."""

    def __init__(self) -> None:
        self.admitted_requests = 0
        self.aged_admissions = 0
        self.total_wait_time_seconds = 0.0
        self.max_wait_time_seconds = 0.0
        self.admitted_by_priority: Counter[int] = Counter()
        self._lock = Lock()

    def record_admission(
        self,
        request: InferenceRequest,
        now: float,
        aging_interval: float,
    ) -> tuple[float, int, bool]:
        """Record one admission and return its scheduling details."""
        wait_time = request_wait_time(request, now)
        admission_priority = effective_priority(request, now, aging_interval)
        aged = admission_priority > request.priority

        with self._lock:
            self.admitted_requests += 1
            self.aged_admissions += int(aged)
            self.total_wait_time_seconds += wait_time
            self.max_wait_time_seconds = max(self.max_wait_time_seconds, wait_time)
            self.admitted_by_priority[request.priority] += 1
        return wait_time, admission_priority, aged

    def snapshot(self) -> dict:
        """Return a JSON-serializable statistics snapshot."""
        with self._lock:
            average_wait_time = (
                self.total_wait_time_seconds / self.admitted_requests
                if self.admitted_requests
                else 0.0
            )
            return {
                "admitted_requests": self.admitted_requests,
                "aged_admissions": self.aged_admissions,
                "average_wait_time_seconds": average_wait_time,
                "max_wait_time_seconds": self.max_wait_time_seconds,
                "admitted_by_priority": dict(sorted(self.admitted_by_priority.items())),
            }


def mark_request_enqueued(
    request: InferenceRequest,
    sequence: int,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Record stable scheduling metadata on a request's first enqueue."""
    if request.scheduling_enqueue_time is None:
        request.scheduling_enqueue_time = clock()
    if request.scheduling_sequence is None:
        request.scheduling_sequence = sequence


def effective_priority(
    request: InferenceRequest,
    now: float,
    aging_interval: float,
) -> int:
    """Return the request priority after applying wait-time aging."""
    if aging_interval <= 0:
        raise ValueError("`aging_interval` must be greater than zero.")
    wait_time = request_wait_time(request, now)
    return request.priority + int(wait_time // aging_interval)


def request_wait_time(request: InferenceRequest, now: float) -> float:
    """Return how long a request has waited for admission."""
    enqueue_time = request.scheduling_enqueue_time
    if enqueue_time is None:
        enqueue_time = now
    return max(0.0, now - enqueue_time)


def drain_priority_ordered(
    sync_queue,
    aging_interval: float,
    clock: Callable[[], float] = time.monotonic,
) -> list[InferenceRequest]:
    """Drain a finite queue snapshot and return requests in admission order."""
    requests = []
    for _ in range(sync_queue.qsize()):
        try:
            requests.append(sync_queue.get_nowait())
        except queue.Empty:
            break

    now = clock()
    requests.sort(
        key=lambda request: (
            -effective_priority(request, now, aging_interval),
            request.scheduling_sequence
            if request.scheduling_sequence is not None
            else float("inf"),
        )
    )
    return requests
