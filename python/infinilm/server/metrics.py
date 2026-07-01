"""In-process Prometheus-style metrics for inference_server."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple


# TTFT / E2E histogram buckets (seconds).
_DEFAULT_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    float("inf"),
)

_MAX_SAMPLES = 4096


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


class _Counter:
    def __init__(self, name: str, help_text: str, label_names: Tuple[str, ...] = ()):
        self.name = name
        self.help = help_text
        self.label_names = label_names
        self._values: Dict[Tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, **labels: str) -> None:
        key = tuple(labels.get(k, "") for k in self.label_names)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def get(self, **labels: str) -> float:
        key = tuple(labels.get(k, "") for k in self.label_names)
        with self._lock:
            return self._values.get(key, 0.0)

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            out: Dict[str, float] = {}
            for key, val in self._values.items():
                if self.label_names:
                    suffix = "_".join(
                        f"{n}_{v}" for n, v in zip(self.label_names, key) if v
                    )
                    out[suffix or "total"] = val
                else:
                    out["total"] = val
            return out

    def prometheus_lines(self) -> List[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            if not self._values:
                if self.label_names:
                    label_str = ",".join(f'{n}=""' for n in self.label_names)
                    lines.append(f"{self.name}{{{label_str}}} 0")
                else:
                    lines.append(f"{self.name} 0")
                return lines
            for key, val in sorted(self._values.items()):
                if self.label_names:
                    parts = [
                        f'{n}="{_escape_label(str(v))}"'
                        for n, v in zip(self.label_names, key)
                    ]
                    lines.append(f"{self.name}{{{','.join(parts)}}} {val}")
                else:
                    lines.append(f"{self.name} {val}")
        return lines


class _Histogram:
    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: Iterable[float] = _DEFAULT_BUCKETS,
    ):
        self.name = name
        self.help = help_text
        self.buckets = tuple(buckets)
        self._count = 0
        self._sum = 0.0
        self._bucket_counts: Dict[float, int] = {b: 0 for b in self.buckets}
        self._samples: Deque[float] = deque(maxlen=_MAX_SAMPLES)
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._count += 1
            self._sum += value
            self._samples.append(value)
            for b in self.buckets:
                if value <= b:
                    self._bucket_counts[b] += 1

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            samples = sorted(self._samples)
            return {
                "count": float(self._count),
                "sum": self._sum,
                "p50": _percentile(samples, 0.50),
                "p99": _percentile(samples, 0.99),
            }

    def prometheus_lines(self) -> List[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            cumulative = 0
            for b in sorted(b for b in self.buckets if b != float("inf")):
                cumulative += self._bucket_counts.get(b, 0)
                le = "+Inf" if b == float("inf") else str(b)
                lines.append(f'{self.name}_bucket{{le="{le}"}} {cumulative}')
            cumulative = self._count
            lines.append(f'{self.name}_bucket{{le="+Inf"}} {cumulative}')
            lines.append(f"{self.name}_sum {self._sum}")
            lines.append(f"{self.name}_count {self._count}")
        return lines


class _Gauge:
    def __init__(self, name: str, help_text: str, label_names: Tuple[str, ...] = ()):
        self.name = name
        self.help = help_text
        self.label_names = label_names
        self._values: Dict[Tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **labels: str) -> None:
        key = tuple(labels.get(k, "") for k in self.label_names)
        with self._lock:
            self._values[key] = value

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            out: Dict[str, float] = {}
            for key, val in self._values.items():
                if self.label_names:
                    suffix = "_".join(
                        f"{n}_{v}" for n, v in zip(self.label_names, key) if v
                    )
                    out[suffix or "value"] = val
                else:
                    out["value"] = val
            return out

    def prometheus_lines(self) -> List[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        with self._lock:
            if not self._values:
                if self.label_names:
                    label_str = ",".join(f'{n}=""' for n in self.label_names)
                    lines.append(f"{self.name}{{{label_str}}} 0")
                else:
                    lines.append(f"{self.name} 0")
                return lines
            for key, val in sorted(self._values.items()):
                if self.label_names:
                    parts = [
                        f'{n}="{_escape_label(str(v))}"'
                        for n, v in zip(self.label_names, key)
                    ]
                    lines.append(f"{self.name}{{{','.join(parts)}}} {val}")
                else:
                    lines.append(f"{self.name} {val}")
        return lines


class MetricsRegistry:
    """Thread-safe metrics registry for inference_server."""

    def __init__(self, server_id: str):
        self.server_id = server_id
        self.requests_total = _Counter(
            "infinilm_requests_total",
            "Total inference requests by terminal status",
            ("status",),
        )
        self.request_ttft_seconds = _Histogram(
            "infinilm_request_ttft_seconds",
            "Time to first token in seconds",
        )
        self.request_e2e_seconds = _Histogram(
            "infinilm_request_e2e_seconds",
            "End-to-end request latency in seconds",
        )
        self.request_itl_seconds = _Histogram(
            "infinilm_request_itl_seconds",
            "Inter-token latency in seconds",
        )
        self.request_tokens_total = _Counter(
            "infinilm_request_tokens_total",
            "Token counts by kind",
            ("kind",),
        )
        self.engine_step_seconds = _Histogram(
            "infinilm_engine_step_seconds",
            "Engine step loop duration in seconds",
        )
        self.engine_queue_size = _Gauge(
            "infinilm_engine_queue_size",
            "Scheduler queue depth",
            ("state",),
        )
        self.engine_free_blocks = _Gauge(
            "infinilm_engine_free_blocks",
            "Free KV cache blocks",
        )
        self.engine_used_blocks = _Gauge(
            "infinilm_engine_used_blocks",
            "Used KV cache blocks",
        )

    def record_request_finish(
        self,
        *,
        status: str,
        arrival_time: float,
        finished_time: Optional[float],
        first_token_time: Optional[float],
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        self.requests_total.inc(status=status)
        self.request_tokens_total.inc(prompt_tokens, kind="prompt")
        self.request_tokens_total.inc(completion_tokens, kind="completion")
        if first_token_time is not None:
            self.request_ttft_seconds.observe(max(0.0, first_token_time - arrival_time))
        if finished_time is not None:
            self.request_e2e_seconds.observe(max(0.0, finished_time - arrival_time))

    def record_inter_token_latency(self, itl_seconds: float) -> None:
        if itl_seconds > 0.0:
            self.request_itl_seconds.observe(itl_seconds)

    def refresh_engine_gauges(self, engine: Any) -> None:
        """Best-effort read of scheduler cache/queue stats."""
        if engine is None:
            return
        try:
            inner = getattr(engine, "engine", engine)
            sched = getattr(inner, "scheduler", None)
            if sched is None:
                return
            if hasattr(sched, "get_cache_stats"):
                stats = sched.get_cache_stats()
                self.engine_free_blocks.set(float(stats.get("num_free_blocks", 0)))
                self.engine_used_blocks.set(float(stats.get("num_used_blocks", 0)))
            waiting = getattr(getattr(sched, "waiting_queue", None), "sync_q", None)
            running = getattr(getattr(sched, "running_queue", None), "sync_q", None)
            if waiting is not None:
                self.engine_queue_size.set(float(waiting.qsize()), state="waiting")
            if running is not None:
                self.engine_queue_size.set(float(running.qsize()), state="running")
        except Exception:
            pass

    def prometheus_text(self, engine: Any = None) -> str:
        if engine is not None:
            self.refresh_engine_gauges(engine)
        parts: List[str] = []
        for metric in (
            self.requests_total,
            self.request_ttft_seconds,
            self.request_e2e_seconds,
            self.request_itl_seconds,
            self.request_tokens_total,
            self.engine_step_seconds,
            self.engine_queue_size,
            self.engine_free_blocks,
            self.engine_used_blocks,
        ):
            parts.extend(metric.prometheus_lines())
        return "\n".join(parts) + "\n"

    def json_snapshot(self, engine: Any = None) -> Dict[str, Any]:
        if engine is not None:
            self.refresh_engine_gauges(engine)
        counters: Dict[str, float] = {}
        for status in ("ok", "error", "canceled", "timeout"):
            val = self.requests_total.get(status=status)
            if val:
                counters[f"requests_total_{status}"] = val
        for kind in ("prompt", "completion"):
            val = self.request_tokens_total.get(kind=kind)
            if val:
                counters[f"tokens_{kind}_total"] = val

        gauges: Dict[str, float] = {}
        free = self.engine_free_blocks.snapshot().get("value")
        used = self.engine_used_blocks.snapshot().get("value")
        if free is not None:
            gauges["engine_free_blocks"] = free
        if used is not None:
            gauges["engine_used_blocks"] = used
        for state in ("waiting", "running"):
            key = f"state_{state}"
            val = self.engine_queue_size.snapshot().get(key)
            if val is not None:
                gauges[f"engine_queue_{state}"] = val

        return {
            "server_id": self.server_id,
            "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "counters": counters,
            "histograms": {
                "request_ttft_seconds": self.request_ttft_seconds.snapshot(),
                "request_e2e_seconds": self.request_e2e_seconds.snapshot(),
                "request_itl_seconds": self.request_itl_seconds.snapshot(),
            },
            "gauges": gauges,
        }
