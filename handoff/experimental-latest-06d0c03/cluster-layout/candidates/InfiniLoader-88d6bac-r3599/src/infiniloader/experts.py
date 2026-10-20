"""Bounded, lease-protected GPU expert copies over a canonical CPU TensorStore.

This is an engine integration API and reference dense MoE, not a KTransformers
kernel/quantization adapter. CPU cold experts execute without GPU admission.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field

from .distributed import DTYPE_BYTES


@dataclass
class _Entry:
    weights: dict
    nbytes: int
    users: int = 0
    events: list = field(default_factory=list)


@dataclass
class ExpertLease:
    weights: dict
    device: object
    placement: str
    _streams: list = field(default_factory=list)

    def record_stream(self, stream):
        """Register additional CUDA consumer streams before leaving the lease."""
        if self.placement != "gpu":
            raise ValueError("CPU leases do not use CUDA streams")
        if stream.device != self.device:
            raise ValueError("consumer stream belongs to another device")
        if stream not in self._streams:
            self._streams.append(stream)


class ExpertPool:
    def __init__(self, store, groups: dict[str, dict[str, str]], *, max_gpu_bytes: int, device=None):
        self.torch, self.store = store.torch, store
        if type(max_gpu_bytes) is not int or max_gpu_bytes < 0:
            raise ValueError("max_gpu_bytes must be a nonnegative integer")
        if not groups or any(not isinstance(k, str) or not v for k, v in groups.items()):
            raise ValueError("expert groups require string IDs and nonempty parameter maps")
        self.groups = {k: dict(v) for k, v in groups.items()}
        self.sizes = {k: sum(math.prod(store.catalog["tensors"][n]["shape"]) *
                             DTYPE_BYTES[store.catalog["tensors"][n]["dtype"]] for n in v.values())
                      for k, v in self.groups.items()}
        self.limit = max_gpu_bytes
        self.device = self.torch.device(device) if device is not None else None
        if self.device is not None:
            if self.device.type != "cuda" or not self.torch.cuda.is_available():
                raise ValueError("GPU expert pool requires an available CUDA/ROCm device")
            index = self.device.index if self.device.index is not None else self.torch.cuda.current_device()
            self.device = self.torch.device("cuda", index)
        self.entries = OrderedDict()
        self._lock = threading.RLock()
        self._users = 0
        self._closed = False
        self._stats = dict(gpu_hits=0, cpu_hits=0, promotions=0, evictions=0, uploaded_bytes=0)

    def _check(self):
        if self._closed:
            raise RuntimeError("expert pool is closed")

    def _cpu(self, expert):
        return {role: self.store.tensor(name) for role, name in self.groups[expert].items()}

    def _evict(self, expert):
        entry = self.entries[expert]
        if entry.users:
            raise RuntimeError("expert is leased")
        for event in entry.events:
            event.synchronize()
        del self.entries[expert]
        self._stats["evictions"] += 1

    def promote(self, expert: str) -> bool:
        """Admit the complete expert or return False (CPU fallback is available).

        Transfers are synchronous. The lock reserves capacity across the entire
        allocation, preventing concurrent duplicate uploads and partial publication.
        """
        with self._lock:
            self._check()
            size = self.sizes[expert]
            if expert in self.entries:
                self.entries.move_to_end(expert)
                return True
            if self.device is None or size > self.limit:
                return False
            while sum(e.nbytes for e in self.entries.values()) + size > self.limit:
                victim = next((k for k, e in self.entries.items() if not e.users), None)
                if victim is None:
                    return False
                self._evict(victim)
            weights = {}
            try:
                with self.torch.cuda.device(self.device):
                    for role, tensor in self._cpu(expert).items():
                        weights[role] = tensor.to(self.device, non_blocking=False)
                    self.torch.cuda.current_stream(self.device).synchronize()
            except BaseException:
                if weights:
                    self.torch.cuda.synchronize(self.device)
                weights.clear()
                raise
            self.entries[expert] = _Entry(weights, size)
            self._stats["promotions"] += 1
            self._stats["uploaded_bytes"] += size
            return True

    @contextmanager
    def acquire(self, expert: str, *, stream=None):
        """Execute inside this scope; GPU work defaults to the selected stream.

        Additional streams must be registered using lease.record_stream(). Tensor
        references retained outside the lease must not be used for new operations.
        """
        with self._lock:
            self._check()
            entry = self.entries.get(expert)
            if entry:
                consumer = stream or self.torch.cuda.current_stream(self.device)
                if getattr(consumer, "device", None) != self.device:
                    raise ValueError("consumer stream belongs to another device")
                lease = ExpertLease(dict(entry.weights), self.device, "gpu", [consumer])
                entry.users += 1
                self.entries.move_to_end(expert)
            else:
                if stream is not None:
                    raise ValueError("CPU fallback does not accept a CUDA stream")
                lease = ExpertLease(self._cpu(expert), self.torch.device("cpu"), "cpu")
            self._users += 1
            self._stats["gpu_hits" if entry else "cpu_hits"] += 1
        try:
            if entry:
                with self.torch.cuda.device(self.device), self.torch.cuda.stream(consumer):
                    yield lease
            else:
                yield lease
        finally:
            with self._lock:
                if entry:
                    try:
                        for consumer_stream in lease._streams:
                            event = self.torch.cuda.Event()
                            event.record(consumer_stream)
                            entry.events.append(event)
                            for tensor in entry.weights.values():
                                tensor.record_stream(consumer_stream)
                        # Completed events do not need to accumulate over hot hits.
                        entry.events[:] = [e for e in entry.events if not e.query()]
                    except BaseException:
                        self.torch.cuda.synchronize(self.device)
                        raise
                    finally:
                        lease.weights.clear()
                        entry.users -= 1
                        self._users -= 1
                else:
                    lease.weights.clear()
                    self._users -= 1

    def evict(self, expert):
        with self._lock:
            self._check()
            if expert in self.entries:
                self._evict(expert)

    def status(self):
        with self._lock:
            return dict(self._stats, gpu_weight_bytes=sum(e.nbytes for e in self.entries.values()),
                        max_gpu_bytes=self.limit, active_leases=self._users,
                        resident_experts=list(self.entries), device=str(self.device))

    def close(self):
        with self._lock:
            if self._users:
                raise RuntimeError("expert pool has active leases")
            for expert in list(self.entries):
                self._evict(expert)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def hybrid_moe(pool: ExpertPool, inputs, expert_ids, gates):
    """Reference ReLU MoE with externally provided top-k routing, including repeats.

    Each group maps up/down and optionally up_bias/down_bias to tensor names.
    Returns output on the input device. Engine adapters can instead use acquire()
    around their own CPU/GPU kernels, packed formats and routing implementation.
    """
    torch = pool.torch
    if (inputs.ndim != 2 or expert_ids.ndim != 2 or gates.shape != expert_ids.shape
            or expert_ids.shape[0] != inputs.shape[0] or expert_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("expected inputs[tokens,dim], integer IDs and gates[tokens,top_k]")
    ids = expert_ids.to("cpu")
    if any(str(i) not in pool.groups for i in ids.unique().tolist()):
        raise ValueError("unknown routed expert")
    first = next(iter(pool.groups.values()))
    out_dim = pool.store.catalog["tensors"][first["down"]]["shape"][0]
    output = torch.zeros((inputs.shape[0], out_dim), dtype=inputs.dtype, device=inputs.device)
    with torch.inference_mode():
        for expert in ids.unique().tolist():
            tokens, slots = (ids == expert).nonzero(as_tuple=True)
            with pool.acquire(str(expert)) as lease:
                weights = lease.weights
                activation = inputs.index_select(0, tokens.to(inputs.device)).to(lease.device)
                hidden = torch.nn.functional.linear(activation, weights["up"], weights.get("up_bias")).relu()
                result = torch.nn.functional.linear(hidden, weights["down"], weights.get("down_bias"))
                factors = gates[tokens.to(gates.device), slots.to(gates.device)].to(result.device)
                result = (result * factors[:, None]).to(device=output.device, dtype=output.dtype)
                output.index_add_(0, tokens.to(output.device), result)
    return output
