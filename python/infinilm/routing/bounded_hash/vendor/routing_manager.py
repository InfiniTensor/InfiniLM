# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 InfiniLM Contributors

from __future__ import annotations

import asyncio
import bisect
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

SUPPORTED_POLICIES = frozenset(
    {"round_robin", "consistent_hashing", "bounded_consistent_hashing"}
)


class NoHealthyWorkerError(RuntimeError):
    pass


def _stable_hash(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def routing_key_digest(value: str) -> str:
    """Return a short, non-reversible identifier suitable for logs."""
    return hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()


@dataclass
class WorkerState:
    member_id: str
    client_info: dict[str, Any] = field(repr=False)
    role: str
    dp_rank: int | None = None
    healthy: bool = True
    active_load: int = 0
    selections: int = 0


class StableHashRing:
    """Cached consistent-hash ring with deterministic virtual nodes."""

    def __init__(self, workers: Sequence[WorkerState], virtual_nodes: int = 128):
        if virtual_nodes < 1:
            raise ValueError("virtual_nodes must be >= 1")

        worker_by_id = {worker.member_id: worker for worker in workers}
        if len(worker_by_id) != len(workers):
            raise ValueError("worker member_id values must be unique")

        entries = [
            (_stable_hash(f"{worker.member_id}:{vnode}"), worker.member_id)
            for worker in workers
            for vnode in range(virtual_nodes)
        ]
        entries.sort()

        self._worker_by_id = worker_by_id
        self._entries = entries
        self._points = [point for point, _ in entries]

    def walk(self, key: str) -> Iterator[WorkerState]:
        if not self._entries:
            return

        start = bisect.bisect_left(self._points, _stable_hash(key))
        seen: set[str] = set()
        for offset in range(len(self._entries)):
            _, member_id = self._entries[(start + offset) % len(self._entries)]
            if member_id in seen:
                continue
            seen.add(member_id)
            yield self._worker_by_id[member_id]
            if len(seen) == len(self._worker_by_id):
                return


@dataclass(frozen=True)
class Selection:
    role: str
    worker: WorkerState
    preferred_member_id: str
    branch: str
    selected_load_before: int


class _WorkerPool:
    def __init__(self, workers: Sequence[WorkerState], virtual_nodes: int):
        if not workers:
            raise ValueError("worker pool must not be empty")
        self.workers = list(workers)
        self.ring = StableHashRing(self.workers, virtual_nodes=virtual_nodes)
        self._round_robin_index = 0

    def _select_round_robin(self) -> tuple[WorkerState, str]:
        for _ in range(len(self.workers)):
            worker = self.workers[self._round_robin_index % len(self.workers)]
            self._round_robin_index = (self._round_robin_index + 1) % len(
                self.workers
            )
            if worker.healthy:
                return worker, "round_robin"
        raise NoHealthyWorkerError("no healthy worker is available")

    def _preferred(self, key: str) -> WorkerState:
        for worker in self.ring.walk(key):
            if worker.healthy:
                return worker
        raise NoHealthyWorkerError("no healthy worker is available")

    def select(
        self,
        role: str,
        policy: str,
        key: str | None,
        max_load_skew: float,
        min_load_gap: int,
    ) -> Selection:
        if policy not in SUPPORTED_POLICIES:
            raise ValueError(f"unsupported routing policy: {policy}")

        if policy == "round_robin" or key is None:
            worker, branch = self._select_round_robin()
            return Selection(
                role=role,
                worker=worker,
                preferred_member_id=worker.member_id,
                branch=branch,
                selected_load_before=worker.active_load,
            )

        preferred = self._preferred(key)
        if policy == "consistent_hashing":
            return Selection(
                role=role,
                worker=preferred,
                preferred_member_id=preferred.member_id,
                branch="consistent_hash_hit",
                selected_load_before=preferred.active_load,
            )

        healthy = [worker for worker in self.workers if worker.healthy]
        loads = [worker.active_load for worker in healthy]
        min_healthy_load = min(loads)
        mean_healthy_load = sum(loads) / len(loads)
        should_spill = (
            preferred.active_load - min_healthy_load > min_load_gap
            and preferred.active_load > mean_healthy_load * max_load_skew
        )
        if not should_spill:
            return Selection(
                role=role,
                worker=preferred,
                preferred_member_id=preferred.member_id,
                branch="bounded_hash_hit",
                selected_load_before=preferred.active_load,
            )

        max_candidate_load = mean_healthy_load * max_load_skew
        candidate = next(
            (
                worker
                for worker in self.ring.walk(key)
                if worker.healthy and worker.active_load <= max_candidate_load
            ),
            None,
        )
        if candidate is None:
            return Selection(
                role=role,
                worker=preferred,
                preferred_member_id=preferred.member_id,
                branch="bounded_no_candidate",
                selected_load_before=preferred.active_load,
            )

        return Selection(
            role=role,
            worker=candidate,
            preferred_member_id=preferred.member_id,
            branch=(
                "bounded_hash_hit"
                if candidate.member_id == preferred.member_id
                else "bounded_spill"
            ),
            selected_load_before=candidate.active_load,
        )


class PairLease:
    """Own active-load reservations for one prefill/decode request pair."""

    def __init__(
        self,
        manager: RoutingManager,
        prefill: Selection,
        decode: Selection,
    ):
        self.manager = manager
        self.prefill = prefill
        self.decode = decode
        self._prefill_released = False
        self._decode_released = False
        self._release_lock = asyncio.Lock()

    async def release_prefill(self) -> None:
        async with self._release_lock:
            if self._prefill_released:
                return
            await self.manager._release(self.prefill.worker)
            self._prefill_released = True

    async def release_decode(self) -> None:
        async with self._release_lock:
            if self._decode_released:
                return
            await self.manager._release(self.decode.worker)
            self._decode_released = True

    async def release_all(self) -> None:
        await self.release_prefill()
        await self.release_decode()


class RoutingManager:
    def __init__(
        self,
        prefill_workers: Sequence[WorkerState],
        decode_workers: Sequence[WorkerState],
        *,
        policy: str = "round_robin",
        virtual_nodes: int = 128,
        prefill_max_load_skew: float = 1.5,
        prefill_min_load_gap: int = 2,
        decode_max_load_skew: float = 1.5,
        decode_min_load_gap: int = 2,
    ):
        if policy not in SUPPORTED_POLICIES:
            raise ValueError(f"unsupported routing policy: {policy}")
        if prefill_max_load_skew < 1.0 or decode_max_load_skew < 1.0:
            raise ValueError("max_load_skew must be >= 1.0")
        if prefill_min_load_gap < 0 or decode_min_load_gap < 0:
            raise ValueError("min_load_gap must be >= 0")

        self.policy = policy
        self.prefill_max_load_skew = prefill_max_load_skew
        self.prefill_min_load_gap = prefill_min_load_gap
        self.decode_max_load_skew = decode_max_load_skew
        self.decode_min_load_gap = decode_min_load_gap
        self.prefill_pool = _WorkerPool(prefill_workers, virtual_nodes)
        self.decode_pool = _WorkerPool(decode_workers, virtual_nodes)
        self._lock = asyncio.Lock()
        self._branch_counts: Counter[str] = Counter()

    async def acquire(self, key: str | None) -> PairLease:
        async with self._lock:
            prefill = self.prefill_pool.select(
                "prefill",
                self.policy,
                key,
                self.prefill_max_load_skew,
                self.prefill_min_load_gap,
            )
            decode = self.decode_pool.select(
                "decode",
                self.policy,
                key,
                self.decode_max_load_skew,
                self.decode_min_load_gap,
            )

            prefill.worker.active_load += 1
            prefill.worker.selections += 1
            decode.worker.active_load += 1
            decode.worker.selections += 1
            self._branch_counts[f"prefill:{prefill.branch}"] += 1
            self._branch_counts[f"decode:{decode.branch}"] += 1
            return PairLease(self, prefill, decode)

    async def _release(self, worker: WorkerState) -> None:
        async with self._lock:
            if worker.active_load <= 0:
                raise RuntimeError(f"active-load underflow for {worker.member_id}")
            worker.active_load -= 1

    async def set_client_health(
        self, role: str, client_info: dict[str, Any], healthy: bool
    ) -> None:
        pool = self.prefill_pool if role == "prefill" else self.decode_pool
        async with self._lock:
            for worker in pool.workers:
                if worker.client_info is client_info:
                    worker.healthy = healthy

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            def pool_snapshot(pool: _WorkerPool) -> list[dict[str, Any]]:
                return [
                    {
                        "member_id": worker.member_id,
                        "healthy": worker.healthy,
                        "active_load": worker.active_load,
                        "selections": worker.selections,
                    }
                    for worker in pool.workers
                ]

            return {
                "policy": self.policy,
                "prefill": pool_snapshot(self.prefill_pool),
                "decode": pool_snapshot(self.decode_pool),
                "branches": dict(self._branch_counts),
            }
