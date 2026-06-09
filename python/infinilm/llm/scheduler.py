"""
Scheduler - Request scheduling and batch management with Paged Attention KV Cache.
"""

import os
import queue
import janus
import logging
from dataclasses import dataclass
from typing import List, Optional

from infinilm.compile.env import (
    long_prefill_threshold,
    max_num_batched_tokens,
    v1_scheduler_enabled,
)
from infinilm.llm.request import RequestStatus, InferenceRequest
from infinilm.llm.cache_manager import BlockManager

logger = logging.getLogger(__name__)

# Piecewise CG power-ladder cap (matches piecewise_bucket_policy.hpp).
_PACK_BUCKET_CAP = 8192


def _padded_bucket(total_q: int) -> int:
    """Pad total Q token count to the next power-of-two bucket (512..8192)."""
    if total_q <= 0:
        return 0
    if total_q > _PACK_BUCKET_CAP:
        return total_q
    bucket = 512
    while bucket < total_q:
        bucket *= 2
    return bucket


@dataclass
class ScheduledRow:
    """Per-request scheduling metadata for one engine step."""

    request: InferenceRequest
    num_scheduled_tokens: int
    is_prefill_row: bool
    is_final_prefill_chunk: bool


class SchedulingBudget:
    """Token and sequence budget with per-request deduplication."""

    def __init__(self, token_budget: int, max_num_seqs: int):
        self._token_budget = token_budget
        self._max_num_seqs = max_num_seqs
        self._num_scheduled_tokens = 0
        self._scheduled_req_ids: set[str] = set()

    def remaining_tokens(self) -> int:
        return self._token_budget - self._num_scheduled_tokens

    def remaining_seqs(self) -> int:
        return self._max_num_seqs - len(self._scheduled_req_ids)

    def can_add(self, request_id: str, num_tokens: int) -> bool:
        if num_tokens <= 0:
            return False
        if (
            request_id not in self._scheduled_req_ids
            and self.remaining_seqs() <= 0
        ):
            return False
        return num_tokens <= self.remaining_tokens()

    def add(self, request_id: str, num_tokens: int) -> None:
        self._num_scheduled_tokens += num_tokens
        self._scheduled_req_ids.add(request_id)


class SchedulerOutput:
    """Scheduler output containing scheduled requests and execution phase info."""

    def __init__(
        self,
        scheduled_requests: List[InferenceRequest],
        is_prefill: bool = False,
        rows: Optional[List[ScheduledRow]] = None,
    ):
        self.scheduled_requests = scheduled_requests
        self.num_requests = len(scheduled_requests)
        self.is_prefill = is_prefill
        self.rows: List[ScheduledRow] = rows or []

    @property
    def total_scheduled_tokens(self) -> int:
        if self.rows:
            return sum(r.num_scheduled_tokens for r in self.rows)
        return 0

    def is_homogeneous_prefill(self) -> bool:
        return bool(self.rows) and all(r.is_prefill_row for r in self.rows)

    def is_homogeneous_decode(self) -> bool:
        return bool(self.rows) and all(not r.is_prefill_row for r in self.rows)

    @property
    def scheduling_mode(self) -> str:
        if not self.rows:
            return "PREFILL" if self.is_prefill else "DECODE"
        if self.is_homogeneous_prefill():
            return "PREFILL"
        if self.is_homogeneous_decode():
            return "DECODE"
        return "MIXED"


class Scheduler:
    """Request scheduler with integrated BlockManager for KV cache management.

    Legacy (``INFINI_V1_SCHEDULER=0``) scheduling priority:
      1. New prefill (waiting_queue) — minimize TTFT for new requests.
      2. Decode (running_queue).
      3. Continue chunked-prefill (chunking_queue) — pack up to max_batch_size.

    V1 (``INFINI_V1_SCHEDULER=1``) uses a token budget and two queues only:
      1. Running (decode + chunked-prefill continuations).
      2. Waiting (new prefills into remaining budget).
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        num_blocks: int = 512,
        block_size: int = 256,
        max_prefill_batch_size: Optional[int] = None,
        enable_prefix_cache: bool = True,
        max_waiting_yields: int = 4,
    ):
        self.waiting_queue = janus.Queue()
        self.running_queue = janus.Queue()
        self.chunking_queue = janus.Queue()
        self.max_batch_size = max_batch_size
        env_prefill_batch = os.environ.get("INFINI_MAX_PREFILL_BATCH")
        if max_prefill_batch_size is None and env_prefill_batch:
            max_prefill_batch_size = int(env_prefill_batch)
        self.max_prefill_batch_size = max_prefill_batch_size
        self.max_capture_req = max_prefill_batch_size or max_batch_size

        self.cache_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
            enable_prefix_cache=enable_prefix_cache,
        )
        self.block_size = block_size

        self._waiting_yields_in_a_row: int = 0
        self.max_waiting_yields: int = max_waiting_yields

    def add_request(self, request: InferenceRequest):
        if request is not None:
            request.status = RequestStatus.WAITING
            self.waiting_queue.sync_q.put(request)

    def schedule(self) -> Optional[SchedulerOutput]:
        """Schedule and return batch of requests to execute."""
        if v1_scheduler_enabled():
            return self._schedule_v1()
        return self._schedule_legacy()

    def _schedule_legacy(self) -> Optional[SchedulerOutput]:
        if self._waiting_yields_in_a_row >= self.max_waiting_yields:
            chunking_out = self._try_schedule_chunking()
            if chunking_out is not None:
                self._waiting_yields_in_a_row = 0
                return chunking_out

        chunking_was_nonempty = self.chunking_queue.sync_q.qsize() > 0
        waiting_out = self._try_schedule_waiting()
        if waiting_out is not None:
            if chunking_was_nonempty:
                self._waiting_yields_in_a_row += 1
            else:
                self._waiting_yields_in_a_row = 0
            return waiting_out

        chunking_was_nonempty = self.chunking_queue.sync_q.qsize() > 0
        decode_out = self._try_schedule_decode()
        if decode_out is not None:
            if chunking_was_nonempty:
                self._waiting_yields_in_a_row += 1
            else:
                self._waiting_yields_in_a_row = 0
            return decode_out

        chunking_out = self._try_schedule_chunking()
        if chunking_out is not None:
            self._waiting_yields_in_a_row = 0
            return chunking_out

        return None

    def _schedule_v1(self) -> Optional[SchedulerOutput]:
        budget = SchedulingBudget(
            max_num_batched_tokens(),
            self.max_batch_size,
        )
        rows: List[ScheduledRow] = []
        scheduled_requests: List[InferenceRequest] = []
        chunk_size_hint = 0

        running_snapshot = self._drain_running_queue()
        deferred_running: List[InferenceRequest] = []
        decode_pending: List[InferenceRequest] = []

        for req in running_snapshot:
            if req.is_finished():
                self.complete_requests([req])
                continue
            if req.is_prefill and req.prefill_debt() > 0:
                q = self._next_prefill_chunk_len(req, budget)
                if q <= 0 or not budget.can_add(req.request_id, q):
                    deferred_running.append(req)
                    continue
                if rows and not self._can_add_v1_prefill_row(
                    rows, req, chunk_size_hint
                ):
                    deferred_running.append(req)
                    continue
                if not scheduled_requests and req.chunk_size > 0:
                    chunk_size_hint = req.chunk_size
                rows.append(
                    ScheduledRow(
                        req,
                        q,
                        True,
                        self._is_final_prefill_chunk(req, q),
                    )
                )
                scheduled_requests.append(req)
                budget.add(req.request_id, q)
            elif not req.is_prefill:
                decode_pending.append(req)
            else:
                deferred_running.append(req)

        prefill_batch_cap = min(
            self.max_batch_size,
            self.max_prefill_batch_size or self.max_batch_size,
        )

        while len(scheduled_requests) < prefill_batch_cap:
            try:
                req = self.waiting_queue.sync_q.get_nowait()
            except queue.Empty:
                break

            if req.is_finished():
                self.complete_requests([req])
                continue

            if not self.can_accept_request(req):
                self.waiting_queue.sync_q.put(req)
                break

            req_tokens = req.get_input_tokens()
            num_required_blocks = req.get_num_blocks_required(self.block_size)

            if not self.cache_manager.can_allocate(num_required_blocks):
                if not self.cache_manager.try_free_blocks(num_required_blocks):
                    raise RuntimeError("No available cache blocks for new request")

            if not req.block_table:
                req.block_table, req.slot_mapping, req.num_cached_tokens = (
                    self.cache_manager.allocate_blocks(req_tokens, req.block_table)
                )

            req.num_blocks = len(req.block_table)
            req.status = RequestStatus.RUNNING

            remaining = req.prompt_length - req.num_cached_tokens
            if req.chunk_size > 0 and remaining > req.chunk_size:
                req.chunk_prefill_offset = req.num_cached_tokens

            q = self._next_prefill_chunk_len(req, budget)
            if q <= 0 or not budget.can_add(req.request_id, q):
                req.status = RequestStatus.WAITING
                self.waiting_queue.sync_q.put(req)
                break

            if scheduled_requests:
                if chunk_size_hint == 0:
                    chunk_size_hint = req.chunk_size
                if not self._can_add_v1_prefill_row(rows, req, chunk_size_hint):
                    req.status = RequestStatus.WAITING
                    self.waiting_queue.sync_q.put(req)
                    break
            elif req.chunk_size > 0:
                chunk_size_hint = req.chunk_size

            rows.append(
                ScheduledRow(
                    req,
                    q,
                    True,
                    self._is_final_prefill_chunk(req, q),
                )
            )
            scheduled_requests.append(req)
            budget.add(req.request_id, q)

        for req in decode_pending:
            if len(scheduled_requests) >= self.max_batch_size:
                deferred_running.append(req)
                continue
            if not budget.can_add(req.request_id, 1):
                deferred_running.append(req)
                continue
            try:
                req.block_table, new_slot = self.cache_manager.append_slot(
                    req.block_table,
                    req.get_total_length(),
                    req.get_all_token_ids(),
                )
                req.slot_mapping = [new_slot]
                req.num_blocks = len(req.block_table)
                req.num_cached_tokens = req.get_total_length() - 1
            except RuntimeError:
                deferred_running.append(req)
                continue
            rows.append(ScheduledRow(req, 1, False, False))
            scheduled_requests.append(req)
            budget.add(req.request_id, 1)

        for req in deferred_running:
            self.running_queue.sync_q.put(req)

        if not scheduled_requests:
            return None

        prefill_rows = [r.request for r in rows if r.is_prefill_row]
        self._log_prefill_pack(prefill_rows)
        self._log_v1_step(rows)

        return SchedulerOutput(
            scheduled_requests=scheduled_requests,
            is_prefill=all(r.is_prefill_row for r in rows),
            rows=rows,
        )

    def _drain_running_queue(self) -> List[InferenceRequest]:
        reqs: List[InferenceRequest] = []
        size = self.running_queue.sync_q.qsize()
        for _ in range(size):
            try:
                reqs.append(self.running_queue.sync_q.get_nowait())
            except queue.Empty:
                break
        return reqs

    def _next_prefill_chunk_len(
        self, req: InferenceRequest, budget: SchedulingBudget
    ) -> int:
        q = self._prefill_compute_len(req)
        if q <= 0:
            return 0
        threshold = long_prefill_threshold()
        return min(q, threshold, budget.remaining_tokens())

    def _is_final_prefill_chunk(self, req: InferenceRequest, q: int) -> bool:
        return req.num_computed_tokens + q >= req.prompt_length

    def _can_add_v1_prefill_row(
        self,
        rows: List[ScheduledRow],
        candidate: InferenceRequest,
        chunk_size: int,
    ) -> bool:
        if not rows:
            return True
        prefill_rows = [r for r in rows if r.is_prefill_row]
        if any(not r.is_prefill_row for r in rows):
            return len(prefill_rows) == 0
        scheduled = [r.request for r in prefill_rows]
        if not self._can_add_to_prefill_pack(
            scheduled, candidate, chunk_size=chunk_size
        ):
            return False
        # Only pack full chunk steps (avoids sub-bucket native CG shape mismatch).
        if chunk_size > 0:
            pack = scheduled + [candidate]
            if any(self._prefill_compute_len(r) != chunk_size for r in pack):
                return False
        return True

    def _log_v1_step(self, rows: List[ScheduledRow]) -> None:
        n_prefill = sum(1 for r in rows if r.is_prefill_row)
        n_decode = len(rows) - n_prefill
        total = sum(r.num_scheduled_tokens for r in rows)
        mode = (
            SchedulerOutput(scheduled_requests=[], rows=rows).scheduling_mode
            if rows
            else "NONE"
        )
        logger.info(
            "scheduled v1 step n_req=%d total_tokens=%d n_prefill=%d n_decode=%d mode=%s",
            len(rows),
            total,
            n_prefill,
            n_decode,
            mode,
        )

    def _pack_token_budget(self, chunk_size: int) -> int:
        """Max sum(compute_len) per prefill pack step (piecewise bucket cap)."""
        return _PACK_BUCKET_CAP

    @staticmethod
    def _prefill_compute_len(req: InferenceRequest) -> int:
        remaining = req.prompt_length - req.num_cached_tokens
        if req.is_chunking():
            start = req.chunk_prefill_offset
            end = min(start + req.chunk_size, len(req.get_input_tokens()))
            return end - start
        return remaining

    @staticmethod
    def _prefill_is_final_chunk(req: InferenceRequest) -> bool:
        if req.chunk_size > 0 and req.is_prefill and req.is_chunking():
            return req.chunk_is_last()
        return True

    @staticmethod
    def _prefix_cache_pack_compatible(reqs: List[InferenceRequest]) -> bool:
        """v1: reject pack when prefix-cache hit lengths differ."""
        cached = [r.num_cached_tokens for r in reqs]
        if any(c > 0 for c in cached):
            return len(set(cached)) == 1
        return True

    def _can_add_to_prefill_pack(
        self,
        scheduled: List[InferenceRequest],
        candidate: InferenceRequest,
        *,
        chunk_size: int,
    ) -> bool:
        if not scheduled:
            return True
        if candidate.chunk_size != scheduled[0].chunk_size:
            return False
        if scheduled[0].chunk_prefill_offset != candidate.chunk_prefill_offset:
            return False
        if not self._prefix_cache_pack_compatible(scheduled + [candidate]):
            return False

        phase_final = self._prefill_is_final_chunk(scheduled[0])
        if self._prefill_is_final_chunk(candidate) != phase_final:
            return False

        total_q = sum(self._prefill_compute_len(r) for r in scheduled)
        cand_q = self._prefill_compute_len(candidate)
        if chunk_size > 0 and cand_q > chunk_size:
            return False
        total_q += cand_q
        budget = self._pack_token_budget(chunk_size)
        if total_q > budget:
            return False
        if _padded_bucket(total_q) > _PACK_BUCKET_CAP:
            return False
        return True

    def _log_prefill_pack(self, scheduled: List[InferenceRequest]) -> None:
        if len(scheduled) <= 1:
            return
        total_q = sum(self._prefill_compute_len(r) for r in scheduled)
        bucket = _padded_bucket(total_q)
        phase = (
            "final"
            if self._prefill_is_final_chunk(scheduled[0])
            else "mid"
        )
        logger.info(
            "scheduled prefill pack n_req=%s total_q=%s bucket=%s phase=%s",
            len(scheduled),
            total_q,
            bucket,
            phase,
        )

    def _prefill_pack_cap(self) -> int:
        """Max requests per prefill pack (matches piecewise CG capture width)."""
        return min(self.max_batch_size, self.max_capture_req)

    def _try_schedule_chunking(self) -> Optional[SchedulerOutput]:
        scheduled: List[InferenceRequest] = []
        chunk_size = 0
        prefill_cap = self._prefill_pack_cap()
        while len(scheduled) < prefill_cap:
            try:
                req = self.chunking_queue.sync_q.get_nowait()
            except queue.Empty:
                break
            if req.is_finished():
                self.complete_requests([req])
                continue
            if not scheduled:
                chunk_size = req.chunk_size
            elif not self._can_add_to_prefill_pack(
                scheduled, req, chunk_size=chunk_size
            ):
                self.chunking_queue.sync_q.put(req)
                break
            scheduled.append(req)
        if scheduled:
            self._log_prefill_pack(scheduled)
            return SchedulerOutput(scheduled, is_prefill=True)
        return None

    def _try_schedule_waiting(self) -> Optional[SchedulerOutput]:
        scheduled_requests: List[InferenceRequest] = []
        prefill_batch_cap = self._prefill_pack_cap()
        chunk_size = 0

        while len(scheduled_requests) < prefill_batch_cap:
            try:
                req = self.waiting_queue.sync_q.get_nowait()
            except queue.Empty:
                break

            if req.is_finished():
                self.complete_requests([req])
                continue

            if not self.can_accept_request(req):
                self.waiting_queue.sync_q.put(req)
                break

            req_tokens = req.get_input_tokens()
            num_required_blocks = req.get_num_blocks_required(self.block_size)

            if not self.cache_manager.can_allocate(num_required_blocks):
                if not self.cache_manager.try_free_blocks(num_required_blocks):
                    raise RuntimeError("No available cache blocks for new request")

            if not req.block_table:
                req.block_table, req.slot_mapping, req.num_cached_tokens = (
                    self.cache_manager.allocate_blocks(req_tokens, req.block_table)
                )

            req.num_blocks = len(req.block_table)
            req.status = RequestStatus.RUNNING

            remaining = req.prompt_length - req.num_cached_tokens
            if req.chunk_size > 0 and remaining > req.chunk_size:
                req.chunk_prefill_offset = req.num_cached_tokens

            if scheduled_requests:
                if chunk_size == 0:
                    chunk_size = req.chunk_size
                if not self._can_add_to_prefill_pack(
                    scheduled_requests, req, chunk_size=chunk_size
                ):
                    req.status = RequestStatus.WAITING
                    self.waiting_queue.sync_q.put(req)
                    break
            else:
                chunk_size = req.chunk_size

            scheduled_requests.append(req)

        if scheduled_requests:
            self._log_prefill_pack(scheduled_requests)
            return SchedulerOutput(
                scheduled_requests=scheduled_requests,
                is_prefill=True,
            )
        return None

    def _try_schedule_decode(self) -> Optional[SchedulerOutput]:
        scheduled_requests: List[InferenceRequest] = []

        while len(scheduled_requests) < self.max_batch_size:
            try:
                req = self.running_queue.sync_q.get_nowait()
            except queue.Empty:
                break

            if req.is_finished():
                self.complete_requests([req])
                continue

            if req.is_prefill:
                logger.warning(
                    "decode scheduler: request %s still in prefill; requeue",
                    req.request_id[:8],
                )
                if req.is_chunking():
                    self.requeue_chunking(req)
                else:
                    self.waiting_queue.sync_q.put(req)
                continue

            try:
                req.block_table, new_slot = self.cache_manager.append_slot(
                    req.block_table, req.get_total_length(), req.get_all_token_ids()
                )
                req.slot_mapping = [new_slot]
                req.num_blocks = len(req.block_table)
                req.num_cached_tokens = req.get_total_length() - 1
                scheduled_requests.append(req)
            except RuntimeError as e:
                raise RuntimeError("No available cache blocks for new token") from e

        if scheduled_requests:
            return SchedulerOutput(
                scheduled_requests=scheduled_requests,
                is_prefill=False,
            )
        return None

    def requeue_chunking(self, req: InferenceRequest):
        """Put a request back into the chunking queue after a chunk has run (legacy)."""
        self.chunking_queue.sync_q.put(req)

    def requeue_running(self, req: InferenceRequest):
        """Put a request back into the running queue after a forward step (v1)."""
        self.running_queue.sync_q.put(req)

    def complete_requests(self, requests: List[InferenceRequest]):
        """Handle completed requests and free their blocks."""
        for req in requests:
            if req.status in [
                RequestStatus.FINISHED,
                RequestStatus.CANCELED,
                RequestStatus.FAILED,
                RequestStatus.TIMEOUT,
            ]:
                if req.block_table:
                    self.cache_manager.free_blocks(req.block_table)

                if req.status == RequestStatus.CANCELED:
                    logger.info(
                        f"Request {req.request_id[:8]}... canceled: {req.finish_reason}"
                    )
                elif req.status == RequestStatus.FAILED:
                    logger.error(
                        f"Request {req.request_id[:8]}... failed: {req.finish_reason}"
                    )
                elif req.status == RequestStatus.TIMEOUT:
                    logger.error(
                        f"Request {req.request_id[:8]}... timed out: {req.finish_reason}"
                    )
            else:
                self.running_queue.sync_q.put(req)

    def can_accept_request(self, request: InferenceRequest) -> bool:
        total_required_blocks = 0

        running_queue_size = self.running_queue.sync_q.qsize()
        for _ in range(running_queue_size):
            req = self.running_queue.sync_q.get()
            remaining_tokens = (
                req.sampling_params.max_tokens - req.get_num_generated_tokens()
            )
            num_blocks_needed = (
                remaining_tokens + self.block_size - 1
            ) // self.block_size
            total_required_blocks += num_blocks_needed
            self.running_queue.sync_q.put(req)

        total_length = request.get_prompt_length()
        total_length += request.sampling_params.max_tokens
        num_blocks_needed = (total_length + self.block_size - 1) // self.block_size
        total_required_blocks += num_blocks_needed

        return total_required_blocks <= self.cache_manager.get_total_usable_blocks()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "num_blocks": self.cache_manager.num_blocks,
            "block_size": self.cache_manager.block_size,
            "num_free_blocks": self.cache_manager.get_num_free_blocks(),
            "num_req_blocks": len(self.cache_manager.req_block_ids),
            "num_used_blocks": len(self.cache_manager.used_block_ids),
        }
