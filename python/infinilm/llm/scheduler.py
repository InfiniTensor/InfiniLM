"""
Scheduler - Request scheduling and batch management with Paged Attention KV Cache.
"""

import logging
import os
import queue
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Set

import janus

from infinilm.compile.env import (
    long_prefill_threshold,
    max_num_batched_tokens,
    moe_aot_step_max_tokens,
    moe_aot_step_total_allowed,
    schedule_no_mixed_enabled,
    schedule_preempt_enabled,
    v1_scheduler_enabled,
)
from infinilm.llm.cache_manager import BlockManager, MambaCacheManager
from infinilm.llm.request import InferenceRequest, RequestStatus

logger = logging.getLogger(__name__)

# Piecewise CG power-ladder cap.
_PACK_BUCKET_CAP = 8192
_PREEMPT_WARN_EVERY = 10


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


class _WaitingQueueCompat:
    """Thin janus-like facade over Scheduler waiting deque for legacy callers."""

    def __init__(self, scheduler: "Scheduler"):
        self._scheduler = scheduler

    @property
    def sync_q(self) -> "_WaitingQueueCompat":
        return self

    def qsize(self) -> int:
        return self._scheduler.waiting_size()

    def put(self, request: InferenceRequest) -> None:
        self._scheduler.waiting_append(request)

    def get_nowait(self) -> InferenceRequest:
        if self._scheduler.waiting_size() == 0:
            raise queue.Empty
        return self._scheduler.waiting_popleft()

    def empty(self) -> bool:
        return self._scheduler.waiting_size() == 0


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


class SpeculativeCacheOps:
    """Limited cache operations needed by speculative verification."""

    def __init__(self, cache_manager: BlockManager):
        self._cache_manager = cache_manager

    def append_verify_slots(
        self,
        block_table: List[int],
        start_length: int,
        num_slots: int,
    ):
        return self._cache_manager.append_slots(
            block_table,
            start_length,
            num_slots,
        )

    def rollback_to_length(self, block_table: List[int], keep_tokens: int):
        return self._cache_manager.truncate_blocks(block_table, keep_tokens)


class SchedulerOutput:
    """Scheduler output containing scheduled requests and execution phase info."""

    def __init__(
        self,
        scheduled_requests: List[InferenceRequest],
        is_prefill: bool = False,
        speculative_cache_ops: Optional[SpeculativeCacheOps] = None,
        rows: Optional[List[ScheduledRow]] = None,
    ):
        self.scheduled_requests = scheduled_requests
        self.num_requests = len(scheduled_requests)
        self.is_prefill = is_prefill
        self.speculative_cache_ops = speculative_cache_ops
        self.kv_connector_metadata = None
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

    Default (``INFINI_V1_SCHEDULER=0``): POST schedule — waiting prefill first,
    then running decode, with prefix-cache / mamba / remote-KV / speculative hooks.

    V1 (``INFINI_V1_SCHEDULER=1``): token-budget continuous mixed batching with
    mid-chunk prefill rows, pack gates, and optional RECOMPUTE preemption.
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        num_blocks: int = 512,
        block_size: int = 256,
        max_num_batched_tokens: int = 1024,
        connector=None,
        has_mamba_cache: bool = False,
        num_mamba_cache_blocks: int | None = None,
        enable_prefix_caching: bool = True,
        enable_prefix_cache: bool | None = None,
        max_prefill_batch_size: int | None = None,
    ):
        if enable_prefix_cache is not None:
            enable_prefix_caching = enable_prefix_cache

        self._waiting: Deque[InferenceRequest] = deque()
        self.running_queue = janus.Queue()
        self.chunking_queue = janus.Queue()
        self.max_batch_size = max_batch_size

        env_prefill_batch = os.environ.get("INFINI_MAX_PREFILL_BATCH")
        if max_prefill_batch_size is None and env_prefill_batch:
            max_prefill_batch_size = int(env_prefill_batch)
        self.max_prefill_batch_size = max_prefill_batch_size
        self.max_capture_req = max_prefill_batch_size or max_batch_size

        self.finished_receiving_kv_req_ids: set[str] = set()
        self.failed_receiving_kv_req_ids: set[str] = set()
        self.pending_free_blocks: dict[str, list[int]] = {}
        self.pending_kv_decode_blocks: int = 0
        self.remote_kv_requests: dict[str, InferenceRequest] = {}

        self.cache_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
        self.has_mamba_cache = has_mamba_cache
        self.mamba_cache_manager = (
            MambaCacheManager(num_mamba_cache_blocks or max(2, num_blocks // 4))
            if has_mamba_cache
            else None
        )
        self.speculative_cache_ops = SpeculativeCacheOps(self.cache_manager)
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.connector = connector
        self.enable_prefix_caching = enable_prefix_caching
        self.chunk_size: int = 0
        self.num_preemptions_total: int = 0

    # --- waiting deque helpers ---

    def waiting_append(self, request: InferenceRequest) -> None:
        self._waiting.append(request)

    def waiting_prepend(self, request: InferenceRequest) -> None:
        self._waiting.appendleft(request)

    def waiting_popleft(self) -> InferenceRequest:
        if not self._waiting:
            raise queue.Empty
        return self._waiting.popleft()

    def waiting_size(self) -> int:
        return len(self._waiting)

    @property
    def waiting_queue(self):
        """Compat shim: ``.sync_q.qsize()`` / ``.put`` / ``.get_nowait``."""
        return _WaitingQueueCompat(self)

    def add_request(self, request: InferenceRequest):
        if request is not None:
            # TODO: Remove the multimodal exclusion once media-aware prefix
            # hashing and model-side cache-boundary handling are supported.
            request.initialize_block_hashes(
                self.block_size,
                self.enable_prefix_caching
                and not self.has_mamba_cache
                and not request.has_multimodal_inputs,
            )
            request.status = RequestStatus.WAITING
            self.waiting_append(request)

    def _exceeds_token_budget(
        self,
        current_num_batched_tokens: int,
        num_tokens_this_step: int,
        num_scheduled_requests: int,
    ) -> bool:
        """Return True when adding this request should be deferred for token budget.

        A single request is always allowed to make progress, even if it is larger
        than max_num_batched_tokens.
        """
        if num_scheduled_requests == 0:
            return False
        return (
            current_num_batched_tokens + num_tokens_this_step
            > self.max_num_batched_tokens
        )

    def schedule(self) -> Optional[SchedulerOutput]:
        """Schedule and return batch of requests to execute."""
        if v1_scheduler_enabled():
            return self._schedule_v1()
        return self._schedule_post()

    def _schedule_post(self) -> Optional[SchedulerOutput]:
        """POST schedule path (prefill-priority, non-mixed)."""
        deferred_requests = []
        scheduled_requests = []
        is_prefill = False
        current_num_batched_tokens = 0
        current_prefill_extra_blocks = 0

        # Process Waiting queue (prefill phase)
        while (
            len(scheduled_requests) < self.max_batch_size
            and current_num_batched_tokens < self.max_num_batched_tokens
        ):
            try:
                req = self.waiting_popleft()
            except queue.Empty:
                break
            # Skip requests that were already finished (e.g., timed out/canceled while waiting)
            if req.is_finished():
                self.complete_requests([req])
                continue

            if req.num_computed_tokens == 0:
                if self.has_mamba_cache:
                    cached_block_table = []
                    num_local_computed_tokens = 0
                    load_kv_async = False
                    num_external_computed_tokens = 0
                else:
                    if self.enable_prefix_caching:
                        (
                            cached_block_table,
                            num_local_computed_tokens,
                        ) = self.cache_manager.get_computed_blocks(
                            req.block_hashes, req.get_prompt_length() - 1
                        )
                    else:
                        cached_block_table = []
                        num_local_computed_tokens = 0
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                req, num_local_computed_tokens
                            )
                        )
                        num_external_computed_tokens = ext_tokens
                    else:
                        load_kv_async = False
                        num_external_computed_tokens = 0

                available_cached_tokens = (
                    num_local_computed_tokens + num_external_computed_tokens
                )
                num_computed_tokens = min(
                    available_cached_tokens,
                    max(req.get_prompt_length() - 1, 0),
                )
                num_new_tokens = req.get_prompt_length() - num_computed_tokens

                # Early token budget check: skip can_accept_request and allocate_slots
                # for requests that would exceed the per-schedule token budget.
                if not load_kv_async:
                    num_tokens_this_step = req.get_prompt_length() - num_computed_tokens
                    if self._exceeds_token_budget(
                        current_num_batched_tokens,
                        num_tokens_this_step,
                        len(scheduled_requests),
                    ):
                        if num_local_computed_tokens > 0:
                            self.cache_manager.free_blocks(cached_block_table)
                        deferred_requests.append(req)
                        break

                if not self.can_accept_request(
                    req,
                    num_local_computed_tokens,
                    current_prefill_extra_blocks,
                    _post=True,
                ):
                    logger.warning(
                        "Insufficient KV cache blocks for request %s, deferring.",
                        req.request_id,
                    )

                    if num_local_computed_tokens > 0:
                        self.cache_manager.free_blocks(cached_block_table)
                    deferred_requests.append(req)
                    break

                allocation = self.cache_manager.allocate_slots(
                    num_new_tokens,
                    num_computed_tokens=num_computed_tokens,
                    cached_block_table=cached_block_table,
                )

                if allocation is None:
                    logger.warning(
                        "Failed to allocate KV cache blocks for request: %s",
                        req.request_id,
                    )
                    if num_local_computed_tokens > 0:
                        self.cache_manager.free_blocks(cached_block_table)
                    deferred_requests.append(req)
                    break
                req_blocks, slot_mapping = allocation

                if self.has_mamba_cache and req.mamba_cache_index is None:
                    req.mamba_cache_index = self.mamba_cache_manager.allocate()
                    if req.mamba_cache_index is None:
                        self.cache_manager.free_blocks(req_blocks)
                        logger.warning(
                            "Insufficient mamba cache rows for request %s, deferring.",
                            req.request_id,
                        )
                        deferred_requests.append(req)
                        break

                req.block_table = req_blocks
                req.slot_mapping = slot_mapping
                req.num_blocks = len(req_blocks)
                req.num_local_cached_tokens = (
                    num_local_computed_tokens if load_kv_async else num_computed_tokens
                )
                req.num_cache_indexed_blocks = len(cached_block_table)
                req.num_computed_tokens = num_computed_tokens

                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        req,
                        req.block_table,
                        num_external_computed_tokens,
                        self.block_size,
                    )
            else:
                load_kv_async = False
                num_tokens_this_step = (
                    req.get_prompt_length() - req.num_local_cached_tokens
                )
                if self._exceeds_token_budget(
                    current_num_batched_tokens,
                    num_tokens_this_step,
                    len(scheduled_requests),
                ):
                    deferred_requests.append(req)
                    break
                self.commit_computed_tokens(req, req.num_computed_tokens)

            if load_kv_async:
                req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                self.remote_kv_requests[req.request_id] = req
                self.pending_kv_decode_blocks += (
                    req.sampling_params.max_tokens + self.block_size - 1
                ) // self.block_size
                continue

            current_prefill_extra_blocks += self._get_prefill_extra_blocks(req)
            scheduled_requests.append(req)

            num_tokens_this_step = req.get_prompt_length() - req.num_local_cached_tokens
            current_num_batched_tokens += num_tokens_this_step

            req.status = RequestStatus.RUNNING
            req.is_prefill = True

        if deferred_requests:
            for req in deferred_requests:
                self.waiting_append(req)

        # Return prefill batch if any waiting requests were scheduled
        if scheduled_requests:
            is_prefill = True
            scheduler_output = SchedulerOutput(
                scheduled_requests=scheduled_requests,
                is_prefill=is_prefill,
                speculative_cache_ops=self.speculative_cache_ops,
            )
            if self.connector is not None:
                meta = self.connector.build_connector_meta()
                scheduler_output.kv_connector_metadata = meta
            return scheduler_output

        # Process Running queue (decode phase)
        while len(scheduled_requests) < self.max_batch_size:
            try:
                req = self.running_queue.sync_q.get_nowait()
            except queue.Empty:
                break
            # Skip requests that were already finished (e.g., timed out/canceled while running)
            if req.is_finished():
                self.complete_requests([req])
                continue

            # Decode phase: allocate slot for newly generated token
            req.block_table, new_slot = self.cache_manager.append_slot(
                req.block_table, req.get_total_length()
            )
            req.slot_mapping = [new_slot]
            req.num_blocks = len(req.block_table)
            req.num_local_cached_tokens = req.get_total_length() - 1
            req.is_prefill = False
            scheduled_requests.append(req)

        # Promote completed remote KV transfers (lower priority than running queue).
        # Cleanup (is_finished, failed re-queue) runs unconditionally; batch append only if slots remain.
        if self.connector is not None and self.remote_kv_requests:
            for req_id in list(self.remote_kv_requests.keys()):
                req = self.remote_kv_requests[req_id]
                if req.is_finished():
                    self.complete_requests([req])
                    continue
                if req_id in self.failed_receiving_kv_req_ids:
                    logger.warning(
                        f"Request {req_id[:8]}... failed receiving KV, re-queuing for prefill."
                    )
                    self.update_waiting_for_remote_kv(req)
                    req.status = RequestStatus.WAITING
                    self.waiting_append(req)
                elif req_id in self.finished_receiving_kv_req_ids:
                    if len(scheduled_requests) < self.max_batch_size:
                        logger.info(
                            f"Request {req_id[:8]}... finished receiving KV, scheduling for decode."
                        )
                        self.update_waiting_for_remote_kv(req)
                        req.status = RequestStatus.RUNNING
                        scheduled_requests.append(req)
                    else:
                        break  # Defer promotion to next schedule() if batch is full

        # Return decode batch if any running requests were scheduled
        if scheduled_requests:
            is_prefill = False
            scheduler_output = SchedulerOutput(
                scheduled_requests=scheduled_requests,
                is_prefill=is_prefill,
                speculative_cache_ops=self.speculative_cache_ops,
            )

            if self.connector is not None:
                meta = self.connector.build_connector_meta()
                scheduler_output.kv_connector_metadata = meta
            return scheduler_output

        if self.connector is not None:
            scheduler_output = SchedulerOutput(
                scheduled_requests=[],
                speculative_cache_ops=self.speculative_cache_ops,
            )
            meta = self.connector.build_connector_meta()
            scheduler_output.kv_connector_metadata = meta
            return scheduler_output

        return None


    def _schedule_v1(self) -> Optional[SchedulerOutput]:
        return self._schedule_v1_mixed()

    def _schedule_v1_mixed(self) -> Optional[SchedulerOutput]:
        """vLLM-like continuous batch: RUNNING (decode + mid-prefill) then WAITING."""
        budget = SchedulingBudget(
            max_num_batched_tokens(default=self.max_num_batched_tokens),
            self.max_batch_size,
        )
        rows: List[ScheduledRow] = []
        scheduled_requests: List[InferenceRequest] = []
        chunk_size_hint = 0
        deferred_running: List[InferenceRequest] = []
        no_mixed = schedule_no_mixed_enabled()

        running_snapshot = self._drain_running_queue()
        if no_mixed:
            has_decode = any(
                (not req.is_finished()) and (not req.is_prefill)
                for req in running_snapshot
            )
            filtered: List[InferenceRequest] = []
            for req in running_snapshot:
                if req.is_finished():
                    filtered.append(req)
                    continue
                if has_decode:
                    if req.is_prefill:
                        deferred_running.append(req)
                    else:
                        filtered.append(req)
                else:
                    if req.is_prefill and req.prefill_debt() > 0:
                        filtered.append(req)
                    elif req.is_prefill:
                        deferred_running.append(req)
                    else:
                        filtered.append(req)
            running_snapshot = filtered

        for req in running_snapshot:
            if req.is_finished():
                self.complete_requests([req])
                continue
            if req.status != RequestStatus.RUNNING:
                continue
            if req.is_prefill and req.prefill_debt() > 0:
                chunk_size_hint_ref = [chunk_size_hint]
                if self._try_schedule_v1_prefill_row(
                    req,
                    rows,
                    scheduled_requests,
                    budget,
                    chunk_size_hint_ref,
                    defer_fn=deferred_running.append,
                    from_waiting=False,
                ):
                    chunk_size_hint = chunk_size_hint_ref[0]
            elif not req.is_prefill:
                self._try_schedule_v1_decode_row(
                    req,
                    rows,
                    scheduled_requests,
                    budget,
                    deferred_running,
                    running_pool=running_snapshot,
                )
            else:
                deferred_running.append(req)

        has_decode_row = any(not r.is_prefill_row for r in rows)
        skip_waiting_prefills = no_mixed and has_decode_row
        prefill_batch_cap = min(
            self.max_batch_size,
            self.max_prefill_batch_size or self.max_batch_size,
        )
        n_prefill = sum(1 for r in rows if r.is_prefill_row)
        while (
            not skip_waiting_prefills
            and n_prefill < prefill_batch_cap
            and len(scheduled_requests) < self.max_batch_size
        ):
            if self.waiting_size() == 0:
                break
            req = self.waiting_popleft()

            if req.is_finished():
                self.complete_requests([req])
                continue

            if not self._v1_can_accept_request(req):
                self.waiting_prepend(req)
                break

            if not self._v1_allocate_waiting_prefill(req):
                self.waiting_prepend(req)
                break

            remaining = req.prefill_seq_len() - req.num_computed_tokens
            if req.chunk_size > 0 and remaining > req.chunk_size:
                req.chunk_prefill_offset = req.num_computed_tokens

            chunk_size_hint_ref = [chunk_size_hint]
            ok = self._try_schedule_v1_prefill_row(
                req,
                rows,
                scheduled_requests,
                budget,
                chunk_size_hint_ref,
                defer_fn=None,
                from_waiting=True,
            )
            chunk_size_hint = chunk_size_hint_ref[0]
            if not ok:
                break
            n_prefill += 1

        for req in deferred_running:
            self.running_queue.sync_q.put(req)

        if not scheduled_requests:
            # Still allow remote-KV connector meta / empty step like POST.
            if self.connector is not None:
                out = SchedulerOutput(
                    scheduled_requests=[],
                    speculative_cache_ops=self.speculative_cache_ops,
                )
                out.kv_connector_metadata = self.connector.build_connector_meta()
                return out
            return None

        prefill_rows = [r.request for r in rows if r.is_prefill_row]
        self._log_prefill_pack(prefill_rows)

        out = SchedulerOutput(
            scheduled_requests=scheduled_requests,
            is_prefill=all(r.is_prefill_row for r in rows),
            speculative_cache_ops=self.speculative_cache_ops,
            rows=rows,
        )
        if self.connector is not None:
            out.kv_connector_metadata = self.connector.build_connector_meta()
        return out

    def _v1_allocate_waiting_prefill(self, req: InferenceRequest) -> bool:
        """Allocate KV (and optional mamba) for a WAITING prefill using POST APIs."""
        if req.block_table:
            req.status = RequestStatus.RUNNING
            return True

        if self.has_mamba_cache:
            cached_block_table: List[int] = []
            num_local_computed_tokens = 0
            load_kv_async = False
            num_external_computed_tokens = 0
        else:
            if self.enable_prefix_caching:
                cached_block_table, num_local_computed_tokens = (
                    self.cache_manager.get_computed_blocks(
                        req.block_hashes, req.get_prompt_length() - 1
                    )
                )
            else:
                cached_block_table = []
                num_local_computed_tokens = 0
            if self.connector is not None:
                ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
                    req, num_local_computed_tokens
                )
                num_external_computed_tokens = ext_tokens
            else:
                load_kv_async = False
                num_external_computed_tokens = 0

        available_cached_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_computed_tokens = min(
            available_cached_tokens,
            max(req.prefill_seq_len() - 1, 0),
        )
        num_new_tokens = req.prefill_seq_len() - num_computed_tokens

        if not self.can_accept_request(
            req,
            num_local_computed_tokens,
            current_prefill_extra_blocks=0,
            _post=True,
        ):
            if num_local_computed_tokens > 0:
                self.cache_manager.free_blocks(cached_block_table)
            return False

        allocation = self.cache_manager.allocate_slots(
            num_new_tokens,
            num_computed_tokens=num_computed_tokens,
            cached_block_table=cached_block_table,
        )
        if allocation is None:
            if num_local_computed_tokens > 0:
                self.cache_manager.free_blocks(cached_block_table)
            return False
        req_blocks, slot_mapping = allocation

        if self.has_mamba_cache and req.mamba_cache_index is None:
            req.mamba_cache_index = self.mamba_cache_manager.allocate()
            if req.mamba_cache_index is None:
                self.cache_manager.free_blocks(req_blocks)
                return False

        req.block_table = req_blocks
        req.slot_mapping = slot_mapping
        req.num_blocks = len(req_blocks)
        req.num_local_cached_tokens = (
            num_local_computed_tokens if load_kv_async else num_computed_tokens
        )
        req.num_cache_indexed_blocks = len(cached_block_table)
        req.num_computed_tokens = num_computed_tokens
        req.is_prefill = True

        if self.connector is not None:
            self.connector.update_state_after_alloc(
                req, req.block_table, num_external_computed_tokens, self.block_size
            )

        if load_kv_async:
            req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            self.remote_kv_requests[req.request_id] = req
            self.pending_kv_decode_blocks += (
                req.sampling_params.max_tokens + self.block_size - 1
            ) // self.block_size
            return False

        req.status = RequestStatus.RUNNING
        return True

    def _step_scheduled_tokens(self, rows: List[ScheduledRow]) -> int:
        return sum(r.num_scheduled_tokens for r in rows)

    def _shape_gate_allows(self, rows: List[ScheduledRow], add_tokens: int) -> bool:
        return moe_aot_step_total_allowed(
            self._step_scheduled_tokens(rows) + add_tokens
        )

    def _shape_headroom(self, rows: List[ScheduledRow]) -> int:
        return max(0, moe_aot_step_max_tokens() - self._step_scheduled_tokens(rows))

    def _preempt_request(self, req: InferenceRequest) -> int:
        blocks_freed = len(req.block_table) if req.block_table else 0
        if req.block_table:
            # POST free_blocks only drops refcounts; reclaim into free_block_ids.
            self.cache_manager.free_blocks(req.block_table)
            self.cache_manager.try_free_blocks(max(1, blocks_freed))
        if self.mamba_cache_manager is not None:
            self.mamba_cache_manager.free(req.mamba_cache_index)
            req.mamba_cache_index = None
        chunk = self.chunk_size if self.chunk_size > 0 else req.chunk_size
        req.prepare_for_recompute(chunk)
        self.waiting_prepend(req)
        self.num_preemptions_total += 1
        if self.num_preemptions_total == 1 or (
            self.num_preemptions_total % _PREEMPT_WARN_EVERY == 0
        ):
            logger.warning(
                "RECOMPUTE preempted request %s (num_preemptions_total=%d). "
                "Chronic preemption usually means undersized NUM_BLOCKS / too-high "
                "max concurrency — raise NUM_BLOCKS or lower MC.",
                req.request_id[:16],
                self.num_preemptions_total,
            )
        return blocks_freed

    def _preempt_until_free(
        self,
        need_blocks: int,
        *,
        protected: Set[int],
        candidate_pool: Optional[List[InferenceRequest]] = None,
        deferred: Optional[List[InferenceRequest]] = None,
    ) -> bool:
        if not schedule_preempt_enabled():
            return False
        if self.cache_manager.can_allocate(need_blocks):
            return True

        deferred = deferred or []
        candidate_pool = candidate_pool or []
        protected_ids = set(protected)

        def _candidates() -> List[InferenceRequest]:
            seen: Set[int] = set()
            pool: List[InferenceRequest] = []
            for r in list(deferred) + list(candidate_pool):
                rid = id(r)
                if rid in protected_ids or rid in seen:
                    continue
                if r.is_finished() or not r.block_table:
                    continue
                seen.add(rid)
                pool.append(r)
            decode = [r for r in pool if not r.is_prefill]
            mid_prefill = [r for r in pool if r.is_prefill]
            decode.sort(key=lambda r: r.arrival_time, reverse=True)
            mid_prefill.sort(key=lambda r: r.arrival_time, reverse=True)
            return decode + mid_prefill

        while not self.cache_manager.can_allocate(need_blocks):
            cands = _candidates()
            if not cands:
                return False
            victim = cands[0]
            try:
                deferred.remove(victim)
            except ValueError:
                pass
            try:
                candidate_pool.remove(victim)
            except ValueError:
                pass
            self._preempt_request(victim)
        return True

    def _try_schedule_v1_decode_row(
        self,
        req: InferenceRequest,
        rows: List[ScheduledRow],
        scheduled_requests: List[InferenceRequest],
        budget: SchedulingBudget,
        deferred: List[InferenceRequest],
        running_pool: Optional[List[InferenceRequest]] = None,
    ) -> bool:
        if len(scheduled_requests) >= self.max_batch_size:
            deferred.append(req)
            return False
        if not budget.can_add(req.request_id, 1):
            deferred.append(req)
            return False
        if not self._shape_gate_allows(rows, 1):
            deferred.append(req)
            return False

        def _append_once() -> None:
            req.block_table, new_slot = self.cache_manager.append_slot(
                req.block_table,
                req.get_total_length(),
            )
            req.slot_mapping = [new_slot]
            req.num_blocks = len(req.block_table)
            req.num_local_cached_tokens = req.get_total_length() - 1
            req.num_computed_tokens = req.get_total_length() - 1

        try:
            _append_once()
        except RuntimeError:
            protected = {id(req)} | {id(r) for r in scheduled_requests}
            freed = self._preempt_until_free(
                1,
                protected=protected,
                candidate_pool=running_pool,
                deferred=deferred,
            )
            if not freed:
                deferred.append(req)
                return False
            try:
                _append_once()
            except RuntimeError:
                deferred.append(req)
                return False

        rows.append(ScheduledRow(req, 1, False, False))
        scheduled_requests.append(req)
        budget.add(req.request_id, 1)
        return True

    def _try_schedule_v1_prefill_row(
        self,
        req: InferenceRequest,
        rows: List[ScheduledRow],
        scheduled_requests: List[InferenceRequest],
        budget: SchedulingBudget,
        chunk_size_hint_ref: List[int],
        *,
        defer_fn,
        from_waiting: bool,
    ) -> bool:
        q = self._next_prefill_chunk_len(req, budget)
        headroom = self._shape_headroom(rows)
        if headroom <= 0:
            if from_waiting:
                req.status = RequestStatus.WAITING
                self.waiting_append(req)
            elif defer_fn is not None:
                defer_fn(req)
            return False
        q = min(q, headroom)
        if q <= 0 or not budget.can_add(req.request_id, q):
            if from_waiting:
                req.status = RequestStatus.WAITING
                self.waiting_append(req)
            elif defer_fn is not None:
                defer_fn(req)
            return False
        if not self._shape_gate_allows(rows, q):
            if from_waiting:
                req.status = RequestStatus.WAITING
                self.waiting_append(req)
            elif defer_fn is not None:
                defer_fn(req)
            return False

        chunk_size_hint = chunk_size_hint_ref[0]
        prefill_rows = [r for r in rows if r.is_prefill_row]
        if prefill_rows:
            if chunk_size_hint == 0:
                chunk_size_hint = req.chunk_size
            if not self._can_add_v1_prefill_row(rows, req, chunk_size_hint):
                if from_waiting:
                    req.status = RequestStatus.WAITING
                    self.waiting_append(req)
                elif defer_fn is not None:
                    defer_fn(req)
                return False
        elif req.chunk_size > 0 and chunk_size_hint == 0:
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
        chunk_size_hint_ref[0] = chunk_size_hint
        return True

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
        remaining = budget.remaining_tokens()
        threshold = long_prefill_threshold()
        if threshold > 0:
            return min(q, threshold, remaining)
        return min(q, remaining)

    def _is_final_prefill_chunk(self, req: InferenceRequest, q: int) -> bool:
        progress = (
            req.chunk_prefill_offset
            if req.chunk_prefill_offset > 0
            else req.num_computed_tokens
        )
        return progress + q >= req.prefill_seq_len()

    def _can_add_v1_prefill_row(
        self,
        rows: List[ScheduledRow],
        candidate: InferenceRequest,
        chunk_size: int,
    ) -> bool:
        prefill_rows = [r for r in rows if r.is_prefill_row]
        if not prefill_rows:
            return True
        scheduled = [r.request for r in prefill_rows]
        if not self._can_add_to_prefill_pack(
            scheduled, candidate, chunk_size=chunk_size
        ):
            return False
        # Homogeneous Q only (allows 2048+2048 under a 4096 chunk budget; rejects
        # mixing mid-chunk 4096 with a short final remainder).
        if chunk_size > 0:
            pack = scheduled + [candidate]
            lens = [self._prefill_compute_len(r) for r in pack]
            if len(set(lens)) != 1:
                return False
        return True

    def _pack_token_budget(self, chunk_size: int) -> int:
        if chunk_size > 0:
            return min(_PACK_BUCKET_CAP, chunk_size)
        return _PACK_BUCKET_CAP

    @staticmethod
    def _prefill_compute_len(req: InferenceRequest) -> int:
        remaining = req.prefill_seq_len() - req.num_computed_tokens
        if req.is_chunking() or (
            req.chunk_size > 0
            and req.is_prefill
            and req.chunk_prefill_offset > 0
            and req.prefill_seq_len() - req.chunk_prefill_offset > 0
        ):
            start = (
                req.chunk_prefill_offset
                if req.chunk_prefill_offset > 0
                else req.num_computed_tokens
            )
            end = min(start + req.chunk_size, req.prefill_seq_len())
            return max(0, end - start)
        if req.chunk_size > 0 and remaining > req.chunk_size:
            return req.chunk_size
        return remaining

    @staticmethod
    def _prefill_is_final_chunk(req: InferenceRequest) -> bool:
        if req.chunk_size > 0 and req.is_prefill and (
            req.is_chunking() or req.chunk_prefill_offset > 0
        ):
            return req.chunk_is_last()
        return True

    @staticmethod
    def _prefix_cache_pack_compatible(reqs: List[InferenceRequest]) -> bool:
        cached = [r.num_computed_tokens for r in reqs]
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
        if _padded_bucket(total_q) > budget:
            return False
        return True

    def _log_prefill_pack(self, scheduled: List[InferenceRequest]) -> None:
        if len(scheduled) <= 1:
            return
        total_q = sum(self._prefill_compute_len(r) for r in scheduled)
        bucket = _padded_bucket(total_q)
        phase = "final" if self._prefill_is_final_chunk(scheduled[0]) else "mid"
        logger.info(
            "scheduled prefill pack n_req=%s total_q=%s bucket=%s phase=%s",
            len(scheduled),
            total_q,
            bucket,
            phase,
        )

    def _prefill_pack_cap(self) -> int:
        return min(self.max_batch_size, self.max_capture_req)

    def requeue_chunking(self, req: InferenceRequest) -> None:
        """Put a request back into the chunking queue (legacy pack / tests)."""
        self.chunking_queue.sync_q.put(req)

    def requeue_running(self, req: InferenceRequest) -> None:
        """Put a request back into the running queue after a forward step (v1)."""
        self.running_queue.sync_q.put(req)

    def _remaining_generation_tokens(self, req: InferenceRequest) -> int:
        max_tokens = req.sampling_params.max_tokens
        if max_tokens is None:
            return 0
        return max(0, max_tokens - req.get_num_generated_tokens())

    def _kv_completion_tokens(self, req: InferenceRequest) -> int:
        max_tokens = req.sampling_params.max_tokens
        if max_tokens is None:
            return req.get_total_length()
        if (
            req.status == RequestStatus.WAITING
            and req.num_preemptions > 0
            and req.generated_token_ids
        ):
            return req.get_prompt_length() + max_tokens
        return req.get_prompt_length() + self._remaining_generation_tokens(req)

    def _v1_can_accept_request(self, request: InferenceRequest) -> bool:
        """Aug-style admission used by V1 waiting + unit tests."""
        total_required_blocks = 0

        running_queue_size = self.running_queue.sync_q.qsize()
        for _ in range(running_queue_size):
            req = self.running_queue.sync_q.get()
            if req.block_table:
                blocks_for_req = len(req.block_table)
            else:
                blocks_for_req = req.get_num_blocks_required(self.block_size)
            total_at_completion = (
                self._kv_completion_tokens(req) + self.block_size - 1
            ) // self.block_size
            total_required_blocks += max(blocks_for_req, total_at_completion)
            self.running_queue.sync_q.put(req)

        total_length = self._kv_completion_tokens(request)
        num_blocks_needed = (total_length + self.block_size - 1) // self.block_size
        total_required_blocks += num_blocks_needed
        total_required_blocks += self.pending_kv_decode_blocks

        return total_required_blocks <= self.cache_manager.get_total_usable_blocks()

    # Alias for tests that call can_accept_request(req) with one arg under V1.
    def can_accept_request_v1(self, request: InferenceRequest) -> bool:
        return self._v1_can_accept_request(request)

    def update_waiting_for_remote_kv(self, request: InferenceRequest):
        self.remote_kv_requests.pop(request.request_id, None)
        self.pending_kv_decode_blocks -= (
            request.sampling_params.max_tokens + self.block_size - 1
        ) // self.block_size
        if request.request_id in self.failed_receiving_kv_req_ids:
            if request.num_computed_tokens:
                self.commit_computed_tokens(request, request.num_computed_tokens)
                request.slot_mapping = self.cache_manager.update_blocks_slot(
                    request.block_table,
                    request.num_computed_tokens,
                    request.get_prompt_length(),
                )
                request.num_local_cached_tokens = request.num_computed_tokens
            else:
                self.cache_manager.free_blocks(request.block_table)
                request.block_table = []
                request.slot_mapping = []
                request.num_local_cached_tokens = 0
            self.failed_receiving_kv_req_ids.discard(request.request_id)
        else:
            self.commit_computed_tokens(request, request.num_computed_tokens)
            request.num_local_cached_tokens = request.num_computed_tokens
        self.finished_receiving_kv_req_ids.discard(request.request_id)

    def complete_requests(self, requests: List[InferenceRequest]):
        """Handle completed requests and free their blocks."""
        for req in requests:
            if req.status in [
                RequestStatus.FINISHED,
                RequestStatus.CANCELED,
                RequestStatus.FAILED,
                RequestStatus.TIMEOUT,
            ]:
                delay_free_blocks = False
                if self.connector is not None:
                    delay_free_blocks, _ = self.connector.request_finished(
                        req, req.block_table, self.block_size
                    )

                if req.request_id in self.remote_kv_requests:
                    self.pending_kv_decode_blocks -= (
                        req.sampling_params.max_tokens + self.block_size - 1
                    ) // self.block_size
                    self.remote_kv_requests.pop(req.request_id, None)
                    if req.request_id in self.finished_receiving_kv_req_ids:
                        self.finished_receiving_kv_req_ids.discard(req.request_id)
                        self.failed_receiving_kv_req_ids.discard(req.request_id)
                    else:
                        delay_free_blocks = True
                if req.block_table and not delay_free_blocks:
                    self.cache_manager.free_blocks(req.block_table)
                elif req.block_table and delay_free_blocks:
                    self.pending_free_blocks[req.request_id] = list(req.block_table)
                if self.mamba_cache_manager is not None:
                    self.mamba_cache_manager.free(req.mamba_cache_index)
                    req.mamba_cache_index = None

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
                # Still running, put back in running queue
                self.running_queue.sync_q.put(req)

    def can_accept_request(
        self,
        request: InferenceRequest,
        num_local_computed_tokens: int = 0,
        current_prefill_extra_blocks: int = 0,
        *,
        _post: bool = False,
    ) -> bool:
        """Admission check.

        Under V1 (and one-arg test calls), use Aug-style ``_v1_can_accept_request``
        unless ``_post=True`` (POST schedule path).
        """
        if v1_scheduler_enabled() and not _post:
            return self._v1_can_accept_request(request)

        if (
            self.mamba_cache_manager is not None
            and request.mamba_cache_index is None
            and not self.mamba_cache_manager.can_allocate()
        ):
            return False

        total_required_blocks = 0

        # Calculate blocks needed for running requests
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

        # Calculate blocks needed for the new request
        total_length = request.get_prompt_length() - num_local_computed_tokens
        total_length += request.sampling_params.max_tokens
        num_blocks_needed = (total_length + self.block_size - 1) // self.block_size
        total_required_blocks += num_blocks_needed

        # Include decode headroom for WAITING_FOR_REMOTE_KVS requests, which
        # hold prompt blocks but will also need decode blocks once promoted.
        total_required_blocks += self.pending_kv_decode_blocks

        # Include decode headroom for requests accepted earlier in this batch.
        total_required_blocks += current_prefill_extra_blocks

        # Compare with total usable blocks in cache manager
        return total_required_blocks <= self.cache_manager.get_total_usable_blocks()

    def _get_prefill_extra_blocks(self, request: InferenceRequest) -> int:
        total_length = request.get_prompt_length()
        total_length += request.sampling_params.max_tokens
        total_required_blocks = (total_length + self.block_size - 1) // self.block_size
        return max(total_required_blocks - len(request.block_table), 0)

    def commit_computed_tokens(
        self, request: InferenceRequest, num_computed_tokens: int
    ) -> None:
        if not self.enable_prefix_caching or self.has_mamba_cache:
            return

        indexed_blocks = request.num_cache_indexed_blocks
        target_blocks = min(
            num_computed_tokens // self.block_size,
            len(request.block_table),
            len(request.block_hashes),
        )
        if target_blocks == indexed_blocks:
            return

        request.num_cache_indexed_blocks = self.cache_manager.publish_computed_blocks(
            request.block_table,
            request.block_hashes,
            request.num_cache_indexed_blocks,
            num_computed_tokens,
        )

    def update_from_output(self, model_output):
        if self.connector is None or model_output.kv_connector_output is None:
            return

        finished_recving_req_ids = (
            getattr(model_output.kv_connector_output, "finished_recving", None) or []
        )
        finished_sending_req_ids = (
            getattr(model_output.kv_connector_output, "finished_sending", None) or []
        )
        failed_recving_req_ids = (
            getattr(model_output.kv_connector_output, "failed_recving", None) or []
        )
        invalid_block_ids = (
            getattr(model_output.kv_connector_output, "invalid_block_ids", None) or []
        )

        for req_id in finished_recving_req_ids:
            if req_id in self.pending_free_blocks:
                # Aborted request: transfer complete, now safe to free blocks.
                self.cache_manager.free_blocks(self.pending_free_blocks.pop(req_id))
            elif req_id in self.remote_kv_requests:
                # Active request: mark ready for promotion in schedule().
                self.finished_receiving_kv_req_ids.add(req_id)
            # else: already processed or unknown, discard to avoid stale entries.
        for req_id in finished_sending_req_ids:
            self.cache_manager.free_blocks(self.pending_free_blocks.pop(req_id, []))

        invalid_set = set(invalid_block_ids)
        for req_id in failed_recving_req_ids:
            req = self.remote_kv_requests.get(req_id)
            if req is None:
                continue

            self.failed_receiving_kv_req_ids.add(req_id)
            if req.has_multimodal_inputs:
                # A physical block boundary can split a media span. Recompute
                # the prompt until multimodal cache-boundary handling is supported.
                req.num_computed_tokens = 0
                continue

            trusted_boundary = req.num_local_cached_tokens
            start_block_idx = req.num_cache_indexed_blocks
            for block_idx, block_id in enumerate(
                req.block_table[start_block_idx:], start=start_block_idx
            ):
                if block_id in invalid_set:
                    trusted_boundary = min(
                        req.num_computed_tokens,
                        block_idx * self.block_size,
                    )
                    break
            req.num_computed_tokens = trusted_boundary

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            "num_blocks": self.cache_manager.num_blocks,
            "block_size": self.cache_manager.block_size,
            "num_free_blocks": self.cache_manager.get_num_free_blocks(),
            "usable_blocks": self.cache_manager.get_total_usable_blocks(),
            "num_used_blocks": len(self.cache_manager.used_block_ids),
            "num_preemptions_total": self.num_preemptions_total,
        }
        if self.mamba_cache_manager is not None:
            stats.update(
                {
                    "num_mamba_cache_blocks": self.mamba_cache_manager.num_blocks,
                    "num_free_mamba_cache_blocks": self.mamba_cache_manager.get_num_free_blocks(),
                    "num_used_mamba_cache_blocks": len(
                        self.mamba_cache_manager.used_block_ids
                    ),
                }
            )
        return stats
