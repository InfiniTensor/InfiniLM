"""
Scheduler - Request scheduling and batch management with Paged Attention KV Cache.
"""

import logging
import queue
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, Optional

import janus

from infinilm.llm.cache_manager import BlockManager, MambaCacheManager
from infinilm.llm.request import InferenceRequest, RequestStatus

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class ScheduledRequestWork:
    request: InferenceRequest
    start_token: int
    num_scheduled_tokens: int

    @property
    def end_token(self) -> int:
        return self.start_token + self.num_scheduled_tokens

    @property
    def requires_sampling(self) -> bool:
        prompt_length = self.request.get_prompt_length()
        return self.start_token >= prompt_length or self.end_token == prompt_length


class SchedulerOutput:
    """Scheduler output containing scheduled requests and execution phase info."""

    def __init__(
        self,
        scheduled_requests: List[InferenceRequest],
        is_prefill: bool = False,
        speculative_cache_ops: Optional[SpeculativeCacheOps] = None,
        work_items: Sequence[ScheduledRequestWork] | None = None,
    ):
        if work_items is not None:
            work_items = tuple(work_items)

        self.work_items = work_items
        self.scheduled_requests = scheduled_requests
        self.num_requests = (
            len(work_items) if work_items is not None else len(scheduled_requests)
        )
        self.is_prefill = is_prefill
        self.speculative_cache_ops = speculative_cache_ops
        self.kv_connector_metadata = None


class Scheduler:
    """Request scheduler with integrated BlockManager for KV cache management.

    Scheduling logic:
    1. Running queue: Check for new blocks needed, update slot_mapping
    2. Waiting queue: Try block reuse (prefix caching), allocate new blocks
    3. Reference counting: Free blocks when requests complete
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
        enable_chunked_prefill: bool = False,
    ):
        self.waiting_queue = janus.Queue()
        self.running_queue = janus.Queue()
        self.max_batch_size = max_batch_size

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
        self.enable_chunked_prefill = enable_chunked_prefill
        if self.enable_chunked_prefill and self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")

        self._active_requests: deque[InferenceRequest] = deque()
        self._waiting_requests: deque[InferenceRequest] = deque()
        self._schedule_impl = (
            self._schedule_chunked
            if self.enable_chunked_prefill
            else self._schedule_legacy
        )
        self._complete_requests_impl = (
            self._complete_chunked_requests
            if self.enable_chunked_prefill
            else self._complete_legacy_requests
        )

    def add_request(self, request: InferenceRequest):
        if request is not None:
            if self.enable_chunked_prefill:
                if request.has_multimodal_inputs:
                    raise RuntimeError(
                        "Chunked prefill does not support multimodal requests yet."
                    )
                max_tokens = request.sampling_params.max_tokens
                if max_tokens is None or max_tokens < 1:
                    raise ValueError("chunked prefill requires request max_tokens >= 1")
                required_blocks = self._blocks_for_tokens(self._max_kv_tokens(request))
                if required_blocks > self.cache_manager.num_blocks:
                    raise ValueError(
                        f"Request {request.request_id} requires "
                        f"{required_blocks} KV blocks, but the cache has only "
                        f"{self.cache_manager.num_blocks}"
                    )
            # TODO: Remove the multimodal exclusion once media-aware prefix
            # hashing and model-side cache-boundary handling are supported.
            request.initialize_block_hashes(
                self.block_size,
                self.enable_prefix_caching
                and not self.has_mamba_cache
                and not request.has_multimodal_inputs,
            )
            request.status = RequestStatus.WAITING
            self.waiting_queue.sync_q.put(request)

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
        return self._schedule_impl()

    def _schedule_legacy(self) -> Optional[SchedulerOutput]:
        """Schedule and return batch of requests to execute."""
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
                req = self.waiting_queue.sync_q.get_nowait()
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

        if deferred_requests:
            for req in deferred_requests:
                self.waiting_queue.sync_q.put(req)

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
                    self.waiting_queue.sync_q.put(req)
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

    def _drain_waiting_ingress(self) -> None:
        for _ in range(self.waiting_queue.sync_q.qsize()):
            self._waiting_requests.append(self.waiting_queue.sync_q.get_nowait())

    def _prune_finished_chunked_requests(self) -> None:
        active_requests: deque[InferenceRequest] = deque()
        for request in self._active_requests:
            if request.is_finished():
                self._complete_terminal_request(request)
            else:
                active_requests.append(request)
        self._active_requests = active_requests

        while self._waiting_requests and self._waiting_requests[0].is_finished():
            self._complete_terminal_request(self._waiting_requests.popleft())

    def _blocks_for_tokens(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def _max_kv_tokens(self, request: InferenceRequest) -> int:
        # The newest sampled token has not been forwarded into KV yet.
        return request.get_prompt_length() + request.sampling_params.max_tokens - 1

    def _completion_missing_blocks(self, request: InferenceRequest) -> int:
        return max(
            self._blocks_for_tokens(self._max_kv_tokens(request))
            - len(request.block_table),
            0,
        )

    def _get_completion_slack(self) -> int:
        """Return usable blocks not reserved for active requests to finish."""
        completion_reservation = sum(
            self._completion_missing_blocks(request)
            for request in self._active_requests
        )
        usable_blocks = self.cache_manager.get_total_usable_blocks()
        if usable_blocks < completion_reservation:
            raise RuntimeError(
                "Resident requests exceed the guaranteed KV completion capacity"
            )
        return usable_blocks - completion_reservation

    def _allocate_chunked_work(
        self,
        request: InferenceRequest,
        block_table: List[int],
        start_token: int,
        end_token: int,
    ) -> ScheduledRequestWork:
        work = ScheduledRequestWork(
            request=request,
            start_token=start_token,
            num_scheduled_tokens=end_token - start_token,
        )
        allocation = self.cache_manager.allocate_slot_range(
            block_table,
            work.start_token,
            work.end_token,
        )

        request.block_table = block_table
        request.block_table.extend(allocation.new_blocks)
        request.slot_mapping = allocation.slot_mapping
        request.num_blocks = len(request.block_table)
        request.status = RequestStatus.RUNNING
        return work

    def _schedule_active_chunked_requests(
        self, token_budget: int
    ) -> tuple[List[ScheduledRequestWork], int]:
        work_items: List[ScheduledRequestWork] = []

        while (
            self._active_requests
            and token_budget > 0
            and len(work_items) < self.max_batch_size
        ):
            request = self._active_requests[0]
            prompt_length = request.get_prompt_length()
            if request.num_computed_tokens < prompt_length:
                remaining = prompt_length - request.num_computed_tokens
                end_token = request.num_computed_tokens + min(remaining, token_budget)
            else:
                if request.get_total_length() != request.num_computed_tokens + 1:
                    raise RuntimeError(
                        f"Request {request.request_id} does not have exactly "
                        "one uncomputed ordinary decode token"
                    )
                end_token = request.num_computed_tokens + 1
            work = self._allocate_chunked_work(
                request,
                request.block_table,
                request.num_computed_tokens,
                end_token,
            )
            self._active_requests.popleft()
            work_items.append(work)
            token_budget -= work.num_scheduled_tokens

        return work_items, token_budget

    def _admit_waiting_chunked_requests(
        self,
        token_budget: int,
        request_budget: int,
        completion_slack: int,
    ) -> List[ScheduledRequestWork]:
        work_items: List[ScheduledRequestWork] = []

        while self._waiting_requests and token_budget > 0 and request_budget > 0:
            request = self._waiting_requests[0]

            # A request can be canceled after it enters the ingress queue but
            # before it reaches the head of the scheduler-owned deque.
            if request.is_finished():
                self._waiting_requests.popleft()
                self._complete_terminal_request(request)
                continue

            if self.enable_prefix_caching:
                usable_before_lookup = self.cache_manager.get_total_usable_blocks()
                cached_blocks, cached_tokens = self.cache_manager.get_computed_blocks(
                    request.block_hashes,
                    request.get_prompt_length() - 1,
                )
                newly_pinned_blocks = (
                    usable_before_lookup - self.cache_manager.get_total_usable_blocks()
                )
            else:
                cached_blocks = []
                cached_tokens = 0
                newly_pinned_blocks = 0

            completion_missing_blocks = max(
                self._blocks_for_tokens(self._max_kv_tokens(request))
                - len(cached_blocks),
                0,
            )
            admission_cost = newly_pinned_blocks + completion_missing_blocks
            if admission_cost > completion_slack:
                self.cache_manager.free_blocks(cached_blocks)
                break

            num_cached_blocks = len(cached_blocks)
            scheduled_tokens = min(
                request.get_prompt_length() - cached_tokens,
                token_budget,
            )
            work = self._allocate_chunked_work(
                request,
                cached_blocks,
                cached_tokens,
                cached_tokens + scheduled_tokens,
            )
            self._waiting_requests.popleft()
            request.num_computed_tokens = cached_tokens
            request.num_local_cached_tokens = cached_tokens
            request.num_cache_indexed_blocks = num_cached_blocks
            work_items.append(work)
            token_budget -= work.num_scheduled_tokens
            request_budget -= 1
            completion_slack -= admission_cost

        return work_items

    def _schedule_chunked(self) -> Optional[SchedulerOutput]:
        self._drain_waiting_ingress()
        self._prune_finished_chunked_requests()
        completion_slack = self._get_completion_slack()

        # Match vLLM's running-first policy: preserve inter-token latency for
        # resident requests, then use the remaining capacity for new prefills.
        active_work, token_budget = self._schedule_active_chunked_requests(
            self.max_num_batched_tokens
        )
        waiting_work = self._admit_waiting_chunked_requests(
            token_budget,
            self.max_batch_size - len(active_work),
            completion_slack,
        )
        work_items = active_work + waiting_work

        if not work_items:
            return None

        return SchedulerOutput(
            scheduled_requests=[],
            speculative_cache_ops=self.speculative_cache_ops,
            work_items=work_items,
        )

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
        self._complete_requests_impl(requests)

    def _complete_legacy_requests(self, requests: List[InferenceRequest]) -> None:
        for req in requests:
            if req.is_finished():
                self._complete_terminal_request(req)
            else:
                self.running_queue.sync_q.put(req)

    def _complete_chunked_requests(self, requests: List[InferenceRequest]) -> None:
        for req in requests:
            if req.is_finished():
                self._complete_terminal_request(req)
            else:
                self._active_requests.append(req)

    def _complete_terminal_request(self, request: InferenceRequest) -> None:
        delay_free_blocks = False
        if self.connector is not None:
            delay_free_blocks, _ = self.connector.request_finished(
                request, request.block_table, self.block_size
            )

        if request.request_id in self.remote_kv_requests:
            self.pending_kv_decode_blocks -= (
                request.sampling_params.max_tokens + self.block_size - 1
            ) // self.block_size
            self.remote_kv_requests.pop(request.request_id, None)
            if request.request_id in self.finished_receiving_kv_req_ids:
                self.finished_receiving_kv_req_ids.discard(request.request_id)
                self.failed_receiving_kv_req_ids.discard(request.request_id)
            else:
                delay_free_blocks = True
        if request.block_table and not delay_free_blocks:
            self.cache_manager.free_blocks(request.block_table)
        elif request.block_table and delay_free_blocks:
            self.pending_free_blocks[request.request_id] = list(request.block_table)
        if self.mamba_cache_manager is not None:
            self.mamba_cache_manager.free(request.mamba_cache_index)
            request.mamba_cache_index = None

        if request.status == RequestStatus.CANCELED:
            logger.info(
                f"Request {request.request_id[:8]}... canceled: {request.finish_reason}"
            )
        elif request.status == RequestStatus.FAILED:
            logger.error(
                f"Request {request.request_id[:8]}... failed: {request.finish_reason}"
            )
        elif request.status == RequestStatus.TIMEOUT:
            logger.error(
                f"Request {request.request_id[:8]}... timed out: "
                f"{request.finish_reason}"
            )

    def can_accept_request(
        self,
        request: InferenceRequest,
        num_local_computed_tokens: int,
        current_prefill_extra_blocks: int = 0,
    ) -> bool:
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
