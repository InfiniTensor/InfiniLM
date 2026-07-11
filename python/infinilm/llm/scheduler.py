"""
Scheduler - Request scheduling and batch management with Paged Attention KV Cache.
"""

import logging
import queue
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
        token_ids: List[int],
    ):
        return self._cache_manager.append_slots(
            block_table,
            start_length,
            num_slots,
            token_ids,
            update_hash=False,
        )

    def rollback_to_length(self, block_table: List[int], keep_tokens: int):
        return self._cache_manager.truncate_blocks(block_table, keep_tokens)

    def commit_accepted_tokens(
        self, block_table: List[int], token_ids: List[int], num_tokens: int
    ) -> None:
        self._cache_manager.commit_blocks_hash(block_table, token_ids, num_tokens)


@dataclass(frozen=True)
class ExecutionPhase:
    kind: str
    request_indices: tuple[int, ...]
    num_tokens: int


class SchedulerOutput:
    """Scheduler output containing scheduled requests and execution phase info."""

    def __init__(
        self,
        scheduled_requests: List[InferenceRequest],
        is_prefill: bool = False,
        scheduled_prefill_flags: Optional[List[bool]] = None,
        speculative_cache_ops: Optional[SpeculativeCacheOps] = None,
    ):
        self.scheduled_requests = scheduled_requests
        self.num_requests = len(scheduled_requests)
        if scheduled_prefill_flags is None:
            scheduled_prefill_flags = [is_prefill] * self.num_requests
        if len(scheduled_prefill_flags) != self.num_requests:
            raise ValueError(
                "scheduled_prefill_flags must match scheduled_requests length"
            )
        self.scheduled_prefill_flags = scheduled_prefill_flags
        self.speculative_cache_ops = speculative_cache_ops
        # C++ attention currently chooses the prefill/extend path at batch level.
        # A mixed decode+prefill batch must therefore advertise prefill=True.
        self.is_prefill = (
            any(scheduled_prefill_flags) if scheduled_requests else is_prefill
        )
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        for req, request_is_prefill in zip(
            scheduled_requests, scheduled_prefill_flags
        ):
            if request_is_prefill:
                self.num_prefill_tokens += max(
                    0, req.prefill_chunk_end - req.prefill_chunk_start
                )
            else:
                self.num_decode_tokens += 1
        self.num_batched_tokens = self.num_prefill_tokens + self.num_decode_tokens
        self.execution_phases = self._build_execution_phases()
        self.kv_connector_metadata = None

    def request_is_prefill(self, index: int) -> bool:
        return self.scheduled_prefill_flags[index]

    @property
    def is_mixed(self) -> bool:
        return self.num_decode_tokens > 0 and self.num_prefill_tokens > 0

    def _build_execution_phases(self) -> List[ExecutionPhase]:
        decode_indices = tuple(
            i
            for i, is_prefill in enumerate(self.scheduled_prefill_flags)
            if not is_prefill
        )
        prefill_indices = tuple(
            i
            for i, is_prefill in enumerate(self.scheduled_prefill_flags)
            if is_prefill
        )
        phases = []
        if decode_indices:
            phases.append(
                ExecutionPhase(
                    kind="decode",
                    request_indices=decode_indices,
                    num_tokens=len(decode_indices),
                )
            )
        if prefill_indices:
            phases.append(
                ExecutionPhase(
                    kind="prefill",
                    request_indices=prefill_indices,
                    num_tokens=self.num_prefill_tokens,
                )
            )
        return phases

    def make_subset(self, request_indices: List[int]) -> "SchedulerOutput":
        scheduled_requests = [self.scheduled_requests[i] for i in request_indices]
        scheduled_prefill_flags = [
            self.scheduled_prefill_flags[i] for i in request_indices
        ]
        subset = SchedulerOutput(
            scheduled_requests=scheduled_requests,
            scheduled_prefill_flags=scheduled_prefill_flags,
            speculative_cache_ops=self.speculative_cache_ops,
        )
        subset.kv_connector_metadata = self.kv_connector_metadata
        return subset


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
        enable_chunked_prefill: bool = False,
        prefill_chunk_size: Optional[int] = None,
        max_num_partial_prefills: int = 1,
        max_long_partial_prefills: int = 1,
        long_prefill_token_threshold: Optional[int] = None,
        min_prefill_chunk_size: Optional[int] = None,
        decode_priority: bool = False,
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
        self.enable_chunked_prefill = enable_chunked_prefill
        self.prefill_chunk_size = prefill_chunk_size or max_num_batched_tokens
        self.max_num_partial_prefills = max(1, max_num_partial_prefills)
        self.max_long_partial_prefills = min(
            max(1, max_long_partial_prefills), self.max_num_partial_prefills
        )
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.min_prefill_chunk_size = min_prefill_chunk_size or min(
            block_size, self.prefill_chunk_size
        )
        self.decode_priority = decode_priority
        self._stats = {
            "schedule_steps": 0,
            "decode_steps": 0,
            "prefill_steps": 0,
            "mixed_steps": 0,
            "decode_tokens": 0,
            "prefill_tokens": 0,
            "partial_prefill_chunks": 0,
            "full_prefill_chunks": 0,
            "prefill_deferred_by_token_budget": 0,
            "prefill_deferred_by_partial_limit": 0,
            "prefill_deferred_by_long_partial_limit": 0,
            "execution_split_mixed_steps": 0,
            "execution_decode_phases": 0,
            "execution_prefill_phases": 0,
            "execution_decode_phase_tokens": 0,
            "execution_prefill_phase_tokens": 0,
        }

    def _is_partial_prefill(
        self,
        request: InferenceRequest,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> bool:
        return (
            self.enable_chunked_prefill
            and num_new_tokens > 0
            and num_computed_tokens + num_new_tokens < request.get_prompt_length()
        )

    def _is_long_prefill(
        self, request: InferenceRequest, num_computed_tokens: int
    ) -> bool:
        threshold = self.long_prefill_token_threshold
        if threshold is None or threshold <= 0:
            return False
        return (
            request.get_prompt_length() > threshold
            and request.get_prompt_length() - num_computed_tokens > threshold
        )

    def _partial_prefill_limit_reason(
        self,
        request: InferenceRequest,
        num_computed_tokens: int,
        num_new_tokens: int,
        partial_prefill_count: int,
        long_partial_prefill_count: int,
    ) -> Optional[str]:
        if not self._is_partial_prefill(request, num_computed_tokens, num_new_tokens):
            return None
        if partial_prefill_count >= self.max_num_partial_prefills:
            return "partial_limit"
        if (
            self._is_long_prefill(request, num_computed_tokens)
            and long_partial_prefill_count >= self.max_long_partial_prefills
        ):
            return "long_partial_limit"
        return None

    def add_request(self, request: InferenceRequest):
        if request is not None:
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
        """Schedule and return the next batch of requests to execute.

        The default policy preserves the original prefill-first behavior. When
        chunked prefill and decode priority are enabled, it follows vLLM's shape:
        schedule ready decodes first, then fill the remaining token budget with
        prefill chunks in the same batch.
        """
        scheduled_requests: List[InferenceRequest] = []
        scheduled_prefill_flags: List[bool] = []
        current_num_batched_tokens = 0

        if self.decode_priority and self.enable_chunked_prefill:
            current_num_batched_tokens = self._schedule_decode_requests(
                scheduled_requests,
                scheduled_prefill_flags,
                current_num_batched_tokens,
            )
            current_prefill_extra_blocks = sum(
                self._get_prefill_extra_blocks(req) for req in scheduled_requests
            )
            self._schedule_prefill_requests(
                scheduled_requests,
                scheduled_prefill_flags,
                current_num_batched_tokens,
                current_prefill_extra_blocks,
            )
        else:
            self._schedule_prefill_requests(
                scheduled_requests,
                scheduled_prefill_flags,
                current_num_batched_tokens,
            )
            if not scheduled_requests:
                scheduler_output = self._schedule_decode_batch()
                if scheduler_output is not None:
                    return scheduler_output

        if scheduled_requests:
            return self._make_scheduler_output(
                scheduled_requests,
                scheduled_prefill_flags,
            )

        if self.connector is not None:
            return self._make_connector_only_output()

        return None

    def update_waiting_for_remote_kv(self, request: InferenceRequest):
        self.remote_kv_requests.pop(request.request_id, None)
        self.pending_kv_decode_blocks -= (
            request.sampling_params.max_tokens + self.block_size - 1
        ) // self.block_size
        if request.request_id in self.failed_receiving_kv_req_ids:
            if request.num_computed_tokens:
                valid_block_count = request.num_computed_tokens // self.block_size
                self.cache_manager.update_blocks_hash(
                    request.block_table[:valid_block_count],
                    request.num_local_cached_tokens,
                )
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
            self.cache_manager.update_blocks_hash(
                request.block_table, request.num_local_cached_tokens
            )
            request.num_local_cached_tokens = request.num_computed_tokens
        self.finished_receiving_kv_req_ids.discard(request.request_id)

    def complete_prefill_chunk(self, request: InferenceRequest) -> bool:
        """Finalize a prefill chunk and return True when generation may sample."""
        chunk_start = request.prefill_chunk_start
        chunk_end = request.prefill_chunk_end
        self.cache_manager.commit_computed_blocks(
            request.block_table,
            request.get_input_tokens(),
            chunk_start,
            chunk_end,
        )
        request.num_computed_tokens = chunk_end
        request.num_local_cached_tokens = chunk_end
        request.slot_mapping = []

        if chunk_end < request.get_prompt_length():
            request.status = RequestStatus.WAITING
            self.waiting_queue.sync_q.put(request)
            return False

        request.status = RequestStatus.RUNNING
        return True

    def _get_prefill_chunk_num_tokens(
        self,
        request: InferenceRequest,
        num_computed_tokens: int,
        current_num_batched_tokens: int,
        num_scheduled_requests: int,
    ) -> int:
        prompt_len = request.get_prompt_length()
        remaining_tokens = prompt_len - num_computed_tokens
        if remaining_tokens <= 0:
            return 0

        if not self.enable_chunked_prefill:
            return remaining_tokens

        token_budget_left = self.max_num_batched_tokens - current_num_batched_tokens
        if num_scheduled_requests == 0:
            token_budget_left = max(token_budget_left, 1)
        if token_budget_left <= 0:
            return 0

        num_tokens = min(
            remaining_tokens,
            self.prefill_chunk_size,
            token_budget_left,
        )

        if num_computed_tokens % self.block_size != 0:
            next_block_boundary = (
                (num_computed_tokens // self.block_size) + 1
            ) * self.block_size
            tokens_to_block_boundary = next_block_boundary - num_computed_tokens
            num_tokens = min(num_tokens, tokens_to_block_boundary)
        else:
            tokens_to_block_boundary = None

        if 0 < num_tokens < self.min_prefill_chunk_size and num_scheduled_requests > 0:
            final_chunk = num_tokens >= remaining_tokens
            completes_partial_block = (
                tokens_to_block_boundary is not None
                and num_tokens == tokens_to_block_boundary
            )
            if not final_chunk and not completes_partial_block:
                return 0

        return max(num_tokens, 0)

    def _set_prefill_chunk(
        self, request: InferenceRequest, chunk_start: int, chunk_end: int
    ) -> None:
        request.prefill_chunk_start = chunk_start
        request.prefill_chunk_end = chunk_end
        request.prefill_chunk_is_final = chunk_end >= request.get_prompt_length()

    def _make_scheduler_output(
        self,
        scheduled_requests: List[InferenceRequest],
        scheduled_prefill_flags: List[bool],
    ) -> SchedulerOutput:
        scheduler_output = SchedulerOutput(
            scheduled_requests=scheduled_requests,
            scheduled_prefill_flags=scheduled_prefill_flags,
            speculative_cache_ops=self.speculative_cache_ops,
        )
        self._record_scheduler_output(scheduler_output)
        if self.connector is not None:
            scheduler_output.kv_connector_metadata = (
                self.connector.build_connector_meta()
            )
        return scheduler_output

    def _make_connector_only_output(self) -> SchedulerOutput:
        self._stats["schedule_steps"] += 1
        scheduler_output = SchedulerOutput(
            scheduled_requests=[],
            speculative_cache_ops=self.speculative_cache_ops,
        )
        scheduler_output.kv_connector_metadata = self.connector.build_connector_meta()
        return scheduler_output

    def _record_scheduler_output(self, scheduler_output: SchedulerOutput) -> None:
        self._stats["schedule_steps"] += 1
        has_decode = scheduler_output.num_decode_tokens > 0
        has_prefill = scheduler_output.num_prefill_tokens > 0
        if has_decode:
            self._stats["decode_steps"] += 1
            self._stats["decode_tokens"] += scheduler_output.num_decode_tokens
        if has_prefill:
            self._stats["prefill_steps"] += 1
            self._stats["prefill_tokens"] += scheduler_output.num_prefill_tokens
        if has_decode and has_prefill:
            self._stats["mixed_steps"] += 1

        for req, request_is_prefill in zip(
            scheduler_output.scheduled_requests,
            scheduler_output.scheduled_prefill_flags,
        ):
            if not request_is_prefill:
                continue
            if req.prefill_chunk_end < req.get_prompt_length():
                self._stats["partial_prefill_chunks"] += 1
            else:
                self._stats["full_prefill_chunks"] += 1

    def _schedule_prefill_requests(
        self,
        scheduled_requests: List[InferenceRequest],
        scheduled_prefill_flags: List[bool],
        current_num_batched_tokens: int,
        current_prefill_extra_blocks: int = 0,
    ) -> int:
        deferred_requests = []
        partial_prefill_count = sum(
            1
            for req, is_prefill in zip(scheduled_requests, scheduled_prefill_flags)
            if is_prefill and req.prefill_chunk_end < req.get_prompt_length()
        )
        long_partial_prefill_count = sum(
            1
            for req, is_prefill in zip(scheduled_requests, scheduled_prefill_flags)
            if is_prefill
            and req.prefill_chunk_end < req.get_prompt_length()
            and self._is_long_prefill(req, req.prefill_chunk_start)
        )

        while self._has_schedule_capacity(
            len(scheduled_requests), current_num_batched_tokens
        ):
            try:
                req = self.waiting_queue.sync_q.get_nowait()
            except queue.Empty:
                break
            # Skip requests that were already finished (e.g., timed out/canceled while waiting)
            if req.is_finished():
                self.complete_requests([req])
                continue

            req_tokens = req.get_input_tokens()

            if req.num_computed_tokens == 0:
                if self.has_mamba_cache:
                    cached_block_table = []
                    num_local_computed_tokens = 0
                    blocks_blueprint = []
                    load_kv_async = False
                    num_external_computed_tokens = 0
                else:
                    (
                        cached_block_table,
                        num_local_computed_tokens,
                        blocks_blueprint,
                    ) = self.cache_manager.get_computed_blocks(
                        req_tokens, req.get_mm_token_index_mappings()
                    )
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

                num_computed_tokens = (
                    num_local_computed_tokens + num_external_computed_tokens
                )
                if load_kv_async:
                    num_computed_tokens -= 1
                num_new_tokens = req.get_prompt_length() - num_computed_tokens

                if self.enable_chunked_prefill and not load_kv_async:
                    num_new_tokens = self._get_prefill_chunk_num_tokens(
                        req,
                        num_computed_tokens,
                        current_num_batched_tokens,
                        len(scheduled_requests),
                    )

                limit_reason = self._partial_prefill_limit_reason(
                    req,
                    num_computed_tokens,
                    num_new_tokens,
                    partial_prefill_count,
                    long_partial_prefill_count,
                )
                if limit_reason is not None:
                    if num_local_computed_tokens > 0:
                        self.cache_manager.free_blocks(cached_block_table)
                    if limit_reason == "partial_limit":
                        self._stats["prefill_deferred_by_partial_limit"] += 1
                    else:
                        self._stats[
                            "prefill_deferred_by_long_partial_limit"
                        ] += 1
                    deferred_requests.append(req)
                    break

                # Early token budget check: skip can_accept_request and allocate_slots
                # for requests that would exceed the per-schedule token budget.
                if not load_kv_async:
                    num_tokens_this_step = num_new_tokens
                    if num_tokens_this_step <= 0 or self._exceeds_token_budget(
                        current_num_batched_tokens,
                        num_tokens_this_step,
                        len(scheduled_requests),
                    ):
                        self._stats["prefill_deferred_by_token_budget"] += 1
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

                req_blocks, slot_mapping = self.cache_manager.allocate_slots(
                    req_tokens,
                    num_new_tokens,
                    num_computed_tokens=num_computed_tokens,
                    cached_block_table=cached_block_table,
                    blocks_blueprint=blocks_blueprint,
                    delay_cache_blocks=True,
                )

                if req_blocks is None:
                    logger.warning(
                        "Failed to allocate KV cache blocks for request: %s",
                        req.request_id,
                    )
                    if num_local_computed_tokens > 0:
                        self.cache_manager.free_blocks(cached_block_table)
                    deferred_requests.append(req)
                    break

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
                req.num_local_cached_tokens = num_local_computed_tokens
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
                num_computed_tokens = req.num_computed_tokens
                num_new_tokens = self._get_prefill_chunk_num_tokens(
                    req,
                    num_computed_tokens,
                    current_num_batched_tokens,
                    len(scheduled_requests),
                )
                num_tokens_this_step = num_new_tokens

                limit_reason = self._partial_prefill_limit_reason(
                    req,
                    num_computed_tokens,
                    num_new_tokens,
                    partial_prefill_count,
                    long_partial_prefill_count,
                )
                if limit_reason is not None:
                    if limit_reason == "partial_limit":
                        self._stats["prefill_deferred_by_partial_limit"] += 1
                    else:
                        self._stats[
                            "prefill_deferred_by_long_partial_limit"
                        ] += 1
                    deferred_requests.append(req)
                    break

                if num_tokens_this_step <= 0 or self._exceeds_token_budget(
                    current_num_batched_tokens,
                    num_tokens_this_step,
                    len(scheduled_requests),
                ):
                    self._stats["prefill_deferred_by_token_budget"] += 1
                    deferred_requests.append(req)
                    break

                req_blocks, slot_mapping = self.cache_manager.allocate_slots(
                    req_tokens,
                    num_new_tokens,
                    num_computed_tokens=num_computed_tokens,
                    cached_block_table=req.block_table,
                    delay_cache_blocks=True,
                )

                if req_blocks is None:
                    logger.warning(
                        "Failed to allocate KV cache blocks for request: %s",
                        req.request_id,
                    )
                    deferred_requests.append(req)
                    break

                req.block_table = req_blocks
                req.slot_mapping = slot_mapping
                req.num_blocks = len(req_blocks)

            if load_kv_async:
                req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                self.remote_kv_requests[req.request_id] = req
                self.pending_kv_decode_blocks += (
                    req.sampling_params.max_tokens + self.block_size - 1
                ) // self.block_size
                continue

            current_prefill_extra_blocks += self._get_prefill_extra_blocks(req)
            self._set_prefill_chunk(
                req,
                req.num_computed_tokens,
                req.num_computed_tokens + num_new_tokens,
            )
            scheduled_requests.append(req)
            scheduled_prefill_flags.append(True)

            num_tokens_this_step = num_new_tokens
            current_num_batched_tokens += num_tokens_this_step

            if self._is_partial_prefill(req, req.num_computed_tokens, num_new_tokens):
                partial_prefill_count += 1
                if self._is_long_prefill(req, req.num_computed_tokens):
                    long_partial_prefill_count += 1

            req.status = RequestStatus.RUNNING

        if deferred_requests:
            for req in deferred_requests:
                self.waiting_queue.sync_q.put(req)

        return current_num_batched_tokens

    def _has_schedule_capacity(
        self,
        num_scheduled_requests: int,
        current_num_batched_tokens: int,
    ) -> bool:
        return (
            num_scheduled_requests < self.max_batch_size
            and current_num_batched_tokens < self.max_num_batched_tokens
        )

    def _schedule_decode_requests(
        self,
        scheduled_requests: List[InferenceRequest],
        scheduled_prefill_flags: List[bool],
        current_num_batched_tokens: int = 0,
    ) -> int:
        while self._has_schedule_capacity(
            len(scheduled_requests), current_num_batched_tokens
        ):
            try:
                req = self.running_queue.sync_q.get_nowait()
            except queue.Empty:
                break
            if req.is_finished():
                self.complete_requests([req])
                continue

            try:
                req.block_table, new_slot = self.cache_manager.append_slot(
                    req.block_table, req.get_total_length(), req.get_all_token_ids()
                )
                req.slot_mapping = [new_slot]
                req.num_blocks = len(req.block_table)
                req.num_local_cached_tokens = req.get_total_length() - 1
                scheduled_requests.append(req)
                scheduled_prefill_flags.append(False)
                current_num_batched_tokens += 1
            except RuntimeError as e:
                raise RuntimeError("No available cache blocks for new token") from e

        return self._promote_remote_kv_requests(
            scheduled_requests,
            scheduled_prefill_flags,
            current_num_batched_tokens,
        )

    def _schedule_decode_batch(self) -> Optional[SchedulerOutput]:
        scheduled_requests: List[InferenceRequest] = []
        scheduled_prefill_flags: List[bool] = []
        self._schedule_decode_requests(scheduled_requests, scheduled_prefill_flags)

        if not scheduled_requests:
            return None

        return self._make_scheduler_output(scheduled_requests, scheduled_prefill_flags)

    def _promote_remote_kv_requests(
        self,
        scheduled_requests: List[InferenceRequest],
        scheduled_prefill_flags: Optional[List[bool]] = None,
        current_num_batched_tokens: int = 0,
    ) -> int:
        if self.connector is None or not self.remote_kv_requests:
            return current_num_batched_tokens
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
                if not self._has_schedule_capacity(
                    len(scheduled_requests), current_num_batched_tokens
                ):
                    break
                logger.info(
                    f"Request {req_id[:8]}... finished receiving KV, scheduling for decode."
                )
                self.update_waiting_for_remote_kv(req)
                req.status = RequestStatus.RUNNING
                scheduled_requests.append(req)
                current_num_batched_tokens += 1
                if scheduled_prefill_flags is not None:
                    scheduled_prefill_flags.append(False)
        return current_num_batched_tokens

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

    def update_from_output(self, model_output):
        self._record_execution_stats(getattr(model_output, "execution_stats", None))
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
        for req_id in failed_recving_req_ids:
            # Only track failures for active (non-aborted) requests; aborted
            # requests are handled via pending_free_blocks in finished_recving.
            if req_id in self.remote_kv_requests:
                self.failed_receiving_kv_req_ids.add(req_id)

        if invalid_block_ids:
            invalid_set = set(invalid_block_ids)

            for req in self.remote_kv_requests.values():
                start_block_idx = req.num_local_cached_tokens // self.block_size
                for i, block_id in enumerate(
                    req.block_table[start_block_idx:], start=start_block_idx
                ):
                    if block_id in invalid_set:
                        req.num_computed_tokens = i * self.block_size
                        break
        elif self.failed_receiving_kv_req_ids:
            for req_id in self.failed_receiving_kv_req_ids:
                req = self.remote_kv_requests[req_id]
                req.num_computed_tokens = req.num_local_cached_tokens

    def _record_execution_stats(self, execution_stats: Optional[dict]) -> None:
        if not execution_stats:
            return
        if execution_stats.get("split_mixed_batch"):
            self._stats["execution_split_mixed_steps"] += 1
        for phase in execution_stats.get("phases", []):
            kind = phase.get("kind")
            num_tokens = int(phase.get("num_tokens") or 0)
            if kind == "decode":
                self._stats["execution_decode_phases"] += 1
                self._stats["execution_decode_phase_tokens"] += num_tokens
            elif kind == "prefill":
                self._stats["execution_prefill_phases"] += 1
                self._stats["execution_prefill_phase_tokens"] += num_tokens

    def get_cache_stats(self) -> dict:
        """Get cache and scheduler statistics."""
        stats = {
            "num_blocks": self.cache_manager.num_blocks,
            "block_size": self.cache_manager.block_size,
            "num_free_blocks": self.cache_manager.get_num_free_blocks(),
            "usable_blocks": self.cache_manager.get_total_usable_blocks(),
            "num_pending_blocks": len(self.cache_manager.pending_block_ids),
            "num_used_blocks": len(self.cache_manager.used_block_ids),
            "chunked_prefill": {
                "enabled": self.enable_chunked_prefill,
                "decode_priority": self.decode_priority,
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "prefill_chunk_size": self.prefill_chunk_size,
                "min_prefill_chunk_size": self.min_prefill_chunk_size,
                "max_num_partial_prefills": self.max_num_partial_prefills,
                "max_long_partial_prefills": self.max_long_partial_prefills,
                "long_prefill_token_threshold": self.long_prefill_token_threshold,
            },
            "scheduler": dict(self._stats),
        }
        stats.update(self.cache_manager.get_stats())
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
