"""
Scheduler - Request scheduling and batch management with Paged Attention KV Cache.
"""

import logging
import os
import queue
from typing import List, Optional

import janus

from infinilm.llm.cache_manager import BlockManager, MambaCacheManager
from infinilm.llm.request import InferenceRequest, RequestStatus

logger = logging.getLogger(__name__)


class RequestCapacityError(ValueError):
    """Raised when one request can never fit in the configured KV cache."""


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


class SchedulerOutput:
    """Scheduler output containing scheduled requests and execution phase info."""

    def __init__(
        self,
        scheduled_requests: List[InferenceRequest],
        is_prefill: bool = False,
        speculative_cache_ops: Optional[SpeculativeCacheOps] = None,
        prefill_request_ids: Optional[set[str]] = None,
    ):
        self.scheduled_requests = scheduled_requests
        self.num_requests = len(scheduled_requests)
        if prefill_request_ids is None:
            prefill_request_ids = (
                {req.request_id for req in scheduled_requests} if is_prefill else set()
            )
        scheduled_request_ids = {req.request_id for req in scheduled_requests}
        unknown_ids = set(prefill_request_ids) - scheduled_request_ids
        if unknown_ids:
            raise ValueError(
                f"Prefill request IDs are not part of the scheduled batch: {unknown_ids}"
            )

        self.prefill_request_ids = frozenset(prefill_request_ids)
        self.is_prefill = bool(scheduled_requests) and (
            len(self.prefill_request_ids) == len(scheduled_requests)
        )
        self.is_mixed = bool(self.prefill_request_ids) and not self.is_prefill
        self.is_decode_only = bool(scheduled_requests) and not self.prefill_request_ids
        self.speculative_cache_ops = speculative_cache_ops
        self.kv_connector_metadata = None

    def is_prefill_request(self, request: InferenceRequest) -> bool:
        """Return whether a request contributes prompt tokens in this batch."""
        return request.request_id in self.prefill_request_ids


class Scheduler:
    """Request scheduler with integrated BlockManager for KV cache management.

    Scheduling logic:
    1. Running queue: Schedule one decode token per active request
    2. Waiting queue: Fill remaining batch/token budget with prefills
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
        allow_mixed_batch: bool = False,
        max_model_len: int | None = None,
        max_num_mixed_prefill_tokens: int | None = None,
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
        cache_token_capacity = num_blocks * block_size
        self.max_model_len = min(
            max_model_len if max_model_len is not None else cache_token_capacity,
            cache_token_capacity,
        )
        if max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be positive")
        self.max_num_batched_tokens = max_num_batched_tokens
        # Keep one long request from entering the vendor sparse-prefill kernel
        # as a single pathological launch while preserving the full batch budget.
        max_prefill_chunk_tokens = int(
            os.environ.get("INFINILM_MAX_PREFILL_CHUNK_TOKENS", "2048")
        )
        if max_prefill_chunk_tokens < 1:
            raise ValueError("INFINILM_MAX_PREFILL_CHUNK_TOKENS must be positive")
        self.max_prefill_chunk_tokens = min(
            max_num_batched_tokens, max_prefill_chunk_tokens
        )
        if max_num_mixed_prefill_tokens is None:
            max_num_mixed_prefill_tokens = max_num_batched_tokens
        if max_num_mixed_prefill_tokens < 1:
            raise ValueError("max_num_mixed_prefill_tokens must be positive")
        self.max_num_mixed_prefill_tokens = max_num_mixed_prefill_tokens
        self.connector = connector
        self.allow_mixed_batch = allow_mixed_batch
        self._schedule_batch_id = 0

    def validate_request(self, request: InferenceRequest) -> None:
        max_tokens = request.sampling_params.max_tokens or 0
        requested_length = request.get_prompt_length() + max_tokens
        if requested_length > self.max_model_len:
            raise RequestCapacityError(
                f"This request requires up to {requested_length} tokens "
                f"({request.get_prompt_length()} prompt + {max_tokens} output), "
                f"but the server's effective max model length is "
                f"{self.max_model_len} tokens"
            )

    def add_request(self, request: InferenceRequest):
        if request is not None:
            self.validate_request(request)
            request.status = RequestStatus.WAITING
            self.waiting_queue.sync_q.put(request)

    def should_coalesce_prefill(self) -> bool:
        """Return True only at an idle-to-prefill scheduling boundary."""
        return (
            self.running_queue.sync_q.qsize() == 0
            and self.waiting_queue.sync_q.qsize() > 0
        )

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

    def _prefill_chunk_size(
        self,
        remaining_tokens: int,
        current_num_batched_tokens: int,
        current_num_prefill_tokens: int,
        num_decode_requests: int,
    ) -> int:
        available = self.max_num_batched_tokens - current_num_batched_tokens
        if num_decode_requests:
            available = min(
                available,
                self.max_num_mixed_prefill_tokens - current_num_prefill_tokens,
            )
        chunk_size = min(remaining_tokens, self.max_prefill_chunk_tokens, available)
        if chunk_size < remaining_tokens:
            # Intermediate boundaries must be scheduler-block aligned so that
            # completed prefix blocks can be committed without partial hashes.
            chunk_size = (chunk_size // self.block_size) * self.block_size
        return max(chunk_size, 0)

    def requeue_prefill_chunk(self, request: InferenceRequest, chunk_end: int):
        request.num_local_cached_tokens = chunk_end
        request.num_computed_tokens = chunk_end
        request.slot_mapping = []
        request.prefill_chunk_end = None
        request.status = RequestStatus.WAITING
        self.waiting_queue.sync_q.put(request)

    def _exceeds_mixed_prefill_budget(
        self,
        current_num_prefill_tokens: int,
        num_tokens_this_step: int,
        num_decode_requests: int,
    ) -> bool:
        if num_decode_requests == 0:
            return False
        return (
            current_num_prefill_tokens + num_tokens_this_step
            > self.max_num_mixed_prefill_tokens
        )

    def schedule(self) -> Optional[SchedulerOutput]:
        """Schedule and return batch of requests to execute."""
        deferred_requests = []
        scheduled_requests = []
        prefill_request_ids = set()
        current_num_batched_tokens = 0
        current_num_prefill_tokens = 0
        current_reserved_extra_blocks = 0

        # Protect inter-token latency by scheduling active decodes first. Waiting
        # prefills can still join the same forward when batch/token budget remains.
        while (
            len(scheduled_requests) < self.max_batch_size
            and current_num_batched_tokens < self.max_num_batched_tokens
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
            except RuntimeError as e:
                raise RuntimeError("No available cache blocks for new token") from e

            req.slot_mapping = [new_slot]
            req.num_blocks = len(req.block_table)
            req.num_local_cached_tokens = req.get_total_length() - 1
            scheduled_requests.append(req)
            current_num_batched_tokens += 1
            current_reserved_extra_blocks += self._get_prefill_extra_blocks(req)

        num_decode_requests = len(scheduled_requests)

        # Fill the rest of the batch with waiting prefills. This forms a mixed
        # prefill/decode batch whenever running requests leave capacity.
        while (
            (self.allow_mixed_batch or not scheduled_requests)
            and len(scheduled_requests) < self.max_batch_size
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
                num_tokens_this_step = self._prefill_chunk_size(
                    num_new_tokens,
                    current_num_batched_tokens,
                    current_num_prefill_tokens,
                    num_decode_requests,
                )

                if not load_kv_async and num_tokens_this_step == 0:
                    if num_local_computed_tokens > 0:
                        self.cache_manager.free_blocks(cached_block_table)
                    deferred_requests.append(req)
                    break

                if not self.can_accept_request(
                    req,
                    num_local_computed_tokens,
                    current_reserved_extra_blocks,
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
                    delay_cache_blocks=(
                        load_kv_async or num_tokens_this_step < num_new_tokens
                    ),
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
                chunk_end = num_computed_tokens + num_tokens_this_step
                req.slot_mapping = self.cache_manager.update_blocks_slot(
                    req_blocks, num_computed_tokens, chunk_end
                )
                req.prefill_chunk_end = chunk_end
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
                remaining_tokens = req.get_prompt_length() - req.num_local_cached_tokens
                num_tokens_this_step = self._prefill_chunk_size(
                    remaining_tokens,
                    current_num_batched_tokens,
                    current_num_prefill_tokens,
                    num_decode_requests,
                )
                if num_tokens_this_step == 0:
                    deferred_requests.append(req)
                    break
                chunk_end = req.num_local_cached_tokens + num_tokens_this_step
                req.slot_mapping = self.cache_manager.update_blocks_slot(
                    req.block_table, req.num_local_cached_tokens, chunk_end
                )
                req.prefill_chunk_end = chunk_end

            if load_kv_async:
                req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                self.remote_kv_requests[req.request_id] = req
                self.pending_kv_decode_blocks += (
                    req.sampling_params.max_tokens + self.block_size - 1
                ) // self.block_size
                continue

            current_reserved_extra_blocks += self._get_prefill_extra_blocks(req)
            scheduled_requests.append(req)
            prefill_request_ids.add(req.request_id)

            current_num_batched_tokens += num_tokens_this_step
            current_num_prefill_tokens += num_tokens_this_step

            req.status = RequestStatus.RUNNING

        if deferred_requests:
            for req in deferred_requests:
                self.waiting_queue.sync_q.put(req)

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
                        current_num_batched_tokens += 1
                        current_reserved_extra_blocks += self._get_prefill_extra_blocks(
                            req
                        )
                        scheduled_requests.append(req)
                    else:
                        break  # Defer promotion to next schedule() if batch is full

        # Return decode batch if any running requests were scheduled
        if scheduled_requests:
            self._schedule_batch_id += 1
            if prefill_request_ids:
                prefill_tokens = sum(
                    req.prefill_chunk_end - req.num_local_cached_tokens
                    for req in scheduled_requests
                    if req.request_id in prefill_request_ids and req.prefill_chunk_end
                )
                logger.info(
                    "Scheduler prefill batch: id=%s requests=%s prefills=%s "
                    "prefill_tokens=%s waiting=%s mixed=%s",
                    self._schedule_batch_id,
                    len(scheduled_requests),
                    len(prefill_request_ids),
                    prefill_tokens,
                    self.waiting_queue.sync_q.qsize(),
                    bool(len(prefill_request_ids) != len(scheduled_requests)),
                )
            scheduler_output = SchedulerOutput(
                scheduled_requests=scheduled_requests,
                speculative_cache_ops=self.speculative_cache_ops,
                prefill_request_ids=prefill_request_ids,
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
            total_required_blocks += self._get_prefill_extra_blocks(req)
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

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            "num_blocks": self.cache_manager.num_blocks,
            "block_size": self.cache_manager.block_size,
            "num_free_blocks": self.cache_manager.get_num_free_blocks(),
            "usable_blocks": self.cache_manager.get_total_usable_blocks(),
            "num_pending_blocks": len(self.cache_manager.pending_block_ids),
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
