"""
Scheduler - Request scheduling and batch management with Paged Attention KV Cache.
"""

import queue
import janus
import logging
from typing import List, Optional
from infinilm.llm.request import RequestStatus, InferenceRequest
from infinilm.llm.cache_manager import BlockManager

logger = logging.getLogger(__name__)


class SchedulerOutput:
    """Scheduler output containing scheduled requests and execution phase info."""

    def __init__(
        self,
        scheduled_requests: List[InferenceRequest],
        is_prefill: bool = False,
    ):
        self.scheduled_requests = scheduled_requests
        self.num_requests = len(scheduled_requests)
        self.is_prefill = is_prefill
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
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.connector = connector

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

            req_tokens = req.get_input_tokens()

            if req.num_computed_tokens == 0:
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

                # Early token budget check: skip can_accept_request and allocate_slots
                # for requests that would exceed the per-schedule token budget.
                if not load_kv_async:
                    num_tokens_this_step = (
                        req.get_prompt_length() - num_local_computed_tokens
                    )
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

                req_blocks, slot_mapping = self.cache_manager.allocate_slots(
                    req_tokens,
                    num_new_tokens,
                    num_computed_tokens=num_computed_tokens,
                    cached_block_table=cached_block_table,
                    blocks_blueprint=blocks_blueprint,
                    delay_cache_blocks=load_kv_async,
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
                self.cache_manager.update_blocks_hash(
                    req.block_table, req.num_local_cached_tokens
                )

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
            try:
                req.block_table, new_slot = self.cache_manager.append_slot(
                    req.block_table, req.get_total_length(), req.get_all_token_ids()
                )
                req.slot_mapping = [new_slot]
                req.num_blocks = len(req.block_table)
                req.num_local_cached_tokens = req.get_total_length() - 1
                scheduled_requests.append(req)

            except RuntimeError as e:
                raise RuntimeError("No available cache blocks for new token") from e

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
            )

            if self.connector is not None:
                meta = self.connector.build_connector_meta()
                scheduler_output.kv_connector_metadata = meta
            return scheduler_output

        if self.connector is not None:
            scheduler_output = SchedulerOutput(scheduled_requests=[])
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
        return {
            "num_blocks": self.cache_manager.num_blocks,
            "block_size": self.cache_manager.block_size,
            "num_free_blocks": self.cache_manager.get_num_free_blocks(),
            "usable_blocks": self.cache_manager.get_total_usable_blocks(),
            "num_pending_blocks": len(self.cache_manager.pending_block_ids),
            "num_used_blocks": len(self.cache_manager.used_block_ids),
        }
