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
    ):
        self.waiting_queue = janus.Queue()
        self.running_queue = janus.Queue()
        self.max_batch_size = max_batch_size

        self.cache_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
        self.block_size = block_size

        self.max_num_batched_tokens = max_num_batched_tokens

    def add_request(self, request: InferenceRequest):
        if request is not None:
            request.status = RequestStatus.WAITING
            self.waiting_queue.sync_q.put(request)

    def schedule(self) -> Optional[SchedulerOutput]:
        """Schedule and return batch of requests to execute."""
        scheduled_requests = []
        is_prefill = False
        current_num_batched_tokens = 0

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

            if not self.can_accept_request(req):
                self.waiting_queue.sync_q.put(req)
                break

            # Skip requests that were already finished (e.g., timed out/canceled while waiting)
            if req.is_finished():
                self.complete_requests([req])
                continue

            req_tokens = req.get_input_tokens()
            num_required_blocks = req.get_num_blocks_required(self.block_size)

            if not self.cache_manager.can_allocate(num_required_blocks):
                if not self.cache_manager.try_free_blocks(num_required_blocks):
                    raise RuntimeError("No available cache blocks for new request")

            # Allocate blocks with automatic prefix caching support
            req.block_table, req.slot_mapping, req.num_cached_tokens = (
                self.cache_manager.allocate_blocks(
                    req_tokens, req.block_table, req.get_mm_token_index_mappings()
                )
            )

            num_tokens_this_step = req.get_prompt_length() - req.num_cached_tokens
            if (
                current_num_batched_tokens + num_tokens_this_step
                >= self.max_num_batched_tokens
            ):
                if req.num_cached_tokens > 0:
                    self.cache_manager.free_blocks(req.block_table)
                    req.block_table = []
                    req.slot_mapping = []
                    req.num_cached_tokens = 0

                self.waiting_queue.sync_q.put(req)
                break

            current_num_batched_tokens += num_tokens_this_step
            req.num_blocks = len(req.block_table)
            req.status = RequestStatus.RUNNING
            scheduled_requests.append(req)

        # Return prefill batch if any waiting requests were scheduled
        if scheduled_requests:
            is_prefill = True
            return SchedulerOutput(
                scheduled_requests=scheduled_requests,
                is_prefill=is_prefill,
            )

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
                req.num_cached_tokens = req.get_total_length() - 1
                scheduled_requests.append(req)

            except RuntimeError as e:
                raise RuntimeError("No available cache blocks for new token") from e

        # Return decode batch if any running requests were scheduled
        if scheduled_requests:
            is_prefill = False
            return SchedulerOutput(
                scheduled_requests=scheduled_requests,
                is_prefill=is_prefill,
            )

        return None

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
                # Still running, put back in running queue
                self.running_queue.sync_q.put(req)

    def can_accept_request(self, request: InferenceRequest) -> bool:
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
        total_length = request.get_prompt_length()
        total_length += request.sampling_params.max_tokens
        num_blocks_needed = (total_length + self.block_size - 1) // self.block_size
        total_required_blocks += num_blocks_needed

        # Compare with total usable blocks in cache manager
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
