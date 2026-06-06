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

    Scheduling priority:
      1. New prefill (waiting_queue) — minimize TTFT for new requests.
      2. Decode (running_queue).
      3. Continue chunked-prefill (chunking_queue) — single-request batch.

    Anti-starvation: after `max_waiting_yields` consecutive steps where
    waiting/decode won over a non-empty chunking_queue, chunking is forced.
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
        self.max_prefill_batch_size = max_prefill_batch_size

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

    def _try_schedule_chunking(self) -> Optional[SchedulerOutput]:
        scheduled: List[InferenceRequest] = []
        while len(scheduled) < self.max_batch_size:
            try:
                req = self.chunking_queue.sync_q.get_nowait()
            except queue.Empty:
                break
            if req.is_finished():
                self.complete_requests([req])
                continue
            # Last chunk runs alone (sampling + block commit).
            if req.chunk_is_last():
                if not scheduled:
                    return SchedulerOutput([req], is_prefill=True)
                self.chunking_queue.sync_q.put(req)
                break
            scheduled.append(req)
        if scheduled:
            return SchedulerOutput(scheduled, is_prefill=True)
        return None

    def _try_schedule_waiting(self) -> Optional[SchedulerOutput]:
        scheduled_requests: List[InferenceRequest] = []
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
                if scheduled_requests:
                    for already in scheduled_requests:
                        already.status = RequestStatus.WAITING
                        self.waiting_queue.sync_q.put(already)
                return SchedulerOutput([req], is_prefill=True)

            scheduled_requests.append(req)

        if scheduled_requests:
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
        """Put a request back into the chunking queue after a chunk has run."""
        self.chunking_queue.sync_q.put(req)

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
