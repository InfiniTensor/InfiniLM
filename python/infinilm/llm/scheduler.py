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

    Scheduling priority (option A + B):
      1. Decode (running_queue) — latency-sensitive, never starves anyone.
      2. New prefill (waiting_queue) — preempts in-flight chunking so newly
         arrived short requests don't wait for an entire long prefill.
      3. Continue chunked-prefill (chunking_queue) — single-request batch.

    Anti-starvation (option B):
      After `max_waiting_yields` consecutive steps where waiting_queue won
      over a non-empty chunking_queue, the next step is forced onto the
      chunking_queue. This bounds the worst-case delay a long-prompt request
      can suffer when there is a steady inflow of new short requests.
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        num_blocks: int = 512,
        block_size: int = 256,
        max_waiting_yields: int = 4,
    ):
        self.waiting_queue = janus.Queue()
        self.running_queue = janus.Queue()
        # Requests in the middle of chunked-prefill — single-request batch only
        # (matches the C++ ChunkPrefillCompiler graph signature).
        self.chunking_queue = janus.Queue()
        self.max_batch_size = max_batch_size

        self.cache_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
        self.block_size = block_size

        # ---- Anti-starvation state ----
        # How many times waiting_queue has won over a non-empty chunking_queue
        # since the last time chunking actually ran. Reset to 0 every time we
        # run a chunking step.
        self._waiting_yields_in_a_row: int = 0
        # Upper bound on _waiting_yields_in_a_row before chunking is forced.
        self.max_waiting_yields: int = max_waiting_yields

    def add_request(self, request: InferenceRequest):
        if request is not None:
            request.status = RequestStatus.WAITING
            self.waiting_queue.sync_q.put(request)

    # ------------------------------------------------------------------ #
    #  Main scheduling entrypoint                                        #
    # ------------------------------------------------------------------ #
    def schedule(self) -> Optional[SchedulerOutput]:
        """Priority order:
        0. Forced chunking after too many consecutive waiting yields.
        1. New prefill (waiting_queue) — protect new-arrival TTFT.
        2. Continue chunked-prefill (chunking_queue) — advance long-prompt
            TTFT whenever waiting is idle; pay decode TPOT.
        3. Decode (running_queue) — lowest priority.
        """
        # 0) Forced chunking
        if self._waiting_yields_in_a_row >= self.max_waiting_yields:
            chunking_out = self._try_schedule_chunking()
            if chunking_out is not None:
                self._waiting_yields_in_a_row = 0
                return chunking_out

        # 1) New prefill
        chunking_was_nonempty = self.chunking_queue.sync_q.qsize() > 0
        waiting_out = self._try_schedule_waiting()
        if waiting_out is not None:
            if chunking_was_nonempty:
                self._waiting_yields_in_a_row += 1
            else:
                self._waiting_yields_in_a_row = 0
            return waiting_out

        # 2) Chunking (raised above decode).
        chunking_out = self._try_schedule_chunking()
        if chunking_out is not None:
            self._waiting_yields_in_a_row = 0
            return chunking_out

        # 3) Decode (lowest). If we reached here, chunking_queue was empty,
        # so the yield counter naturally resets.
        decode_out = self._try_schedule_decode()
        if decode_out is not None:
            self._waiting_yields_in_a_row = 0
            return decode_out

        return None

    # ------------------------------------------------------------------ #
    #  Per-queue schedulers                                              #
    # ------------------------------------------------------------------ #
    def _try_schedule_chunking(self) -> Optional[SchedulerOutput]:
        """Drain chunking_queue and form a batch of uniform chunk-kind.

        Invariant (enforced by llm._update_requests' chunk_mid_step check):
        a batch must be either ALL middle-chunks (no sample, no commit) OR
        ALL last-chunks (sample + commit). Mixing them is unsafe.

        Strategy: greedy drain. The first non-finished request seen fixes
        the batch's kind. Subsequent same-kind requests are added up to
        max_batch_size. Mismatched requests are buffered and re-enqueued at
        the end so they get handled in the next schedule cycle. Order within
        each kind is preserved.
        """
        scheduled: List[InferenceRequest] = []
        deferred: List[InferenceRequest] = []
        kind_is_last: Optional[bool] = None

        while len(scheduled) < self.max_batch_size:
            try:
                req = self.chunking_queue.sync_q.get_nowait()
            except queue.Empty:
                break

            if req.is_finished():
                self.complete_requests([req])
                continue

            cur_is_last = req.chunk_is_last()

            if kind_is_last is None:
                kind_is_last = cur_is_last
                scheduled.append(req)
            elif cur_is_last == kind_is_last:
                scheduled.append(req)
            else:
                deferred.append(req)

        # Re-enqueue deferred reqs preserving their relative order so they
        # aren't permanently overtaken by newcomers.
        for r in deferred:
            self.chunking_queue.sync_q.put(r)

        if scheduled:
            return SchedulerOutput(scheduled, is_prefill=True)
        return None

    def _try_schedule_waiting(self) -> Optional[SchedulerOutput]:
        """Pull new prefill requests from waiting_queue and form a prefill batch.

        If any request triggers chunked-prefill (prompt_length > chunk_size > 0),
        it's emitted alone as a single-request batch (the chunking graph requires
        a uniform chunk_size across the batch, and we don't mix chunking with
        regular prefill in the same batch).
        """
        scheduled_requests: List[InferenceRequest] = []

        while len(scheduled_requests) < self.max_batch_size:
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

            # Start chunked-prefill: emit a single-request batch immediately
            # to keep the C++ graph signature stable.
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
        """Pull running_queue requests into a decode batch."""
        scheduled_requests: List[InferenceRequest] = []

        while len(scheduled_requests) < self.max_batch_size:
            try:
                req = self.running_queue.sync_q.get_nowait()
            except queue.Empty:
                break

            # Skip requests that were already finished (timed out / canceled while running).
            if req.is_finished():
                self.complete_requests([req])
                continue

            # Decode phase: allocate slot for newly generated token.
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

    # ------------------------------------------------------------------ #
    #  External hooks (unchanged behavior)                               #
    # ------------------------------------------------------------------ #
    def requeue_chunking(self, req: InferenceRequest):
        """Put a request back into the chunking queue after a chunk has run."""
        self.chunking_queue.sync_q.put(req)

    def complete_requests(self, requests: List[InferenceRequest]):
        """Handle completed requests and free their blocks.

        Active (non-finished) requests passed here are re-enqueued into the
        running_queue — this is how prefill-finished requests migrate into
        the decode pipeline.
        """
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
        logger.info(f"accepted={total_required_blocks <= self.cache_manager.get_total_usable_blocks()}")
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