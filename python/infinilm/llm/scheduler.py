"""
Scheduler - Request scheduling and batch management with Paged Attention KV Cache.
"""

import os
import queue
import janus
import logging
from typing import List, Optional
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
      3. Continue chunked-prefill (chunking_queue) — pack up to max_batch_size.

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
        env_prefill_batch = os.environ.get("INFINI_MAX_PREFILL_BATCH")
        if max_prefill_batch_size is None and env_prefill_batch:
            max_prefill_batch_size = int(env_prefill_batch)
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

    def _try_schedule_chunking(self) -> Optional[SchedulerOutput]:
        scheduled: List[InferenceRequest] = []
        chunk_size = 0
        while len(scheduled) < self.max_batch_size:
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
        prefill_batch_cap = min(
            self.max_batch_size,
            self.max_prefill_batch_size or self.max_batch_size,
        )
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
