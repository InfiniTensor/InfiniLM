"""
Static Scheduler - Single-batch request scheduling for Static KV Cache.
"""

import logging
import time
from collections import deque
from itertools import count
from typing import Callable, List, Optional

import janus

from infinilm.config.engine_config import (
    DEFAULT_PRIORITY_AGING_INTERVAL,
    validate_priority_aging_interval,
)
from infinilm.llm.prefix_cache import BlockHash
from infinilm.llm.priority_scheduling import (
    PrioritySchedulingStats,
    drain_priority_ordered,
    mark_request_enqueued,
)
from infinilm.llm.request import (
    FinishReason,
    InferenceRequest,
    RequestStatus,
    TokenOutput,
)

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 16


class StaticSchedulerOutput:
    """Static scheduler output containing single request and execution phase info."""

    def __init__(
        self,
        scheduled_requests: List[InferenceRequest],
        is_prefill: bool = False,
        prefix_hit_len: int = 0,
    ):
        self.scheduled_requests = scheduled_requests
        self.num_requests = len(scheduled_requests)
        self.is_prefill = is_prefill
        self.prefix_hit_len = prefix_hit_len
        self.kv_connector_metadata = None


class StaticScheduler:
    """Request scheduler for Static KV Cache with batch_size=1.

    Simplified scheduling logic:
    - Only handles one request at a time
    - No cache block management needed
    - Simple waiting queue for incoming requests
    - Prefix cache reuse via chained block hashing (block size = _BLOCK_SIZE)
    """

    def __init__(
        self,
        max_cache_len: int = 4096,
        enable_prefix_caching: bool = True,
        priority_aging_interval: float = DEFAULT_PRIORITY_AGING_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.priority_aging_interval = validate_priority_aging_interval(
            priority_aging_interval
        )
        self.waiting_queue = janus.Queue()
        self.running_request: Optional[InferenceRequest] = None
        self.max_cache_len = max_cache_len
        self.enable_prefix_caching = enable_prefix_caching
        self.cached_block_hashes: List[BlockHash] = []
        self._clock = clock
        self._enqueue_sequence = count()
        self.priority_scheduling_stats = PrioritySchedulingStats()

    def add_request(self, request: InferenceRequest):
        if request is not None:
            # TODO: Remove the multimodal exclusion once media-aware prefix
            # hashing and model-side cache-boundary handling are supported.
            request.initialize_block_hashes(
                _BLOCK_SIZE,
                self.enable_prefix_caching and not request.has_multimodal_inputs,
            )
            request.status = RequestStatus.WAITING
            mark_request_enqueued(request, next(self._enqueue_sequence), self._clock)
            self.waiting_queue.sync_q.put(request)

    def schedule(self) -> Optional[StaticSchedulerOutput]:
        """Schedule and return single request to execute."""
        waiting_requests = None
        while True:
            # Case 1: Continue running request (decode phase)
            if self.running_request is not None:
                req = self.running_request

                if req.is_finished():
                    self.running_request = None
                    continue

                if req.get_total_length() > self.max_cache_len:
                    logger.warning(
                        f"Request {req.request_id} exceeds max_cache_len={self.max_cache_len}, "
                        "completing request."
                    )
                    self.running_request = None
                    req.mark_failed(FinishReason.LENGTH)
                    output = TokenOutput(
                        request_id=req.request_id,
                        token_id=-1,
                        token_text="",
                        finished=True,
                        finish_reason=req.finish_reason,
                        generated_text=req.generated_text,
                    )
                    try:
                        req.output_queue.sync_q.put(output)
                    except Exception as e:
                        logger.warning(
                            f"Failed to put completion token for {req.request_id}: {e}. "
                            f"Likely due to client disconnecting or request cancelation."
                        )
                    continue

                return StaticSchedulerOutput(scheduled_requests=[req], is_prefill=False)

            # Case 2: Get new request from waiting queue (prefill phase)
            if waiting_requests is None:
                waiting_requests = deque(
                    drain_priority_ordered(
                        self.waiting_queue.sync_q,
                        self.priority_aging_interval,
                        self._clock,
                    )
                )
            if not waiting_requests:
                return None
            req = waiting_requests.popleft()

            if req.is_finished():
                continue

            prompt_len = req.get_prompt_length()

            if prompt_len > self.max_cache_len:
                logger.error(
                    f"Request {req.request_id} prompt length {prompt_len} "
                    f"exceeds max_cache_len={self.max_cache_len}. Request rejected."
                )

                req.mark_failed(FinishReason.LENGTH)
                output = TokenOutput(
                    request_id=req.request_id,
                    token_id=-1,
                    token_text="",
                    finished=True,
                    finish_reason=req.finish_reason,
                    generated_text=req.generated_text,
                )
                try:
                    req.output_queue.sync_q.put(output)
                except Exception as e:
                    logger.warning(
                        f"Failed to put completion token for {req.request_id}: {e}. "
                        f"Likely due to client disconnecting or request cancelation."
                    )
                continue

            matched = 0

            if self.enable_prefix_caching:
                for block_idx in range(len(req.block_hashes)):
                    if (
                        block_idx >= len(self.cached_block_hashes)
                        or req.block_hashes[block_idx]
                        != self.cached_block_hashes[block_idx]
                    ):
                        break
                    matched += 1
                self.cached_block_hashes = self.cached_block_hashes[:matched]
            else:
                self.cached_block_hashes.clear()

            # Leave the last prompt token for a non-empty model input.
            prefix_hit_len = min(matched * _BLOCK_SIZE, max(prompt_len - 1, 0))

            logger.info(
                f"Prefill cache match: {matched}/{len(req.block_hashes)} blocks "
                f"({prefix_hit_len} tokens reused)"
            )

            req.status = RequestStatus.RUNNING
            self.running_request = req
            now = self._clock()
            wait_time, admission_priority, aged = (
                self.priority_scheduling_stats.record_admission(
                    req, now, self.priority_aging_interval
                )
            )
            logger.debug(
                "Admitting request %s with priority=%d, effective_priority=%d, "
                "wait_time=%.3fs, waiting_queue_size=%d, aged=%s.",
                req.request_id[:8],
                req.priority,
                admission_priority,
                wait_time,
                len(waiting_requests) + self.waiting_queue.sync_q.qsize(),
                aged,
            )
            for waiting_req in waiting_requests:
                self.waiting_queue.sync_q.put(waiting_req)
            return StaticSchedulerOutput(
                scheduled_requests=[req], is_prefill=True, prefix_hit_len=prefix_hit_len
            )

    def commit_computed_tokens(
        self, request: InferenceRequest, num_computed_tokens: int
    ) -> None:
        """Publish the current request's computed static-cache prefix."""
        if not self.enable_prefix_caching:
            self.cached_block_hashes.clear()
            return
        num_computed_blocks = min(
            num_computed_tokens // _BLOCK_SIZE,
            len(request.block_hashes),
        )
        if len(self.cached_block_hashes) > num_computed_blocks:
            del self.cached_block_hashes[num_computed_blocks:]
        start_block = len(self.cached_block_hashes)
        if start_block == num_computed_blocks:
            return

        self.cached_block_hashes.extend(
            request.block_hashes[start_block:num_computed_blocks]
        )

    def update_from_output(self, model_output):
        """Static cache has no scheduler-side connector state to update."""
        return None

    def complete_requests(self, requests: List[InferenceRequest]):
        """Handle completed requests."""
        for req in requests:
            if req.is_finished() and req == self.running_request:
                self.running_request = None
                logger.debug(f"Completed request {req.request_id}")

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "max_cache_len": self.max_cache_len,
            "cached_blocks": len(self.cached_block_hashes),
            "cached_tokens": len(self.cached_block_hashes) * _BLOCK_SIZE,
            "running_request": (
                self.running_request.request_id if self.running_request else None
            ),
            "waiting_queue_size": self.waiting_queue.sync_q.qsize(),
            "priority_scheduling": self.priority_scheduling_stats.snapshot(),
        }
