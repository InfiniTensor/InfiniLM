"""
Static Scheduler - Single-batch request scheduling for Static KV Cache.
"""

import logging
import queue
import janus
from typing import List, Optional

from infinilm.llm.cache_manager import BlockManager
from infinilm.llm.request import (
    RequestStatus,
    InferenceRequest,
    FinishReason,
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

    def __init__(self, max_cache_len: int = 4096):
        self.waiting_queue = janus.Queue()
        self.running_request: Optional[InferenceRequest] = None
        self.max_cache_len = max_cache_len
        self.cached_block_hashes: List[int] = []
        self.pending_block_hashes: List[int] = []

    def add_request(self, request: InferenceRequest):
        if request is not None:
            request.status = RequestStatus.WAITING
            self.waiting_queue.sync_q.put(request)

    def schedule(self) -> Optional[StaticSchedulerOutput]:
        """Schedule and return single request to execute."""
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

                total_length = req.get_total_length()
                if total_length % _BLOCK_SIZE == 1 and total_length > _BLOCK_SIZE:
                    block_index = total_length // _BLOCK_SIZE - 1
                    if len(self.cached_block_hashes) <= block_index:
                        all_tokens = req.get_all_token_ids()
                        block_tokens = all_tokens[-(_BLOCK_SIZE + 1) : -1]
                        prev_h = (
                            self.cached_block_hashes[-1]
                            if self.cached_block_hashes
                            else -1
                        )
                        new_h = BlockManager.compute_hash(block_tokens, prev_h)
                        self.cached_block_hashes.append(new_h)
                        logger.debug(
                            f"Decode: appended block hash at index {block_index}"
                        )

                return StaticSchedulerOutput(scheduled_requests=[req], is_prefill=False)

            # Case 2: Get new request from waiting queue (prefill phase)
            try:
                req = self.waiting_queue.sync_q.get_nowait()
            except queue.Empty:
                return None

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

            tokens = req.prompt_token_ids
            num_full_blocks = prompt_len // _BLOCK_SIZE
            matched = 0

            self.pending_block_hashes.clear()

            for i in range(num_full_blocks):
                prev_h = self.cached_block_hashes[i - 1] if i > 0 else -1
                h = BlockManager.compute_hash(
                    tokens[i * _BLOCK_SIZE : (i + 1) * _BLOCK_SIZE], prev_h
                )
                if (
                    i < len(self.cached_block_hashes)
                    and h == self.cached_block_hashes[i]
                ):
                    matched = i + 1
                else:
                    del self.cached_block_hashes[i:]
                    cur_h = h
                    self.pending_block_hashes.append(cur_h)
                    for j in range(i + 1, num_full_blocks):
                        cur_h = BlockManager.compute_hash(
                            tokens[j * _BLOCK_SIZE : (j + 1) * _BLOCK_SIZE],
                            cur_h,
                        )
                        self.pending_block_hashes.append(cur_h)
                    break
            else:
                del self.cached_block_hashes[matched:]

            prefix_hit_len = matched * _BLOCK_SIZE

            # Prevent 100% prefix cache hits to avoid an empty prefill input.
            # When prefix_hit_len equals the total token length, the remaining tokens
            # for prefill would be zero (input_ids becomes empty), leading to crashes
            # or incorrect state transitions in the downstream processor.
            #
            # This fix is directly inspired by SGLang's approach in their core scheduler
            # logic (specifically the `adjust_max_prefix_ids` method). SGLang explicitly
            # trims the prefix matching length to ensure at least one token is left:
            #
            #   # To work around some bugs in logprob computation, we need to
            #   # ensure each request has at least one token. Later, we can relax
            #   # this requirement and use `input_len`.
            #   max_prefix_len = input_len - 1
            #
            # By forcing the last token to be "missed" from the cache hit, we guarantee
            # that every request has at least one token to go through the prefill forward
            # pass. This elegantly avoids the complexity of hijacking the request into
            # the decode phase and handles edge cases in logprob computation gracefully.
            #
            # Mathematically and architecturally, this 2-line adjustment is exactly
            # equivalent to SGLang's robust production strategy.
            #
            if prefix_hit_len >= len(tokens):
                prefix_hit_len = len(tokens) - 1

            logger.info(
                f"Prefill cache match: {matched}/{num_full_blocks} blocks "
                f"({prefix_hit_len} tokens reused, {len(self.pending_block_hashes)} pending)"
            )

            req.status = RequestStatus.RUNNING
            self.running_request = req
            return StaticSchedulerOutput(
                scheduled_requests=[req], is_prefill=True, prefix_hit_len=prefix_hit_len
            )

    def update_cache(self):
        """Commit hashes computed during prefill into the confirmed cache hash list."""
        self.cached_block_hashes.extend(self.pending_block_hashes)
        self.pending_block_hashes.clear()
        logger.debug(
            f"update_cache: cached_block_hashes now has {len(self.cached_block_hashes)} blocks"
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
        }
