"""
Static Scheduler - Single-batch request scheduling for Static KV Cache.
"""

import logging
import queue
import janus
from typing import List, Optional

from infinilm.llm.request import RequestStatus, InferenceRequest, FinishReason

logger = logging.getLogger(__name__)


class StaticSchedulerOutput:
    """Static scheduler output containing single request and execution phase info."""

    def __init__(
        self,
        scheduled_requests: List[InferenceRequest],
        is_prefill: bool = False,
    ):
        self.scheduled_requests = scheduled_requests
        self.num_requests = len(scheduled_requests)
        self.is_prefill = is_prefill

    def build_model_inputs(
        self, temperature: float = 1.0, top_p: float = 0.8, top_k: int = 1
    ):
        """Construct model inputs for prefill or decode phase.

        Static cache model inputs:

        Prefill phase:
            - input_ids: All prompt tokens [1, prompt_length]
            - position_ids: [0, 1, 2, ..., prompt_length-1]
            - past_kv_lengths: [0] (no cached tokens initially)
            - total_kv_lengths: [prompt_length]

        Decode phase:
            - input_ids: Only the last generated token [1, 1]
            - position_ids: [current_position] (position in full sequence)
            - past_kv_lengths: [num_cached_tokens]
            - total_kv_lengths: [total_tokens]
            -
        """
        req = self.scheduled_requests[0]

        if self.is_prefill:
            # Prefill: send all prompt tokens
            tokens = req.get_input_tokens()
            input_ids = [tokens]
            position_ids = [list(range(len(tokens)))]
            past_kv_len = 0
            total_kv_len = len(tokens)
            input_offsets = [0, len(tokens)]
        else:
            # Decode: send only the last generated token
            last_token = req.generated_token_ids[-1]
            current_position = req.get_total_length() - 1
            input_ids = [[last_token]]
            position_ids = [[current_position]]
            past_kv_len = current_position
            total_kv_len = req.get_total_length()
            input_offsets = [0, 1]

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_kv_lengths": [past_kv_len],
            "total_kv_lengths": [total_kv_len],
            "input_offsets": input_offsets,
            "block_tables": None,
            "slot_mapping": None,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }


class StaticScheduler:
    """Request scheduler for Static KV Cache with batch_size=1.

    Simplified scheduling logic:
    - Only handles one request at a time
    - No cache block management needed
    - Simple waiting queue for incoming requests
    """

    def __init__(self, max_cache_len: int = 4096):
        self.waiting_queue = janus.Queue()
        self.running_request: Optional[InferenceRequest] = None
        self.max_cache_len = max_cache_len

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
                    continue

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
                continue

            req.status = RequestStatus.RUNNING
            self.running_request = req
            return StaticSchedulerOutput(scheduled_requests=[req], is_prefill=True)

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
            "running_request": (
                self.running_request.request_id if self.running_request else None
            ),
            "waiting_queue_size": self.waiting_queue.sync_q.qsize(),
        }
