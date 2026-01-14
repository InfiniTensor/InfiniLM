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

    def build_model_inputs(
        self, temperature: float = 1.0, top_p: float = 0.8, top_k: int = 1
    ):
        """Construct model inputs for prefill or decode phase.

        Prefill phase:
            - input_ids: Flattened token list (excluding cached tokens)
            - position_ids: Position IDs for new tokens in complete sequence
            - past_kv_lengths: Number of cached tokens per request
            - total_kv_lengths: Total tokens (cached + new) per request
            - input_offsets: Start position of each request in flattened array
            - block_tables: Padded block_table for each request
            - slot_mapping: Token to slot mappings

        Decode phase:
            - input_ids: Only last generated token per request
            - position_ids: Position of last token in complete sequence
            - past_kv_lengths: Number of cached tokens per request
            - total_kv_lengths: Total sequence length per request
            - input_offsets: Offsets for each request
            - block_tables: Padded block_table for each request
            - slot_mapping: Single slot per request
        """
        if not self.scheduled_requests:
            raise RuntimeError(
                "build_model_inputs called with empty scheduled_requests"
            )

        tokens = []
        seq_lens = []
        seq_offsets = [0]
        block_tables = []
        slot_mapping = []
        cached_lens = []
        position_ids = []

        max_block_table_len = max(
            len(req.block_table) for req in self.scheduled_requests
        )
        current_offset = 0

        for req in self.scheduled_requests:
            num_cached = req.num_cached_tokens
            if self.is_prefill:
                # Prefill phase
                req_tokens = req.get_input_tokens()
                tokens_to_compute = req_tokens[num_cached:]
                tokens.extend(tokens_to_compute)

                seq_len = len(tokens_to_compute)
                seq_lens.append(len(req_tokens))

                current_offset += seq_len
                seq_offsets.append(current_offset)

                slot_mapping.extend(req.slot_mapping)
                cached_lens.append(num_cached)
                position_ids.extend(range(num_cached, num_cached + seq_len))

            else:
                # Decode phase
                last_token = req.generated_token_ids[-1]
                tokens.append(last_token)
                seq_lens.append(req.get_total_length())

                current_offset += 1
                seq_offsets.append(current_offset)

                slot_mapping.extend(req.slot_mapping)
                cached_lens.append(num_cached)
                position_ids.append(req.get_total_length() - 1)

            # Pad block_table to same length
            padded_block_table = req.block_table + [-1] * (
                max_block_table_len - len(req.block_table)
            )
            block_tables.append(padded_block_table)

        return {
            "input_ids": tokens,
            "position_ids": position_ids,
            "past_kv_lengths": cached_lens,
            "total_kv_lengths": seq_lens,
            "input_offsets": seq_offsets,
            "block_tables": block_tables,
            "slot_mapping": slot_mapping,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }


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
        num_blocks: int = 8 * 1024,
        block_size: int = 16,
    ):
        self.waiting_queue = janus.Queue()
        self.running_queue = janus.Queue()
        self.max_batch_size = max_batch_size

        self.cache_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
        self.block_size = block_size

    def add_request(self, request: InferenceRequest):
        if request is not None:
            request.status = RequestStatus.WAITING
            self.waiting_queue.sync_q.put(request)

    def schedule(self) -> Optional[SchedulerOutput]:
        """Schedule and return batch of requests to execute."""
        scheduled_requests = []
        is_prefill = False

        # Process Waiting queue (prefill phase)
        while len(scheduled_requests) < self.max_batch_size:
            try:
                req = self.waiting_queue.sync_q.get_nowait()
            except queue.Empty:
                break

            req_tokens = req.get_input_tokens()
            num_required_blocks = req.get_num_blocks_required(self.block_size)

            if not self.cache_manager.can_allocate(num_required_blocks):
                if not self.cache_manager.try_free_blocks(num_required_blocks):
                    raise RuntimeError("No available cache blocks")

            # Allocate blocks with automatic prefix caching support
            req.block_table, req.slot_mapping, req.num_cached_tokens = (
                self.cache_manager.allocate_blocks(req_tokens, req.block_table)
            )

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
                raise RuntimeError("No available cache blocks") from e

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

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "num_blocks": self.cache_manager.num_blocks,
            "block_size": self.cache_manager.block_size,
            "num_free_blocks": self.cache_manager.get_num_free_blocks(),
            "num_req_blocks": len(self.cache_manager.req_block_ids),
            "num_used_blocks": len(self.cache_manager.used_block_ids),
        }
