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
        input_ids=None,
        position_ids=None,
        past_kv_lengths=None,
        total_kv_lengths=None,
        input_offsets=None,
        cu_seqlens=None,
        block_tables=None,
        slot_mapping=None,
    ):
        self.scheduled_requests = scheduled_requests
        self.num_requests = len(scheduled_requests)
        self.is_prefill = is_prefill
        self.kv_connector_metadata = None
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.past_kv_lengths = past_kv_lengths
        self.total_kv_lengths = total_kv_lengths
        self.input_offsets = input_offsets
        self.cu_seqlens = cu_seqlens
        self.block_tables = block_tables
        self.slot_mapping = slot_mapping

    def add_params(self, temperature, top_p, top_k):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    @property
    def has_model_inputs(self) -> bool:
        """Used by infer_engine.py to skip forward() when there are no model inputs."""
        return self.input_ids is not None


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
        connector=None,
    ):
        self.waiting_queue = janus.Queue()
        self.running_queue = janus.Queue()
        self.max_batch_size = max_batch_size

        self.finished_receiving_kv_req_ids: set[str] = set()
        self.failed_receiving_kv_req_ids: set[str] = set()
        self.pending_free_blocks: dict[str, list[int]] = {}

        self.cache_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
        self.block_size = block_size

        self.connector = connector

    def add_request(self, request: InferenceRequest):
        if request is not None:
            request.status = RequestStatus.WAITING

            self.waiting_queue.sync_q.put(request)

    def schedule(self) -> Optional[SchedulerOutput]:
        """Schedule and return batch of requests to execute."""
        deferred_requests = []
        scheduled_requests = []
        is_prefill = False

        # Process Waiting queue (prefill phase)
        while len(scheduled_requests) < self.max_batch_size:
            try:
                req = self.waiting_queue.sync_q.get_nowait()
            except queue.Empty:
                break
            # Skip requests that were already finished (e.g., timed out/canceled while waiting)
            if req.is_finished():
                self.complete_requests([req])
                continue

            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                if req.request_id in self.finished_receiving_kv_req_ids:
                    self.update_waiting_for_remote_kv(req)
                    req.status = RequestStatus.WAITING
                else:
                    deferred_requests.append(req)
                    continue

            req_tokens = req.get_input_tokens()

            if req.num_computed_tokens == 0:
                cached_block_table, num_local_computed_tokens = (
                    self.cache_manager.get_computed_blocks(req_tokens)
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

                num_new_tokens = req.get_total_length() - num_computed_tokens
                if not self.can_accept_request(
                    req,
                    num_local_computed_tokens,
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
                    num_local_computed_tokens=num_local_computed_tokens,
                    cached_block_table=cached_block_table,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                )

                if req_blocks is None:
                    logger.warning(
                        "Failed to allocate KV cache blocks for request: %s",
                        req.request_id,
                    )
                    deferred_requests.append(req)
                    break

                req.block_table = req_blocks
                req.slot_mapping = slot_mapping
                req.num_blocks = len(req_blocks)
                req.num_cached_tokens = num_local_computed_tokens
                req.num_computed_tokens = num_computed_tokens

                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        req,
                        req.block_table,
                        num_external_computed_tokens,
                        self.block_size,
                    )
            else:
                num_external_computed_tokens = 0
                load_kv_async = False

            if load_kv_async:
                req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                deferred_requests.append(req)
                continue

            scheduled_requests.append(req)
            req.status = RequestStatus.RUNNING

        if deferred_requests:
            for req in deferred_requests:
                self.waiting_queue.sync_q.put(req)

        # Return prefill batch if any waiting requests were scheduled
        if scheduled_requests:
            is_prefill = True
            scheduler_output = self.build_scheduler_output(
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
                req.num_cached_tokens = req.get_total_length() - 1
                scheduled_requests.append(req)

            except RuntimeError as e:
                raise RuntimeError("No available cache blocks for new token") from e

        # Return decode batch if any running requests were scheduled
        if scheduled_requests:
            is_prefill = False
            scheduler_output = self.build_scheduler_output(
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

    def build_scheduler_output(
        self,
        scheduled_requests: List[InferenceRequest],
        is_prefill: bool = False,
    ) -> SchedulerOutput:
        """Construct and return a SchedulerOutput with model inputs populated.

        Prefill phase:
            - input_ids: Flattened token list (excluding cached tokens)
            - position_ids: Position IDs for new tokens in complete sequence
            - past_kv_lengths: Number of cached tokens per request
            - total_kv_lengths: Total tokens (cached + new) per request
            - input_offsets: Start position of each request in flattened input array
            - cu_seqlens: Cumulative full sequence lengths
            - block_tables: Padded block_table for each request
            - slot_mapping: Token to slot mappings

        Decode phase:
            - input_ids: Only last generated token per request
            - position_ids: Position of last token in complete sequence
            - past_kv_lengths: Number of cached tokens per request
            - total_kv_lengths: Total sequence length per request
            - input_offsets: Offsets for each request in input array
            - cu_seqlens: Cumulative full sequence lengths
            - block_tables: Padded block_table for each request
            - slot_mapping: Single slot per request
        """
        if not scheduled_requests:
            return SchedulerOutput(
                scheduled_requests=scheduled_requests,
                is_prefill=is_prefill,
            )

        tokens = []
        seq_lens = []
        input_offsets = [0]
        block_tables = []
        slot_mapping = []
        past_kv_lengths = []
        position_ids = []
        cu_seqlens = [0]

        max_block_table_len = max(len(req.block_table) for req in scheduled_requests)
        current_offset = 0

        for req in scheduled_requests:
            num_cached = req.num_cached_tokens
            if is_prefill:
                # Prefill phase
                req_tokens = req.get_input_tokens()
                tokens_to_compute = req_tokens[num_cached:]
                tokens.extend(tokens_to_compute)

                compute_len = len(tokens_to_compute)
                seq_len = len(req_tokens)
                seq_lens.append(seq_len)

                current_offset += compute_len
                input_offsets.append(current_offset)

                slot_mapping.extend(req.slot_mapping)
                past_kv_lengths.append(num_cached)
                position_ids.extend(range(num_cached, num_cached + compute_len))

            else:
                # Decode phase
                seq_len = req.get_total_length()
                last_token = req.generated_token_ids[-1]
                tokens.append(last_token)
                seq_lens.append(seq_len)

                current_offset += 1
                input_offsets.append(current_offset)

                slot_mapping.extend(req.slot_mapping)
                past_kv_lengths.append(num_cached)
                position_ids.append(seq_len - 1)

            # Pad block_table to same length
            padded_block_table = req.block_table + [-1] * (
                max_block_table_len - len(req.block_table)
            )
            block_tables.append(padded_block_table)
            cu_seqlens.append(cu_seqlens[-1] + seq_len)

        return SchedulerOutput(
            scheduled_requests=scheduled_requests,
            is_prefill=is_prefill,
            input_ids=tokens,
            position_ids=position_ids,
            past_kv_lengths=past_kv_lengths,
            total_kv_lengths=seq_lens,
            input_offsets=input_offsets,
            cu_seqlens=cu_seqlens,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
        )

    def update_waiting_for_remote_kv(self, request: InferenceRequest):
        if request.request_id in self.failed_receiving_kv_req_ids:
            if request.num_computed_tokens:
                self.cache_manager.update_blocks_hash(
                    request.block_table, request.num_cached_tokens
                )
                request.slot_mapping = self.cache_manager.update_blocks_slot(
                    request.block_table,
                    request.num_computed_tokens,
                    request.get_prompt_length(),
                )
                request.num_cached_tokens = request.num_computed_tokens
            else:
                self.cache_manager.free_blocks(request.block_table)
                request.block_table = []
                request.slot_mapping = []
                request.num_cached_tokens = 0
            self.failed_receiving_kv_req_ids.remove(request.request_id)
        else:
            self.cache_manager.update_blocks_hash(
                request.block_table, request.num_cached_tokens
            )
            request.num_cached_tokens = request.num_computed_tokens
        self.finished_receiving_kv_req_ids.remove(request.request_id)

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
        self, request: InferenceRequest, num_local_computed_tokens: int
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

        # Compare with total usable blocks in cache manager
        return total_required_blocks <= self.cache_manager.get_total_usable_blocks()

    def update_from_output(self, model_output):
        if self.connector is None or model_output.kv_connector_output is None:
            return

        finished_recving_req_ids = (
            getattr(model_output.kv_connector_output, "finished_recving", None) or []
        )
        finished_sending_req_ids = (
            getattr(model_output.kv_connector_output, "finished_sending", None) or []
        )

        for req_id in finished_recving_req_ids:
            self.finished_receiving_kv_req_ids.add(req_id)
        for req_id in finished_sending_req_ids:
            self.cache_manager.free_blocks(self.pending_free_blocks.pop(req_id, []))

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "num_blocks": self.cache_manager.num_blocks,
            "block_size": self.cache_manager.block_size,
            "num_free_blocks": self.cache_manager.get_num_free_blocks(),
            "num_req_blocks": len(self.cache_manager.req_block_ids),
            "num_used_blocks": len(self.cache_manager.used_block_ids),
        }
