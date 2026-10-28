"""
Scheduler - Request scheduling and batch management with Paged Attention KV Cache.
"""

import logging
from collections import deque
import queue
from typing import List, Optional

import janus

from infinilm.llm.cache_manager import BlockManager, MambaCacheManager
from infinilm.llm.request import InferenceRequest, RequestStatus

logger = logging.getLogger(__name__)


class SpeculativeCacheOps:
    """Limited cache operations needed by speculative verification."""

    def __init__(self, cache_manager: BlockManager):
        self._cache_manager = cache_manager

    def append_verify_slots(
        self,
        block_table: List[int],
        start_length: int,
        num_slots: int,
    ):
        return self._cache_manager.append_slots(
            block_table,
            start_length,
            num_slots,
        )

    def rollback_to_length(self, block_table: List[int], keep_tokens: int):
        return self._cache_manager.truncate_blocks(block_table, keep_tokens)


class SchedulerOutput:
    """Scheduler output containing scheduled requests and execution phase info."""

    def __init__(
        self,
        scheduled_requests: List[InferenceRequest],
        is_prefill: bool = False,
        speculative_cache_ops: Optional[SpeculativeCacheOps] = None,
        prefill_request_ids: Optional[set[str]] = None,
        scheduled_token_counts: Optional[dict[str, int]] = None,
    ):
        self.scheduled_requests = scheduled_requests
        self.num_requests = len(scheduled_requests)
        self.is_prefill = is_prefill
        self.speculative_cache_ops = speculative_cache_ops
        self.kv_connector_metadata = None
        self.prefill_request_ids = prefill_request_ids
        self.scheduled_token_counts = scheduled_token_counts or {}

    def request_is_prefill(self, request: InferenceRequest) -> bool:
        if self.prefill_request_ids is None:
            return self.is_prefill
        return request.request_id in self.prefill_request_ids

    def request_token_count(self, request: InferenceRequest) -> int:
        if request.request_id in self.scheduled_token_counts:
            return self.scheduled_token_counts[request.request_id]
        return (
            request.get_prompt_length() - request.num_local_cached_tokens
            if self.request_is_prefill(request)
            else 1
        )

    def request_finishes_prefill(self, request: InferenceRequest) -> bool:
        return (
            not self.request_is_prefill(request)
            or request.num_local_cached_tokens + self.request_token_count(request)
            == request.get_prompt_length()
        )


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
        has_mamba_cache: bool = False,
        num_mamba_cache_blocks: int | None = None,
        enable_prefix_caching: bool = True,
        enable_mixed_prefill: bool = False,
        prefill_chunk_size: int = 0,
        partial_prefill_first: bool = False,
        max_prefill_continuations: int = 0,
        max_num_batched_visual_tokens: int = 0,
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
        self.has_mamba_cache = has_mamba_cache
        self.mamba_cache_manager = (
            MambaCacheManager(num_mamba_cache_blocks or max(2, num_blocks // 4))
            if has_mamba_cache
            else None
        )
        self.speculative_cache_ops = SpeculativeCacheOps(self.cache_manager)
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.connector = connector
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_mixed_prefill = enable_mixed_prefill and connector is None
        self.prefill_chunk_size = (
            max(0, prefill_chunk_size) if self.enable_mixed_prefill else 0
        )
        self.partial_prefill_requests: dict[str, InferenceRequest] = {}
        self.partial_prefill_first = partial_prefill_first
        self.max_prefill_continuations = max(0, max_prefill_continuations)
        self._partial_prefill_queue = deque()
        self._prefill_budget = max_num_batched_tokens
        self._chunking_active = False
        self._prefill_heavy_requests: dict[str, int] = {}
        self.max_num_batched_visual_tokens = max(0, max_num_batched_visual_tokens)
        self._visual_prefill_spans: dict[str, list[tuple[int, int, int]]] = {}

    def add_request(self, request: InferenceRequest):
        if request is not None:
            # TODO: Remove the multimodal exclusion once media-aware prefix
            # hashing and model-side cache-boundary handling are supported.
            request.initialize_block_hashes(
                self.block_size,
                self.enable_prefix_caching
                and not self.has_mamba_cache
                and not request.has_multimodal_inputs,
            )
            if (
                self.max_num_batched_visual_tokens
                and request.processed_inputs is not None
            ):
                inputs = request.processed_inputs
                if "image_grid_thw" in inputs and "image_bound" in inputs:

                    def metadata_list(value):
                        if hasattr(value, "tolist"):
                            return value.tolist()
                        return [
                            (
                                metadata_list(v)
                                if isinstance(v, (list, tuple)) or hasattr(v, "tolist")
                                else v
                            )
                            for v in value
                        ]

                    grids = metadata_list(inputs["image_grid_thw"])
                    bounds = metadata_list(inputs["image_bound"])
                    if grids and not isinstance(grids[0], list):
                        grids = [grids]
                    if bounds and isinstance(bounds[0][0], list):
                        bounds = bounds[0]
                    if len(grids) == len(bounds):
                        self._visual_prefill_spans[request.request_id] = [
                            (int(b[0]), int(b[1]), int(g[0]) * int(g[1]) * int(g[2]))
                            for b, g in zip(bounds, grids)
                        ]
            request.status = RequestStatus.WAITING
            if (
                self.prefill_chunk_size
                and self._can_chunk_request(request)
                and request.get_prompt_length() > self.prefill_chunk_size
                and request.get_prompt_length() > 2 * request.sampling_params.max_tokens
            ):
                self._prefill_heavy_requests[request.request_id] = (
                    request.get_prompt_length()
                )
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
        return current_num_batched_tokens + num_tokens_this_step > self._prefill_budget

    def _visual_token_count(
        self, request: InferenceRequest, start: int, count: int
    ) -> int:
        # Count only images actually encoded in this chunk; continuations and
        # decode must not be charged for pictures that already reside in KV.
        return sum(
            cost
            for left, right, cost in self._visual_prefill_spans.get(
                request.request_id, []
            )
            if right > start and left < start + count
        )

    def _exceeds_visual_budget(
        self,
        current: int,
        request: InferenceRequest,
        start: int,
        count: int,
        scheduled: int,
    ) -> bool:
        if not self.max_num_batched_visual_tokens or not scheduled:
            return False
        # At high occupancy keep the throughput-oriented token batch. Smaller
        # waves benefit from completing some vision work earlier; large vision
        # batches can otherwise dominate their first-token latency.
        queued = (
            self.waiting_queue.sync_q.qsize()
            + len(self._partial_prefill_queue)
            + self.running_queue.sync_q.qsize()
            + scheduled
            + 1
        )
        if queued >= max(2, (3 * self.max_batch_size + 3) // 4):
            return False
        return (
            current + self._visual_token_count(request, start, count)
            > self.max_num_batched_visual_tokens
        )

    @staticmethod
    def _can_chunk_request(request: InferenceRequest) -> bool:
        # Incomplete chunks still execute sampling internally. Restrict chunking
        # to deterministic requests so discarded samples cannot advance RNG.
        params = request.sampling_params
        return params.temperature == 0 or params.top_k == 1

    def _prefill_token_count(self, request: InferenceRequest, start: int) -> int:
        end = request.get_prompt_length()
        if self._chunking_active and self._can_chunk_request(request):
            end = min(end, start + self.prefill_chunk_size)
            # A visual embedding span must be consumed atomically. Splitting
            # inside it would either reinsert or omit image embeddings.
            for mapping in request.mm_token_index_mappings or []:
                left, right = mapping["start_index"], mapping["end_index"] + 1
                if left < end < right:
                    end = right
        return end - start

    def complete_prefill_chunk(
        self, request: InferenceRequest, token_count: int
    ) -> None:
        computed = request.num_local_cached_tokens + token_count
        if not request.num_local_cached_tokens < computed < request.get_prompt_length():
            raise RuntimeError("invalid partial prefill boundary")
        self.commit_computed_tokens(request, computed)
        request.num_computed_tokens = computed
        request.num_local_cached_tokens = computed
        if request.is_finished() or request.is_aborted():
            if not request.is_finished():
                request.mark_canceled()
            self.complete_requests([request])
            return
        request.status = RequestStatus.WAITING
        self.partial_prefill_requests[request.request_id] = request
        if self.partial_prefill_first:
            self._partial_prefill_queue.append(request)
        else:
            self.waiting_queue.sync_q.put(request)

    def _should_chunk_prefill(self, scheduled_prefills: int = 0) -> bool:
        return bool(self.prefill_chunk_size) and (
            self.waiting_queue.sync_q.qsize()
            + len(self._partial_prefill_queue)
            + self.running_queue.sync_q.qsize()
            + scheduled_prefills
            >= max(2, (3 * self.max_batch_size + 3) // 4)
            or bool(self.partial_prefill_requests)
            # Prefill-heavy arrivals may fill a token batch before they reach
            # the request-count threshold. Start chunking as soon as their
            # queued work occupies half the budget; never wait to form a batch.
            or (
                self.waiting_queue.sync_q.qsize() + scheduled_prefills >= 2
                and sum(self._prefill_heavy_requests.values())
                > self.max_num_batched_tokens // 2
            )
        )

    def schedule(self) -> Optional[SchedulerOutput]:
        """Schedule and return batch of requests to execute."""
        self._chunking_active = self._should_chunk_prefill()
        decode_reserve = (
            min(self.running_queue.sync_q.qsize(), self.max_batch_size - 1)
            if self._chunking_active and self.partial_prefill_first
            else 0
        )
        self._prefill_budget = self.max_num_batched_tokens - decode_reserve
        scheduled_token_counts = {}
        scheduled_continuations = 0
        deferred_requests = []
        scheduled_requests = []
        is_prefill = False
        current_num_batched_tokens = 0
        current_batched_visual_tokens = 0
        current_prefill_extra_blocks = 0

        # Process Waiting queue (prefill phase)
        while (
            len(scheduled_requests) < self.max_batch_size
            and current_num_batched_tokens < self._prefill_budget
        ):
            # Arrivals can continue while CPU-side allocation builds a batch.
            # Recheck pressure before execution, including already selected
            # prompts, and shorten their metadata together when it rises.
            if not self._chunking_active and self._should_chunk_prefill(
                len(scheduled_requests)
            ):
                self._chunking_active = True
                if self.partial_prefill_first:
                    self._prefill_budget = self.max_num_batched_tokens - min(
                        self.running_queue.sync_q.qsize(), self.max_batch_size - 1
                    )
                current_num_batched_tokens = 0
                current_batched_visual_tokens = 0
                for selected in scheduled_requests:
                    count = self._prefill_token_count(
                        selected, selected.num_local_cached_tokens
                    )
                    scheduled_token_counts[selected.request_id] = count
                    selected.slot_mapping = self.cache_manager.update_blocks_slot(
                        selected.block_table,
                        selected.num_local_cached_tokens,
                        selected.num_local_cached_tokens + count,
                    )
                    current_num_batched_tokens += count
                    current_batched_visual_tokens += self._visual_token_count(
                        selected, selected.num_local_cached_tokens, count
                    )
            try:
                take_continuation = bool(self._partial_prefill_queue) and (
                    not self.max_prefill_continuations
                    or scheduled_continuations < self.max_prefill_continuations
                    or self.waiting_queue.sync_q.empty()
                )
                if take_continuation:
                    req = self._partial_prefill_queue.popleft()
                    scheduled_continuations += 1
                else:
                    req = self.waiting_queue.sync_q.get_nowait()
            except queue.Empty:
                break
            # Skip requests that were already finished (e.g., timed out/canceled while waiting)
            if req.is_finished():
                self.complete_requests([req])
                continue

            if req.num_computed_tokens == 0:
                if self.has_mamba_cache:
                    cached_block_table = []
                    num_local_computed_tokens = 0
                    load_kv_async = False
                    num_external_computed_tokens = 0
                else:
                    if self.enable_prefix_caching:
                        (
                            cached_block_table,
                            num_local_computed_tokens,
                        ) = self.cache_manager.get_computed_blocks(
                            req.block_hashes, req.get_prompt_length() - 1
                        )
                    else:
                        cached_block_table = []
                        num_local_computed_tokens = 0
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

                available_cached_tokens = (
                    num_local_computed_tokens + num_external_computed_tokens
                )
                num_computed_tokens = min(
                    available_cached_tokens,
                    max(req.get_prompt_length() - 1, 0),
                )
                num_new_tokens = req.get_prompt_length() - num_computed_tokens

                # Early token budget check: skip can_accept_request and allocate_slots
                # for requests that would exceed the per-schedule token budget.
                if not load_kv_async:
                    num_tokens_this_step = self._prefill_token_count(
                        req, num_computed_tokens
                    )
                    if self._exceeds_token_budget(
                        current_num_batched_tokens,
                        num_tokens_this_step,
                        len(scheduled_requests),
                    ) or self._exceeds_visual_budget(
                        current_batched_visual_tokens,
                        req,
                        num_computed_tokens,
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
                    {r.request_id for r in scheduled_requests},
                ):
                    logger.warning(
                        "Insufficient KV cache blocks for request %s, deferring.",
                        req.request_id,
                    )

                    if num_local_computed_tokens > 0:
                        self.cache_manager.free_blocks(cached_block_table)
                    deferred_requests.append(req)
                    break

                allocation = self.cache_manager.allocate_slots(
                    num_new_tokens,
                    num_computed_tokens=num_computed_tokens,
                    cached_block_table=cached_block_table,
                )

                if allocation is None:
                    logger.warning(
                        "Failed to allocate KV cache blocks for request: %s",
                        req.request_id,
                    )
                    if num_local_computed_tokens > 0:
                        self.cache_manager.free_blocks(cached_block_table)
                    deferred_requests.append(req)
                    break
                req_blocks, slot_mapping = allocation

                if self.has_mamba_cache and req.mamba_cache_index is None:
                    req.mamba_cache_index = self.mamba_cache_manager.allocate()
                    if req.mamba_cache_index is None:
                        self.cache_manager.free_blocks(req_blocks)
                        logger.warning(
                            "Insufficient mamba cache rows for request %s, deferring.",
                            req.request_id,
                        )
                        deferred_requests.append(req)
                        break

                req.block_table = req_blocks
                req.slot_mapping = slot_mapping
                req.num_blocks = len(req_blocks)
                req.num_local_cached_tokens = (
                    num_local_computed_tokens if load_kv_async else num_computed_tokens
                )
                req.num_cache_indexed_blocks = len(cached_block_table)
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
                num_tokens_this_step = self._prefill_token_count(
                    req, req.num_local_cached_tokens
                )
                if self._exceeds_token_budget(
                    current_num_batched_tokens,
                    num_tokens_this_step,
                    len(scheduled_requests),
                ) or self._exceeds_visual_budget(
                    current_batched_visual_tokens,
                    req,
                    req.num_local_cached_tokens,
                    num_tokens_this_step,
                    len(scheduled_requests),
                ):
                    deferred_requests.append(req)
                    break
                self.commit_computed_tokens(req, req.num_computed_tokens)

            if load_kv_async:
                req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                self.remote_kv_requests[req.request_id] = req
                self.pending_kv_decode_blocks += (
                    req.sampling_params.max_tokens + self.block_size - 1
                ) // self.block_size
                continue

            current_prefill_extra_blocks += self._get_prefill_extra_blocks(req)
            scheduled_requests.append(req)

            num_tokens_this_step = self._prefill_token_count(
                req, req.num_local_cached_tokens
            )
            scheduled_token_counts[req.request_id] = num_tokens_this_step
            req.slot_mapping = self.cache_manager.update_blocks_slot(
                req.block_table,
                req.num_local_cached_tokens,
                req.num_local_cached_tokens + num_tokens_this_step,
            )
            current_num_batched_tokens += num_tokens_this_step
            current_batched_visual_tokens += self._visual_token_count(
                req, req.num_local_cached_tokens, num_tokens_this_step
            )

            req.status = RequestStatus.RUNNING

        if deferred_requests:
            for req in reversed(deferred_requests):
                if (
                    self.partial_prefill_first
                    and req.request_id in self.partial_prefill_requests
                ):
                    self._partial_prefill_queue.appendleft(req)
                else:
                    self.waiting_queue.sync_q.put(req)

        # Return prefill batch if any waiting requests were scheduled
        if scheduled_requests:
            prefill_request_ids = {req.request_id for req in scheduled_requests}
            # Include existing decode requests in the same eager forward. The
            # token budget and request ceiling include both phases. Each decode
            # row retains its own KV/Mamba state and consumes exactly one token.
            has_partial_prefill = any(
                req.num_local_cached_tokens + scheduled_token_counts[req.request_id]
                < req.get_prompt_length()
                for req in scheduled_requests
            )
            if self.enable_mixed_prefill and (
                not self.prefill_chunk_size
                or has_partial_prefill
                or self.partial_prefill_requests
            ):
                while (
                    len(scheduled_requests) < self.max_batch_size
                    and current_num_batched_tokens < self.max_num_batched_tokens
                ):
                    try:
                        req = self.running_queue.sync_q.get_nowait()
                    except queue.Empty:
                        break
                    if req.is_finished():
                        self.complete_requests([req])
                        continue
                    req.block_table, new_slot = self.cache_manager.append_slot(
                        req.block_table, req.get_total_length()
                    )
                    req.slot_mapping = [new_slot]
                    req.num_blocks = len(req.block_table)
                    req.num_local_cached_tokens = req.get_total_length() - 1
                    scheduled_requests.append(req)
                    current_num_batched_tokens += 1
            is_prefill = True
            scheduler_output = SchedulerOutput(
                scheduled_requests=scheduled_requests,
                is_prefill=is_prefill,
                speculative_cache_ops=self.speculative_cache_ops,
                prefill_request_ids=prefill_request_ids,
                scheduled_token_counts=scheduled_token_counts,
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
            req.block_table, new_slot = self.cache_manager.append_slot(
                req.block_table, req.get_total_length()
            )
            req.slot_mapping = [new_slot]
            req.num_blocks = len(req.block_table)
            req.num_local_cached_tokens = req.get_total_length() - 1
            scheduled_requests.append(req)

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
                speculative_cache_ops=self.speculative_cache_ops,
            )

            if self.connector is not None:
                meta = self.connector.build_connector_meta()
                scheduler_output.kv_connector_metadata = meta
            return scheduler_output

        if self.connector is not None:
            scheduler_output = SchedulerOutput(
                scheduled_requests=[],
                speculative_cache_ops=self.speculative_cache_ops,
            )
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
                self.commit_computed_tokens(request, request.num_computed_tokens)
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
            self.commit_computed_tokens(request, request.num_computed_tokens)
            request.num_local_cached_tokens = request.num_computed_tokens
        self.finished_receiving_kv_req_ids.discard(request.request_id)

    def complete_requests(self, requests: List[InferenceRequest]):
        """Handle completed requests and free their blocks."""
        for req in requests:
            self._prefill_heavy_requests.pop(req.request_id, None)
            self._visual_prefill_spans.pop(req.request_id, None)
            self.partial_prefill_requests.pop(req.request_id, None)
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
                if self.mamba_cache_manager is not None:
                    self.mamba_cache_manager.free(req.mamba_cache_index)
                    req.mamba_cache_index = None

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
        scheduled_request_ids: Optional[set[str]] = None,
    ) -> bool:
        if (
            self.mamba_cache_manager is not None
            and request.mamba_cache_index is None
            and not self.mamba_cache_manager.can_allocate()
        ):
            return False

        total_required_blocks = sum(
            self._get_prefill_extra_blocks(req)
            for key, req in self.partial_prefill_requests.items()
            if key not in (scheduled_request_ids or set()) and key != request.request_id
        )

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

    def commit_computed_tokens(
        self, request: InferenceRequest, num_computed_tokens: int
    ) -> None:
        if not self.enable_prefix_caching or self.has_mamba_cache:
            return

        indexed_blocks = request.num_cache_indexed_blocks
        target_blocks = min(
            num_computed_tokens // self.block_size,
            len(request.block_table),
            len(request.block_hashes),
        )
        if target_blocks == indexed_blocks:
            return

        request.num_cache_indexed_blocks = self.cache_manager.publish_computed_blocks(
            request.block_table,
            request.block_hashes,
            request.num_cache_indexed_blocks,
            num_computed_tokens,
        )

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

        invalid_set = set(invalid_block_ids)
        for req_id in failed_recving_req_ids:
            req = self.remote_kv_requests.get(req_id)
            if req is None:
                continue

            self.failed_receiving_kv_req_ids.add(req_id)
            if req.has_multimodal_inputs:
                # A physical block boundary can split a media span. Recompute
                # the prompt until multimodal cache-boundary handling is supported.
                req.num_computed_tokens = 0
                continue

            trusted_boundary = req.num_local_cached_tokens
            start_block_idx = req.num_cache_indexed_blocks
            for block_idx, block_id in enumerate(
                req.block_table[start_block_idx:], start=start_block_idx
            ):
                if block_id in invalid_set:
                    trusted_boundary = min(
                        req.num_computed_tokens,
                        block_idx * self.block_size,
                    )
                    break
            req.num_computed_tokens = trusted_boundary

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            "num_blocks": self.cache_manager.num_blocks,
            "block_size": self.cache_manager.block_size,
            "num_free_blocks": self.cache_manager.get_num_free_blocks(),
            "usable_blocks": self.cache_manager.get_total_usable_blocks(),
            "num_used_blocks": len(self.cache_manager.used_block_ids),
        }
        if self.mamba_cache_manager is not None:
            stats.update(
                {
                    "num_mamba_cache_blocks": self.mamba_cache_manager.num_blocks,
                    "num_free_mamba_cache_blocks": self.mamba_cache_manager.get_num_free_blocks(),
                    "num_used_mamba_cache_blocks": len(
                        self.mamba_cache_manager.used_block_ids
                    ),
                }
            )
        return stats
