"""
Scheduler - Request scheduling and batch management with Paged Attention KV Cache.
"""

import logging
import queue
import threading
from contextlib import ExitStack, contextmanager
from typing import List, Optional

import janus

from infinilm.llm.cache_manager import BlockManager, MambaCacheManager
from infinilm.llm.request import InferenceRequest, RequestStatus

from infinilm.llm.nsa_paged_owner import OwnerClock, claim_first, capture_owner_batch

logger = logging.getLogger(__name__)


class _SpeculativeCacheReservation:
    """One verification's fixed bindings; never reused by another batch."""

    def __init__(self, ops):
        self.ops = ops
        self.rows = []  # (request, base length, original table, provisional table, draft count)
        self.submitted = False
        self.completed = False

    def _assert_active(self):
        if (self.ops._scheduler._scheduler_thread_id != threading.get_ident()
                or self.ops._pending is not self):
            raise RuntimeError("verification belongs to its active Scheduler thread")

    def mark_submitted(self):
        self._assert_active()
        if self.submitted:
            raise RuntimeError("verification reservation cannot be submitted twice")
        self.submitted = True

    def mark_completed(self):
        self._assert_active()
        if not self.submitted or self.completed:
            raise RuntimeError("verification has no unfinished submitted reservation")
        self.completed = True

    def commit(self, keep_lengths):
        self._assert_active()
        if not self.completed:
            raise RuntimeError("verification must complete before cache publication")
        if len(keep_lengths) != len(self.rows):
            raise ValueError("one accepted length is required per verification row")
        for (_, base, _, _, count), length in zip(self.rows, keep_lengths):
            if type(length) is not int or not base <= length <= base + count:
                raise ValueError("accepted length exceeds its verification reservation")
        with ExitStack() as locks:
            for req, _, _, _, _ in self.rows:
                locks.enter_context(req._terminal_lock)
            for req, base, original, _, _ in self.rows:
                if (self.ops._scheduler._active_requests.get(req.request_id) is not req
                        or req.get_total_length() != base
                        or tuple(req.block_table) != original):
                    raise RuntimeError("verification request changed before cache publication")
            lengths = [base if req.is_aborted() else length
                       for (req, base, _, _, _), length in zip(self.rows, keep_lengths)]
            tables = self.ops._cache_manager.truncate_blocks_many(
                [table for _, _, _, table, _ in self.rows], lengths)
            for (req, _, _, _, _), table in zip(self.rows, tables):
                req.block_table = table
                req.num_blocks = len(table)
                req.slot_mapping = []
            self.ops._pending = None


class SpeculativeCacheOps:
    """Scheduler-owned reservation, publication and retirement of verify pages."""

    def __init__(self, scheduler):
        self._scheduler = scheduler
        self._cache_manager = scheduler.cache_manager
        self._pending = None

    def assert_idle(self):
        if self._pending is not None:
            raise RuntimeError("speculative verification still owns cache reservations")

    @contextmanager
    def verify_batch(self, candidates):
        self._scheduler._ensure_scheduler_thread()
        requests = [candidate["req"] for candidate in candidates]
        if not requests:
            raise ValueError("verification requires a non-empty batch")
        if len({id(req) for req in requests}) != len(requests):
            raise ValueError("verification batch repeats a request")
        for candidate in candidates:
            req = candidate["req"]
            if (self._scheduler._active_requests.get(req.request_id) is not req
                    or req._paged_completion_state != "active"
                    or candidate["base_len"] != req.get_total_length()):
                raise RuntimeError("verification requires current Scheduler request ownership")
            self._cache_manager._validate_truncate_blocks(req.block_table, candidate["base_len"])
        batch = _SpeculativeCacheReservation(self)
        self._pending = batch
        try:
            for candidate in candidates:
                req, base = candidate["req"], candidate["base_len"]
                count = len(candidate["draft_tokens"])
                original = tuple(req.block_table)
                table, slots = self._cache_manager.append_slots(list(original), base + 1, count)
                batch.rows.append((req, base, original, tuple(table), count))
                candidate["block_table"] = table
                candidate["slot_mapping"] = slots
            slots = [slot for row in candidates for slot in row["slot_mapping"]]
            if len(slots) != len(set(slots)):
                raise RuntimeError("verification batch overlaps writable cache slots")
            batch.rows = tuple(batch.rows)
            yield batch
            if self._pending is batch:
                raise RuntimeError("verification batch exited without a commit")
        finally:
            if self._pending is batch:
                if batch.submitted and not batch.completed:
                    # Native failure is not proof of stream/collective quiescence.
                    # Retain original and provisional pages and prevent reuse.
                    self._scheduler._completion_failed = True
                else:
                    try:
                        self._cache_manager.truncate_blocks_many(
                            [table for _, _, _, table, _ in batch.rows],
                            [base for _, base, _, _, _ in batch.rows])
                    except BaseException:
                        self._scheduler._completion_failed = True
                        raise
                    self._pending = None


class SchedulerOutput:
    """Scheduler output containing scheduled requests and execution phase info."""

    def __init__(
        self,
        scheduled_requests: List[InferenceRequest],
        is_prefill: bool = False,
        speculative_cache_ops: Optional[SpeculativeCacheOps] = None,
        nsa_owner=None,
    ):
        self.scheduled_requests = scheduled_requests
        self.num_requests = len(scheduled_requests)
        self.is_prefill = is_prefill
        self.speculative_cache_ops = speculative_cache_ops
        self.kv_connector_metadata = None
        self.nsa_owner = nsa_owner


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
        nsa_owner_pool_epoch: int | None = None,
    ):
        if nsa_owner_pool_epoch is not None and (connector is not None or has_mamba_cache):
            raise ValueError("NSA owner slice excludes connectors and recurrent cache")
        self.waiting_queue = janus.Queue()
        self.running_queue = janus.Queue()
        self.max_batch_size = max_batch_size

        self.finished_receiving_kv_req_ids: set[str] = set()
        self.failed_receiving_kv_req_ids: set[str] = set()
        self.pending_free_blocks: dict[str, list[int]] = {}
        self.pending_kv_decode_blocks: int = 0
        self.remote_kv_requests: dict[str, InferenceRequest] = {}
        # Only live requests are retained; remote pending pages remain keyed by
        # request ID until their connector event arrives.
        self._active_requests: dict[str, InferenceRequest] = {}
        self._dequeued_request: InferenceRequest | None = None
        self._scheduler_thread_id: int | None = None
        self._completion_failed = False
        self._nsa_pending_handoff = None

        self._nsa_owner_clock = (OwnerClock(nsa_owner_pool_epoch)
                                 if nsa_owner_pool_epoch is not None else None)
        self.cache_manager = BlockManager(
            num_blocks=num_blocks, block_size=block_size,
            **({"nsa_owner_clock": self._nsa_owner_clock}
               if self._nsa_owner_clock is not None else {}))
        self.has_mamba_cache = has_mamba_cache
        self.mamba_cache_manager = (
            MambaCacheManager(num_mamba_cache_blocks or max(2, num_blocks // 4))
            if has_mamba_cache
            else None
        )
        self.speculative_cache_ops = SpeculativeCacheOps(self)
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.connector = connector
        self.enable_prefix_caching = enable_prefix_caching

    def add_request(self, request: InferenceRequest):
        if request is not None:
            if self._completion_failed:
                raise RuntimeError("paged scheduler cannot accept after a failed completion")
            # add_request is a producer-side method. It only protects this
            # object's single submission; ID ownership is claimed on the
            # scheduler thread when the queue is consumed.
            with request._terminal_lock:
                if (request._paged_submitted or request._paged_completion_state is not None
                        or request.is_finished() or request.is_aborted()
                        or request.status != RequestStatus.WAITING):
                    raise ValueError("paged scheduler requires a new waiting request")
                # Advisory producer-side rejection. _claim_request repeats the
                # check on the single consumer thread, closing the race with
                # a concurrent completion or another queued producer.
                if (request.request_id in self._active_requests
                        or request.request_id in self.pending_free_blocks):
                    raise ValueError("paged request ID still owns cache resources")
                # TODO: Remove the multimodal exclusion once media-aware prefix
                # hashing and model-side cache-boundary handling are supported.
                request.initialize_block_hashes(
                    self.block_size,
                    self.enable_prefix_caching
                    and not self.has_mamba_cache
                    and not request.has_multimodal_inputs,
                )
                request._paged_submitted = True
                try:
                    self.waiting_queue.sync_q.put_nowait(request)
                except BaseException:
                    request._paged_submitted = False
                    raise

    def _ensure_scheduler_thread(self) -> None:
        """All cache/connector mutations run on one scheduler consumer thread."""
        thread_id = threading.get_ident()
        if self._scheduler_thread_id is None:
            self._scheduler_thread_id = thread_id
        elif self._scheduler_thread_id != thread_id:
            self._completion_failed = True
            raise RuntimeError("paged scheduler resource mutation changed thread")
        if self._completion_failed:
            raise RuntimeError("paged scheduler stopped after failed completion")
        self.speculative_cache_ops.assert_idle()

    def _claim_request(self, request: InferenceRequest) -> None:
        owner = self._active_requests.get(request.request_id)
        if (not request._paged_submitted
                or request._paged_completion_state not in (None, "active")
                or request.request_id in self.pending_free_blocks
                or (owner is not None and owner is not request)):
            self._completion_failed = True
            raise RuntimeError("paged request ID or object has conflicting resource ownership")
        if owner is None and self._nsa_owner_clock is not None:
            claim_first(self._nsa_owner_clock, request)
        self._active_requests[request.request_id] = request
        request._paged_completion_state = "active"

    def _publish_nsa_handoff(self, output: SchedulerOutput) -> SchedulerOutput:
        """One current managed handoff; not a native ownership credential."""
        if self._nsa_owner_clock is None:
            return output
        if self._nsa_pending_handoff is not None or output.nsa_owner is None:
            raise RuntimeError("NSA managed handoff is missing or already pending")
        self._nsa_pending_handoff = (
            output, output.nsa_owner, tuple(output.scheduled_requests), output.is_prefill)
        return output

    def take_nsa_owner_batch(self, output: SchedulerOutput):
        """Managed consumer checks against the ACTUAL live Scheduler authority.

        This only guards the bound Python route. Public native DTO inputs still
        have no Scheduler provenance check. Private Python state is trusted.
        """
        self._ensure_scheduler_thread()
        handoff = self._nsa_pending_handoff
        if self._nsa_owner_clock is None or handoff is None or handoff[0] is not output:
            raise RuntimeError("NSA output was not published by this Scheduler or was already consumed")
        _, batch, requests, is_prefill = handoff
        actual = output.scheduled_requests
        if (output.nsa_owner is not batch or output.is_prefill != is_prefill
                or output.num_requests != len(requests) or len(actual) != len(requests)
                or any(a is not b for a, b in zip(actual, requests))):
            raise RuntimeError("NSA published batch was replaced or reordered")
        # Cancellation after publication does not revoke live ownership.
        # Execute this bounded batch; existing step handles abort after return.
        for request in requests:
            if (self._active_requests.get(request.request_id) is not request
                    or not request._paged_submitted
                    or request._paged_completion_state != "active"
                    or request.request_id in self.pending_free_blocks):
                raise RuntimeError("NSA batch lacks current live request ownership")
        live = capture_owner_batch(self.cache_manager, requests, is_prefill)
        if live != batch:
            raise RuntimeError("NSA live request/page/slot authority changed after publication")
        # Consume before processor/native submission. A retry cannot reuse this
        # current handoff; no history, event queue, registry or completion claim.
        self._nsa_pending_handoff = None
        return batch

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
        return (
            current_num_batched_tokens + num_tokens_this_step
            > self.max_num_batched_tokens
        )

    def schedule(self) -> Optional[SchedulerOutput]:
        """Schedule and return batch of requests to execute."""
        self._ensure_scheduler_thread()
        if self._nsa_owner_clock is not None and self._nsa_pending_handoff is not None:
            raise RuntimeError("NSA previous managed batch must be consumed before another schedule")
        deferred_requests = []
        scheduled_requests = []
        seen_request_objects = set()
        is_prefill = False
        current_num_batched_tokens = 0
        current_prefill_extra_blocks = 0

        # Process Waiting queue (prefill phase)
        waiting_to_inspect = self.waiting_queue.sync_q.qsize()
        while (
            waiting_to_inspect > 0
            and len(scheduled_requests) < self.max_batch_size
            and current_num_batched_tokens < self.max_num_batched_tokens
        ):
            waiting_to_inspect -= 1
            try:
                req = self.waiting_queue.sync_q.get_nowait()
                self._dequeued_request = req
            except queue.Empty:
                break
            if id(req) in seen_request_objects:
                self._completion_failed = True
                raise RuntimeError("paged request appeared twice in one schedule")
            seen_request_objects.add(id(req))
            self._claim_request(req)
            self._dequeued_request = None
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
                    num_tokens_this_step = req.get_prompt_length() - num_computed_tokens
                    if self._exceeds_token_budget(
                        current_num_batched_tokens,
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
                num_tokens_this_step = (
                    req.get_prompt_length() - req.num_local_cached_tokens
                )
                if self._exceeds_token_budget(
                    current_num_batched_tokens,
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

            num_tokens_this_step = req.get_prompt_length() - req.num_local_cached_tokens
            current_num_batched_tokens += num_tokens_this_step

            req.status = RequestStatus.RUNNING

        if deferred_requests:
            for req in deferred_requests:
                self.waiting_queue.sync_q.put(req)

        # Return prefill batch if any waiting requests were scheduled
        if scheduled_requests:
            is_prefill = True
            if self._nsa_owner_clock is not None:
                self.cache_manager.validate_nsa_request_writes(
                    scheduled_requests, decode=not is_prefill)
            scheduler_output = SchedulerOutput(
                scheduled_requests=scheduled_requests,
                is_prefill=is_prefill,
                speculative_cache_ops=self.speculative_cache_ops,
                nsa_owner=(capture_owner_batch(self.cache_manager, scheduled_requests, is_prefill)
                           if self._nsa_owner_clock is not None else None),
            )
            if self.connector is not None:
                meta = self.connector.build_connector_meta()
                scheduler_output.kv_connector_metadata = meta
            return self._publish_nsa_handoff(scheduler_output)

        # Process Running queue (decode phase)
        running_to_inspect = self.running_queue.sync_q.qsize()
        while running_to_inspect > 0 and len(scheduled_requests) < self.max_batch_size:
            running_to_inspect -= 1
            try:
                req = self.running_queue.sync_q.get_nowait()
                self._dequeued_request = req
            except queue.Empty:
                break
            if id(req) in seen_request_objects:
                self._completion_failed = True
                raise RuntimeError("paged request appeared twice in one schedule")
            seen_request_objects.add(id(req))
            self._claim_request(req)
            self._dequeued_request = None
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
            if self._nsa_owner_clock is not None:
                self.cache_manager.validate_nsa_request_writes(
                    scheduled_requests, decode=not is_prefill)
            scheduler_output = SchedulerOutput(
                scheduled_requests=scheduled_requests,
                is_prefill=is_prefill,
                speculative_cache_ops=self.speculative_cache_ops,
                nsa_owner=(capture_owner_batch(self.cache_manager, scheduled_requests, is_prefill)
                           if self._nsa_owner_clock is not None else None),
            )

            if self.connector is not None:
                meta = self.connector.build_connector_meta()
                scheduler_output.kv_connector_metadata = meta
            return self._publish_nsa_handoff(scheduler_output)

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
        self._ensure_scheduler_thread()
        if len({id(req) for req in requests}) != len(requests):
            self._completion_failed = True
            raise RuntimeError("paged completion repeated an object in one batch")
        for req in requests:
            if req._paged_completion_state == "done":
                continue
            if (self._active_requests.get(req.request_id) is not req
                    or req._paged_completion_state != "active"):
                self._completion_failed = True
                raise RuntimeError("paged completion does not own this request object")
            if req.status in [
                RequestStatus.FINISHED,
                RequestStatus.CANCELED,
                RequestStatus.FAILED,
                RequestStatus.TIMEOUT,
            ]:
                req._paged_completion_state = "releasing"
                try:
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
                        transfer_received = (
                            req.request_id in self.finished_receiving_kv_req_ids
                        )
                        self.finished_receiving_kv_req_ids.discard(req.request_id)
                        self.failed_receiving_kv_req_ids.discard(req.request_id)
                        if not transfer_received:
                            delay_free_blocks = True
                    if req.block_table and not delay_free_blocks:
                        self.cache_manager.free_blocks(req.block_table)
                    elif req.block_table and delay_free_blocks:
                        self.pending_free_blocks[req.request_id] = list(req.block_table)
                    if self.mamba_cache_manager is not None:
                        self.mamba_cache_manager.free(req.mamba_cache_index)
                        req.mamba_cache_index = None
                except BaseException:
                    req._paged_completion_state = "failed"
                    self._completion_failed = True
                    raise
                req._paged_completion_state = "done"
                del self._active_requests[req.request_id]

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

    def notify_failed_requests(self, finalize_request) -> None:
        """Notify after async fail-stop without retiring resource ownership."""
        for req in list(self._active_requests.values()) + list(self.remote_kv_requests.values()):
            finalize_request(req)
        if self._dequeued_request is not None:
            finalize_request(self._dequeued_request)
        for work_queue in (self.waiting_queue.sync_q, self.running_queue.sync_q):
            while True:
                try:
                    req = work_queue.get_nowait()
                except queue.Empty:
                    break
                finalize_request(req)

    def shutdown_drain(self, finalize_request) -> None:
        """Retire all requests on the existing scheduler consumer thread.

        A connector transfer still holding pages has no proven quiescence
        barrier here. Notify waiters, but retain native/cache ownership and
        fail closed instead of freeing pages that a transfer may still read.
        """
        self._ensure_scheduler_thread()
        seen = set()
        for work_queue in (self.waiting_queue.sync_q, self.running_queue.sync_q):
            while True:
                try:
                    req = work_queue.get_nowait()
                    self._dequeued_request = req
                except queue.Empty:
                    break
                if id(req) in seen:
                    self._completion_failed = True
                    raise RuntimeError("paged shutdown found a duplicate queued request")
                seen.add(id(req))
                self._claim_request(req)
                self._dequeued_request = None

        active = list(self._active_requests.values())
        for req in active:
            finalize_request(req)

        if (self.remote_kv_requests or self.pending_free_blocks
                or self.pending_kv_decode_blocks):
            self._completion_failed = True
            raise RuntimeError(
                "paged connector transfer pending during shutdown; "
                "native cache ownership retained"
            )

        self.complete_requests(active)
        if self._active_requests or self.pending_free_blocks:
            self._completion_failed = True
            raise RuntimeError("paged shutdown left cache ownership active")

    def can_accept_request(
        self,
        request: InferenceRequest,
        num_local_computed_tokens: int,
        current_prefill_extra_blocks: int = 0,
    ) -> bool:
        if (
            self.mamba_cache_manager is not None
            and request.mamba_cache_index is None
            and not self.mamba_cache_manager.can_allocate()
        ):
            return False

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

    def accept_model_output(self, requests, generations, physical_rows) -> None:
        """Qualify the whole returned batch using current paged ownership."""
        self._ensure_scheduler_thread()
        if physical_rows is not None or len(generations) != len(requests):
            raise RuntimeError("paged model output has incompatible cache bindings")
        if len({id(req) for req in requests}) != len(requests):
            raise RuntimeError("paged model output repeated a request")
        for req, generation in zip(requests, generations):
            if (generation is not None
                    or self._active_requests.get(req.request_id) is not req
                    or req._paged_completion_state != "active"
                    or req.request_id in self.pending_free_blocks):
                raise RuntimeError("paged model output does not own its request object")

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
        self._ensure_scheduler_thread()
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
                try:
                    self.cache_manager.free_blocks(self.pending_free_blocks[req_id])
                except BaseException:
                    self._completion_failed = True
                    raise
                del self.pending_free_blocks[req_id]
            elif req_id in self.remote_kv_requests:
                # Active request: mark ready for promotion in schedule().
                self.finished_receiving_kv_req_ids.add(req_id)
            # else: already processed or unknown, discard to avoid stale entries.
        for req_id in finished_sending_req_ids:
            if req_id in self.pending_free_blocks:
                try:
                    self.cache_manager.free_blocks(self.pending_free_blocks[req_id])
                except BaseException:
                    self._completion_failed = True
                    raise
                del self.pending_free_blocks[req_id]

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
