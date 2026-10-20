"""
Static Scheduler - row-leased forwards for Static KV Cache.
"""

import logging
import queue
from typing import List, Optional

import janus

from infinilm.llm.prefix_cache import BlockHash
from infinilm.llm.request import (
    FinishReason,
    InferenceRequest,
    RequestStatus,
    TokenOutput,
)

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 16


class StaticSchedulerOutput:
    """Requests, their physical KV rows, and execution phase."""

    def __init__(
        self,
        scheduled_requests: List[InferenceRequest],
        cache_row_length: int,
        is_prefill: bool = False,
        prefix_hit_len: int = 0,
        physical_row: int = 0,
        cache_capacity: int = 1,
        physical_rows: Optional[List[int]] = None,
    ):
        rows = [physical_row] if physical_rows is None else list(physical_rows)
        if (not scheduled_requests or len(rows) != len(scheduled_requests)
                or len(set(rows)) != len(rows)
                or any(row < 0 or row >= cache_capacity for row in rows)
                or (is_prefill and len(rows) != 1)):
            raise ValueError("static scheduler output requires one unique cache row per request")
        self.scheduled_requests = scheduled_requests
        self.num_requests = len(scheduled_requests)
        self.is_prefill = is_prefill
        self.prefix_hit_len = prefix_hit_len
        self.physical_row = rows[0]
        self.physical_rows = rows
        self.cache_row_length = cache_row_length
        self.cache_capacity = cache_capacity
        self.kv_connector_metadata = None


class _RowLease:
    """One request's ownership of a physical static-cache row."""

    def __init__(self, row: int, generation: int, request: InferenceRequest):
        self.row = row
        self.generation = generation
        self.request = request
        self.in_flight = True
        self.forward_returned = False


class StaticScheduler:
    """Request scheduler for separate prefills and up to two-row decode.

    Simplified scheduling logic:
    - Prefill is one request; two ready leases may decode together
    - No cache block management needed
    - Simple waiting queue for incoming requests
    - Prefix cache reuse via chained block hashing (block size = _BLOCK_SIZE)
    """

    def __init__(
        self,
        max_cache_len: int = 4096,
        enable_prefix_caching: bool = True,
        cache_capacity: int = 1,
        on_aborted_request=None,
    ):
        if cache_capacity not in (1, 2):
            raise ValueError("static cache capacity must be 1 or 2")
        self.waiting_queue = janus.Queue()
        self._queued_request_ids: set[int] = set()
        self._leases_by_row: dict[int, _RowLease] = {}
        self._dequeued_request: InferenceRequest | None = None
        self._next_generation = 0
        self._next_row = 0
        self._poisoned_rows: set[int] = set()
        self.cache_capacity = cache_capacity
        self._on_aborted_request = on_aborted_request
        self.max_cache_len = max_cache_len
        self.enable_prefix_caching = enable_prefix_caching
        self._cached_block_hashes_by_row: dict[int, List[BlockHash]] = {
            row: [] for row in range(cache_capacity)
        }

    def _owned_lease(self, request: InferenceRequest) -> Optional[_RowLease]:
        return next(
            (lease for lease in self._leases_by_row.values()
             if self._owns_row(lease, request)),
            None,
        )

    def _release_row(self, request: InferenceRequest) -> None:
        lease = self._owned_lease(request)
        if lease is None or lease.in_flight:
            raise RuntimeError("static cache row cannot be released before its forward completes")
        request._static_cache_lease_generation = None
        del self._leases_by_row[lease.row]

    @staticmethod
    def _owns_row(lease: _RowLease, request: InferenceRequest) -> bool:
        return (lease.request is request
                and getattr(request, "_static_cache_lease_generation", None)
                == lease.generation)

    def add_request(self, request: InferenceRequest):
        if request is not None:
            if (id(request) in self._queued_request_ids
                    or any(lease.request is request
                           for lease in self._leases_by_row.values())):
                raise RuntimeError("static request object is already queued or owns a cache row")
            if (request.status != RequestStatus.WAITING
                    or request.is_finished() or request.is_aborted()):
                raise ValueError("static scheduler requires a new, non-aborted waiting request")
            # TODO: Remove the multimodal exclusion once media-aware prefix
            # hashing and model-side cache-boundary handling are supported.
            request.initialize_block_hashes(
                _BLOCK_SIZE,
                self.enable_prefix_caching and not request.has_multimodal_inputs,
            )
            self._queued_request_ids.add(id(request))
            try:
                self.waiting_queue.sync_q.put(request)
            except BaseException:
                self._queued_request_ids.remove(id(request))
                raise

    def schedule(self) -> Optional[StaticSchedulerOutput]:
        """Schedule one prefill or one/two ready decode requests."""
        while True:
            if self._poisoned_rows:
                raise RuntimeError("static cache row has an unresolved failed forward")
            # A call to schedule while any forward is unresolved means the
            # synchronous runner failed. Do not guess which row stayed valid.
            unresolved = [lease.row for lease in self._leases_by_row.values()
                          if lease.in_flight]
            if unresolved:
                self._poisoned_rows.update(unresolved)
                raise RuntimeError("static cache row has an unresolved forward")

            # Retire terminal/overlength requests before considering new work.
            for lease in list(self._leases_by_row.values()):
                req = lease.request
                # close() can set the abort bit before AsyncLLMEngine drains
                # its cancellation queue. The engine callback owns terminal
                # status/final output; never schedule this row again.
                if req.is_aborted() or req.is_finished():
                    if (req.is_aborted() and not req.is_finished()
                            and self._on_aborted_request is not None):
                        self._on_aborted_request(req)
                    self._release_row(req)
                    continue
                if req.get_total_length() > self.max_cache_len:
                    with req._terminal_lock:
                        if req.is_aborted() or req.is_finished():
                            if (req.is_aborted() and not req.is_finished()
                                    and self._on_aborted_request is not None):
                                self._on_aborted_request(req)
                            self._release_row(req)
                            continue
                        logger.warning(
                            f"Request {req.request_id} exceeds max_cache_len={self.max_cache_len}, "
                            "completing request."
                        )
                        req.mark_failed(FinishReason.LENGTH)
                        self._release_row(req)
                        output = TokenOutput(
                            request_id=req.request_id, token_id=-1,
                            token_text="", finished=True,
                            finish_reason=req.finish_reason,
                            generated_text=req.generated_text,
                        )
                        try:
                            req.output_queue.sync_q.put(output)
                        except Exception as e:
                            logger.warning(
                                f"Failed to put completion token for {req.request_id}: {e}. "
                                "Likely due to client disconnecting or request cancelation."
                            )
            free_rows = [row for row in range(self.cache_capacity)
                         if row not in self._leases_by_row]
            # A steady stream of short prefills must not starve an existing
            # decoder. Give the row at the round-robin cursor its turn first.
            if free_rows and self._next_row not in self._leases_by_row:
                try:
                    req = self.waiting_queue.sync_q.get_nowait()
                    self._dequeued_request = req
                except queue.Empty:
                    req = None
                if req is not None:
                    self._queued_request_ids.remove(id(req))
                    if req.is_aborted() or req.is_finished():
                        if (req.is_aborted() and not req.is_finished()
                                and self._on_aborted_request is not None):
                            self._on_aborted_request(req)
                        self._dequeued_request = None
                        continue
                    prompt_len = req.get_prompt_length()
                    if prompt_len > self.max_cache_len:
                        with req._terminal_lock:
                            if req.is_aborted() or req.is_finished():
                                if (req.is_aborted() and not req.is_finished()
                                        and self._on_aborted_request is not None):
                                    self._on_aborted_request(req)
                                self._dequeued_request = None
                                continue
                            logger.error(
                                f"Request {req.request_id} prompt length {prompt_len} "
                                f"exceeds max_cache_len={self.max_cache_len}. Request rejected."
                            )
                            req.mark_failed(FinishReason.LENGTH)
                            output = TokenOutput(
                                request_id=req.request_id, token_id=-1,
                                token_text="", finished=True,
                                finish_reason=req.finish_reason,
                                generated_text=req.generated_text,
                            )
                            try:
                                req.output_queue.sync_q.put(output)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to put completion token for {req.request_id}: {e}. "
                                    "Likely due to client disconnecting or request cancelation."
                                )
                        self._dequeued_request = None
                        continue

                    def match_count(row: int) -> int:
                        if not self.enable_prefix_caching:
                            return 0
                        cached = self._cached_block_hashes_by_row[row]
                        count = 0
                        for wanted, existing in zip(req.block_hashes, cached):
                            if wanted != existing:
                                break
                            count += 1
                        return count

                    row = min(free_rows, key=lambda r: (-match_count(r), r))
                    matched = match_count(row)
                    cached = self._cached_block_hashes_by_row[row]
                    del cached[matched:]
                    prefix_hit_len = min(matched * _BLOCK_SIZE,
                                         max(prompt_len - 1, 0))
                    logger.info(
                        "Prefill cache match: %s/%s blocks (%s tokens reused) in row %s",
                        matched, len(req.block_hashes), prefix_hit_len, row,
                    )
                    req.status = RequestStatus.RUNNING
                    self._next_generation += 1
                    req._static_cache_lease_generation = self._next_generation
                    self._leases_by_row[row] = _RowLease(row, self._next_generation, req)
                    self._dequeued_request = None
                    self._next_row = (row + 1) % self.cache_capacity
                    return StaticSchedulerOutput(
                        scheduled_requests=[req], cache_row_length=self.max_cache_len,
                        is_prefill=True,
                        prefix_hit_len=prefix_hit_len, physical_row=row,
                        cache_capacity=self.cache_capacity,
                    )

            if not self._leases_by_row:
                return None
            rows = sorted(self._leases_by_row)
            first_row = next((r for r in rows if r >= self._next_row), rows[0])
            decode_rows = [first_row]
            if self.cache_capacity == 2 and len(rows) == 2:
                decode_rows.append(rows[0] if rows[1] == first_row else rows[1])
            requests = [self._leases_by_row[row].request for row in decode_rows]
            for row in decode_rows:
                lease = self._leases_by_row[row]
                lease.in_flight = True
                lease.forward_returned = False
            self._next_row = (first_row + 1) % self.cache_capacity
            return StaticSchedulerOutput(
                scheduled_requests=requests,
                cache_row_length=self.max_cache_len, is_prefill=False,
                physical_row=first_row, physical_rows=decode_rows,
                cache_capacity=self.cache_capacity,
            )

    def accept_model_output(self, requests, generations, physical_rows) -> None:
        """Confirm all returned row leases before publishing any one of them."""
        if (physical_rows is None or len(requests) != len(generations)
                or len(requests) != len(physical_rows)
                or len({id(req) for req in requests}) != len(requests)
                or len(set(physical_rows)) != len(physical_rows)):
            raise RuntimeError("static model output has conflicting row bindings")
        leases = []
        for req, generation, row in zip(requests, generations, physical_rows):
            lease = self._leases_by_row.get(row)
            if (lease is None or not self._owns_row(lease, req)
                    or lease.generation != generation or not lease.in_flight
                    or lease.forward_returned or row in self._poisoned_rows):
                raise RuntimeError("static model output does not own its active row lease")
            leases.append(lease)
        for lease in leases:
            lease.forward_returned = True

    def commit_computed_tokens(
        self, request: InferenceRequest, num_computed_tokens: int
    ) -> None:
        """Publish KV only after the synchronous model forward has returned."""
        lease = self._owned_lease(request)
        if lease is None or not self._owns_row(lease, request) or not lease.in_flight:
            raise RuntimeError("static cache prefix commit requires the active request lease")
        if num_computed_tokens < 0:
            raise ValueError("computed token count must be nonnegative")
        if not lease.forward_returned:
            raise RuntimeError("static prefix commit requires a qualified returned batch")
        cached_block_hashes = self._cached_block_hashes_by_row[lease.row]
        if not self.enable_prefix_caching:
            cached_block_hashes.clear()
            return
        num_computed_blocks = min(
            num_computed_tokens // _BLOCK_SIZE,
            len(request.block_hashes),
        )
        if len(cached_block_hashes) > num_computed_blocks:
            del cached_block_hashes[num_computed_blocks:]
        start_block = len(cached_block_hashes)
        if start_block == num_computed_blocks:
            return

        cached_block_hashes.extend(
            request.block_hashes[start_block:num_computed_blocks]
        )

    def update_from_output(self, model_output):
        """Static cache has no scheduler-side connector state to update."""
        return None

    def complete_requests(self, requests: List[InferenceRequest]):
        """Handle completed requests."""
        if len({id(req) for req in requests}) != len(requests):
            raise RuntimeError("static cache completion repeated a request")
        leases = []
        for req in requests:
            lease = self._owned_lease(req)
            if lease is None:
                raise RuntimeError("static cache completion requires the active request lease")
            if not lease.in_flight or not lease.forward_returned:
                raise RuntimeError("static cache completion requires a returned forward")
            leases.append((req, lease))
        for req, lease in leases:
            lease.in_flight = False
            if req.is_finished():
                self._release_row(req)
                logger.debug(f"Completed request {req.request_id}")

    def notify_failed_requests(self, finalize_request) -> None:
        """Notify after async fail-stop without releasing any cache row."""
        for lease in list(self._leases_by_row.values()):
            finalize_request(lease.request)
        if self._dequeued_request is not None:
            finalize_request(self._dequeued_request)
        while True:
            try:
                req = self.waiting_queue.sync_q.get_nowait()
            except queue.Empty:
                break
            finalize_request(req)

    def shutdown_drain(self, finalize_request) -> None:
        """Retire accepted requests after the current forward has returned.

        AsyncLLMEngine calls this only on its existing step thread. It does
        not submit another forward or change the normal row-lease protocol.
        """
        seen = set()
        while True:
            try:
                req = self.waiting_queue.sync_q.get_nowait()
                self._dequeued_request = req
            except queue.Empty:
                break
            self._queued_request_ids.remove(id(req))
            if id(req) in seen:
                raise RuntimeError("static shutdown found a duplicate request")
            seen.add(id(req))
            finalize_request(req)
            self._dequeued_request = None

        leases = list(self._leases_by_row.values())
        for req in [lease.request for lease in leases]:
            if id(req) in seen:
                raise RuntimeError("static shutdown found a duplicate request")
            seen.add(id(req))
            finalize_request(req)

        if self._poisoned_rows or any(lease.in_flight for lease in leases):
            raise RuntimeError(
                "static shutdown has an unresolved forward; native cache retained"
            )
        for lease in leases:
            self._release_row(lease.request)
        if self._leases_by_row or self._queued_request_ids:
            raise RuntimeError("static shutdown left a cache row owned")

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        cached_by_row = {row: len(blocks)
                         for row, blocks in self._cached_block_hashes_by_row.items()}
        running_by_row = {row: lease.request.request_id
                          for row, lease in self._leases_by_row.items()}
        return {
            "max_cache_len": self.max_cache_len,
            "cache_capacity": self.cache_capacity,
            "cached_blocks": sum(cached_by_row.values()),
            "cached_tokens": sum(cached_by_row.values()) * _BLOCK_SIZE,
            "cached_blocks_by_row": cached_by_row,
            "running_request": next(iter(running_by_row.values()), None),
            "running_requests_by_row": running_by_row,
            "poisoned_rows": sorted(self._poisoned_rows),
            "waiting_queue_size": self.waiting_queue.sync_q.qsize(),
        }
