"""
Request and Output - Data structures for inference requests and outputs.
"""

import asyncio
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import janus

from infinilm.llm.prefix_cache import (
    EMPTY_BLOCK_HASH,
    BlockHash,
    hash_block_tokens,
)
from infinilm.llm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class _SequenceView(Sequence):
    """Live read-only view over an internally mutable list."""

    def __init__(self, values: list) -> None:
        self._values = values

    def __getitem__(self, index):
        return self._values[index]

    def __iter__(self) -> Iterator:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr(self._values)

    def __eq__(self, other: object) -> bool:
        return (
            list(self._values) == list(other) if isinstance(other, Sequence) else False
        )


class RequestStatus(Enum):
    """Status of an inference request."""

    # Pending
    WAITING = "waiting"
    WAITING_FOR_REMOTE_KVS = "waiting_for_remote_kvs"

    # Active
    RUNNING = "running"

    # Successful terminal
    FINISHED = "finished"

    # Abnormal terminal
    CANCELED = "canceled"
    TIMEOUT = "timeout"
    FAILED = "failed"


class FinishReason(Enum):
    """Reason for finishing generation."""

    # Normal completion
    EOS_TOKEN = "eos_token"
    STOP_STRING = "stop_string"
    STOP = "stop"

    # Controlled truncation
    LENGTH = "length"

    # Abnormal termination
    CANCELED = "canceled"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class RequestOutput:
    """Output from a single generation request.

    Attributes:
        request_id: Unique identifier for the request.
        prompt: Original prompt text.
        prompt_token_ids: Token IDs of the prompt.
        outputs: List of generated outputs (for beam search, multiple outputs possible).
        finished: Whether generation is complete.
        finish_reason: Reason for finishing.
    """

    request_id: str
    prompt: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None
    outputs: List["CompletionOutput"] = field(default_factory=list)
    finished: bool = False
    finish_reason: Optional[FinishReason] = None


@dataclass
class CompletionOutput:
    """Single completion output.

    Attributes:
        index: Index of this output (for beam search).
        text: Generated text.
        token_ids: Generated token IDs.
        finish_reason: Reason for finishing.
    """

    index: int = 0
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    finish_reason: Optional[FinishReason] = None


@dataclass
class TokenOutput:
    """Output for a single generated token.

    Attributes:
        request_id: Unique identifier for the request.
        token_id: Generated token ID.
        token_text: Decoded text of the token.
        finished: Whether generation is complete.
        finish_reason: Reason for finishing.
        generated_text: Full generated text so far.
    """

    request_id: str
    token_id: int
    token_text: str
    finished: bool = False
    finish_reason: Optional[FinishReason] = None
    generated_text: str = ""


class InferenceRequest:
    """Internal inference request object for managing generation state and resources."""

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        processed_inputs: Optional[dict] = None,
        mm_token_index_mappings: Optional[List[dict]] = None,
        sampling_params: Optional[SamplingParams] = None,
        eos_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        # For server use
        request_data: Optional[dict] = None,
        *,
        has_multimodal_inputs: bool = False,
    ):
        self.arrival_time: float = arrival_time or time.time()
        self.finished_time: Optional[float] = None

        # Request metadata
        self.request_id: str = request_id
        self.prompt: Optional[str] = prompt
        self._prompt_token_ids: tuple[int, ...] = (
            tuple(prompt_token_ids) if prompt_token_ids is not None else ()
        )
        self.prompt_length: int = len(self._prompt_token_ids)
        self.processed_inputs: Optional[dict] = processed_inputs
        self.mm_token_index_mappings: Optional[List[dict]] = mm_token_index_mappings
        self.has_multimodal_inputs: bool = has_multimodal_inputs or bool(
            mm_token_index_mappings
        )
        self.priority: int = 0

        # Sampling & stopping criteria
        self.sampling_params: SamplingParams = sampling_params or SamplingParams()
        self.eos_token_ids: List[int] = (
            eos_token_ids if eos_token_ids is not None else []
        )

        # Generation state
        self._generated_token_ids: List[int] = []
        self._generated_token_ids_view = _SequenceView(self._generated_token_ids)
        self.generated_text: str = ""  # generated_text == tokenizer.decode(generated_token_ids[:_token_decode_offset])
        self.status: RequestStatus = RequestStatus.WAITING
        self.finish_reason: Optional[FinishReason] = None

        # KV cache state
        self.block_table: List[int] = []  # Logical block to physical page ID
        self.slot_mapping: List[int] = []
        self.num_local_cached_tokens: int = (
            0  # Number of cached tokens visible to the current model step
        )
        self.num_computed_tokens: int = 0  # Total computed boundary, local + remote
        self.num_blocks: int = 0
        # Incremental hashes for complete logical token blocks.
        self._block_hashes: List[BlockHash] = []
        self._block_hashes_view = _SequenceView(self._block_hashes)
        self._hash_block_size: Optional[int] = None
        self._prefix_caching_enabled: bool = False
        self._hash_tail_token_ids: List[int] = []
        # Number of leading full blocks published to the prefix-cache index.
        self.num_cache_indexed_blocks: int = 0

        # Mamba cache management. None means no mamba cache row is currently owned.
        self.mamba_cache_index: Optional[int] = None

        # Qwen-style MRoPE decode offset. It is zero for pure text requests.
        self.mrope_position_delta: int = 0

        # PD disaggregation support
        self.kv_transfer_params: Optional[dict] = (
            None  # KV transfer parameters from the router
        )

        # For server use
        self.request_data: Optional[dict] = request_data

        # Async output & streaming
        self._output_queue: Optional[janus.Queue] = None
        self._aborted: bool = False
        self._text_output_offset: int = 0
        self._token_decode_offset: int = 0

    @property
    def output_queue(self) -> janus.Queue:
        """Lazy initialization of output queue."""
        if self._output_queue is None:
            self._output_queue = janus.Queue()
        return self._output_queue

    @property
    def prompt_token_ids(self) -> tuple[int, ...]:
        return self._prompt_token_ids

    @property
    def generated_token_ids(self) -> Sequence[int]:
        return self._generated_token_ids_view

    @property
    def block_hashes(self) -> Sequence[BlockHash]:
        return self._block_hashes_view

    def get_prompt_length(self) -> int:
        return self.prompt_length

    def get_input_tokens(self) -> Sequence[int]:
        return self._prompt_token_ids

    def get_num_generated_tokens(self) -> int:
        return len(self._generated_token_ids)

    def get_total_length(self) -> int:
        return self.prompt_length + len(self._generated_token_ids)

    def get_all_token_ids(self) -> List[int]:
        return list(self._prompt_token_ids) + self._generated_token_ids

    def get_token_slice(self, start: int, end: int) -> Sequence[int]:
        """Return the existing logical tokens in [start, end)."""
        total_length = self.get_total_length()
        if not 0 <= start <= end <= total_length:
            raise ValueError(
                f"invalid token range [{start}, {end}) for length {total_length}"
            )
        if end <= self.prompt_length:
            return self._prompt_token_ids[start:end]
        if start < self.prompt_length:
            raise RuntimeError("scheduled token range crosses the prompt boundary")
        offset = self.prompt_length
        return self._generated_token_ids[start - offset : end - offset]

    def initialize_block_hashes(
        self, block_size: int, enable_prefix_caching: bool
    ) -> None:
        """Initialize incremental block hashing for this request."""
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if self._hash_block_size is not None:
            if (
                self._hash_block_size != block_size
                or self._prefix_caching_enabled != enable_prefix_caching
            ):
                raise RuntimeError("request block hashing is already initialized")
            return

        self._hash_block_size = block_size
        self._prefix_caching_enabled = enable_prefix_caching
        if not enable_prefix_caching:
            return

        num_full_blocks = self.prompt_length // block_size
        parent_hash = EMPTY_BLOCK_HASH
        for block_idx in range(num_full_blocks):
            start = block_idx * block_size
            end = start + block_size
            parent_hash = hash_block_tokens(
                self.prompt_token_ids[start:end], parent_hash
            )
            self._block_hashes.append(parent_hash)

        tail_start = num_full_blocks * block_size
        self._hash_tail_token_ids = list(self._prompt_token_ids[tail_start:])

    def append_generated_token_id(self, token_id: int) -> None:
        """Append one output token and extend the hash chain at block boundaries."""
        self._generated_token_ids.append(token_id)
        if not self._prefix_caching_enabled:
            return

        self._hash_tail_token_ids.append(token_id)
        if len(self._hash_tail_token_ids) == self._hash_block_size:
            parent_hash = (
                self._block_hashes[-1] if self._block_hashes else EMPTY_BLOCK_HASH
            )
            self._block_hashes.append(
                hash_block_tokens(self._hash_tail_token_ids, parent_hash)
            )
            self._hash_tail_token_ids.clear()

    def get_num_blocks_required(self, block_size: int) -> int:
        total_tokens = self.get_total_length()
        return (total_tokens + block_size - 1) // block_size

    def get_max_tokens(self) -> Optional[int]:
        return self.sampling_params.max_tokens

    def get_mm_token_index_mappings(self) -> Optional[List[dict]]:
        return self.mm_token_index_mappings

    def is_finished(self) -> bool:
        return self.status in [
            RequestStatus.FINISHED,
            RequestStatus.CANCELED,
            RequestStatus.FAILED,
            RequestStatus.TIMEOUT,
        ]

    def abort(self):
        """Signal that the request has been aborted and should stop generation."""
        self._aborted = True

    def is_aborted(self) -> bool:
        """Check if the request has been aborted."""
        return self._aborted

    def mark_finished(self, reason: FinishReason):
        """Mark the request as finished with the given reason."""
        self.status = RequestStatus.FINISHED
        self.finish_reason = reason
        self.finished_time = time.time()

    def mark_failed(self, reason: FinishReason = FinishReason.ERROR):
        """Mark the request as failed."""
        self.abort()
        self.status = RequestStatus.FAILED
        self.finish_reason = reason
        self.finished_time = time.time()

    def mark_canceled(self):
        """Mark the request as canceled."""
        self.abort()
        self.status = RequestStatus.CANCELED
        self.finish_reason = FinishReason.CANCELED
        self.finished_time = time.time()

    def mark_timeout(self):
        """Mark the request as timed out."""
        self.abort()
        self.status = RequestStatus.TIMEOUT
        self.finish_reason = FinishReason.TIMEOUT
        self.finished_time = time.time()

    async def close(self):
        """Close the output queue and clean up resources."""
        if self._output_queue is not None:
            self.abort()
            try:
                while not self._output_queue.async_q.empty():
                    try:
                        self._output_queue.async_q.get_nowait()
                        self._output_queue.async_q.task_done()
                    except asyncio.QueueEmpty:
                        break
            except Exception as e:
                logger.error(
                    f"Error while clearing output queue for request {self.request_id}: {e}"
                )
                pass

            self._output_queue.close()
            try:
                await asyncio.wait_for(self._output_queue.wait_closed(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("wait_closed timeout, force close")

    def to_request_output(self) -> RequestOutput:
        """Convert to RequestOutput for external use."""
        return RequestOutput(
            request_id=self.request_id,
            prompt=self.prompt,
            prompt_token_ids=list(self._prompt_token_ids),
            outputs=[
                CompletionOutput(
                    index=0,
                    text=self.generated_text,
                    token_ids=list(self._generated_token_ids),
                    finish_reason=self.finish_reason,
                )
            ],
            finished=self.is_finished(),
            finish_reason=self.finish_reason,
        )
