"""
Request and Output - Data structures for inference requests and outputs.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import janus

from infinilm.llm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


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
    ):
        self.arrival_time: float = arrival_time or time.time()
        self.finished_time: Optional[float] = None

        # Request metadata
        self.request_id: str = request_id
        self.prompt: Optional[str] = prompt
        self.prompt_token_ids: List[int] = (
            prompt_token_ids if prompt_token_ids is not None else []
        )
        self.prompt_length: int = len(self.prompt_token_ids)
        self.processed_inputs: Optional[dict] = processed_inputs
        self.mm_token_index_mappings: Optional[List[dict]] = mm_token_index_mappings
        self.priority: int = 0

        # Sampling & stopping criteria
        self.sampling_params: SamplingParams = sampling_params or SamplingParams()
        self.eos_token_ids: List[int] = (
            eos_token_ids if eos_token_ids is not None else []
        )

        # Generation state
        self.generated_token_ids: List[int] = []
        self.generated_text: str = ""  # generated_text == tokenizer.decode(generated_token_ids[:_token_decode_offset])
        self.status: RequestStatus = RequestStatus.WAITING
        self.finish_reason: Optional[FinishReason] = None

        # KV cache management
        self.block_table: List[int] = []
        self.slot_mapping: List[int] = []
        self.num_local_cached_tokens: int = (
            0  # Number of locally cached (prefix-hit) tokens
        )
        self.num_computed_tokens: int = 0  # Total tokens computed (local + remote)
        self.num_blocks: int = 0

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

    def get_prompt_length(self) -> int:
        return self.prompt_length

    def get_input_tokens(self) -> List[int]:
        return self.prompt_token_ids

    def get_num_generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    def get_total_length(self) -> int:
        return self.prompt_length + len(self.generated_token_ids)

    def get_all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.generated_token_ids

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
            prompt_token_ids=self.prompt_token_ids,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=self.generated_text,
                    token_ids=self.generated_token_ids.copy(),
                    finish_reason=self.finish_reason,
                )
            ],
            finished=self.is_finished(),
            finish_reason=self.finish_reason,
        )
