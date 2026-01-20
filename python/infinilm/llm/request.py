"""
Request and Output - Data structures for inference requests and outputs.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Any
import time
import janus

from infinilm.llm.sampling_params import SamplingParams


class RequestStatus(Enum):
    """Status of an inference request."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELED = "canceled"
    FAILED = "failed"
    TIMEOUT = "timeout"


class FinishReason(Enum):
    """Reason for finishing generation."""

    STOP = "stop"
    LENGTH = "length"
    EOS_TOKEN = "eos_token"
    STOP_STRING = "stop_string"
    TIMEOUT = "timeout"
    CANCELED = "canceled"
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
        sampling_params: Optional[SamplingParams] = None,
        eos_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        # For server use
        request_data: Optional[dict] = None,
        http_request: Optional[Any] = None,
    ):
        # Request metadata
        self.request_id: str = request_id
        self.prompt: Optional[str] = prompt
        self.prompt_token_ids: List[int] = prompt_token_ids or []
        self.prompt_length: int = len(self.prompt_token_ids)
        self.arrival_time: float = arrival_time or time.time()
        self.finished_time: Optional[float] = None

        # Sampling parameters
        self.sampling_params: SamplingParams = sampling_params or SamplingParams()

        # EOS token IDs (from model config)
        self.eos_token_ids: List[int] = eos_token_ids or []

        # Generation state
        self.generated_token_ids: List[int] = []
        self.generated_text: str = ""
        self.is_prefill: bool = True
        self.status: RequestStatus = RequestStatus.WAITING
        self.finish_reason: Optional[FinishReason] = None
        self.priority: int = 0

        # KV cache management
        self.cache_id: Optional[int] = None
        self.block_table: List[int] = []
        self.slot_mapping: List[int] = []
        self.num_cached_tokens: int = 0
        self.num_blocks: int = 0

        # For server use
        self.request_data: Optional[dict] = request_data
        self.http_request: Optional[Any] = http_request

        # Output management (for async streaming)
        self._output_queue: Optional[janus.Queue] = None

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

    def is_finished(self) -> bool:
        return self.status in [
            RequestStatus.FINISHED,
            RequestStatus.CANCELED,
            RequestStatus.FAILED,
            RequestStatus.TIMEOUT,
        ]

    def mark_finished(self, reason: FinishReason):
        """Mark the request as finished with the given reason."""
        self.status = RequestStatus.FINISHED
        self.finish_reason = reason
        self.finished_time = time.time()

    def mark_failed(self, reason: FinishReason = FinishReason.ERROR):
        """Mark the request as failed."""
        self.status = RequestStatus.FAILED
        self.finish_reason = reason
        self.finished_time = time.time()

    def mark_canceled(self):
        """Mark the request as canceled."""
        self.status = RequestStatus.CANCELED
        self.finish_reason = FinishReason.CANCELED
        self.finished_time = time.time()

    def mark_timeout(self):
        """Mark the request as timed out."""
        self.status = RequestStatus.TIMEOUT
        self.finish_reason = FinishReason.TIMEOUT
        self.finished_time = time.time()

    async def close(self):
        """Close the output queue and clean up resources."""
        if self._output_queue is not None:
            await self._output_queue.async_q.join()
            self._output_queue.close()
            await self._output_queue.wait_closed()

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
