from enum import Enum
import time
import janus
from typing import List, Optional


class RequestStatus(Enum):
    Waiting = "waiting"
    Running = "running"
    Finished = "finished"
    Canceled = "canceled"
    Failed = "failed"
    Timeout = "timeout"


class SamplingParams:
    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k


class InferenceRequest:
    """Inference request object for managing generation state and resources."""

    def __init__(
        self,
        request_id: str,
        request_data: dict,
        request,
        input_tokens: Optional[List[int]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 0.8,
        eos_token_id: Optional[List[int]] = None,
        end_strings: Optional[List[str]] = None,
    ):
        # Request metadata
        self.request_id: str = request_id
        self.request_data: dict = request_data
        self.request = request
        self.messages: List[dict] = request_data.get("messages", [])
        self.model: Optional[str] = request_data.get("model", None)
        self.stream: bool = request_data.get("stream", False)
        self.created_time: float = time.time()
        self.finished_time: Optional[float] = None

        # Input data
        self.input_tokens: List[int] = input_tokens or []
        self.prompt_length: int = len(self.input_tokens)

        # Generation configuration
        self.max_new_tokens: Optional[int] = max_new_tokens
        self.eos_token_id: Optional[List[int]] = eos_token_id
        self.end_strings: List[str] = end_strings or []
        self.sampling_params: SamplingParams = SamplingParams(
            temperature=temperature, top_p=top_p, top_k=top_k
        )

        # Generation state
        self.generated_token_ids: List[int] = []
        self.generated_text: str = ""
        self.is_prefill: bool = True
        self.status: RequestStatus = RequestStatus.Waiting
        self.finish_reason: Optional[str] = None
        self.priority: int = 0

        # KV cache management (Paged Attention)
        self.cache_id = None
        self.block_table: List[int] = []
        self.slot_mapping: List[int] = []
        self.num_cached_tokens: int = 0
        self.num_blocks: int = 0

        # Output management
        self.output_queue: janus.Queue = janus.Queue()

    def get_prompt_length(self) -> int:
        return self.prompt_length

    def get_input_tokens(self) -> List[int]:
        return self.input_tokens

    def get_num_generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    def get_total_length(self) -> int:
        return self.prompt_length + len(self.generated_token_ids)

    def get_all_token_ids(self) -> List[int]:
        return self.input_tokens + self.generated_token_ids

    def get_num_blocks_required(self, block_size: int) -> int:
        total_tokens = self.get_total_length()
        return (total_tokens + block_size - 1) // block_size

    async def close(self):
        await self.output_queue.async_q.join()
        self.output_queue.close()
        await self.output_queue.wait_closed()


class RequestOutput:
    """Output object for inference request results."""

    def __init__(
        self,
        request_id: str,
        token_id: int,
        token_text: str,
        status: RequestStatus,
        finish_reason: Optional[str] = None,
        generated_text: str = "",
    ):
        self.request_id = request_id
        self.token_id = token_id
        self.token_text = token_text
        self.status = status
        self.finish_reason = finish_reason
        self.generated_text = generated_text
