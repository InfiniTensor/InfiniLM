"""
Sampling Parameters - Configuration for text generation sampling.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SamplingParams:
    """Sampling parameters for text generation."""

    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 1
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None

    def __post_init__(self):
        if self.stop is None:
            self.stop = []
        if self.stop_token_ids is None:
            self.stop_token_ids = []

    def clone(self) -> "SamplingParams":
        """Create a copy of this SamplingParams instance."""
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            stop=self.stop.copy() if self.stop else None,
            stop_token_ids=self.stop_token_ids.copy() if self.stop_token_ids else None,
        )
