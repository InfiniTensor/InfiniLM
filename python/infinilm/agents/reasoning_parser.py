"""
Parser for reasoning/thinking content extraction.
"""

from typing import List, Optional, Tuple


class ReasoningStreamingParseResult:
    """Result of reasoning content parsing."""

    def __init__(
        self, reasoning_content: str = "", normal_text: str = "", complete: bool = False
    ):
        self.reasoning_content = reasoning_content
        self.normal_text = normal_text
        self.complete = complete

    def __repr__(self):
        return (
            f"ReasoningStreamingParseResult("
            f"reasoning_content={self.reasoning_content!r}, "
            f"normal_text={self.normal_text!r}, "
            f"complete={self.complete})"
        )


class BaseReasoningFormatDetector:
    """Abstract base for reasoning format detectors."""

    def __init__(
        self,
        start_token: str,
        end_token: str,
        stream_start_prefill: bool = False,
        include_start_token: bool = False,
        include_end_token: bool = False,
        starts_with_start_token: bool = True,
        is_full_suffix: bool = True,
    ):
        self.start_token = start_token
        self.end_token = end_token
        self.stream_start_prefill = stream_start_prefill
        self.include_start_token = include_start_token
        self.include_end_token = include_end_token
        self.starts_with_start_token = starts_with_start_token
        self.is_full_suffix = is_full_suffix
        self._buffer = ""
        self.found_reasoning_end = False
        self.reasoning_started = stream_start_prefill

    def clear(self):
        self._buffer = ""
        self.found_reasoning_end = False
        self.reasoning_started = self.stream_start_prefill

    def detect_and_parse(self, text: str) -> Tuple[ReasoningStreamingParseResult, ...]:
        return self._detect_and_parse_impl(text)

    def parse_streaming_increment(
        self, text: str, delta_text: str
    ) -> ReasoningStreamingParseResult:
        return self._parse_streaming_increment_impl(text, delta_text)

    def _detect_and_parse_impl(
        self, text: str
    ) -> Tuple[ReasoningStreamingParseResult, ...]:
        if self.start_token not in text:
            return (
                ReasoningStreamingParseResult(
                    reasoning_content="", normal_text=text, complete=True
                ),
            )
        start_idx = text.index(self.start_token)
        after_start = text[start_idx + len(self.start_token) :]
        if self.end_token not in after_start:
            reasoning = after_start
            normal = ""
            return (
                ReasoningStreamingParseResult(
                    reasoning_content=reasoning, normal_text=normal, complete=False
                ),
            )
        end_idx = after_start.index(self.end_token)
        reasoning = after_start[:end_idx]
        normal = after_start[end_idx + len(self.end_token) :]
        return (
            ReasoningStreamingParseResult(
                reasoning_content=reasoning, normal_text="", complete=True
            ),
            ReasoningStreamingParseResult(
                reasoning_content="", normal_text=normal, complete=True
            ),
        )

    def _parse_streaming_increment_impl(
        self, text: str, delta_text: str
    ) -> ReasoningStreamingParseResult:
        self._buffer += delta_text
        if not self.reasoning_started:
            if self.start_token in self._buffer:
                self.reasoning_started = True
                idx = self._buffer.index(self.start_token)
                normal_before = self._buffer[:idx]
                self._buffer = self._buffer[idx + len(self.start_token) :]
                out = self._buffer
                self._buffer = ""
                return ReasoningStreamingParseResult(
                    reasoning_content=out, normal_text=normal_before, complete=False
                )
            if len(self._buffer) >= len(self.start_token):
                for i in range(1, len(self.start_token)):
                    if self._buffer.endswith(self.start_token[:i]):
                        out = self._buffer[:-i]
                        self._buffer = self._buffer[-i:]
                        if out:
                            return ReasoningStreamingParseResult(
                                reasoning_content="", normal_text=out, complete=True
                            )
                        return ReasoningStreamingParseResult()
                out = self._buffer
                self._buffer = ""
                return ReasoningStreamingParseResult(
                    reasoning_content="", normal_text=out, complete=True
                )
            return ReasoningStreamingParseResult()
        if not self.found_reasoning_end:
            if self.end_token in self._buffer:
                self.found_reasoning_end = True
                idx = self._buffer.index(self.end_token)
                reasoning = self._buffer[:idx]
                normal = self._buffer[idx + len(self.end_token) :]
                self._buffer = ""
                return ReasoningStreamingParseResult(
                    reasoning_content=reasoning, normal_text=normal, complete=True
                )
            if len(self._buffer) > len(self.end_token):
                reasoning = self._buffer[: -len(self.end_token)]
                self._buffer = self._buffer[-len(self.end_token) :]
                return ReasoningStreamingParseResult(
                    reasoning_content=reasoning, normal_text="", complete=False
                )
            return ReasoningStreamingParseResult()
        normal = self._buffer
        self._buffer = ""
        return ReasoningStreamingParseResult(
            reasoning_content="", normal_text=normal, complete=True
        )


class Glm45Detector(BaseReasoningFormatDetector):
    """Detector for GLM-4.5 thinking tags (PUA unicode start/end tokens)."""

    def __init__(self, **kwargs):
        super().__init__(
            start_token="think",
            end_token="/think",
            stream_start_prefill=True,
            include_start_token=False,
            include_end_token=False,
            **kwargs,
        )


class ThinkTagDetector(BaseReasoningFormatDetector):
    """Generic detector for standard 'thinking' / 'response' reasoning tags."""

    def __init__(self, **kwargs):
        super().__init__(
            start_token="<thinking>",
            end_token="</thinking>",
            stream_start_prefill=False,
            include_start_token=False,
            include_end_token=False,
            **kwargs,
        )


class DeepSeekR1Detector(BaseReasoningFormatDetector):
    """Detector for DeepSeek-R1 / QwQ style <think>...</think> reasoning tags."""

    def __init__(self, **kwargs):
        super().__init__(
            start_token="<think>",
            end_token="</think>",
            stream_start_prefill=False,
            include_start_token=False,
            include_end_token=False,
            **kwargs,
        )


class ReasoningParser:
    """Parses reasoning/thinking content from LLM outputs."""

    def __init__(self, reasoning_parser_name: Optional[str] = None):
        self.reasoning_parser_name = reasoning_parser_name
        self.format_detectors: List[BaseReasoningFormatDetector] = []
        if reasoning_parser_name:
            name = reasoning_parser_name.lower()
            if name in {
                "glm4",
                "glm45",
                "glm-4",
                "glm-4.5",
                "glm-4.5-air",
                "glm-4.5-flash",
            }:
                self.format_detectors = [Glm45Detector()]
            elif name in {"think", "thinking"}:
                self.format_detectors = [ThinkTagDetector()]
            elif name in {
                "deepseek",
                "deepseek-r1",
                "deepseek_r1",
                "qwq",
                "qwq-32b",
                "qwen3",
                "qwen3-thinking",
            }:
                # DeepSeek-R1, QwQ and Qwen3 thinking models use the short <think>...</think> tags,
                # not <thinking>...</thinking>.
                self.format_detectors = [DeepSeekR1Detector()]
            else:
                self.format_detectors = []
        else:
            self.format_detectors = []

    def clear(self):
        for detector in self.format_detectors:
            detector.clear()

    def extract_reasoning_content_streaming(
        self, previous_text: str, current_text: str, delta_text: str
    ) -> ReasoningStreamingParseResult:
        if not self.format_detectors:
            return ReasoningStreamingParseResult(
                reasoning_content="", normal_text=delta_text, complete=True
            )
        return self.format_detectors[0].parse_streaming_increment(
            current_text, delta_text
        )

    def extract_reasoning_content(
        self, text: str
    ) -> Tuple[ReasoningStreamingParseResult, ...]:
        if not self.format_detectors:
            return (
                ReasoningStreamingParseResult(
                    reasoning_content="", normal_text=text, complete=True
                ),
            )
        return self.format_detectors[0].detect_and_parse(text)
