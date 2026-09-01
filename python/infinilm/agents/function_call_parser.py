"""
Parser for function/tool calls in model outputs.
Supports GLM-4 and Llama-3.1/3.2 tool call formats.
"""

import inspect
import logging
from typing import Dict, List, Optional, Tuple, Type

from infinilm.agents.base_detector import BaseFormatDetector
from infinilm.agents.detectors import (
    Glm4Chat0414Detector,
    Glm4MoeDetector,
    Llama32Detector,
    Qwen3XmlDetector,
)
from infinilm.agents.protocol import Function, Tool
from infinilm.agents.types import StreamingParseResult, ToolCallItem

logger = logging.getLogger(__name__)


def _normalize_tools(tools: Optional[list]) -> List[Tool]:
    """Normalize raw dict/list tools into Tool model instances."""
    if not tools:
        return []
    normalized: List[Tool] = []
    for t in tools:
        if isinstance(t, Tool):
            normalized.append(t)
        elif isinstance(t, dict):
            function_data = t.get("function", {})
            function = Function(
                description=function_data.get("description"),
                name=function_data.get("name", ""),
                parameters=function_data.get("parameters"),
                strict=function_data.get("strict"),
            )
            normalized.append(Tool(type=t.get("type", "function"), function=function))
        else:
            raise ValueError(f"Invalid tool type: {type(t)}")
    return normalized


class FunctionCallParser:
    """
    Parser for function/tool calls in model outputs.
    Handles both streaming and non-streaming parsing using a detector.
    """

    ToolCallParserEnum: Dict[str, Type[BaseFormatDetector]] = {
        "glm": Glm4MoeDetector,
        "glm45": Glm4MoeDetector,
        "glm47": Glm4MoeDetector,
        # GLM-4-9B-Chat-0414 "metadata" format: function name on one line,
        # JSON arguments on the next line (no xml <tool_call> tags).
        "glm4": Glm4Chat0414Detector,
        "glm49b": Glm4Chat0414Detector,
        "glm4-9b-0414": Glm4Chat0414Detector,
        "glm-4-9b-0414": Glm4Chat0414Detector,
        "llama3": Llama32Detector,
        "llama32": Llama32Detector,
        "llama31": Llama32Detector,
        # Qwen3 xml format:  {"name": "...", "arguments": {...}}
        "qwen3": Qwen3XmlDetector,
        "qwen3-30b-a3b": Qwen3XmlDetector,
    }

    def __init__(
        self, tool_call_parser: str, tools: Optional[list] = None, tokenizer=None
    ):
        detector_class = self.ToolCallParserEnum.get(tool_call_parser)
        if detector_class:
            kwargs = {}
            if tokenizer is not None:
                sig = inspect.signature(detector_class)
                if "tokenizer" in sig.parameters:
                    kwargs["tokenizer"] = tokenizer
            detector = detector_class(**kwargs)
        else:
            raise ValueError(f"Unsupported tool_call_parser: {tool_call_parser}")

        self.detector = detector
        self.tools = _normalize_tools(tools)

    def _ensure_tools(self, tools: Optional[list] = None) -> List[Tool]:
        if tools is not None:
            return _normalize_tools(tools)
        return self.tools

    def has_tool_call(self, text: str) -> bool:
        if not self.tools:
            return False
        return self.detector.has_tool_call(text)

    def parse_non_stream(
        self, full_text: str, tools: Optional[list] = None
    ) -> Tuple[str, list[ToolCallItem]]:
        tools = self._ensure_tools(tools)
        if not tools:
            return full_text, []
        has_tool_call = self.detector.has_tool_call(full_text)
        parsed_result = self.detector.detect_and_parse(full_text, tools)
        tool_call_list = parsed_result.calls
        if tool_call_list or has_tool_call:
            return parsed_result.normal_text, tool_call_list
        else:
            return full_text, []

    def parse_streaming_increment(
        self, text: str, delta_text: str, tools: Optional[list] = None
    ) -> StreamingParseResult:
        """Streaming increment wrapper; delegates to detector.

        Args:
            text: Accumulated text so far (kept for API compatibility).
            delta_text: New text chunk from this streaming step.
            tools: Optional override list of tools.
        """
        tools = self._ensure_tools(tools)
        if not tools:
            return StreamingParseResult(normal_text=delta_text)
        sp_result = self.detector.parse_streaming_increment(delta_text, tools)
        return sp_result

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, list[ToolCallItem]]:
        if not self.tools:
            return chunk_text, []
        final_normal_text = ""
        final_calls = []

        sp_result = self.detector.parse_streaming_increment(chunk_text, self.tools)
        if sp_result.normal_text:
            final_normal_text = sp_result.normal_text
        if sp_result.calls:
            final_calls.extend(sp_result.calls)
            final_normal_text = sp_result.normal_text

        return final_normal_text, final_calls

    def parse_stream_end(
        self, tools: Optional[list] = None
    ) -> Tuple[str, list[ToolCallItem]]:
        """Flush any buffered state at the end of a stream.

        Args:
            tools: Optional override list of tools; falls back to the tools
                bound at construction time.
        """
        tools = self._ensure_tools(tools)
        if not tools:
            return "", []
        sp_result = self.detector.finish(tools)
        return sp_result.normal_text, sp_result.calls
