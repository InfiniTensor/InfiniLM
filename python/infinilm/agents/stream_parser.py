"""
Per-request parsing of generated text into agent-style output.

The model forward pass produces plain text tokens. These helpers convert that
text stream into the chat-completion protocol fields (``reasoning_content``,
``content`` and OpenAI-format ``tool_calls``), so all parser state and
protocol formatting stays inside ``infinilm.agents`` instead of being spread
across the HTTP layer.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from infinilm.agents.function_call_parser import FunctionCallParser
from infinilm.agents.protocol import chunk_json
from infinilm.agents.reasoning_parser import ReasoningParser
from infinilm.agents.types import ToolCallItem


@dataclass
class AgentDelta:
    """One increment of parsed model output, ready for the API protocol."""

    reasoning_content: str = ""
    content: str = ""
    tool_calls: List[Dict] = field(default_factory=list)


def format_streaming_tool_calls(calls: List[ToolCallItem]) -> List[Dict]:
    """Format parsed calls as OpenAI streaming ``delta.tool_calls`` items."""
    return [
        {
            "index": call.tool_index,
            "id": f"call_{call.tool_index}",
            "type": "function",
            "function": {
                "name": call.name or "",
                "arguments": call.parameters,
            },
        }
        for call in calls
    ]


def format_tool_calls(calls: List[ToolCallItem]) -> List[Dict]:
    """Format parsed calls as OpenAI ``message.tool_calls`` items."""
    return [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {
                "name": call.name or "",
                "arguments": call.parameters,
            },
        }
        for index, call in enumerate(calls)
    ]


class AgentStreamParser:
    """Stateful per-request parser for a streamed generation.

    Splits each generated token increment into reasoning content, visible
    content and tool-call deltas. Create one instance per request; instances
    must not be shared across concurrent streams.
    """

    def __init__(
        self,
        tool_call_parser: Optional[str] = None,
        reasoning_parser: Optional[str] = None,
        tools: Optional[list] = None,
    ):
        self._reasoning_parser = (
            ReasoningParser(reasoning_parser_name=reasoning_parser)
            if reasoning_parser
            else None
        )
        self._tool_call_parser = (
            FunctionCallParser(tool_call_parser=tool_call_parser, tools=tools or [])
            if tool_call_parser
            else None
        )
        self._text = ""
        self.has_tool_calls = False

    def process_delta(self, token_text: str) -> AgentDelta:
        """Parse one generated token increment."""
        self._text += token_text
        delta_reasoning = ""
        delta_normal = token_text

        if self._reasoning_parser:
            previous_text = self._text[: -len(token_text)] if token_text else self._text
            result = self._reasoning_parser.extract_reasoning_content_streaming(
                previous_text=previous_text,
                current_text=self._text,
                delta_text=token_text,
            )
            delta_reasoning = result.reasoning_content or ""
            delta_normal = result.normal_text or ""

        delta_content = delta_normal
        tool_calls: List[Dict] = []
        if self._tool_call_parser and delta_normal:
            result = self._tool_call_parser.parse_streaming_increment(
                self._text, delta_normal
            )
            delta_content = result.normal_text
            if result.calls:
                tool_calls = format_streaming_tool_calls(result.calls)
                self.has_tool_calls = True

        return AgentDelta(
            reasoning_content=delta_reasoning,
            content=delta_content,
            tool_calls=tool_calls,
        )

    def flush(self) -> AgentDelta:
        """Flush buffered state at the end of the stream."""
        if not self._tool_call_parser:
            return AgentDelta()
        normal_text, calls = self._tool_call_parser.parse_stream_end()
        if calls:
            self.has_tool_calls = True
        return AgentDelta(
            content=normal_text,
            tool_calls=format_streaming_tool_calls(calls),
        )

    def delta_events(self, token_text: str, request_id: str, model: str) -> List[str]:
        """Parse one token increment and render it as OpenAI SSE lines.

        Returns an empty list when the increment carries nothing to emit.
        """
        return delta_to_sse_chunks(request_id, model, self.process_delta(token_text))

    def flush_events(self, request_id: str, model: str) -> List[str]:
        """Flush buffered state and render it as OpenAI SSE lines."""
        return delta_to_sse_chunks(request_id, model, self.flush())


def delta_to_sse_chunks(request_id: str, model: str, delta: AgentDelta) -> List[str]:
    """Render an ``AgentDelta`` as OpenAI streaming SSE ``data:`` lines."""
    if not (delta.reasoning_content or delta.content or delta.tool_calls):
        return []
    chunk = chunk_json(
        request_id,
        content=delta.content or None,
        reasoning_content=delta.reasoning_content or None,
        tool_calls=delta.tool_calls or None,
        model=model,
    )
    return [f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"]


def parse_full_response(
    text: str,
    tool_call_parser: Optional[str] = None,
    reasoning_parser: Optional[str] = None,
    tools: Optional[list] = None,
) -> Tuple[Optional[str], str, List[Dict]]:
    """One-shot parse of a complete (non-streaming) response.

    Returns ``(reasoning_content, content, tool_calls)`` where
    ``tool_calls`` is a list of OpenAI-format tool-call dicts.
    """
    reasoning_content: Optional[str] = None
    normal_text = text

    if reasoning_parser:
        results = ReasoningParser(
            reasoning_parser_name=reasoning_parser
        ).extract_reasoning_content(text)
        if len(results) >= 2:
            reasoning_content = results[0].reasoning_content or None
            normal_text = results[1].normal_text or ""
        elif len(results) == 1:
            if results[0].reasoning_content:
                reasoning_content = results[0].reasoning_content or None
                normal_text = results[0].normal_text or ""
            else:
                normal_text = results[0].normal_text or text

    tool_calls: List[Dict] = []
    if tool_call_parser:
        parser = FunctionCallParser(
            tool_call_parser=tool_call_parser, tools=tools or []
        )
        normal_text_after, call_list = parser.parse_non_stream(normal_text)
        if call_list:
            tool_calls = format_tool_calls(call_list)
            normal_text = normal_text_after

    return reasoning_content, normal_text, tool_calls
