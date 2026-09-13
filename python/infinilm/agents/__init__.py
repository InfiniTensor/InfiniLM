"""
Agent support for InfiniLM: tool calls and reasoning parsing.
Supports GLM-4, Llama-3.1+, and Qwen3 tool call formats.
"""

from .function_call_parser import FunctionCallParser
from .message_adapter import adapt_messages
from .protocol import DeltaMessage, Function, Tool, ToolChoice
from .reasoning_parser import ReasoningParser
from .stream_parser import (
    AgentDelta,
    AgentStreamParser,
    parse_full_response,
)
from .types import StreamingParseResult, ToolCallItem

__all__ = [
    "ToolCallItem",
    "StreamingParseResult",
    "Tool",
    "ToolChoice",
    "Function",
    "DeltaMessage",
    "FunctionCallParser",
    "ReasoningParser",
    "AgentDelta",
    "AgentStreamParser",
    "parse_full_response",
    "adapt_messages",
]
