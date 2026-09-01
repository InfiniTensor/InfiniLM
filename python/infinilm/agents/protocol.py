"""
OpenAI-compatible protocol models and response builders for agent support.
Minimal subset needed for tools, tool_choice, and reasoning_content.
"""

import time
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class Function(BaseModel):
    description: Optional[str] = None
    name: str
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: Function


class ToolChoiceFuncName(BaseModel):
    name: str


class ToolChoice(BaseModel):
    type: Literal["function"] = "function"
    function: ToolChoiceFuncName


class FunctionResponse(BaseModel):
    name: str
    arguments: str


class ChatCompletionMessageToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionResponse


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


def chunk_json(
    id_,
    content=None,
    role=None,
    finish_reason=None,
    model: str = "unknown",
    reasoning_content=None,
    tool_calls=None,
    usage=None,
):
    """Generate JSON chunk for streaming response."""
    delta = {}
    if content is not None:
        delta["content"] = content
    if role:
        delta["role"] = role
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if tool_calls:
        delta["tool_calls"] = tool_calls
    chunk = {
        "id": id_,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage:
        chunk["usage"] = usage
    return chunk


def completion_json(
    id_,
    content,
    role="assistant",
    finish_reason="stop",
    model: str = "unknown",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    reasoning_content=None,
    tool_calls=None,
):
    """Generate JSON response for non-streaming completion."""
    message = {
        "role": role,
        "content": content,
    }
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": id_,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
