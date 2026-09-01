"""
Anthropic Messages API protocol support.

Everything needed to speak the Anthropic protocol on top of the internal
OpenAI-format pipeline: request/response models, request conversion, and the
streaming SSE converter. Kept free of HTTP/engine concerns so it stays
unit-testable in isolation.
"""

import json
import uuid
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

# ---------- request models ----------


class AnthropicTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict = Field(default_factory=dict)


class AnthropicToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: Optional[str] = None
    content: Optional[Union[str, list["AnthropicContentBlock"]]] = None
    is_error: Optional[bool] = None


AnthropicContentBlock = Union[
    AnthropicTextBlock, AnthropicToolUseBlock, AnthropicToolResultBlock
]


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, list[AnthropicContentBlock]]


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    metadata: Optional[dict] = None
    stop_sequences: Optional[list[str]] = None
    stream: Optional[bool] = False
    system: Optional[Union[str, list[AnthropicTextBlock]]] = None
    temperature: Optional[float] = None
    tool_choice: Optional[dict] = None
    tools: Optional[list[dict]] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None


# ---------- SSE helpers ----------


def anthropic_sse_event(event_type: str, data: dict) -> str:
    """Wrap a dict as an Anthropic-style SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def anthropic_error_body(message: str) -> dict:
    """Body of an Anthropic-style error response."""
    return {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": message},
    }


def parse_openai_sse_line(raw: str) -> Optional[dict]:
    """Parse one OpenAI SSE line into its chunk dict; None if not a data chunk."""
    if not raw.startswith("data: ") or raw.startswith("data: [DONE]"):
        return None
    try:
        return json.loads(raw[6:].strip())
    except (json.JSONDecodeError, ValueError):
        return None


# ---------- request / response conversion ----------


def convert_anthropic_request(anthropic_req: AnthropicMessagesRequest) -> dict:
    """Convert an Anthropic Messages request to an OpenAI chat completion dict."""
    openai_messages = []

    # --- System message ---
    system_parts = []
    if anthropic_req.system is not None:
        if isinstance(anthropic_req.system, str):
            if anthropic_req.system.strip():
                system_parts.append(anthropic_req.system)
        else:
            for block in anthropic_req.system:
                if block.type == "text" and block.text:
                    system_parts.append(block.text)
    # Also pick up inline system messages
    for msg in anthropic_req.messages:
        if msg.role == "system":
            if isinstance(msg.content, str) and msg.content.strip():
                system_parts.append(msg.content)
            else:
                for block in msg.content or []:
                    if isinstance(block, AnthropicTextBlock) and block.text:
                        system_parts.append(block.text)
    if system_parts:
        openai_messages.append({"role": "system", "content": "\n".join(system_parts)})

    # --- User / Assistant messages ---
    for msg in anthropic_req.messages:
        if msg.role == "system":
            continue
        if isinstance(msg.content, str):
            openai_messages.append({"role": msg.role, "content": msg.content})
            continue

        openai_msg: dict = {"role": msg.role}
        content_parts: list[dict] = []
        tool_calls: list[dict] = []

        for block in msg.content:
            if block.type == "text" and block.text is not None:
                content_parts.append({"type": "text", "text": block.text})
            elif block.type == "image":
                # Best-effort image passthrough
                content_parts.append(block.model_dump(exclude_none=True))
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id or f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": block.name or "",
                            "arguments": json.dumps(block.input or {}),
                        },
                    }
                )
            elif block.type == "tool_result":
                tool_content = block.content
                tool_text = ""
                if isinstance(tool_content, str):
                    tool_text = tool_content
                elif isinstance(tool_content, list):
                    texts = [
                        b.text
                        for b in tool_content
                        if isinstance(b, AnthropicTextBlock)
                    ]
                    tool_text = "\n".join(texts)

                tool_call_id = block.tool_use_id or ""
                # Flush any pending user content first
                if content_parts and msg.role == "user":
                    if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                        openai_messages.append(
                            {"role": "user", "content": content_parts[0]["text"]}
                        )
                    else:
                        openai_messages.append(
                            {"role": "user", "content": list(content_parts)}
                        )
                    content_parts.clear()

                if msg.role == "user":
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": tool_text,
                        }
                    )
                else:
                    content_parts.append(
                        {"type": "text", "text": f"Tool result: {tool_text}"}
                    )

        if tool_calls:
            openai_msg["tool_calls"] = tool_calls
        if content_parts:
            if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                openai_msg["content"] = content_parts[0]["text"]
            else:
                openai_msg["content"] = content_parts
        elif tool_calls:
            pass  # assistant message with only tool_calls
        elif msg.role == "user":
            continue  # already emitted as tool messages
        else:
            openai_msg["content"] = ""  # empty assistant placeholder

        openai_messages.append(openai_msg)

    data: dict = {
        "messages": openai_messages,
        "model": anthropic_req.model,
        "max_tokens": anthropic_req.max_tokens,
        "stream": anthropic_req.stream or False,
    }
    if anthropic_req.temperature is not None:
        data["temperature"] = anthropic_req.temperature
    if anthropic_req.top_p is not None:
        data["top_p"] = anthropic_req.top_p
    if anthropic_req.top_k is not None:
        data["top_k"] = anthropic_req.top_k
    if anthropic_req.stop_sequences is not None:
        data["stop"] = anthropic_req.stop_sequences

    # Tools
    if anthropic_req.tools:
        openai_tools = []
        for tool in anthropic_req.tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
            )
        data["tools"] = openai_tools
        tc = anthropic_req.tool_choice
        if tc is None:
            data["tool_choice"] = "auto"
        elif tc.get("type") == "none":
            data["tool_choice"] = "none"
        elif tc.get("type") == "any":
            data["tool_choice"] = "required"
        elif tc.get("type") == "tool":
            data["tool_choice"] = {
                "type": "function",
                "function": {"name": tc.get("name", "")},
            }
        else:
            data["tool_choice"] = "auto"

    return data


def convert_openai_to_anthropic_response(response: dict, model_id: str) -> dict:
    """Convert an OpenAI chat completion response to an Anthropic Messages response."""
    choices = response.get("choices", [])
    if not choices:
        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
            "model": model_id,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    choice = choices[0]
    message = choice.get("message", {})
    content: list[dict] = []

    # Reasoning content -> thinking block (best-effort)
    reasoning = message.get("reasoning_content")
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})

    # Text content
    text = message.get("content", "")
    if text:
        content.append({"type": "text", "text": text})

    # Tool calls -> tool_use blocks
    for tc in message.get("tool_calls", []):
        raw_args = tc.get("function", {}).get("arguments", "")
        try:
            tool_input = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError):
            tool_input = {}
        content.append(
            {
                "type": "tool_use",
                "id": tc.get("id", f"call_{uuid.uuid4().hex}"),
                "name": tc.get("function", {}).get("name", ""),
                "input": tool_input,
            }
        )

    if not content:
        content.append({"type": "text", "text": ""})

    finish_reason = choice.get("finish_reason") or "stop"
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")

    usage = response.get("usage", {})
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model_id,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------- streaming conversion ----------


class AnthropicStreamConverter:
    """Convert OpenAI chat-completion stream chunks into Anthropic SSE events.

    Content blocks are tracked with an explicit active block type and a
    monotonically increasing block index: whenever the kind of output changes
    (thinking -> text -> tool_use, or a new tool call starts), the active
    block is closed and a new one is opened at the next index, so the emitted
    ``content_block_start/delta/stop`` sequence is always valid.
    """

    STOP_REASON_MAP = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }

    def __init__(self, message_id: str, model: str):
        self._message_id = message_id
        self._model = model
        # Type of the currently open content block, if any:
        # None | "thinking" | "text" | "tool_use"
        self._active_block: Optional[str] = None
        # Index of the currently open block; increases monotonically.
        self._block_index = -1
        # OpenAI tool_calls index of the currently open tool_use block.
        self._open_tool_idx: Optional[int] = None
        self._finish_reason: Optional[str] = None
        self._usage: Optional[dict] = None

    def begin(self) -> list:
        """Events to emit before the first chunk (message_start)."""
        return [
            anthropic_sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self._message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self._model,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        ]

    def feed(self, chunk: dict) -> list:
        """Events for one OpenAI-format stream chunk."""
        events = []
        choices = chunk.get("choices") or []
        if not choices:
            return events

        delta = choices[0].get("delta") or {}
        self._finish_reason = choices[0].get("finish_reason") or self._finish_reason
        if chunk.get("usage"):
            self._usage = chunk["usage"]

        # -- Reasoning content (thinking block) --
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if self._active_block != "thinking":
                events.extend(
                    self._switch_block("thinking", {"type": "thinking", "thinking": ""})
                )
            events.append(
                anthropic_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._block_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    },
                )
            )

        # -- Text content --
        text = delta.get("content")
        if text:
            if self._active_block != "text":
                events.extend(self._switch_block("text", {"type": "text", "text": ""}))
            events.append(
                anthropic_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._block_index,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            )

        # -- Tool calls --
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            if self._active_block != "tool_use" or self._open_tool_idx != idx:
                events.extend(
                    self._switch_block(
                        "tool_use",
                        {
                            "type": "tool_use",
                            "id": tc.get("id") or f"call_{idx}",
                            "name": tc.get("function", {}).get("name", ""),
                            "input": {},
                        },
                    )
                )
                self._open_tool_idx = idx

            args = tc.get("function", {}).get("arguments", "")
            if args:
                events.append(
                    anthropic_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._block_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": args,
                            },
                        },
                    )
                )

        return events

    def end(self) -> list:
        """Closing events after the chunk stream ends."""
        events = []
        if self._active_block is not None:
            events.append(
                anthropic_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self._block_index},
                )
            )
        stop_reason = self.STOP_REASON_MAP.get(
            self._finish_reason or "stop", "end_turn"
        )
        usage = self._usage or {}
        events.append(
            anthropic_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason},
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    },
                },
            )
        )
        events.append(anthropic_sse_event("message_stop", {"type": "message_stop"}))
        return events

    def _switch_block(self, block_type: str, content_block: dict) -> list:
        """Close the active block (if any) and open a new one."""
        events = []
        if self._active_block is not None:
            events.append(
                anthropic_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self._block_index},
                )
            )
        self._block_index += 1
        self._active_block = block_type
        if block_type != "tool_use":
            self._open_tool_idx = None
        events.append(
            anthropic_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": content_block,
                },
            )
        )
        return events


async def convert_openai_sse_stream(openai_stream, message_id: str, model: str):
    """Convert an OpenAI SSE line stream into Anthropic SSE events.

    ``openai_stream`` yields the raw ``data: ...`` lines produced by the
    OpenAI-format chat stream; this generator yields Anthropic-format SSE
    event strings.
    """
    converter = AnthropicStreamConverter(message_id=message_id, model=model)
    for event in converter.begin():
        yield event
    async for raw in openai_stream:
        if raw.startswith("data: [DONE]"):
            break
        chunk = parse_openai_sse_line(raw)
        if chunk is not None:
            for event in converter.feed(chunk):
                yield event
    for event in converter.end():
        yield event
