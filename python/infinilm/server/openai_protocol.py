"""Helpers for translating model tool-call text to OpenAI protocol objects."""

import json
import re
import uuid
from typing import Optional

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _new_tool_call(name: str, arguments) -> dict:
    if isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments_json = arguments
        else:
            arguments_json = json.dumps(
                parsed_arguments, ensure_ascii=False, separators=(",", ":")
            )
    else:
        arguments_json = json.dumps(
            arguments, ensure_ascii=False, separators=(",", ":")
        )

    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name.strip(),
            "arguments": arguments_json,
        },
    }


def _decode_parameter(value: str):
    value = value.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_tool_block(body: str) -> list[dict]:
    calls = []
    for function_match in _FUNCTION_RE.finditer(body):
        arguments = {}
        for parameter_match in _PARAMETER_RE.finditer(function_match.group(2)):
            arguments[parameter_match.group(1).strip()] = _decode_parameter(
                parameter_match.group(2)
            )
        calls.append(_new_tool_call(function_match.group(1), arguments))

    if calls:
        return calls

    try:
        payload = json.loads(body.strip())
    except json.JSONDecodeError:
        return []

    payloads = payload if isinstance(payload, list) else [payload]
    for item in payloads:
        if not isinstance(item, dict):
            continue
        function = item.get("function", item)
        if not isinstance(function, dict) or not function.get("name"):
            continue
        calls.append(_new_tool_call(function["name"], function.get("arguments", {})))
    return calls


def parse_tool_calls(text: str) -> tuple[Optional[str], list[dict]]:
    """Parse Qwen/Hermes tool-call blocks and return OpenAI tool calls."""
    tool_calls = []
    content_parts = []
    cursor = 0

    for match in _TOOL_BLOCK_RE.finditer(text):
        parsed = _parse_tool_block(match.group(1))
        if not parsed:
            continue
        content_parts.append(text[cursor : match.start()])
        cursor = match.end()
        tool_calls.extend(parsed)

    if not tool_calls:
        content = text.strip()
        return content or None, []

    content_parts.append(text[cursor:])
    content = "".join(content_parts).strip()
    return content or None, tool_calls


class ToolCallStreamParser:
    """Incrementally hide XML markers and emit parsed OpenAI tool calls."""

    def __init__(self):
        self._buffer = ""
        self.has_tool_calls = False

    @staticmethod
    def _marker_suffix_length(value: str) -> int:
        max_length = min(len(value), len(_TOOL_CALL_OPEN) - 1)
        for length in range(max_length, 0, -1):
            if _TOOL_CALL_OPEN.startswith(value[-length:]):
                return length
        return 0

    def feed(self, chunk: str) -> tuple[list[str], list[dict]]:
        self._buffer += chunk
        content_parts = []
        tool_calls = []

        while self._buffer:
            start = self._buffer.find(_TOOL_CALL_OPEN)
            if start < 0:
                keep = self._marker_suffix_length(self._buffer)
                safe_length = len(self._buffer) - keep
                if safe_length:
                    content_parts.append(self._buffer[:safe_length])
                    self._buffer = self._buffer[safe_length:]
                break

            if start:
                content_parts.append(self._buffer[:start])
                self._buffer = self._buffer[start:]

            end = self._buffer.find(_TOOL_CALL_CLOSE)
            if end < 0:
                break
            end += len(_TOOL_CALL_CLOSE)
            block = self._buffer[:end]
            self._buffer = self._buffer[end:]
            _, parsed = parse_tool_calls(block)
            if parsed:
                self.has_tool_calls = True
                tool_calls.extend(parsed)
            else:
                content_parts.append(block)

        return content_parts, tool_calls

    def finalize(self) -> tuple[list[str], list[dict]]:
        if not self._buffer:
            return [], []
        content, tool_calls = parse_tool_calls(self._buffer)
        self._buffer = ""
        if tool_calls:
            self.has_tool_calls = True
        return ([content] if content else []), tool_calls
