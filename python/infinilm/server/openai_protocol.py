"""Helpers for translating model tool-call text to OpenAI protocol objects."""

import json
import re
import uuid
from typing import Optional

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def strip_reasoning_markers(text: str) -> str:
    """Remove model reasoning delimiters from visible chat content."""
    return text.replace("<think>", "").replace("</think>", "")


_TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_COMPAT_FUNCTION_RE = re.compile(
    r"^\s*<function=([^>\n]+)>\s*(.*?)(?:\s*</function>)?\s*$", re.DOTALL
)
_COMPAT_PARAMETER_RE = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</(?:parameter|arg_value)>", re.DOTALL
)
_GLM_FUNCTION_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.:-]*)\s*(.*)$", re.DOTALL)
_GLM_ARGUMENT_RE = re.compile(
    r"<arg_key>\s*([^<>]+?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)


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


def _parse_tool_block(body: str, complete_outer: bool = False) -> list[dict]:
    # GLM's native template uses a bare function name followed by key/value tags.
    # Require a complete outer block and complete, unique arguments before execution.
    native = _GLM_FUNCTION_RE.fullmatch(body) if complete_outer else None
    if native:
        argument_body = native.group(2)
        matches = list(_GLM_ARGUMENT_RE.finditer(argument_body))
        remainder = _GLM_ARGUMENT_RE.sub("", argument_body).strip()
        keys = [match.group(1).strip() for match in matches]
        if not remainder and len(keys) == len(set(keys)):
            arguments = {}
            for key, match in zip(keys, matches):
                value = match.group(2)
                try:
                    arguments[key] = json.loads(value)
                except json.JSONDecodeError:
                    arguments[key] = value
            return [_new_tool_call(native.group(1), arguments)]
    calls = []
    for function_match in _FUNCTION_RE.finditer(body):
        arguments = {}
        function_body = function_match.group(2)
        parameter_matches = list(_PARAMETER_RE.finditer(function_body))
        if function_body.count("<parameter=") != len(parameter_matches):
            continue
        for parameter_match in parameter_matches:
            arguments[parameter_match.group(1).strip()] = _decode_parameter(
                parameter_match.group(2)
            )
        calls.append(_new_tool_call(function_match.group(1), arguments))

    if calls:
        return calls

    # Some GLM tool templates close the final argument with </arg_value> and
    # omit </function>. Accept that only inside an already complete outer
    # <tool_call> block, and only when every parameter has an explicit close.
    function_match = _COMPAT_FUNCTION_RE.fullmatch(body)
    if function_match:
        function_body = function_match.group(2)
        parameter_matches = list(_COMPAT_PARAMETER_RE.finditer(function_body))
        if parameter_matches and function_body.count("<parameter=") == len(
            parameter_matches
        ):
            remainder = _COMPAT_PARAMETER_RE.sub("", function_body).strip()
            if not remainder:
                arguments = {
                    match.group(1).strip(): _decode_parameter(match.group(2))
                    for match in parameter_matches
                }
                return [_new_tool_call(function_match.group(1), arguments)]

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


def _hide_incomplete_tool_prefix(text: str) -> str:
    marker_start = text.find(_TOOL_CALL_OPEN)
    if marker_start >= 0:
        text = text[:marker_start]
    for length in range(min(len(text), len(_TOOL_CALL_OPEN) - 1), 0, -1):
        if text.endswith(_TOOL_CALL_OPEN[:length]):
            return text[:-length]
    return text


def parse_tool_calls(
    text: str, allowed_tool_names: Optional[set[str]] = None
) -> tuple[Optional[str], list[dict]]:
    """Parse GLM/Qwen/Hermes tool-call blocks and return OpenAI tool calls."""
    tool_calls = []
    content_parts = []
    cursor = 0
    saw_tool_block = False

    for match in _TOOL_BLOCK_RE.finditer(text):
        parsed = _parse_tool_block(match.group(1), complete_outer=True)
        if not parsed:
            if allowed_tool_names is not None:
                saw_tool_block = True
                content_parts.append(text[cursor : match.start()])
                cursor = match.end()
            continue
        saw_tool_block = True
        content_parts.append(text[cursor : match.start()])
        cursor = match.end()
        if allowed_tool_names is not None:
            parsed = [
                call
                for call in parsed
                if call["function"]["name"] in allowed_tool_names
            ]
        tool_calls.extend(parsed)

    if not saw_tool_block:
        start = text.find(_TOOL_CALL_OPEN)
        if start >= 0:
            parsed = _parse_tool_block(text[start + len(_TOOL_CALL_OPEN) :])
            if parsed:
                if allowed_tool_names is not None:
                    parsed = [
                        call
                        for call in parsed
                        if call["function"]["name"] in allowed_tool_names
                    ]
                content = text[:start].strip()
                return content or None, parsed

        content = (
            _hide_incomplete_tool_prefix(text)
            if allowed_tool_names is not None
            else text
        ).strip()
        return content or None, []

    tail = text[cursor:]
    if allowed_tool_names is not None:
        tail = _hide_incomplete_tool_prefix(tail)
    content_parts.append(tail)
    content = "".join(content_parts).strip()
    return content or None, tool_calls


def select_tool_calls(
    calls: list[dict], already_emitted: int, parallel_tool_calls: Optional[bool]
) -> list[dict]:
    """Match OpenAI's single-call limit when parallel tools are disabled."""
    if parallel_tool_calls is False:
        return calls[: max(0, 1 - already_emitted)]
    return calls


class ToolCallStreamParser:
    """Incrementally hide XML markers and emit parsed OpenAI tool calls."""

    def __init__(self, allowed_tool_names: Optional[set[str]] = None):
        self._buffer = ""
        self.has_tool_calls = False
        self.allowed_tool_names = allowed_tool_names

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
            content, parsed = parse_tool_calls(
                block, allowed_tool_names=self.allowed_tool_names
            )
            if parsed:
                self.has_tool_calls = True
                tool_calls.extend(parsed)
            elif self.allowed_tool_names is not None:
                # A complete tool block was rejected as malformed or because
                # its function was not exposed. Never leak its XML markup.
                pass
            else:
                content_parts.append(content if content is not None else block)

        return content_parts, tool_calls

    def finalize(self) -> tuple[list[str], list[dict]]:
        if not self._buffer:
            return [], []
        buffered = self._buffer
        self._buffer = ""
        content, tool_calls = parse_tool_calls(
            buffered, allowed_tool_names=self.allowed_tool_names
        )
        if tool_calls:
            self.has_tool_calls = True
            return ([content] if content else []), tool_calls
        if buffered.startswith(_TOOL_CALL_OPEN) or _TOOL_CALL_OPEN.startswith(buffered):
            return [], []
        return ([content] if content else []), tool_calls
