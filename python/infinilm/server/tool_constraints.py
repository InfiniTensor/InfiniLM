"""Request-local tool constraints for OpenAI-compatible agent clients."""

import json
import re
from collections import Counter
from typing import Optional

_ALLOW_PATTERNS = (
    re.compile(r"(?:仅|只)(?:允许|能|可)?(?:使用|用)\s*([^；;。\n]+)", re.I),
    re.compile(r"\bonly\s+(?:use|allow)\s+([^.;\n]+)", re.I),
)
_BLOCK_PATTERNS = (
    re.compile(r"(?:禁止|不得|禁用|不要使用)\s*([^；;。\n]+)", re.I),
    re.compile(r"\b(?:do\s+not\s+use|don't\s+use|forbid|disable)\s+([^.;\n]+)", re.I),
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_NAMED_FILE_ZH_RE = re.compile(
    r"(?:写入|写到|保存到)\s*[^\s；;。\n]+\.[A-Za-z0-9]+", re.I
)
_NAMED_FILE_EN_RE = re.compile(
    r"\b(?:write|save)\b[^.\n]*\b(?:to|into)\s+[^\s.;\n]+\.[A-Za-z0-9]+",
    re.I,
)
_STOP_AFTER_WRITE_ZH_RE = re.compile(
    r"(?:写完|完成后)[^；;。\n]*(?:立即)?(?:停止|停下)", re.I
)
_STOP_AFTER_WRITE_EN_RE = re.compile(
    r"(?:\b(?:stop|wait)\b[^.\n]*\bafter\b[^.\n]*(?:writ|sav)|"
    r"\bafter\b[^.\n]*(?:writ|sav)[^.\n]*\b(?:stop|wait)\b)",
    re.I,
)
_FAILED_TOOL_RESULT_MARKERS = (
    "tool_error",
    "tool execution failed",
    '"iserror":true',
    '"is_error":true',
)


def _normalize_tool_name(value: str) -> str:
    return re.sub(r"[-_]", "", value.lower())


def _tool_aliases(name: str) -> set[str]:
    normalized = _normalize_tool_name(name)
    aliases = {normalized}
    if normalized.endswith("file"):
        aliases.add(normalized[:-4])
    return aliases


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


_SYNTHETIC_USER_RE = re.compile(
    r"^(?:Output token limit hit[.]|Your previous response was empty "
    r"[(]thinking only, no visible text[)][.])",
    re.I,
)


def _latest_user_text(messages: list) -> Optional[str]:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _message_text(message)
        if text and not _SYNTHETIC_USER_RE.search(text):
            return text
    return None


def _extract_tool_names(text: str, patterns, tools: list) -> set[str]:
    lookup = {}
    for tool in tools:
        name = tool.get("function", {}).get("name")
        if not isinstance(name, str):
            continue
        for alias in _tool_aliases(name):
            lookup[alias] = name

    names = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            for identifier in _IDENTIFIER_RE.findall(match.group(1)):
                name = lookup.get(_normalize_tool_name(identifier))
                if name:
                    names.add(name)
    return names


def _is_direct_write_then_stop(text: str) -> bool:
    requests_file = bool(
        _NAMED_FILE_ZH_RE.search(text) or _NAMED_FILE_EN_RE.search(text)
    )
    requests_stop = bool(
        _STOP_AFTER_WRITE_ZH_RE.search(text) or _STOP_AFTER_WRITE_EN_RE.search(text)
    )
    return requests_file and requests_stop


def _canonical_arguments(arguments) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return json.dumps(
        arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _tool_history(messages: list) -> tuple[Counter, dict[str, str]]:
    signatures = Counter()
    names_by_id = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            name = function.get("name")
            if not isinstance(name, str):
                continue
            signatures[(name, _canonical_arguments(function.get("arguments", {})))] += 1
            call_id = call.get("id")
            if isinstance(call_id, str):
                names_by_id[call_id] = name
    return signatures, names_by_id


def _has_successful_write(messages: list, names_by_id: dict[str, str]) -> bool:
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        if names_by_id.get(message.get("tool_call_id")) != "write_file":
            continue
        result = _message_text(message).lower().replace(" ", "")
        if not any(
            marker.replace(" ", "") in result for marker in _FAILED_TOOL_RESULT_MARKERS
        ):
            return True
    return False


def constrain_tools(messages: list, tools: list) -> list:
    """Filter schemas using explicit user constraints and request history."""
    if not tools:
        return tools
    text = _latest_user_text(messages)
    if not text:
        return tools

    allowed = _extract_tool_names(text, _ALLOW_PATTERNS, tools)
    blocked = _extract_tool_names(text, _BLOCK_PATTERNS, tools)
    constrained = [
        tool
        for tool in tools
        if (not allowed or tool["function"]["name"] in allowed)
        and tool["function"]["name"] not in blocked
    ]

    signatures, names_by_id = _tool_history(messages)
    repeated_names = {name for (name, _), count in signatures.items() if count >= 2}
    constrained = [
        tool for tool in constrained if tool["function"]["name"] not in repeated_names
    ]

    if _is_direct_write_then_stop(text):
        if _has_successful_write(messages, names_by_id):
            return []
        write_tools = [
            tool for tool in constrained if tool["function"]["name"] == "write_file"
        ]
        if write_tools:
            return write_tools
    return constrained


_NAMED_FILE_TARGET_ZH_RE = re.compile(
    r"(?:写入|写到|保存到)\s*([^\s；;。\n]+\.[A-Za-z0-9]+)", re.I
)
_NAMED_FILE_TARGET_EN_RE = re.compile(
    r"\b(?:write|save)\b[^.\n]*\b(?:to|into)\s+([^\s.;\n]+\.[A-Za-z0-9]+)",
    re.I,
)


def forced_write_tool_prefix(messages: list, tools: list) -> Optional[str]:
    exposed = {
        tool.get("function", {}).get("name") for tool in tools if isinstance(tool, dict)
    }
    if exposed != {"write_file"}:
        return None

    text = _latest_user_text(messages)
    if not text or not _is_direct_write_then_stop(text):
        return None

    match = _NAMED_FILE_TARGET_ZH_RE.search(text) or _NAMED_FILE_TARGET_EN_RE.search(
        text
    )
    if not match:
        return None
    target = match.group(1).strip()
    if not target:
        return None

    lines = [
        "<tool_call>",
        "<function=write_file>",
        "<parameter=file_path>",
        target,
        "</parameter>",
        "<parameter=content>",
    ]
    return chr(10).join(lines) + chr(10)


def direct_write_complete(messages: list, tools: list) -> bool:
    if not tools:
        return False

    text = _latest_user_text(messages)
    if not text or not _is_direct_write_then_stop(text):
        return False

    _, names_by_id = _tool_history(messages)
    return _has_successful_write(messages, names_by_id)
