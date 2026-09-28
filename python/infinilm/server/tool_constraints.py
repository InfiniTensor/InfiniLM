"""Request-local tool constraints for OpenAI-compatible agent clients."""

import json
import re
from collections import Counter
from typing import Optional
from xml.sax.saxutils import escape

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
    r"(?:写入|写到|保存到|创建|新建)\s*[^\s；;。\n]+\.[A-Za-z0-9]+", re.I
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
_READ_FILE_ZH_RE = re.compile(r"(?:读取|查看|打开)\s*[^\s；;。\n]+\.[A-Za-z0-9]+", re.I)
_READ_FILE_EN_RE = re.compile(r"\bread\b[^.\n]*\b(?:file|from)\b[^.\n]*\.[A-Za-z0-9]+", re.I)
_EDIT_FILE_ZH_RE = re.compile(r"(?:编辑|修改)\s*[^\s；;。\n]+\.[A-Za-z0-9]+", re.I)
_EDIT_FILE_EN_RE = re.compile(r"\bedit\b[^.\n]*\.[A-Za-z0-9]+", re.I)
_RUN_COMMAND_ZH_RE = re.compile(r"(?:执行|运行)\s*(?:命令|shell|bash|脚本)", re.I)
_RUN_COMMAND_EN_RE = re.compile(r"\b(?:run|execute)\b[^.\n]*\b(?:command|shell|bash|script)\b", re.I)
_READ_FILE_TARGET_ZH_RE = re.compile(r"(?:读取|查看|打开)\s*([^\s；;。\n]+\.[A-Za-z0-9]+)", re.I)
_READ_FILE_TARGET_EN_RE = re.compile(r"\bread\s+([^\s.;\n]+\.[A-Za-z0-9]+)", re.I)
_EDIT_FILE_TARGET_ZH_RE = re.compile(r"(?:编辑|修改)\s*([^\s；;。\n]+\.[A-Za-z0-9]+)", re.I)
_EDIT_FILE_TARGET_EN_RE = re.compile(r"\bedit\s+([^\s.;\n]+\.[A-Za-z0-9]+)", re.I)
_EDIT_CHANGE_ZH_RE = re.compile(r"把\s*([^，。；;\n]+?)\s*(?:改成|改为|替换为)\s*([^，。；;\n]+)", re.I)
_EDIT_CHANGE_EN_RE = re.compile(r"\bchange\s+(.+?)\s+to\s+(.+?)(?:[.;\n]|$)", re.I)
_BASH_COMMAND_ZH_RE = re.compile(r"(?:执行|运行)\s*(?:bash\s*)?命令\s*(.+)", re.I | re.S)
_BASH_COMMAND_EN_RE = re.compile(r"\b(?:run|execute)\s+(?:the\s+)?(?:bash\s+)?command\s+(.+)", re.I | re.S)


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
        text = text.split("\n\n<InfiniLM tool contract>", 1)[0].strip()
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

def _is_direct_read(text: str) -> bool:
    candidate = text.strip()
    while candidate.startswith(("仅", "只", "禁止", "不得", "不要使用")) and "。" in candidate:
        candidate = candidate.split("。", 1)[1].strip()
    return bool(re.fullmatch(
        r"\s*(?:请\s*)?(?:(?:读取|查看|打开)\s*[^\s；;。\n]+\.[A-Za-z0-9]+|"
        r"read\s+[^\s.;\n]+\.[A-Za-z0-9]+)[。.]*\s*",
        candidate, re.I,
    ))


def _current_turn_messages(messages: list) -> list:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "user":
            text = _message_text(message).split("\n\n<InfiniLM tool contract>", 1)[0]
            if text and not _SYNTHETIC_USER_RE.search(text):
                return messages[index:]
    return messages


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
    for message in _current_turn_messages(messages):
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
    if _is_direct_read(text):
        read_tools = [
            tool for tool in constrained if tool["function"]["name"] == "read_file"
        ]
        if read_tools:
            return read_tools

    if _is_direct_edit_then_stop(text):
        edit_tools = [
            tool
            for tool in constrained
            if tool["function"]["name"] in {"edit_file", "read_file"}
        ]
        if edit_tools:
            return edit_tools

    if _bash_command(text) and _STOP_AFTER_WRITE_ZH_RE.search(text):
        command_tools = [
            tool
            for tool in constrained
            if tool["function"]["name"] in {"bash", "write_file", "read_file"}
        ]
        if command_tools:
            return command_tools
    return constrained


_NAMED_FILE_TARGET_ZH_RE = re.compile(
    r"(?:写入|写到|保存到|创建|新建)\s*([^\s；;。\n]+\.[A-Za-z0-9]+)", re.I
)
_NAMED_FILE_TARGET_EN_RE = re.compile(
    r"\b(?:write|save)\b[^.\n]*\b(?:to|into)\s+([^\s.;\n]+\.[A-Za-z0-9]+)",
    re.I,
)


def direct_literal_write_arguments(messages: list, tools: list) -> Optional[dict]:
    exposed = {
        tool.get("function", {}).get("name") for tool in tools if isinstance(tool, dict)
    }
    if exposed != {"write_file"}:
        return None
    text = _latest_user_text(messages)
    if not text or not _is_direct_write_then_stop(text):
        return None
    target = _NAMED_FILE_TARGET_ZH_RE.search(text)
    if not target:
        return None
    label = re.search(r"内容(?:为|是)?(?:一行|两行)?\s*[：:]\s*", text[target.end():])
    if not label:
        return None
    body = text[target.end() + label.end():]
    stop = re.search(r"[。.!！]\s*(?:写完|完成后)[^。.!！\n]*(?:停止|停下)", body)
    if not stop:
        return None
    content = body[:stop.start()].strip()
    if not content or len(content) > 2048:
        return None
    return {"file_path": target.group(1), "content": content}


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




def direct_read_arguments(messages: list, tools: list) -> Optional[dict]:
    exposed = {
        tool.get("function", {}).get("name") for tool in tools if isinstance(tool, dict)
    }
    if exposed != {"read_file"}:
        return None
    text = _latest_user_text(messages)
    if not text or not _is_direct_read(text):
        return None
    match = _READ_FILE_TARGET_ZH_RE.search(text) or _READ_FILE_TARGET_EN_RE.search(text)
    if not match:
        return None
    target = match.group(1).strip()
    if not target:
        return None
    signatures, _ = _tool_history(messages)
    if any(name == "read_file" for name, _ in signatures):
        return None
    return {"file_path": target}


def forced_read_tool_prefix(messages: list, tools: list) -> Optional[str]:
    arguments = direct_read_arguments(messages, tools)
    if arguments is None:
        return None

    return chr(10).join([
        "<tool_call>",
        "<function=read_file>",
        "<parameter=file_path>",
        arguments["file_path"],
        "</parameter>",
        "</function>",
    ]) + chr(10)



def direct_read_result(messages: list, tools: list) -> Optional[str]:
    """Return a successful read_file result for a direct read request."""
    if not any(
        isinstance(tool, dict) and tool.get("function", {}).get("name") == "read_file"
        for tool in tools
    ):
        return None

    text = _latest_user_text(messages)
    if not text or not _is_direct_read(text):
        return None

    _, names_by_id = _tool_history(messages)
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        if names_by_id.get(message.get("tool_call_id")) != "read_file":
            continue
        result = _message_text(message)
        normalized = result.lower().replace(" ", "")
        if any(marker.replace(" ", "") in normalized for marker in _FAILED_TOOL_RESULT_MARKERS):
            continue
        if "fileunchangedsince" in normalized:
            continue
        return result
    return None



def _is_direct_edit_then_stop(text: str) -> bool:
    target = _EDIT_FILE_TARGET_ZH_RE.search(text) or _EDIT_FILE_TARGET_EN_RE.search(text)
    return bool(target and _STOP_AFTER_WRITE_ZH_RE.search(text))


def _has_successful_tool(
    messages: list, names_by_id: dict[str, str], tool_name: str
) -> bool:
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        if names_by_id.get(message.get("tool_call_id")) != tool_name:
            continue
        result = _message_text(message).lower().replace(" ", "")
        if not any(
            marker.replace(" ", "") in result
            for marker in _FAILED_TOOL_RESULT_MARKERS
        ):
            return True
    return False


def forced_read_for_edit_prefix(messages: list, tools: list) -> Optional[str]:
    exposed = {
        tool.get("function", {}).get("name") for tool in tools if isinstance(tool, dict)
    }
    if not {"edit_file", "read_file"}.issubset(exposed):
        return None
    text = _latest_user_text(messages)
    if not text or not _is_direct_edit_then_stop(text):
        return None
    _, names_by_id = _tool_history(messages)
    if _has_successful_tool(messages, names_by_id, "read_file"):
        return None
    match = _EDIT_FILE_TARGET_ZH_RE.search(text) or _EDIT_FILE_TARGET_EN_RE.search(text)
    if not match:
        return None
    target = match.group(1).strip()
    return chr(10).join([
        "<tool_call>",
        "<function=read_file>",
        "<parameter=file_path>",
        json.dumps(target, ensure_ascii=False),
        "</parameter>",
        "</function>",
    ]) + chr(10)


def direct_edit_step(messages: list, tools: list) -> Optional[dict]:
    exposed = {
        tool.get("function", {}).get("name") for tool in tools if isinstance(tool, dict)
    }
    if not {"read_file", "edit_file"}.issubset(exposed):
        return None
    text = _latest_user_text(messages)
    if not text or not _is_direct_edit_then_stop(text):
        return None
    target_match = _EDIT_FILE_TARGET_ZH_RE.search(text) or _EDIT_FILE_TARGET_EN_RE.search(text)
    change_match = _EDIT_CHANGE_ZH_RE.search(text) or _EDIT_CHANGE_EN_RE.search(text)
    if not target_match or not change_match:
        return None
    target = target_match.group(1).strip()
    old_string = change_match.group(1).strip().strip("'\"")
    new_string = change_match.group(2).strip().strip("'\"")
    if not target or not old_string or not new_string or old_string == new_string:
        return None

    signatures, names_by_id = _tool_history(messages)
    attempted = {name for name, _ in signatures}
    if "edit_file" in attempted:
        return None
    if _has_successful_tool(messages, names_by_id, "read_file"):
        return {
            "name": "edit_file",
            "arguments": {
                "file_path": target,
                "old_string": old_string,
                "new_string": new_string,
            },
        }
    if "read_file" not in attempted:
        return {"name": "read_file", "arguments": {"file_path": target}}
    return None


def forced_edit_tool_prefix(messages: list, tools: list) -> Optional[str]:
    exposed = {
        tool.get("function", {}).get("name") for tool in tools if isinstance(tool, dict)
    }
    if "edit_file" not in exposed:
        return None
    text = _latest_user_text(messages)
    if not text or not _is_direct_edit_then_stop(text):
        return None
    signatures, names_by_id = _tool_history(messages)
    if any(name == "edit_file" for name, _ in signatures):
        return None
    if not _has_successful_tool(messages, names_by_id, "read_file"):
        return None
    if _has_successful_tool(messages, names_by_id, "edit_file"):
        return None

    target_match = _EDIT_FILE_TARGET_ZH_RE.search(text) or _EDIT_FILE_TARGET_EN_RE.search(text)
    change_match = _EDIT_CHANGE_ZH_RE.search(text) or _EDIT_CHANGE_EN_RE.search(text)
    if not target_match or not change_match:
        return None
    target = target_match.group(1).strip()
    old_string = change_match.group(1).strip().strip("'''")
    new_string = change_match.group(2).strip().strip("'''")
    if not target or not old_string or not new_string or old_string == new_string:
        return None

    return chr(10).join([
        "<tool_call>",
        "<function=edit_file>",
        "<parameter=file_path>",
        json.dumps(target, ensure_ascii=False),
        "</parameter>",
        "<parameter=old_string>",
        json.dumps(old_string, ensure_ascii=False),
        "</parameter>",
        "<parameter=new_string>",
        json.dumps(new_string, ensure_ascii=False),
        "</parameter>",
        "</function>",
    ]) + chr(10)


def direct_edit_complete(messages: list, tools: list) -> bool:
    if not any(
        isinstance(tool, dict) and tool.get("function", {}).get("name") == "edit_file"
        for tool in tools
    ):
        return False
    text = _latest_user_text(messages)
    if not text or not _is_direct_edit_then_stop(text):
        return False
    _, names_by_id = _tool_history(messages)
    return _has_successful_tool(messages, names_by_id, "edit_file")



def _bash_command(text: str) -> Optional[str]:
    match = _BASH_COMMAND_ZH_RE.search(text) or _BASH_COMMAND_EN_RE.search(text)
    if not match:
        return None
    command = re.split(
        r"[。；;]\s*(?:完成后|执行后|运行后)[^\n]*(?:停止|停下)[。.]?\s*$",
        match.group(1),
        maxsplit=1,
    )[0].strip().removesuffix("。").strip()
    return command or None


def direct_bash_arguments(messages: list, tools: list) -> Optional[dict]:
    if not any(
        isinstance(tool, dict) and tool.get("function", {}).get("name") == "bash"
        for tool in tools
    ):
        return None
    text = _latest_user_text(messages)
    command = _bash_command(text) if text else None
    if not command or not _STOP_AFTER_WRITE_ZH_RE.search(text) or len(command) > 2048:
        return None
    return {"command": command}


def forced_bash_tool_prefix(messages: list, tools: list) -> Optional[str]:
    if not any(
        isinstance(tool, dict) and tool.get("function", {}).get("name") == "bash"
        for tool in tools
    ):
        return None
    text = _latest_user_text(messages)
    command = _bash_command(text) if text else None
    if not command or not _STOP_AFTER_WRITE_ZH_RE.search(text):
        return None
    return chr(10).join([
        "<tool_call>",
        "<function=bash>",
        "<parameter=command>",
        json.dumps(command, ensure_ascii=False),
        "</parameter>",
        "</function>",
    ]) + chr(10)


def direct_bash_result(messages: list, tools: list) -> Optional[str]:
    if not any(
        isinstance(tool, dict) and tool.get("function", {}).get("name") == "bash"
        for tool in tools
    ):
        return None
    text = _latest_user_text(messages)
    if not text or not (_bash_command(text) and _STOP_AFTER_WRITE_ZH_RE.search(text)):
        return None
    _, names_by_id = _tool_history(messages)
    if not _has_successful_tool(messages, names_by_id, "bash"):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        if names_by_id.get(message.get("tool_call_id")) != "bash":
            continue
        result = _message_text(message)
        normalized = result.lower().replace(" ", "")
        if not any(
            marker.replace(" ", "") in normalized
            for marker in _FAILED_TOOL_RESULT_MARKERS
        ):
            return result
    return None

def direct_write_complete(messages: list, tools: list) -> bool:
    if not tools:
        return False

    text = _latest_user_text(messages)
    if not text or not _is_direct_write_then_stop(text):
        return False

    _, names_by_id = _tool_history(messages)
    return _has_successful_write(messages, names_by_id)
