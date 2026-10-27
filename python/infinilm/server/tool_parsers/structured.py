"""Typed DSML and XTML calls used by DeepSeek V4 and Kimi K3."""

import html
import json
import re

from jsonschema import Draft202012Validator

from ..agent_protocol import AgentProtocolError, make_tool_call


def members(text, pattern):
    offset = 0
    for match in re.finditer(pattern, text, re.S):
        if text[offset : match.start()].strip():
            raise AgentProtocolError("Malformed tool markup")
        offset = match.end()
        yield match
    if text[offset:].strip():
        raise AgentProtocolError("Malformed or truncated tool markup")


def typed_value(raw, kind):
    if kind == "string":
        return raw
    if kind not in {"number", "integer", "boolean", "null", "array", "object"}:
        raise AgentProtocolError("Unknown tool argument type")
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise AgentProtocolError("Invalid JSON tool argument") from exc
    if not Draft202012Validator({"type": kind}).is_valid(value):
        raise AgentProtocolError("Tool argument does not match its declared type")
    return value


def parse_structured_calls(body, schemas, parser, offset=0):
    if parser == "deepseek_v4":
        call_pattern = r'<｜DSML｜invoke name="([^"]+)">\s*(.*?)\s*</｜DSML｜invoke>'
        param_pattern = r'<｜DSML｜parameter name="([^"]+)" string="(true|false)">(.*?)</｜DSML｜parameter>'
    else:
        call_pattern = r'<\|open\|>call tool="([^"]+)" index="([1-9]\d*)"<\|sep\|>(.*?)<\|close\|>call<\|sep\|>'
        param_pattern = r'<\|open\|>argument key="([^"]+)" type="([^"]+)"<\|sep\|>(.*?)<\|close\|>argument<\|sep\|>'
    calls = []
    for call in members(body, call_pattern):
        name = html.unescape(call[1])
        args = {}
        content = call[2] if parser == "deepseek_v4" else call[3]
        if parser == "kimi_k3" and int(call[2]) != offset + len(calls) + 1:
            raise AgentProtocolError("Kimi call indices must follow call order")
        block = (
            re.fullmatch(
                r'<\|open\|>json type="object"<\|sep\|>(.*?)<\|close\|>json<\|sep\|>',
                content,
                re.S,
            )
            if parser == "kimi_k3"
            else None
        )
        if block:
            args = typed_value(block[1], "object")
        else:
            for param in members(content, param_pattern):
                key = html.unescape(param[1])
                if key in args:
                    raise AgentProtocolError("Duplicate tool argument")
                kind = param[2]
                if parser == "deepseek_v4":
                    kind = "string" if kind == "true" else "json"
                if kind == "json":
                    try:
                        args[key] = json.loads(param[3])
                    except ValueError as exc:
                        raise AgentProtocolError("Invalid JSON tool argument") from exc
                else:
                    args[key] = typed_value(param[3], kind)
        calls.append(make_tool_call(name, args, schemas, parser))
    if not calls:
        raise AgentProtocolError("Empty tool calls")
    return calls
