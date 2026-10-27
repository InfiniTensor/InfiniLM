"""Shared scanners for native tool-call formats; no tool execution."""

import html
import json
import re

from jsonschema import Draft202012Validator
from referencing.exceptions import Unresolvable

from ..agent_protocol import AgentProtocolError, make_tool_call
from .structured import parse_structured_calls

TOOL_CALL_FORMATS = {
    "qwen3_coder": ("<tool_call>", "</tool_call>"),
    "hermes": ("<tool_call>", "</tool_call>"),
    "llama3_json": ("<|python_tag|>", ""),
    "mistral": ("[TOOL_CALLS]", ""),
    "minicpm5": ("<function", "</function>"),
    "glm5": ("<tool_call>", "</tool_call>"),
    "deepseek_v4": ("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>"),
    "kimi_k3": ("<|open|>tools<|sep|>", "<|close|>tools<|sep|>"),
}


def _trim_newline(text):
    return re.sub(r"\r?\n$", "", re.sub(r"^\r?\n", "", text))


def _accepts_string(spec, root, raw):
    # Preserve numeric-looking strings, including local references and unions.
    if not isinstance(spec, dict) or not spec:
        return False
    try:
        return Draft202012Validator(root).evolve(schema=spec).is_valid(raw)
    except Unresolvable as exc:
        raise AgentProtocolError("Unresolved local tool parameter reference") from exc


def parse_xml_call(block, schemas, parser="qwen3_coder"):
    function = r"<tool_call>\s*<function=([^<>\s]+)>(.*?)</function>\s*</tool_call>"
    parameter = r"<parameter=([^<>\s]+)>(.*?)</parameter>"
    if parser == "minicpm5":
        function = r'<function\s+name="([^"<>]+)"\s*>(.*?)</function>'
        parameter = r'<param\s+name="([^"<>]+)"\s*>(<!\[CDATA\[.*?\]\]>|.*?)</param>'
    elif parser == "glm5":
        function = r"<tool_call>\s*([^<>\s]+)\s*(.*?)</tool_call>"
        parameter = r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>"
    match = re.fullmatch(function, block, re.S)
    if not match or match[1] not in schemas:
        raise AgentProtocolError("Malformed or unknown tool call")
    name, body = match[1], match[2]
    schema = schemas[name]
    args, offset = {}, 0
    for param in re.finditer(parameter, body, re.S):
        if body[offset : param.start()].strip() or param[1] in args:
            raise AgentProtocolError("Malformed or duplicate tool parameter")
        key, raw = param[1], param[2]
        if parser == "minicpm5" and raw.startswith("<![CDATA[") and raw.endswith("]]>"):
            raw = raw[9:-3]
        else:
            if parser != "glm5":
                raw = _trim_newline(raw)
            if parser == "minicpm5":
                raw = html.unescape(raw)
        spec = schema.get("properties", {}).get(key, {})
        # The template writes strings verbatim and all other values as JSON.
        args[key] = raw
        if not _accepts_string(spec, schema, raw):
            try:
                args[key] = json.loads(raw)
            except ValueError:
                pass
        offset = param.end()
    if body[offset:].strip():
        raise AgentProtocolError("Malformed tool parameters")
    return make_tool_call(name, args, schemas)


class ToolCallParser:
    """A bounded-prefix scanner; accepts arbitrary token/chunk boundaries."""

    def __init__(
        self,
        schemas,
        thinking=False,
        choice="auto",
        parallel=True,
        parser="qwen3_coder",
    ):
        self.schemas = schemas
        self.mode = "reasoning_content" if thinking else "content"
        self.buffer = ""
        self.calls = []
        self.choice = choice
        self.parallel = parallel
        self.parser = parser
        self.start, self.end = TOOL_CALL_FORMATS[parser]
        self.at_start = True
        self.think_start, self.think_end = "<think>", "</think>"
        self.ignored = []
        if parser == "kimi_k3":
            self.think_start = "<|open|>think<|sep|>"
            self.think_end = "<|close|>think<|sep|>"
            self.ignored = [
                "<|open|>response<|sep|>",
                "<|close|>response<|sep|>",
                "<|close|>message<|sep|>",
                "<|end_of_msg|>",
            ]

    def _parse_calls(self, block):
        if self.parser in ("qwen3_coder", "minicpm5", "glm5"):
            return [parse_xml_call(block, self.schemas, self.parser)]
        body = block.removeprefix(self.start)
        if self.end:
            body = body.removesuffix(self.end)
        if self.parser in ("deepseek_v4", "kimi_k3"):
            return parse_structured_calls(
                body, self.schemas, self.parser, len(self.calls)
            )
        try:
            calls = json.loads(body)
        except ValueError as exc:
            raise AgentProtocolError("Malformed or truncated JSON tool call") from exc
        if (
            self.parser == "llama3_json"
            and not block.startswith(self.start)
            and isinstance(calls, dict)
            and (
                "name" not in calls or not ({"arguments", "parameters"} & calls.keys())
            )
        ):
            return None
        calls = calls if isinstance(calls, list) else [calls]
        if not calls:
            raise AgentProtocolError("Empty tool calls")
        if not all(isinstance(c, dict) for c in calls):
            raise AgentProtocolError("Tool calls must be objects")
        return [
            make_tool_call(
                c.get("name"),
                c.get("arguments", c.get("parameters", {})),
                self.schemas,
                self.parser,
            )
            for c in calls
        ]

    def feed(self, text, final=False):
        self.buffer += text
        events = []
        while self.buffer:
            if (
                self.parser == "llama3_json"
                and self.at_start
                and self.mode == "content"
                and self.schemas
                and self.buffer.lstrip().startswith("{")
            ):
                self.buffer = self.buffer.lstrip()
                self.mode = "tool"
            if self.mode == "tool":
                end = (
                    self.buffer.find(self.end) + len(self.end)
                    if self.end
                    else len(self.buffer)
                )
                if self.parser == "minicpm5":
                    matches = re.finditer(
                        r"<!\[CDATA\[.*?(?:\]\]>|$)|</function>", self.buffer, re.S
                    )
                    end = next((m.end() for m in matches if m[0] == self.end), 0)
                if (self.end and end < len(self.end)) or (not self.end and not final):
                    break
                calls = self._parse_calls(self.buffer[:end])
                if calls is None:
                    events.append({"content": self.buffer[:end]})
                    calls = []
                if len(self.calls) + len(calls) > 1 and not self.parallel:
                    raise AgentProtocolError(
                        "Multiple calls with parallel_tool_calls=false"
                    )
                for call in calls:
                    self.calls.append(call)
                    events.append(
                        {"tool_calls": [dict(index=len(self.calls) - 1, **call)]}
                    )
                self.buffer = self.buffer[end:]
                self.mode = "content"
                self.at_start = False
                continue
            markers = [self.think_start, self.think_end, *self.ignored]
            if self.schemas:
                markers.append(self.start)
            found = [(self.buffer.find(m), m) for m in markers if m in self.buffer]
            if found:
                pos, marker = min(found)
                if pos:
                    events.append({self.mode: self.buffer[:pos]})
                    if self.buffer[:pos].strip():
                        self.at_start = False
                self.buffer = self.buffer[pos:]
                if marker == self.start:
                    if self.mode == "reasoning_content":
                        raise AgentProtocolError("Tool call inside reasoning")
                    self.mode = "tool"
                else:
                    self.buffer = self.buffer[len(marker) :]
                    if marker in self.ignored:
                        continue
                    self.mode = (
                        "reasoning_content" if marker == self.think_start else "content"
                    )
                continue
            # Hold only a suffix which could become a split marker.
            keep = max(
                [
                    n
                    for m in markers
                    for n in range(1, len(m))
                    if self.buffer.endswith(m[:n])
                ]
                or [0]
            )
            count = len(self.buffer) - keep
            if count:
                events.append({self.mode: self.buffer[:count]})
                if self.buffer[:count].strip():
                    self.at_start = False
                self.buffer = self.buffer[count:]
            break
        if final:
            if self.mode == "tool" or (
                self.buffer and self.schemas and self.start.startswith(self.buffer)
            ):
                raise AgentProtocolError("Truncated tool call")
            if self.buffer:
                events.append({self.mode: self.buffer})
                self.buffer = ""
            if self.choice not in ("auto", "none") and not self.calls:
                raise AgentProtocolError("Model did not satisfy tool_choice")
        return events

    def finish_reason(self, original):
        # Length/error/timeout always override a successfully parsed earlier call.
        if original in ("stop", "eos_token", "stop_string") and self.calls:
            return "tool_calls"
        return original
