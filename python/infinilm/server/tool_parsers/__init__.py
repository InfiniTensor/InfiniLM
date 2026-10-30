"""Shared scanners for native tool-call formats; no tool execution."""

import html
import json
import re

from referencing.exceptions import Unresolvable

from ..agent_protocol import AgentProtocolError, make_tool_call, schema_validator
from .structured import members, parse_structured_calls

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
        return schema_validator(root).evolve(schema=spec).is_valid(raw)
    except Unresolvable:
        # A property's own resource scope may be needed; defer to the full schema.
        return False


def _resolve_xml_arguments(args, raw_args, schema):
    # Object constraints may couple fields. Keep alternatives independent and
    # bound the search instead of committing a failed branch's conversions.
    validator = schema_validator(schema)
    values = {}
    for key, value in args.items():
        if value != raw_args[key]:
            values[key] = raw_args[key]
        else:
            try:
                values[key] = json.loads(raw_args[key])
            except ValueError:
                pass
    bits = {key: 1 << index for index, key in enumerate(values)}

    def error_keys(error):
        path = list(error.absolute_path)
        if path and path[0] in bits:
            yield path[0]
        elif not path:
            yield from bits
        for child in error.context:
            yield from error_keys(child)

    def alternatives(mask, keys):
        combined = mask
        for key in keys:
            combined |= bits[key]
        yield combined
        for key in keys:
            yield mask | bits[key]
        # Conditionals can depend on a different field than the error names.
        for key in bits:
            yield mask | bits[key]

    try:
        pending, seen = [iter((0,))], set()
        while pending:
            mask = next(pending[-1], None)
            if mask is None:
                pending.pop()
                continue
            if mask in seen:
                continue
            if len(seen) >= 256:
                raise AgentProtocolError(
                    "XML argument type inference exceeded its schema search limit"
                )
            seen.add(mask)
            candidate = {
                key: values[key] if mask & bits.get(key, 0) else value
                for key, value in args.items()
            }
            errors = list(validator.iter_errors(candidate))
            if not errors:
                try:
                    json.dumps(candidate, allow_nan=False)
                except ValueError:
                    pass
                else:
                    args.update(candidate)
                    return
            keys = dict.fromkeys(key for error in errors for key in error_keys(error))
            # Try the combined correction first, then backtrack to individual fields.
            pending.append(alternatives(mask, keys))
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
    args, raw_args = {}, {}
    for param in members(body, parameter):
        if param[1] in args:
            raise AgentProtocolError("Duplicate tool parameter")
        key, raw = param[1], param[2]
        if parser == "minicpm5" and raw.startswith("<![CDATA[") and raw.endswith("]]>"):
            raw = raw[9:-3]
        else:
            if parser != "glm5":
                raw = _trim_newline(raw)
            if parser == "minicpm5":
                raw = html.unescape(raw)
        raw_args[key] = raw
        spec = schema.get("properties", {}).get(key, {})
        # The template writes strings verbatim and all other values as JSON.
        args[key] = raw
        if not _accepts_string(spec, schema, raw):
            try:
                args[key] = json.loads(raw)
            except ValueError:
                pass
    _resolve_xml_arguments(args, raw_args, schema)
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
            if self.parser == "llama3_json" and not block.startswith(self.start):
                try:
                    calls, _ = json.JSONDecoder().raw_decode(body.lstrip())
                except ValueError:
                    return None
                if (
                    not isinstance(calls, dict)
                    or "name" not in calls
                    or not ({"arguments", "parameters"} & calls.keys())
                ):
                    return None
            calls = json.loads(body)
        except ValueError as exc:
            raise AgentProtocolError("Malformed or truncated JSON tool call") from exc
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

    def feed(self, text, final=False, finish_reason="stop"):
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
                self.mode = "tool"
            if self.mode == "tool":
                end = (
                    self.buffer.find(self.end) + len(self.end)
                    if self.end
                    else len(self.buffer)
                )
                if self.parser == "hermes":
                    body = self.buffer[len(self.start) :].lstrip()
                    try:
                        _, size = json.JSONDecoder().raw_decode(body)
                    except ValueError:
                        if not final:
                            break
                        raise AgentProtocolError(
                            "Malformed or truncated JSON tool call"
                        )
                    end = len(self.buffer) - len(body) + size
                    tail = self.buffer[end:].lstrip()
                    end = len(self.buffer) - len(tail)
                    if not tail.startswith(self.end):
                        if not final and self.end.startswith(tail):
                            break
                        raise AgentProtocolError(
                            "Malformed or truncated JSON tool call"
                        )
                    end += len(self.end)
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
            if self.mode == "tool":
                raise AgentProtocolError("Truncated tool call")
            if self.buffer:
                events.append({self.mode: self.buffer})
                self.buffer = ""
            if (
                finish_reason in ("stop", "eos_token", "stop_string")
                and self.choice not in ("auto", "none")
                and not self.calls
            ):
                raise AgentProtocolError("Model did not satisfy tool_choice")
        return events

    def finish_reason(self, original):
        # Length/error/timeout always override a successfully parsed earlier call.
        if original in ("stop", "eos_token", "stop_string") and self.calls:
            return "tool_calls"
        return original
