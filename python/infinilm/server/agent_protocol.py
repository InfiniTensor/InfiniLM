"""Qwen3.5 XML tool protocol, independent of the native runtime.

Tools are emitted only after a complete, schema-validated call. Text and reasoning
stream incrementally; partial XML is never exposed as an executable tool call.
"""

import copy
import json
import re
import uuid

from jsonschema import Draft202012Validator, SchemaError, ValidationError
from referencing.exceptions import Unresolvable


class AgentProtocolError(ValueError):
    pass


def _object(value, label):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise AgentProtocolError(f"{label} must be a JSON object") from exc
    if not isinstance(value, dict):
        raise AgentProtocolError(f"{label} must be a JSON object")
    return value


def _check_refs(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("$ref", "$dynamicRef") and not str(child).startswith("#"):
                raise AgentProtocolError(
                    "Only local JSON Schema references are supported"
                )
            _check_refs(child)
    elif isinstance(value, list):
        for child in value:
            _check_refs(child)


def _function_name(fn, label="function"):
    name = fn.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name):
        raise AgentProtocolError(f"Invalid {label} name")
    return name


def prepare_agent_request(data, enabled):
    """Validate before SSE headers or GPU submission; preserve caller data."""
    data = copy.deepcopy(data)
    if not isinstance(data.get("messages", []), list):
        raise AgentProtocolError("messages must be an array")
    if data.get("stream_options") is None:
        data["stream_options"] = {}
    if not isinstance(data.get("stream_options", {}), dict):
        raise AgentProtocolError("stream_options must be an object")
    if not isinstance(data.get("stream", False), bool):
        raise AgentProtocolError("stream must be boolean")
    if not isinstance(data["stream_options"].get("include_usage", False), bool):
        raise AgentProtocolError("include_usage must be boolean")
    if not isinstance(data.get("parallel_tool_calls", True), bool):
        raise AgentProtocolError("parallel_tool_calls must be boolean")
    tools = data.get("tools")
    if tools is None:
        tools = []
    if not isinstance(tools, list):
        raise AgentProtocolError("tools must be an array")
    if tools and not enabled:
        raise AgentProtocolError("Enable --tool-call-parser qwen3_coder for tools")
    schemas = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise AgentProtocolError("Only function tools are supported")
        fn = _object(tool.get("function"), "function")
        name = _function_name(fn)
        if name in schemas:
            raise AgentProtocolError("Duplicate function name")
        if fn.get("strict"):
            raise AgentProtocolError(
                "strict=true requires constrained decoding, which is not supported"
            )
        schema = fn.get("parameters", {"type": "object"})
        if not isinstance(schema, dict):
            raise AgentProtocolError("parameters must be a JSON Schema object")
        _check_refs(schema)
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise AgentProtocolError("Invalid tool parameter schema") from exc
        schemas[name] = schema
    choice = data.get("tool_choice", "auto")
    if choice is None:
        choice = "auto"
    selected = None
    if isinstance(choice, dict):
        if choice.get("type") != "function":
            raise AgentProtocolError("Unsupported tool_choice")
        selected = _object(choice.get("function"), "tool_choice.function").get("name")
        if selected not in schemas:
            raise AgentProtocolError("tool_choice names an unavailable function")
    elif choice not in ("auto", "none", "required"):
        raise AgentProtocolError("Unsupported tool_choice")
    if choice == "required" and not schemas:
        raise AgentProtocolError("tool_choice=required needs tools")
    if choice == "none":
        schemas = {}
        tools = []
    elif selected:
        tools = [t for t in tools if t["function"]["name"] == selected]
        schemas = {selected: schemas[selected]}
    kwargs = data.get("chat_template_kwargs")
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise AgentProtocolError("chat_template_kwargs must be an object")
    if "tools" in kwargs:
        raise AgentProtocolError("Pass tools at request top level")
    if enabled:
        kwargs["tools"] = tools
        if "reasoning_effort" in data:
            kwargs["reasoning_effort"] = data["reasoning_effort"]
        if kwargs.get("enable_thinking", True) and kwargs.get(
            "reasoning_effort", "xhigh"
        ) not in ("low", "medium", "xhigh"):
            raise AgentProtocolError("reasoning_effort must be low, medium, or xhigh")
        seen, pending = set(), {}
        for message in data.get("messages", []):
            if not isinstance(message, dict):
                raise AgentProtocolError("messages must contain objects")
            message.setdefault("content", "")
            if message.get("content") is None:
                message["content"] = ""
            role = message.get("role")
            if role == "tool":
                call_id = message.get("tool_call_id")
                if not isinstance(call_id, str) or call_id not in pending:
                    raise AgentProtocolError("Tool result has no pending tool_call_id")
                name = pending.pop(call_id)
                if message.get("name", name) != name:
                    raise AgentProtocolError("Tool result name does not match its call")
                message["name"] = name
            elif pending:
                raise AgentProtocolError("Missing tool results before next message")
            calls = message.get("tool_calls")
            if calls is None:
                calls = []
            if not isinstance(calls, list) or any(
                not isinstance(c, dict) for c in calls
            ):
                raise AgentProtocolError("tool_calls must be an array of objects")
            if calls and role != "assistant":
                raise AgentProtocolError("tool_calls require assistant role")
            for call in calls:
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in seen:
                    raise AgentProtocolError(
                        "Tool call IDs must be nonempty and unique"
                    )
                if call.get("type") != "function":
                    raise AgentProtocolError("Only function tool calls are supported")
                fn = _object(call.get("function"), "tool_call.function")
                name = _function_name(fn, "historical function")
                fn["arguments"] = _object(fn.get("arguments", {}), "tool arguments")
                seen.add(call_id)
                pending[call_id] = name
        if pending:
            raise AgentProtocolError("Missing tool results")
    data["chat_template_kwargs"] = kwargs
    return data, schemas, selected or choice


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


def parse_xml_call(block, schemas):
    match = re.fullmatch(
        r"<tool_call>\s*<function=([^<>\s]+)>(.*?)</function>\s*</tool_call>",
        block,
        re.S,
    )
    if not match or match[1] not in schemas:
        raise AgentProtocolError("Malformed or unknown tool call")
    name, body = match[1], match[2]
    schema = schemas[name]
    args, offset = {}, 0
    for param in re.finditer(r"<parameter=([^<>\s]+)>(.*?)</parameter>", body, re.S):
        if body[offset : param.start()].strip() or param[1] in args:
            raise AgentProtocolError("Malformed or duplicate tool parameter")
        key, raw = param[1], _trim_newline(param[2])
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
    try:
        Draft202012Validator(schema).validate(args)
    except ValidationError as exc:
        raise AgentProtocolError(
            f"Invalid arguments for {name}: {exc.message}"
        ) from exc
    try:
        encoded_args = json.dumps(args, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        raise AgentProtocolError(
            "Tool arguments must contain finite JSON values"
        ) from exc
    return {
        "id": "call_" + uuid.uuid4().hex,
        "type": "function",
        "function": {
            "name": name,
            "arguments": encoded_args,
        },
    }


class QwenAgentParser:
    """A bounded-prefix scanner; accepts arbitrary token/chunk boundaries."""

    def __init__(self, schemas, thinking=True, choice="auto", parallel=True):
        self.schemas = schemas
        self.mode = "reasoning_content" if thinking else "content"
        self.buffer = ""
        self.calls = []
        self.choice = choice
        self.parallel = parallel

    def feed(self, text, final=False):
        self.buffer += text
        events = []
        while self.buffer:
            if self.mode == "tool":
                end = self.buffer.find("</tool_call>")
                if end < 0:
                    break
                end += len("</tool_call>")
                call = parse_xml_call(self.buffer[:end], self.schemas)
                if self.calls and not self.parallel:
                    raise AgentProtocolError(
                        "Multiple calls with parallel_tool_calls=false"
                    )
                self.calls.append(call)
                events.append({"tool_calls": [dict(index=len(self.calls) - 1, **call)]})
                self.buffer = self.buffer[end:]
                self.mode = "content"
                continue
            markers = ["<think>", "</think>"]
            if self.schemas:
                markers.append("<tool_call>")
            found = [(self.buffer.find(m), m) for m in markers if m in self.buffer]
            if found:
                pos, marker = min(found)
                if pos:
                    events.append({self.mode: self.buffer[:pos]})
                self.buffer = self.buffer[pos:]
                if marker == "<tool_call>":
                    if self.mode == "reasoning_content":
                        raise AgentProtocolError("Tool call inside reasoning")
                    self.mode = "tool"
                else:
                    self.buffer = self.buffer[len(marker) :]
                    self.mode = (
                        "reasoning_content" if marker == "<think>" else "content"
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
                self.buffer = self.buffer[count:]
            break
        if final:
            if self.mode == "tool" or (
                self.buffer and self.schemas and "<tool_call>".startswith(self.buffer)
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


def collect_message(events):
    result = {"role": "assistant", "content": ""}
    for event in events:
        for key, value in event.items():
            if key == "role":
                result[key] = value
            elif key == "tool_calls":
                result.setdefault(key, []).extend(
                    {k: v for k, v in c.items() if k != "index"} for c in value
                )
            else:
                result[key] = result.get(key, "") + value
    if result.get("tool_calls") and not result["content"].strip():
        result["content"] = None
    return result
