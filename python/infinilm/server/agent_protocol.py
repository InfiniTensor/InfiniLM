"""Tool protocols, independent of the native runtime.

Tools are emitted only after a complete, schema-validated call. Text and reasoning
stream incrementally; partial XML is never exposed as an executable tool call.
"""

import copy
import json
import re
import uuid

from jsonschema import Draft202012Validator, SchemaError, ValidationError


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


def prepare_agent_request(data, parser="none"):
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
    if tools and parser == "none":
        raise AgentProtocolError("Enable `--tool-call-parser` for tools")
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
    if parser != "none":
        kwargs["tools"] = tools or None
        if "reasoning_effort" in data:
            kwargs["reasoning_effort"] = data["reasoning_effort"]
        if parser == "kimi_k3":
            if "enable_thinking" in kwargs:
                kwargs["thinking"] = kwargs.pop("enable_thinking")
            if "reasoning_effort" in kwargs:
                kwargs["thinking_effort"] = kwargs.pop("reasoning_effort")
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
                if parser == "mistral" and not re.fullmatch(r"[A-Za-z0-9]{9}", call_id):
                    raise AgentProtocolError(
                        "Mistral call IDs must be nine alphanumeric characters"
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


def make_tool_call(name, args, schemas, parser="qwen3_coder"):
    if not isinstance(name, str) or name not in schemas:
        raise AgentProtocolError("Unknown tool call")
    args = _object(args, "tool arguments")
    try:
        Draft202012Validator(schemas[name]).validate(args)
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
        "id": uuid.uuid4().hex[:9]
        if parser == "mistral"
        else "call_" + uuid.uuid4().hex,
        "type": "function",
        "function": {
            "name": name,
            "arguments": encoded_args,
        },
    }


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
