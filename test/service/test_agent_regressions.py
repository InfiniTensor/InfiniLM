import json
from types import SimpleNamespace

import pytest
from infinilm.llm.llm import validate_request_capacity
from infinilm.server.inference_server import InferenceServer
from infinilm.server.openai_protocol import parse_tool_calls
from infinilm.server.tool_constraints import forced_bash_tool_prefix


def test_tool_description_compaction_preserves_full_schema(monkeypatch):
    from infinilm.server.tool_schema import compact_tool_descriptions

    monkeypatch.setenv("INFINILM_TOOL_DESCRIPTION_CHARS", "20")
    monkeypatch.setenv("INFINILM_PARAMETER_DESCRIPTION_CHARS", "10")
    tools = [{"type": "function", "function": {
        "name": "inspect", "description": "d" * 100,
        "parameters": {"type": "object", "required": ["description"], "properties": {
            "description": {"type": "string", "description": "p" * 100, "enum": ["a", "b"]}
        }, "$defs": {"value": {"type": "integer", "minimum": 1}}},
    }}]
    result = compact_tool_descriptions(tools)
    assert result[0]["function"]["name"] == "inspect"
    assert result[0]["function"]["description"] == "d" * 20 + "..."
    schema = result[0]["function"]["parameters"]
    assert schema["required"] == ["description"]
    assert schema["properties"]["description"] == {
        "type": "string", "description": "p" * 10 + "...", "enum": ["a", "b"]
    }
    assert schema["$defs"] == tools[0]["function"]["parameters"]["$defs"]
    assert tools[0]["function"]["description"] == "d" * 100


def test_tool_description_compaction_can_be_disabled(monkeypatch):
    from infinilm.server.tool_schema import compact_tool_descriptions

    monkeypatch.setenv("INFINILM_COMPACT_TOOL_DESCRIPTIONS", "0")
    tools = [{"function": {"description": "d" * 1000}}]
    assert compact_tool_descriptions(tools) is tools


def test_read_then_analyze_keeps_tools_and_model_followup():
    from infinilm.server.tool_constraints import constrain_tools, direct_read_result

    tools = [{"function": {"name": name}} for name in ("read_file", "bash", "edit_file")]
    messages = [{"role": "user", "content": "读取 outline.md，分析内容并修改 other.md。"}]
    assert constrain_tools(messages, tools) == tools
    messages += [
        {"role": "assistant", "tool_calls": [{
            "id": "read", "function": {"name": "read_file", "arguments": "{}"}
        }]},
        {"role": "tool", "tool_call_id": "read", "content": "# Outline"},
    ]
    assert direct_read_result(messages, tools) is None


def test_previous_turn_tool_result_is_not_reused():
    from infinilm.server.tool_constraints import direct_read_result

    tools = [{"function": {"name": "read_file"}}]
    messages = [
        {"role": "user", "content": "读取 previous.md。"},
        {"role": "assistant", "tool_calls": [{
            "id": "old", "function": {"name": "read_file", "arguments": "{}"}
        }]},
        {"role": "tool", "tool_call_id": "old", "content": "old contents"},
        {"role": "user", "content": "读取 current.md。"},
    ]
    assert direct_read_result(messages, tools) is None


def test_edit_shortcut_preserves_special_characters():
    from infinilm.server.tool_constraints import forced_edit_tool_prefix

    tools = [{"function": {"name": "edit_file"}}, {"function": {"name": "read_file"}}]
    messages = [
        {"role": "user", "content": "编辑 target.md，把 A&B 改为 A>B。完成后立即停止。"},
        {"role": "assistant", "tool_calls": [{
            "id": "read", "function": {"name": "read_file", "arguments": "{}"}
        }]},
        {"role": "tool", "tool_call_id": "read", "content": "A&B"},
    ]
    prefix = forced_edit_tool_prefix(messages, tools)
    _, calls = parse_tool_calls(prefix + "</tool_call>")
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "file_path": "target.md", "old_string": "A&B", "new_string": "A>B"
    }


def test_bash_command_does_not_include_stop_instruction_or_contract():
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    data = {
        "messages": [{"role": "user", "content": "执行 bash 命令 printf bash-ok > sample.txt。完成后立即停止。"}],
        "tools": tools,
    }
    _, _, request = InferenceServer.__new__(InferenceServer)._prepare_chat_request(data)
    prefix = request["_infinilm_forced_tool_prefix"]
    _, calls = parse_tool_calls(prefix + "</tool_call>")
    arguments = json.loads(calls[0]["function"]["arguments"])
    assert arguments == {"command": "printf bash-ok > sample.txt"}


def test_bash_command_preserves_shell_special_characters():
    tools = [{"type": "function", "function": {"name": "bash"}}]
    command = 'printf "a&b" > sample.txt; cat sample.txt'
    prefix = forced_bash_tool_prefix(
        [{"role": "user", "content": "执行 bash 命令 " + command + "。完成后立即停止。"}], tools
    )
    _, calls = parse_tool_calls(prefix + "</tool_call>")
    assert json.loads(calls[0]["function"]["arguments"])["command"] == command


def test_prompt_limit_rejects_before_forward(monkeypatch):
    monkeypatch.setenv("INFINILM_MAX_PROMPT_TOKENS", "6144")
    config = SimpleNamespace(cache_type="paged", num_blocks=192, block_size=64)
    with pytest.raises(ValueError, match="configured prompt limit is 6144"):
        validate_request_capacity(config, 6145, 512)
    validate_request_capacity(config, 6144, 512)
