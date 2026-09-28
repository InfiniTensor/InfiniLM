import json
from types import SimpleNamespace

import pytest
from infinilm.llm.llm import validate_request_capacity
from infinilm.server.inference_server import InferenceServer
from infinilm.server.openai_protocol import parse_tool_calls
from infinilm.server.tool_constraints import forced_bash_tool_prefix


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
        [{"role": "user", "content": "执行 bash 命令 " + command}], tools
    )
    _, calls = parse_tool_calls(prefix + "</tool_call>")
    assert json.loads(calls[0]["function"]["arguments"])["command"] == command


def test_prompt_limit_rejects_before_forward(monkeypatch):
    monkeypatch.setenv("INFINILM_MAX_PROMPT_TOKENS", "6144")
    config = SimpleNamespace(cache_type="paged", num_blocks=192, block_size=64)
    with pytest.raises(ValueError, match="configured prompt limit is 6144"):
        validate_request_capacity(config, 6145, 512)
    validate_request_capacity(config, 6144, 512)
