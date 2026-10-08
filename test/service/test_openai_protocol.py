import asyncio
import json

import pytest
from infinilm.processors.processor import normalize_openai_messages
from infinilm.server.inference_server import InferenceServer, completion_json
from infinilm.server.openai_protocol import (
    ToolCallStreamParser,
    parse_tool_calls,
    select_tool_calls,
    strip_reasoning_markers,
)
from infinilm.server.tool_constraints import (
    constrain_tools,
    direct_bash_arguments,
    direct_bash_result,
    direct_edit_complete,
    direct_edit_step,
    direct_literal_write_arguments,
    direct_read_arguments,
    direct_read_result,
    forced_bash_tool_prefix,
    forced_edit_tool_prefix,
    forced_read_for_edit_prefix,
    forced_read_tool_prefix,
    forced_write_tool_prefix,
)
from infinilm.server.tool_contract import apply_tool_contract


def test_parse_glm_native_arguments_preserves_paths_and_values():
    path = r"\\wsl.localhost\Ubuntu\home\wei\Projects\main\InfiniCore"
    content, calls = parse_tool_calls(
        "Inspecting.\n<tool_call>glob<arg_key>pattern</arg_key>"
        "<arg_value>**/*.py</arg_value><arg_key>path</arg_key>"
        f"<arg_value>{path}</arg_value><arg_key>limit</arg_key>"
        "<arg_value>20</arg_value></tool_call>",
        {"glob"},
    )
    assert content == "Inspecting."
    assert calls[0]["function"]["name"] == "glob"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "pattern": "**/*.py",
        "path": path,
        "limit": 20,
    }


def test_stream_glm_native_every_character_and_parallel_calls():
    parser = ToolCallStreamParser({"read_file"})
    source = (
        "Checking. <tool_call>read_file<arg_key>file_path</arg_key>"
        "<arg_value>a.py</arg_value></tool_call>"
        "<tool_call>read_file<arg_key>file_path</arg_key>"
        "<arg_value>b.py</arg_value></tool_call>"
    )
    content, calls = [], []
    for char in source:
        text, parsed = parser.feed(char)
        content.extend(text)
        calls.extend(parsed)
    text, parsed = parser.finalize()
    content.extend(text)
    calls.extend(parsed)
    assert "".join(content) == "Checking. "
    assert parser.has_tool_calls
    assert [
        json.loads(call["function"]["arguments"])["file_path"] for call in calls
    ] == ["a.py", "b.py"]


def test_parallel_tool_calls_false_keeps_only_first_call_across_stream_chunks():
    parser = ToolCallStreamParser({"read_file"})
    emitted = []
    for block in (
        "<tool_call>read_file<arg_key>file_path</arg_key>"
        "<arg_value>a.py</arg_value></tool_call>",
        "<tool_call>read_file<arg_key>file_path</arg_key>"
        "<arg_value>b.py</arg_value></tool_call>",
    ):
        _, calls = parser.feed(block)
        emitted.extend(select_tool_calls(calls, len(emitted), False))
    assert len(emitted) == 1
    assert json.loads(emitted[0]["function"]["arguments"]) == {"file_path": "a.py"}
    assert len(select_tool_calls([{}, {}], 0, False)) == 1
    assert len(select_tool_calls([{}, {}], 0, True)) == 2
    assert len(select_tool_calls([{}, {}], 0, None)) == 2
    InferenceServer._validate_request({"parallel_tool_calls": None})
    with pytest.raises(
        ValueError, match="parallel_tool_calls must be a boolean or null"
    ):
        InferenceServer._validate_request({"parallel_tool_calls": "false"})


@pytest.mark.parametrize(
    "source",
    [
        "<tool_call>glob<arg_key>path</arg_key><arg_value>x</tool_call>",
        "<tool_call>glob<arg_key>path</arg_key><arg_value>x</arg_value>",
        "<tool_call>glob<arg_key>path</arg_key><arg_value>x</arg_value><arg_key>path</arg_key><arg_value>y</arg_value></tool_call>",
        "<tool_call>glob<arg_key>path</arg_key><arg_value>x</arg_value>garbage</tool_call>",
        "<tool_call>not_exposed<arg_key>path</arg_key><arg_value>x</arg_value></tool_call>",
    ],
)
def test_glm_native_rejects_incomplete_duplicate_or_unexposed_calls(source):
    parser = ToolCallStreamParser({"glob"})
    content, calls = parser.feed(source)
    final_content, final_calls = parser.finalize()
    assert not content + final_content
    assert not calls + final_calls


@pytest.mark.parametrize(
    "unfinished",
    [
        "<tool_call><function=write_file><parameter=file_path>outline.md",
        "<tool_call>write_file<arg_key>file_path</arg_key><arg_value>outline.md",
        "<tool_",
    ],
)
def test_nonstream_parser_hides_truncated_tool_call(unfinished):
    content, calls = parse_tool_calls("Visible. " + unfinished, {"write_file"})
    assert content == "Visible."
    assert calls == []


def test_nonstream_parser_hides_truncated_tail_after_complete_call():
    content, calls = parse_tool_calls(
        "Before. <tool_call>read_file<arg_key>file_path</arg_key>"
        "<arg_value>a.py</arg_value></tool_call> After. "
        "<tool_call>write_file<arg_key>file_path</arg_key><arg_value>",
        {"read_file", "write_file"},
    )
    assert content == "Before.  After."
    assert [call["function"]["name"] for call in calls] == ["read_file"]


def test_glm_native_preserves_multiline_content_and_json_values():
    content = "  first line\nlast line  "
    _, calls = parse_tool_calls(
        "<tool_call>write_file<arg_key>content</arg_key>"
        f"<arg_value>{content}</arg_value><arg_key>flags</arg_key>"
        '<arg_value>{"ok":true,"n":null}</arg_value></tool_call>'
    )
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "content": content,
        "flags": {"ok": True, "n": None},
    }


def test_parse_qwen_tool_call_xml():
    content, calls = parse_tool_calls(
        "checking\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nBeijing\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    assert content == "checking"
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "city": "Beijing",
        "days": 3,
    }


def test_parse_json_tool_call_xml():
    content, calls = parse_tool_calls(
        '<tool_call>{"name":"lookup","arguments":{"query":"InfiniLM"}}</tool_call>'
    )

    assert content is None
    assert calls[0]["function"]["name"] == "lookup"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "InfiniLM"}


def test_stream_parser_handles_split_markers_without_leaking_xml():
    parser = ToolCallStreamParser()
    content_parts = []
    calls = []
    chunks = [
        "I will check.\n<tool_",
        "call>\n<function=lookup>\n<parameter=query>\n",
        "PilotDeck\n</parameter>\n</function>\n</tool_call>",
    ]
    for chunk in chunks:
        content, parsed = parser.feed(chunk)
        content_parts.extend(content)
        calls.extend(parsed)
    content, parsed = parser.finalize()
    content_parts.extend(content)
    calls.extend(parsed)

    assert "".join(content_parts) == "I will check.\n"
    assert parser.has_tool_calls
    assert calls[0]["function"]["name"] == "lookup"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "PilotDeck"}


def test_normalize_openai_tool_call_history_arguments():
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city":"Beijing"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]

    normalized = normalize_openai_messages(messages)

    assert normalized[1]["content"] is None
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == {
        "city": "Beijing"
    }
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == (
        '{"city":"Beijing"}'
    )


def test_invalid_tool_history_arguments_are_rejected():
    with pytest.raises(ValueError, match="valid JSON"):
        normalize_openai_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "bad", "arguments": "{"}}],
                }
            ]
        )


def test_openai_completion_contains_native_tool_calls():
    _, calls = parse_tool_calls("<tool_call><function=lookup></function></tool_call>")
    response = completion_json("cmpl-test", None, tool_calls=calls)

    assert response["choices"][0]["message"]["content"] is None
    assert response["choices"][0]["message"]["tool_calls"] == calls


def test_pilotdeck_request_fields_become_template_kwargs():
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    kwargs = InferenceServer._build_chat_template_kwargs(
        {
            "tools": tools,
            "tool_choice": "auto",
            "reasoning_effort": "low",
            "thinking": {"type": "disabled"},
        }
    )

    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["enable_thinking"] is False


def test_parse_multiple_qwen_tool_calls():
    content, calls = parse_tool_calls(
        "<tool_call><function=lookup><parameter=query>one</parameter>"
        "</function></tool_call>\n"
        "<tool_call><function=lookup><parameter=query>two</parameter>"
        "</function></tool_call>"
    )

    assert content is None
    assert [call["function"]["name"] for call in calls] == ["lookup", "lookup"]
    assert [json.loads(call["function"]["arguments"])["query"] for call in calls] == [
        "one",
        "two",
    ]


def test_forced_tool_choice_filters_template_tools():
    tools = [
        {"type": "function", "function": {"name": "first", "parameters": {}}},
        {"type": "function", "function": {"name": "second", "parameters": {}}},
    ]
    kwargs = InferenceServer._build_chat_template_kwargs(
        {
            "tools": tools,
            "tool_choice": {
                "type": "function",
                "function": {"name": "second"},
            },
        }
    )

    assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["second"]


def test_max_completion_tokens_alias():
    server = InferenceServer("/tmp/model", max_tokens=99)

    params = server._build_sampling_params({"max_completion_tokens": 17})

    assert params.max_tokens == 17


def test_ignore_eos_request_overrides_server_default():
    server = InferenceServer("/tmp/model", ignore_eos=False)
    assert not server._build_sampling_params({}).ignore_eos
    assert server._build_sampling_params({"ignore_eos": True}).ignore_eos
    assert server._build_sampling_params(
        {"sampling_params": {"ignore_eos": True}}
    ).ignore_eos

    server = InferenceServer("/tmp/model", ignore_eos=True)
    assert server._build_sampling_params({}).ignore_eos
    assert not server._build_sampling_params({"ignore_eos": False}).ignore_eos


def _tool(name):
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


def test_pilotdeck_chinese_constraints_filter_tool_schemas():
    tools = [
        _tool("write_file"),
        _tool("edit_file"),
        _tool("bash"),
        _tool("read_file"),
        _tool("read_skill"),
        _tool("web_search"),
        _tool("agent"),
    ]
    messages = [
        {
            "role": "user",
            "content": (
                "把大纲写入 outline.md。约束：禁止 read_skill / WebSearch / "
                "Agent；仅用 Write/Edit/Bash/Read；写完立即停止。"
            ),
        }
    ]

    assert [tool["function"]["name"] for tool in constrain_tools(messages, tools)] == [
        "write_file"
    ]


def test_direct_create_file_then_stop_filters_to_write_tool():
    tools = [_tool("write_file"), _tool("read_file"), _tool("bash")]
    messages = [
        {
            "role": "user",
            "content": ("创建 outline.md，内容为验收大纲。写完后立即停止。"),
        }
    ]

    assert [tool["function"]["name"] for tool in constrain_tools(messages, tools)] == [
        "write_file"
    ]


def test_read_file_request_filters_to_read_tool():
    tools = [_tool("write_file"), _tool("read_file"), _tool("bash")]
    messages = [{"role": "user", "content": "读取 outline.md。"}]

    assert [tool["function"]["name"] for tool in constrain_tools(messages, tools)] == [
        "read_file"
    ]

    prefix = forced_read_tool_prefix(messages, [_tool("read_file")])
    assert prefix is not None
    assert "file_path" in prefix
    assert "outline.md" in prefix


def test_edit_file_request_filters_to_edit_and_read_tools():
    tools = [_tool("write_file"), _tool("read_file"), _tool("edit_file"), _tool("bash")]
    messages = [
        {
            "role": "user",
            "content": "编辑 outline.md，把 pending 改成 done。完成后立即停止。",
        }
    ]

    assert {tool["function"]["name"] for tool in constrain_tools(messages, tools)} == {
        "edit_file",
        "read_file",
    }


def test_run_command_request_filters_to_command_tools():
    tools = [_tool("write_file"), _tool("read_file"), _tool("edit_file"), _tool("bash")]
    messages = [
        {"role": "user", "content": "执行 bash 命令 printf ok。完成后立即停止。"}
    ]

    assert {tool["function"]["name"] for tool in constrain_tools(messages, tools)} == {
        "bash",
        "write_file",
        "read_file",
    }


def test_direct_write_then_stop_removes_tools_after_success():
    tools = [_tool("write_file"), _tool("read_file")]
    messages = [
        {
            "role": "user",
            "content": "把结果写入 outline.md。写完 outline.md 后立即停止。",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path":"outline.md","content":"ok"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Successfully wrote outline.md",
        },
    ]

    assert constrain_tools(messages, tools) == []


def test_direct_write_then_stop_allows_retry_after_failed_write():
    tools = [_tool("write_file"), _tool("read_file")]
    messages = [
        {
            "role": "user",
            "content": "把结果写入 outline.md。写完 outline.md 后立即停止。",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "write_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "TOOL_ERROR: permission denied",
        },
    ]

    assert [tool["function"]["name"] for tool in constrain_tools(messages, tools)] == [
        "write_file"
    ]


def test_repeated_identical_calls_remove_the_tool_from_later_rounds():
    tools = [_tool("bash"), _tool("write_file")]
    call = {
        "id": "call_1",
        "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
    }
    messages = [
        {"role": "user", "content": "Inspect and write the result."},
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "assistant", "tool_calls": [{**call, "id": "call_2"}]},
        {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
    ]

    assert [tool["function"]["name"] for tool in constrain_tools(messages, tools)] == [
        "write_file"
    ]


def test_parser_rejects_tool_calls_not_exposed_for_this_request():
    content, calls = parse_tool_calls(
        "<tool_call><function=read_skill><parameter=name>x</parameter>"
        "</function></tool_call>",
        allowed_tool_names={"write_file"},
    )

    assert content is None
    assert calls == []


def test_filtered_tool_schema_adds_explicit_system_contract():
    messages = [
        {"role": "system", "content": "agent instructions"},
        {"role": "user", "content": "write the result"},
    ]
    result = apply_tool_contract(
        messages,
        [_tool("write_file"), _tool("bash")],
        [_tool("write_file")],
    )

    assert "`write_file`" in result[0]["content"]
    assert "`bash`" not in result[1]["content"]
    assert messages[0]["content"] == "agent instructions"


def test_unfiltered_tool_schema_does_not_modify_messages():
    messages = [{"role": "user", "content": "write the result"}]
    tools = [_tool("write_file")]

    assert apply_tool_contract(messages, tools, tools) is messages


def test_stream_parser_hides_rejected_complete_tool_block():
    parser = ToolCallStreamParser(allowed_tool_names={"write_file"})
    content_parts, calls = parser.feed(
        "<tool_call><function=read_file>"
        "<parameter=file_path>x</parameter></function></tool_call>"
    )
    final_content, final_calls = parser.finalize()
    content_parts.extend(final_content)
    calls.extend(final_calls)

    assert content_parts == []
    assert calls == []


def test_constraints_survive_pilotdeck_recovery_messages():
    tools = [_tool("write_file"), _tool("read_file"), _tool("bash")]
    messages = [
        {
            "role": "user",
            "content": "把结果写入 outline.md。写完 outline.md 后立即停止。",
        },
        {"role": "assistant", "content": None},
        {
            "role": "user",
            "content": (
                "Your previous response was empty (thinking only, no visible "
                "text). Please provide your answer as visible text output."
            ),
        },
    ]

    assert [tool["function"]["name"] for tool in constrain_tools(messages, tools)] == [
        "write_file"
    ]


def test_tool_contract_is_repeated_for_current_user_round():
    messages = [
        {"role": "system", "content": "agent instructions"},
        {"role": "user", "content": "write the result"},
    ]
    result = apply_tool_contract(
        messages,
        [_tool("write_file"), _tool("bash")],
        [_tool("write_file")],
    )

    assert result[0]["content"].count("<InfiniLM tool contract>") == 1
    assert result[1]["content"].count("<InfiniLM tool contract>") == 1


def test_forced_write_prefix_constrains_generation_to_write_file():
    messages = [{"role": "user", "content": "把结果写入 outline.md。写完后立即停止。"}]
    prefix = forced_write_tool_prefix(messages, [_tool("write_file")])

    assert prefix == (
        "<tool_call>\n<function=write_file>\n<parameter=file_path>\n"
        "outline.md\n</parameter>\n<parameter=content>\n"
    )

    parser = ToolCallStreamParser(allowed_tool_names={"write_file"})
    parser.feed(prefix)
    content_parts, calls = parser.feed(
        "outline content\n</parameter>\n</function>\n</tool_call>"
    )
    final_content, final_calls = parser.finalize()
    content_parts.extend(final_content)
    calls.extend(final_calls)

    assert content_parts == []
    assert calls[0]["function"]["name"] == "write_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "file_path": "outline.md",
        "content": "outline content",
    }


def test_explicit_multiline_write_preserves_content_and_stream_protocol():
    prompt = (
        "创建 offline.md，内容为两行：# Offline PilotDeck\n"
        "status=pending。写完后立即停止。"
    )
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "tools": [_tool("write_file"), _tool("read_file")],
    }
    arguments = {
        "file_path": "offline.md",
        "content": "# Offline PilotDeck\nstatus=pending",
    }
    assert (
        direct_literal_write_arguments(data["messages"], [_tool("write_file")])
        == arguments
    )

    server = InferenceServer("/tmp/model", skip_load=True)
    _, kwargs, prepared = server._prepare_chat_request(data)
    assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["write_file"]
    assert prepared["_infinilm_direct_tool_call"] == {
        "name": "write_file",
        "arguments": arguments,
    }
    assert "_infinilm_forced_tool_prefix" not in prepared

    response = asyncio.run(server._chat("write-test", data, None))
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert (
        json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"])
        == arguments
    )

    async def stream():
        return [
            chunk async for chunk in server._stream_chat("write-stream", data, None)
        ]

    chunks = asyncio.run(stream())
    events = [json.loads(chunk[6:]) for chunk in chunks if chunk.startswith("data: {")]
    call = events[1]["choices"][0]["delta"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == arguments
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1] == "data: [DONE]\n\n"


def test_explicit_bash_uses_exact_command_and_tool_output_cap(monkeypatch):
    command = 'printf "a&b" > sample.txt; cat sample.txt'
    data = {
        "messages": [
            {
                "role": "user",
                "content": "执行 bash 命令 " + command + "。完成后立即停止。",
            }
        ],
        "tools": [_tool("bash"), _tool("write_file")],
        "max_tokens": 4096,
    }
    monkeypatch.setenv("INFINILM_TOOL_MAX_NEW_TOKENS", "1024")
    server = InferenceServer("/tmp/model", skip_load=True)
    assert direct_bash_arguments(data["messages"], data["tools"]) == {
        "command": command
    }
    assert server._build_sampling_params(data).max_tokens == 1024
    assert server._build_sampling_params({"max_tokens": 4096}).max_tokens == 4096
    _, _, prepared = server._prepare_chat_request(data)
    assert prepared["_infinilm_direct_tool_call"] == {
        "name": "bash",
        "arguments": {"command": command},
    }
    assert "_infinilm_forced_tool_prefix" not in prepared

    response = asyncio.run(server._chat("bash-test", data, None))
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "command": command
    }

    async def stream():
        return [chunk async for chunk in server._stream_chat("bash-stream", data, None)]

    chunks = asyncio.run(stream())
    events = [json.loads(chunk[6:]) for chunk in chunks if chunk.startswith("data: {")]
    call = events[1]["choices"][0]["delta"]["tool_calls"][0]
    assert call["function"]["name"] == "bash"
    assert json.loads(call["function"]["arguments"]) == {"command": command}
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"

    completed = {
        **data,
        "messages": data["messages"]
        + [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "bash-1",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": command}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "bash-1",
                "content": "BASH_RESULT[success] a&b",
            },
        ],
    }
    _, _, prepared = server._prepare_chat_request(completed)
    assert "_infinilm_direct_tool_call" not in prepared
    finished = asyncio.run(server._chat("bash-done", completed, None))
    assert finished["choices"][0]["finish_reason"] == "stop"
    assert finished["choices"][0]["message"].get("tool_calls") is None


def test_explicit_edit_reads_once_then_edits_once():
    prompt = "编辑 sample.md，把 status=pending 改为 status=done。完成后立即停止。"
    tools = [_tool("read_file"), _tool("edit_file"), _tool("bash")]
    messages = [{"role": "user", "content": prompt}]
    server = InferenceServer("/tmp/model", skip_load=True)
    _, _, first = server._prepare_chat_request({"messages": messages, "tools": tools})
    assert first["_infinilm_direct_tool_call"] == {
        "name": "read_file",
        "arguments": {"file_path": "sample.md"},
    }
    messages += [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "read-1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path":"sample.md"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "read-1",
            "content": "# Sample\nstatus=pending",
        },
    ]
    _, _, second = server._prepare_chat_request({"messages": messages, "tools": tools})
    assert second["_infinilm_direct_tool_call"] == {
        "name": "edit_file",
        "arguments": {
            "file_path": "sample.md",
            "old_string": "status=pending",
            "new_string": "status=done",
        },
    }
    assert (
        direct_edit_step(messages, [_tool("read_file"), _tool("edit_file")])
        == (second["_infinilm_direct_tool_call"])
    )
    messages += [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "edit-1",
                    "function": {
                        "name": "edit_file",
                        "arguments": '{"file_path":"sample.md","old_string":"status=pending","new_string":"status=done"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "edit-1", "content": "Updated sample.md"},
    ]
    data = {"messages": messages, "tools": tools}
    _, _, finished = server._prepare_chat_request(data)
    assert finished["_infinilm_direct_edit_complete"] is True
    assert "_infinilm_direct_tool_call" not in finished
    response = asyncio.run(server._chat("edit-done", data, None))
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["choices"][0]["message"].get("tool_calls") is None


def test_constrained_read_calls_only_read_file_and_stops():
    prompt = "仅允许使用 read_file，禁止 bash。读取 sample.md。"
    tools = [_tool("read_file"), _tool("bash")]
    messages = [{"role": "user", "content": prompt}]
    server = InferenceServer("/tmp/model", skip_load=True)
    _, kwargs, first = server._prepare_chat_request(
        {"messages": messages, "tools": tools}
    )
    assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["read_file"]
    assert direct_read_arguments(messages, kwargs["tools"]) == {
        "file_path": "sample.md"
    }
    assert first["_infinilm_direct_tool_call"] == {
        "name": "read_file",
        "arguments": {"file_path": "sample.md"},
    }
    response = asyncio.run(
        server._chat("read-first", {"messages": messages, "tools": tools}, None)
    )
    assert response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == (
        "read_file"
    )

    messages += [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "read-1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path":"sample.md"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "read-1", "content": "# Sample"},
    ]
    data = {"messages": messages, "tools": tools}
    _, _, finished = server._prepare_chat_request(data)
    assert finished["_infinilm_direct_read_result"] == "# Sample"
    assert "_infinilm_direct_tool_call" not in finished
    response = asyncio.run(server._chat("read-done", data, None))
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["choices"][0]["message"].get("tool_calls") is None


def test_prepared_request_carries_forced_tool_prefix():
    server = InferenceServer("/tmp/model", skip_load=True)
    messages, kwargs, request_data = server._prepare_chat_request(
        {
            "messages": [
                {"role": "user", "content": "把结果写入 outline.md。写完后立即停止。"}
            ],
            "tools": [_tool("write_file"), _tool("read_file")],
        }
    )

    assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["write_file"]
    assert request_data["_infinilm_forced_tool_prefix"].startswith("<tool_call>\n")


def test_successful_direct_write_marks_next_round_complete():
    server = InferenceServer("/tmp/model", skip_load=True)
    messages, kwargs, request_data = server._prepare_chat_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "把结果写入 outline.md。写完后立即停止。",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "write_file",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "Created outline.md.",
                },
            ],
            "tools": [_tool("write_file"), _tool("read_file")],
        }
    )

    assert kwargs == {}
    assert request_data["_infinilm_direct_write_complete"] is True


def test_strip_reasoning_markers_from_visible_content():
    assert (
        strip_reasoning_markers("<think>hidden</think>visible</think>")
        == "hiddenvisible"
    )


def test_stream_parser_drops_truncated_tool_block():
    parser = ToolCallStreamParser(allowed_tool_names={"write_file"})
    content_parts, calls = parser.feed(
        "<tool_call>\n<function=write_file>\n"
        "<parameter=file_path>outline.md</parameter>\n"
        "<parameter=content># Outline\n- First section"
    )
    final_content, final_calls = parser.finalize()
    content_parts.extend(final_content)
    calls.extend(final_calls)

    assert content_parts == []
    assert calls == []


def test_stream_parser_drops_truncated_open_marker():
    parser = ToolCallStreamParser(allowed_tool_names={"write_file"})
    content_parts, calls = parser.feed("Visible answer.<tool_")
    final_content, final_calls = parser.finalize()
    content_parts.extend(final_content)
    calls.extend(final_calls)

    assert "".join(content_parts) == "Visible answer."
    assert calls == []


def test_parse_tool_call_with_complete_function_and_missing_outer_close():
    content, calls = parse_tool_calls(
        "<tool_call>\n<function=write_file>\n"
        "<parameter=file_path>outline.md</parameter>\n"
        "<parameter=content># Outline</parameter>\n"
        "</function>"
    )

    assert content is None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "write_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "file_path": "outline.md",
        "content": "# Outline",
    }


def test_stream_parser_accepts_complete_function_without_outer_close():
    parser = ToolCallStreamParser(allowed_tool_names={"write_file"})
    content_parts, calls = parser.feed(
        "<tool_call>\n<function=write_file>\n"
        "<parameter=file_path>outline.md</parameter>\n"
        "<parameter=content># Outline</parameter>\n"
        "</function>"
    )
    final_content, final_calls = parser.finalize()
    content_parts.extend(final_content)
    calls.extend(final_calls)

    assert content_parts == []
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "write_file"


def test_stream_parser_accepts_glm_arg_value_close_in_complete_outer_block():
    parser = ToolCallStreamParser(allowed_tool_names={"write_file"})
    content_parts, calls = parser.feed(
        "<tool_call>\n<function=write_file>\n"
        "<parameter=file_path>\noutline.md\n</parameter>\n"
        "<parameter=content>\n# Outline\n- First section\n"
        "</arg_value></tool_call>"
    )
    final_content, final_calls = parser.finalize()
    content_parts.extend(final_content)
    calls.extend(final_calls)

    assert content_parts == []
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "write_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "file_path": "outline.md",
        "content": "# Outline\n- First section",
    }


def test_stream_parser_rejects_unclosed_parameter_in_complete_outer_block():
    parser = ToolCallStreamParser(allowed_tool_names={"write_file"})
    content_parts, calls = parser.feed(
        "<tool_call>\n<function=write_file>\n"
        "<parameter=file_path>outline.md</parameter>\n"
        "<parameter=content># Outline</tool_call>"
    )
    final_content, final_calls = parser.finalize()
    content_parts.extend(final_content)
    calls.extend(final_calls)


def test_successful_direct_read_marks_next_round_complete():
    tools = [_tool("write_file"), _tool("read_file")]
    messages = [
        {"role": "user", "content": "读取 outline.md。"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_read",
            "content": "1|# Outline",
        },
    ]

    assert direct_read_result(messages, tools) == "1|# Outline"


def test_direct_edit_orchestration_forces_read_then_edit_and_stops():
    tools = [_tool("read_file"), _tool("edit_file")]
    messages = [
        {
            "role": "user",
            "content": "编辑 outline.md，把 status=pending 改为 status=done。完成后立即停止。",
        }
    ]
    read_prefix = forced_read_for_edit_prefix(messages, tools)
    assert read_prefix is not None and "read_file" in read_prefix

    messages += [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_read", "content": "status=pending"},
    ]
    edit_prefix = forced_edit_tool_prefix(messages, tools)
    assert edit_prefix is not None
    assert "edit_file" in edit_prefix
    assert "status=pending" in edit_prefix
    assert "status=done" in edit_prefix

    messages += [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_edit",
                    "type": "function",
                    "function": {"name": "edit_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_edit", "content": "Updated outline.md"},
    ]
    assert direct_edit_complete(messages, tools) is True


def test_direct_bash_command_forces_call_and_completes_with_result():
    tools = [_tool("bash"), _tool("read_file")]
    messages = [
        {"role": "user", "content": "执行 bash 命令 printf bash-ok。完成后立即停止。"}
    ]
    prefix = forced_bash_tool_prefix(messages, tools)
    assert prefix is not None and "printf bash-ok" in prefix

    messages += [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_bash",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_bash",
            "content": "BASH_RESULT[success] bash-ok",
        },
    ]
    assert direct_bash_result(messages, tools) == "BASH_RESULT[success] bash-ok"
