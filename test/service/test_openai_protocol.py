import json

import pytest
from infinilm.processors.processor import normalize_openai_messages
from infinilm.server.inference_server import InferenceServer, completion_json
from infinilm.server.openai_protocol import (
    ToolCallStreamParser,
    parse_tool_calls,
    strip_reasoning_markers,
)
from infinilm.server.tool_contract import apply_tool_contract
from infinilm.server.tool_constraints import constrain_tools, forced_write_tool_prefix


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
