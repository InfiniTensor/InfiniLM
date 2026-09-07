import json

import pytest
from infinilm.processors.processor import normalize_openai_messages
from infinilm.server.inference_server import InferenceServer, completion_json
from infinilm.server.openai_protocol import (
    ToolCallStreamParser,
    parse_tool_calls,
)


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
