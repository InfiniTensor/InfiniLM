"""Protocol and real HTTP-route tests; fake inference is explicitly scoped here."""

import copy
import importlib.util
import json
import sys
import types
from enum import Enum
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "python/infinilm/server"
spec = importlib.util.spec_from_file_location(
    "agent_protocol_under_test", SERVER / "agent_protocol.py"
)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
Error = protocol.AgentProtocolError
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compute",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "label": {"type": "string"},
                    "options": {"type": "object"},
                },
                "required": ["n", "label"],
                "additionalProperties": False,
            },
        },
    }
]
XML = "<tool_call>\n<function=compute>\n<parameter=n>\n42\n</parameter>\n<parameter=label>\n123\n</parameter>\n</function>\n</tool_call>"


def prepared(**fields):
    return protocol.prepare_agent_request(
        dict(messages=[{"role": "user", "content": "compute"}], tools=TOOLS, **fields),
        True,
    )


def tool_history(function_name="compute", **result_fields):
    return [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "calculate",
            "tool_calls": [
                {
                    "id": "call_first",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": '{"n":42,"label":"x"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_first",
            "content": "42",
            **result_fields,
        },
    ]


def post_chat(client, **fields):
    return client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "compute"}], **fields},
    )


def sse_chunks(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.parametrize("width", [1, 2, 3, 7, 13, 64, 10000])
def test_stream_partition_invariance(width):
    data, schemas, choice = prepared()
    text = "reason</think>\nanswer\n" + XML + "\n" + XML
    parser = protocol.QwenAgentParser(schemas, choice=choice)
    events = []
    for i in range(0, len(text), width):
        events.extend(parser.feed(text[i : i + width]))
    events.extend(parser.feed("", final=True))
    message = protocol.collect_message(events)
    assert message["reasoning_content"] == "reason"
    assert message["content"] == "\nanswer\n\n"
    assert len(message["tool_calls"]) == 2
    assert len({x["id"] for x in message["tool_calls"]}) == 2
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "n": 42,
        "label": "123",
    }
    assert parser.finish_reason("stop") == "tool_calls"
    assert parser.finish_reason("length") == "length"


@pytest.mark.parametrize(
    "text",
    [
        XML[:-2],
        XML.replace("compute", "missing"),
        XML.replace("42", "oops"),
        XML.replace("<parameter=n>", "<parameter=unknown>"),
        XML.replace("</function>", "garbage</function>"),
    ],
)
def test_invalid_output_is_not_executable(text):
    _, schemas, _ = prepared()
    parser = protocol.QwenAgentParser(schemas, thinking=False)
    with pytest.raises(Error):
        parser.feed(text, final=True)


def test_every_single_split():
    _, schemas, _ = prepared()
    for i in range(len(XML) + 1):
        p = protocol.QwenAgentParser(schemas, thinking=False)
        events = p.feed(XML[:i]) + p.feed(XML[i:]) + p.feed("", final=True)
        assert len(protocol.collect_message(events)["tool_calls"]) == 1


def test_required_and_parallel():
    _, schemas, _ = prepared()
    with pytest.raises(Error, match="tool_choice"):
        protocol.QwenAgentParser(schemas, thinking=False, choice="required").feed(
            "answer", final=True
        )
    with pytest.raises(Error, match="Multiple calls"):
        protocol.QwenAgentParser(schemas, thinking=False, parallel=False).feed(
            XML + XML, final=True
        )
    with pytest.raises(Error, match="inside reasoning"):
        protocol.QwenAgentParser(schemas).feed(XML, final=True)


def test_tool_choice_none_and_named():
    data, schemas, choice = prepared(tool_choice="none")
    assert schemas == {} and data["chat_template_kwargs"]["tools"] == []
    data, schemas, choice = prepared(
        tool_choice={"type": "function", "function": {"name": "compute"}}
    )
    assert choice == "compute" and list(schemas) == ["compute"]


def test_history_arguments_are_objects_for_jinja_and_ids_preserved():
    messages = [{"role": "user", "content": "compute"}, *tool_history()]
    original = copy.deepcopy(messages)
    data, _, _ = protocol.prepare_agent_request(
        {"messages": messages, "tools": TOOLS}, True
    )
    assert messages == original
    assert data["messages"][1]["tool_calls"][0]["function"]["arguments"]["n"] == 42
    assert data["messages"][1]["tool_calls"][0]["id"] == "call_first"
    assert data["messages"][2]["name"] == "compute"
    with pytest.raises(Error, match="pending"):
        protocol.prepare_agent_request({"messages": [messages[0], messages[2]]}, True)
    with pytest.raises(Error, match="Missing tool results"):
        protocol.prepare_agent_request({"messages": messages[:2]}, True)


def test_request_rejections():
    with pytest.raises(Error, match="Enable"):
        protocol.prepare_agent_request({"tools": TOOLS}, False)
    with pytest.raises(Error, match="unavailable"):
        prepared(tool_choice={"type": "function", "function": {"name": "missing"}})
    with pytest.raises(Error, match="top level"):
        prepared(chat_template_kwargs={"tools": []})
    bad = copy.deepcopy(TOOLS)
    bad[0]["function"]["parameters"]["$ref"] = "https://example.org/schema"
    with pytest.raises(Error, match="local"):
        protocol.prepare_agent_request({"tools": bad}, True)


@pytest.mark.parametrize(
    "field,value",
    [("messages", {}), ("stream_options", []), ("parallel_tool_calls", "false")],
)
def test_invalid_request_field_types(field, value):
    with pytest.raises(Error):
        protocol.prepare_agent_request({field: value}, True)


def test_explicit_think_tags_and_numeric_looking_strings():
    _, schemas, _ = prepared()
    p = protocol.QwenAgentParser(schemas, thinking=False)
    result = protocol.collect_message(p.feed("<think>reason</think>" + XML, final=True))
    assert result["reasoning_content"] == "reason"
    assert (
        json.loads(result["tool_calls"][0]["function"]["arguments"])["label"] == "123"
    )


def test_nested_nonfinite_arguments_rejected():
    _, schemas, _ = prepared()
    xml = XML.replace(
        "</function>", '<parameter=options>\n{"x":NaN}\n</parameter>\n</function>'
    )
    with pytest.raises(Error, match="finite"):
        protocol.QwenAgentParser(schemas, thinking=False).feed(xml, final=True)


@pytest.mark.parametrize(
    "spec",
    [
        {"enum": ["123"]},
        {"anyOf": [{"type": "string"}, {"type": "null"}]},
        {"$ref": "#/$defs/label"},
        {"anyOf": [{"$ref": "#/$defs/label"}, {"type": "null"}]},
        {"$ref": "#/$defs/alias"},
    ],
)
def test_string_schema_variants(spec):
    _, schemas, _ = prepared()
    schemas["compute"]["$defs"] = {
        "label": {"type": "string"},
        "alias": {"$ref": "#/$defs/label"},
    }
    schemas["compute"]["properties"]["label"] = spec
    message = protocol.collect_message(
        protocol.QwenAgentParser(schemas, thinking=False).feed(XML, final=True)
    )
    assert (
        json.loads(message["tool_calls"][0]["function"]["arguments"])["label"] == "123"
    )


def test_unresolved_parameter_reference_is_a_protocol_error():
    _, schemas, _ = prepared()
    schemas["compute"]["properties"]["label"] = {"$ref": "#/$defs/missing"}
    with pytest.raises(Error, match="Unresolved local"):
        protocol.QwenAgentParser(schemas, thinking=False).feed(XML, final=True)


class FinishReason(Enum):
    EOS_TOKEN = "eos_token"
    STOP_STRING = "stop_string"
    LENGTH = "length"
    CANCELED = "canceled"
    ERROR = "error"
    TIMEOUT = "timeout"


class FakeRequest:
    finish_reason = FinishReason.EOS_TOKEN

    def is_finished(self):
        return True

    def get_prompt_length(self):
        return 12

    def get_num_generated_tokens(self):
        return 5

    def get_total_length(self):
        return 17


class FakeEngine:
    def __init__(self, text):
        self.text = text
        self.requests = []

    def add_chat_request(self, **kwargs):
        self.requests.append(kwargs)
        return FakeRequest()

    async def stream_request(self, *args, **kwargs):
        for char in self.text:
            yield types.SimpleNamespace(
                token_text=char, finished=False, finish_reason=None
            )
        yield types.SimpleNamespace(
            token_text="", finished=True, finish_reason=FinishReason.EOS_TOKEN
        )


def make_client(text=XML, parser="qwen3_coder"):
    stubs = {}
    for name in [
        "infinilm",
        "infinilm.base_config",
        "infinilm.config",
        "infinilm.llm",
        "infinilm.moe_config",
        "infinilm.server",
    ]:
        stubs[name] = types.ModuleType(name)
    stubs["infinilm.base_config"].BaseConfig = object
    stubs["infinilm.config"].KVTransferConfig = object
    stubs["infinilm.llm"].AsyncLLMEngine = object
    stubs["infinilm.llm"].FinishReason = FinishReason
    stubs["infinilm.llm"].SamplingParams = lambda **kw: kw
    stubs["infinilm.moe_config"].configure_moe_ep_backend = lambda *a: None
    stubs["infinilm.server.agent_protocol"] = protocol
    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(
            "inference_server_under_test", SERVER / "inference_server.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    server = module.InferenceServer("/models/Qwen3.8-27B", tool_call_parser=parser)
    server.engine = FakeEngine(text)
    app = FastAPI()
    server._register_routes(app)
    return TestClient(app), server


@pytest.mark.parametrize("stream", [False, True])
def test_http_tool_call_and_history_replay(stream):
    client, server = make_client()
    body = {
        "model": "Qwen3.8-27B",
        "messages": [{"role": "user", "content": "compute"}],
        "tools": TOOLS,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    if stream:
        chunks = sse_chunks(response)
        events = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
        message = protocol.collect_message(events)
        assert chunks[-1]["usage"]["total_tokens"] == 17
        assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    else:
        message = response.json()["choices"][0]["message"]
        assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert server.engine.requests[0]["chat_template_kwargs"]["tools"] == TOOLS
    body["messages"] += [
        message,
        {
            "role": "tool",
            "tool_call_id": message["tool_calls"][0]["id"],
            "content": "42",
        },
    ]
    server.engine.text = "The result is 42."
    response = client.post("/v1/chat/completions", json=dict(body, stream=False))
    assert response.json()["choices"][0]["message"]["content"] == "The result is 42."
    assert (
        server.engine.requests[-1]["messages"][1]["tool_calls"][0]["function"][
            "arguments"
        ]["n"]
        == 42
    )


def test_http_malformed_request_before_sse():
    client, server = make_client()
    response = post_chat(client, tools=TOOLS, tool_choice="invalid", stream=True)
    assert response.status_code == 400
    assert not server.engine.requests


@pytest.mark.parametrize("stream", [False, True])
def test_http_malformed_model_output(stream):
    client, _ = make_client(XML[:-4])
    r = post_chat(
        client,
        tools=TOOLS,
        stream=stream,
        chat_template_kwargs={"enable_thinking": False},
    )
    if stream:
        assert "invalid_model_output" in r.text
        assert '"tool_calls"' not in r.text
    else:
        assert r.status_code == 502
        assert r.json()["error"]["type"] == "invalid_model_output"


@pytest.mark.parametrize("stream", [False, True])
def test_disabled_parser_preserves_plain_chat(stream):
    client, server = make_client("plain answer", parser="none")
    response = post_chat(client, stream=stream)
    assert response.status_code == 200
    assert "tools" not in server.engine.requests[0]["chat_template_kwargs"]
    if stream:
        chunks = sse_chunks(response)
        assert (
            "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
            == "plain answer"
        )
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert response.text.count("data: [DONE]") == 1
    else:
        assert response.json()["choices"][0]["message"]["content"] == "plain answer"


@pytest.mark.parametrize("stream", [False, True])
def test_disabled_parser_rejects_tools_before_submission(stream):
    client, server = make_client(parser="none")
    response = post_chat(client, tools=TOOLS, stream=stream)
    assert response.status_code == 400
    assert not server.engine.requests


@pytest.mark.parametrize("stream", [False, True])
def test_disabled_parser_preserves_legacy_error_shape(stream):
    client, server = make_client(parser="none")

    def fail(**kwargs):
        raise RuntimeError("test execution failed")

    server.engine.add_chat_request = fail
    response = post_chat(client, stream=stream)
    if stream:
        assert response.status_code == 200
        assert "[Error: test execution failed]" in response.text
        assert '"finish_reason": "error"' in response.text
    else:
        assert response.status_code == 500
        assert response.json() == {"error": "test execution failed"}


@pytest.mark.parametrize("name", [None, "", "bad name"])
def test_historical_function_name_is_validated(name):
    with pytest.raises(Error, match="historical function name"):
        protocol.prepare_agent_request(
            {"messages": tool_history(name), "tools": TOOLS}, True
        )


def test_tool_result_name_must_match_call_id():
    with pytest.raises(Error, match="name does not match"):
        protocol.prepare_agent_request(
            {"messages": tool_history(name="other"), "tools": TOOLS}, True
        )


def test_arbitrary_precision_integer_arguments():
    _, schemas, _ = prepared()
    value = 10**400
    text = XML.replace("42", str(value))
    message = protocol.collect_message(
        protocol.QwenAgentParser(schemas, thinking=False).feed(text, final=True)
    )
    assert json.loads(message["tool_calls"][0]["function"]["arguments"])["n"] == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("tools", {}),
        ("tools", False),
        ("tools", ""),
        ("chat_template_kwargs", []),
        ("stream", "false"),
        ("stream_options", {"include_usage": "false"}),
    ],
)
def test_falsy_protocol_fields_are_not_silently_coerced(field, value):
    with pytest.raises(Error):
        protocol.prepare_agent_request({field: value}, True)


def test_falsy_historical_calls_are_not_silently_coerced():
    with pytest.raises(Error, match="tool_calls must be an array"):
        protocol.prepare_agent_request(
            {"messages": [{"role": "assistant", "tool_calls": {}}]}, True
        )


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"max_completion_tokens": 7}, 7),
        ({"max_tokens": 3, "max_completion_tokens": 7}, 3),
        ({"max_tokens": None, "max_completion_tokens": 7}, 7),
    ],
)
def test_completion_token_alias_reaches_sampling(fields, expected):
    client, server = make_client("answer")
    response = post_chat(client, **fields)
    assert response.status_code == 200
    assert server.engine.requests[0]["sampling_params"]["max_tokens"] == expected
