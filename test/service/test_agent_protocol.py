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

SERVER = Path(__file__).resolve().parents[2] / "python/infinilm/server"
spec = importlib.util.spec_from_file_location(
    "agent_protocol_under_test", SERVER / "agent_protocol.py"
)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
with patch.dict(
    sys.modules,
    {
        "infinilm": types.ModuleType("infinilm"),
        "infinilm.server": types.ModuleType("infinilm.server"),
        "infinilm.server.agent_protocol": protocol,
    },
):
    spec = importlib.util.spec_from_file_location(
        "infinilm.server.tool_parsers", SERVER / "tool_parsers/__init__.py"
    )
    parsers = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parsers
    spec.loader.exec_module(parsers)
ToolCallParser = parsers.ToolCallParser
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


JSON_CALL = {"name": "compute", "arguments": {"n": 42, "label": "123"}}
WIRES = {
    "qwen3_coder": XML,
    "hermes": "<tool_call>" + json.dumps(JSON_CALL) + "</tool_call>",
    "llama3_json": "<|python_tag|>"
    + json.dumps({"name": "compute", "parameters": JSON_CALL["arguments"]}),
    "mistral": "[TOOL_CALLS]" + json.dumps([JSON_CALL]),
    "minicpm5": '<function name="compute"><param name="n">42</param><param name="label">123</param></function>',
    "glm5": "<tool_call>compute<arg_key>n</arg_key><arg_value>42</arg_value><arg_key>label</arg_key><arg_value>123</arg_value></tool_call>",
    "deepseek_v4": '<｜DSML｜tool_calls><｜DSML｜invoke name="compute"><｜DSML｜parameter name="n" string="false">42</｜DSML｜parameter><｜DSML｜parameter name="label" string="true">123</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>',
    "kimi_k3": '<|open|>tools<|sep|><|open|>call tool="compute" index="1"<|sep|><|open|>argument key="n" type="number"<|sep|>42<|close|>argument<|sep|><|open|>argument key="label" type="string"<|sep|>123<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>',
}


def prepared(**fields):
    return protocol.prepare_agent_request(
        dict(messages=[{"role": "user", "content": "compute"}], tools=TOOLS, **fields),
        "qwen3_coder",
    )


def tool_history(function_name="compute", **result_fields):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_first",
                    "type": "function",
                    "function": {"name": function_name, "arguments": "{}"},
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


def tools_with(**fields):
    tools = copy.deepcopy(TOOLS)
    tools[0]["function"].update(fields)
    return tools


@pytest.mark.parametrize("parser_name", WIRES)
def test_stream_protocol_across_boundaries(parser_name):
    _, schemas, _ = prepared()
    wire = WIRES[parser_name] * 2
    if parser_name in ("mistral", "llama3_json"):
        wire = parsers.TOOL_CALL_FORMATS[parser_name][0] + json.dumps(
            [JSON_CALL, JSON_CALL]
        )
    if parser_name == "kimi_k3":
        start, end = parsers.TOOL_CALL_FORMATS[parser_name]
        call = WIRES[parser_name][len(start) : -len(end)]
        wire = start + call + call.replace('index="1"', 'index="2"') + end
        text = "<|open|>think<|sep|>reason<|close|>think<|sep|>\nanswer\n" + wire
    else:
        text = "<think>reason</think>\nanswer\n" + wire
    partitions = [list(text)] + [[text[:i], text[i:]] for i in range(len(text) + 1)]
    for parts in partitions:
        parser = ToolCallParser(schemas, thinking=False, parser=parser_name)
        events = []
        for part in parts:
            events.extend(parser.feed(part))
        message = protocol.collect_message(events + parser.feed("", final=True))
        assert message["reasoning_content"] == "reason"
        assert message["content"] == "\nanswer\n"
        calls = message["tool_calls"]
        assert len(calls) == 2 and len({c["id"] for c in calls}) == 2
        assert all(
            json.loads(c["function"]["arguments"]) == {"n": 42, "label": "123"}
            for c in calls
        )
    assert parser.finish_reason("stop") == "tool_calls"
    assert parser.finish_reason("length") == "length"


@pytest.mark.parametrize(
    "text",
    [
        XML.replace("compute", "missing"),
        XML.replace("42", "oops"),
        XML.replace("<parameter=n>", "<parameter=unknown>"),
        XML.replace("</function>", "garbage</function>"),
        XML.replace(
            "</function>", '<parameter=options>{"x":NaN}</parameter></function>'
        ),
    ],
    ids=[
        "unknown-tool",
        "wrong-type",
        "extra-parameter",
        "malformed-xml",
        "nonfinite-json",
    ],
)
def test_invalid_output_is_not_executable(text):
    _, schemas, _ = prepared()
    with pytest.raises(Error):
        ToolCallParser(schemas, thinking=False).feed(text, final=True)


def test_tool_choice_and_parallel_contract():
    data, schemas, _ = prepared(tool_choice="none")
    assert schemas == {} and not data["chat_template_kwargs"]["tools"]
    _, schemas, choice = prepared(
        tool_choice={"type": "function", "function": {"name": "compute"}}
    )
    assert choice == "compute" and list(schemas) == ["compute"]
    with pytest.raises(Error):
        ToolCallParser(schemas, thinking=False, choice="required").feed(
            "answer", final=True
        )
    with pytest.raises(Error):
        ToolCallParser(schemas, thinking=False, parallel=False).feed(
            XML + XML, final=True
        )
    with pytest.raises(Error):
        ToolCallParser(schemas, thinking=True).feed(XML, final=True)


@pytest.mark.parametrize(
    "fields",
    [
        {"tools": {}},
        {"stream": "false"},
        {"stream_options": {"include_usage": "false"}},
        {"tool_choice": {"type": "function", "function": {"name": "missing"}}},
        {"chat_template_kwargs": {"tools": []}},
        {"tools": tools_with(strict=True)},
        {"tools": tools_with(parameters={"$ref": "https://example.org/schema"})},
    ],
    ids=[
        "falsy-tools",
        "stream-type",
        "usage-type",
        "unavailable-tool",
        "nested-tools",
        "strict",
        "remote-ref",
    ],
)
def test_invalid_tool_requests(fields):
    with pytest.raises(Error):
        protocol.prepare_agent_request({"tools": TOOLS, **fields}, "qwen3_coder")


@pytest.mark.parametrize(
    "messages",
    [
        tool_history()[1:],
        tool_history()[:1],
        tool_history(name="other"),
        tool_history() * 2,
        tool_history(None),
        tool_history("bad name"),
        [{"role": "assistant", "tool_calls": {}}],
    ],
    ids=[
        "orphan-result",
        "missing-result",
        "wrong-name",
        "duplicate-id",
        "missing-name",
        "invalid-name",
        "falsy-calls",
    ],
)
def test_invalid_tool_history(messages):
    with pytest.raises(Error):
        protocol.prepare_agent_request(
            {"messages": messages, "tools": TOOLS}, "qwen3_coder"
        )


@pytest.mark.parametrize(
    "spec",
    [
        {"enum": ["123"]},
        {"anyOf": [{"$ref": "#/$defs/alias"}, {"type": "null"}]},
    ],
    ids=["enum", "union-with-rooted-ref-chain"],
)
def test_string_parameter_schema_integration(spec):
    _, schemas, _ = prepared()
    schemas["compute"]["$defs"] = {
        "label": {"type": "string"},
        "alias": {"$ref": "#/$defs/label"},
    }
    schemas["compute"]["properties"]["label"] = spec
    message = protocol.collect_message(
        ToolCallParser(schemas, thinking=False).feed(XML, final=True)
    )
    assert (
        json.loads(message["tool_calls"][0]["function"]["arguments"])["label"] == "123"
    )


def test_unresolved_parameter_reference_is_a_protocol_error():
    _, schemas, _ = prepared()
    schemas["compute"]["properties"]["label"] = {"$ref": "#/$defs/missing"}
    with pytest.raises(Error):
        ToolCallParser(schemas, thinking=False).feed(XML, final=True)


def test_arbitrary_precision_integer_arguments():
    _, schemas, _ = prepared()
    value = 10**400
    message = protocol.collect_message(
        ToolCallParser(schemas, thinking=False).feed(
            XML.replace("42", str(value)), final=True
        )
    )
    assert json.loads(message["tool_calls"][0]["function"]["arguments"])["n"] == value


class FinishReason(Enum):
    EOS_TOKEN = "eos_token"
    STOP_STRING = "stop_string"
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
        self.prompt = ""

    def add_chat_request(self, **kwargs):
        self.requests.append(kwargs)
        request = FakeRequest()
        request.prompt = self.prompt
        return request

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
    stubs["infinilm.server.tool_parsers"] = parsers
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
@pytest.mark.parametrize("parser_name", WIRES)
def test_http_tool_call_and_history_replay(stream, parser_name):
    client, server = make_client(WIRES[parser_name], parser=parser_name)
    body = {
        "messages": [{"role": "user", "content": "compute"}],
        "tools": TOOLS,
        "stream": stream,
        "max_completion_tokens": 7,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }
    response = post_chat(client, **body)
    assert response.status_code == 200, response.text
    if stream:
        chunks = sse_chunks(response)
        message = protocol.collect_message(
            [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
        )
        assert chunks[-1]["usage"]["total_tokens"] == 17
        assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
        assert response.text.count("data: [DONE]") == 1
    else:
        message = response.json()["choices"][0]["message"]
        assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert server.engine.requests[0]["chat_template_kwargs"]["tools"] == TOOLS
    assert server.engine.requests[0]["sampling_params"]["max_tokens"] == 7
    call_id = message["tool_calls"][0]["id"]
    body["messages"] += [
        message,
        {"role": "tool", "tool_call_id": call_id, "content": "42"},
    ]
    server.engine.text = "The result is 42."
    response = post_chat(client, **dict(body, stream=False))
    assert response.json()["choices"][0]["message"]["content"] == "The result is 42."
    replay = server.engine.requests[-1]["messages"]
    assert replay[1]["tool_calls"][0]["function"]["arguments"]["n"] == 42
    assert replay[1]["tool_calls"][0]["id"] == call_id
    assert replay[2]["name"] == "compute" and replay[2]["tool_call_id"] == call_id


@pytest.mark.parametrize(
    "parser,fields",
    [
        ("none", {"tools": TOOLS}),
        ("qwen3_coder", {"tools": TOOLS, "tool_choice": "invalid"}),
    ],
)
def test_http_rejects_requests_before_sse(parser, fields):
    client, server = make_client(parser=parser)
    response = post_chat(client, stream=True, **fields)
    assert response.status_code == 400
    assert not server.engine.requests


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("parser_name", WIRES)
def test_http_truncated_model_output(stream, parser_name):
    client, _ = make_client(WIRES[parser_name][:-4], parser=parser_name)
    response = post_chat(
        client,
        tools=TOOLS,
        stream=stream,
        chat_template_kwargs={"enable_thinking": False},
    )
    if stream:
        assert "invalid_model_output" in response.text
        assert '"tool_calls"' not in response.text
        assert response.text.count("data: [DONE]") == 1
    else:
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "invalid_model_output"


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


def test_completion_token_alias_precedence():
    client, server = make_client("answer")
    response = post_chat(client, max_tokens=3, max_completion_tokens=7)
    assert response.status_code == 200
    assert server.engine.requests[0]["sampling_params"]["max_tokens"] == 3


def test_minicpm_cdata_preserves_literal_tags():
    _, schemas, _ = prepared()
    value = "\ncode </function> and </param> & text\n"
    wire = (
        '<function name="compute"><param name="n">42</param><param name="label"><![CDATA['
        + value
        + "]]></param></function>"
    )
    parser = ToolCallParser(schemas, parser="minicpm5")
    events = []
    for char in wire:
        events.extend(parser.feed(char))
    message = protocol.collect_message(events + parser.feed("", final=True))
    assert (
        json.loads(message["tool_calls"][0]["function"]["arguments"])["label"] == value
    )


def test_json_array_validation_is_atomic():
    _, schemas, _ = prepared()
    parser = ToolCallParser(schemas, parser="mistral")
    wire = "[TOOL_CALLS]" + json.dumps(
        [JSON_CALL, {"name": "missing", "arguments": {}}]
    )
    with pytest.raises(Error):
        parser.feed(wire, final=True)
    assert not parser.calls


def test_llama_bare_json_and_ordinary_json():
    _, schemas, _ = prepared()
    for text, is_call in [(json.dumps(JSON_CALL), True), ('{"name":"Alice"}', False)]:
        message = protocol.collect_message(
            ToolCallParser(schemas, parser="llama3_json").feed(text, final=True)
        )
        assert bool(message.get("tool_calls")) == is_call
        if not is_call:
            assert message["content"] == text


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("parser_name", ["hermes", "kimi_k3"])
def test_http_reasoning_starts_from_rendered_prompt(thinking, parser_name):
    start, end = "<think>", "</think>"
    if parser_name == "kimi_k3":
        start, end = "<|open|>think<|sep|>", "<|close|>think<|sep|>"
    client, server = make_client(
        "reason" + end + "42" if thinking else "42", parser=parser_name
    )
    server.engine.prompt = "assistant\n" + start if thinking else "assistant\n"
    response = post_chat(
        client,
        stream=True,
        reasoning_effort="low",
        chat_template_kwargs={"enable_thinking": thinking},
    )
    message = protocol.collect_message(
        [c["choices"][0]["delta"] for c in sse_chunks(response) if c.get("choices")]
    )
    assert message["content"] == "42"
    assert message.get("reasoning_content") == ("reason" if thinking else None)
    if parser_name == "kimi_k3":
        kwargs = server.engine.requests[0]["chat_template_kwargs"]
        assert kwargs["thinking"] == thinking and kwargs["thinking_effort"] == "low"


@pytest.mark.parametrize(
    "parser_name,wire",
    [
        ("kimi_k3", WIRES["kimi_k3"].replace('index="1"', 'index="2"')),
        ("kimi_k3", WIRES["kimi_k3"].replace('type="number"', 'type="unknown"')),
        (
            "deepseek_v4",
            WIRES["deepseek_v4"].replace(
                "</｜DSML｜invoke>",
                '<｜DSML｜parameter name="n" string="false">42</｜DSML｜parameter></｜DSML｜invoke>',
            ),
        ),
    ],
)
def test_typed_markup_rejects_invalid_structure(parser_name, wire):
    _, schemas, _ = prepared()
    parser = ToolCallParser(schemas, parser=parser_name)
    with pytest.raises(Error):
        parser.feed(wire, final=True)
    assert not parser.calls


def test_kimi_json_block_and_response_channels():
    _, schemas, _ = prepared()
    wire = (
        '<|open|>response<|sep|>answer<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="compute" index="1"<|sep|><|open|>json type="object"<|sep|>'
        + json.dumps(JSON_CALL["arguments"])
        + "<|close|>json<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"
    )
    message = protocol.collect_message(
        ToolCallParser(schemas, parser="kimi_k3").feed(wire, final=True)
    )
    assert message["content"] == "answer"
    assert (
        json.loads(message["tool_calls"][0]["function"]["arguments"])
        == JSON_CALL["arguments"]
    )


def test_deepseek_processor_supplies_native_encoder_contract():
    base = types.ModuleType("infinilm.processors.basic_llm_processor")
    base.BasicLLMProcessor = object
    with patch.dict(
        sys.modules,
        {
            "infinilm": types.ModuleType("infinilm"),
            "infinilm.processors": types.ModuleType("infinilm.processors"),
            "infinilm.processors.basic_llm_processor": base,
        },
    ):
        spec = importlib.util.spec_from_file_location(
            "infinilm.processors.deepseek_v4_processor",
            SERVER.parent / "processors/deepseek_v4_processor.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    processor = object.__new__(module.DeepSeekV4Processor)
    captured = {}

    def encode(messages, **kwargs):
        captured.update(messages=messages, kwargs=kwargs)
        return "prompt<｜Assistant｜><think>"

    processor.encode_messages = encode
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "compute"}]},
        *tool_history(),
    ]
    original = copy.deepcopy(messages)
    assert (
        processor.apply_chat_template(
            messages, tools=TOOLS, tokenize=False, add_generation_prompt=False
        )
        == "prompt"
    )
    assert captured["messages"][0]["tools"] == TOOLS
    assert captured["messages"][1]["content"] == "compute"
    assert isinstance(
        captured["messages"][2]["tool_calls"][0]["function"]["arguments"], str
    )
    assert messages == original
