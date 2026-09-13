"""
Regression tests for the agent-related server logic in inference_server.py.

The server module normally pulls in the native engine stack; in CPU-only
environments lightweight stand-ins are registered for those engine modules so
the pure-Python request/response shaping code can still be exercised.
"""

import asyncio
import enum
import json
import sys
import types
import unittest
from pathlib import Path

# Ensure infinilm is importable when running tests directly. The Python
# sources live under <repo root>/python.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
for _path in (str(PYTHON_ROOT), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _ensure_infinilm_stub():
    """Make ``infinilm`` importable without the native extension."""
    try:
        import infinilm  # noqa: F401
    except Exception:
        sys.modules.pop("infinilm", None)
        stub = types.ModuleType("infinilm")
        stub.__path__ = [str(PYTHON_ROOT / "infinilm")]
        sys.modules["infinilm"] = stub


def _install_engine_stubs():
    """Register stand-ins for engine modules that require the native stack."""

    class FinishReason(enum.Enum):
        EOS_TOKEN = "eos_token"
        STOP_STRING = "stop_string"
        STOP = "stop"
        LENGTH = "length"
        CANCELED = "canceled"
        TIMEOUT = "timeout"
        ERROR = "error"

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            for key, value in kwargs.items():
                setattr(self, key, value)

        def clone(self):
            return SamplingParams(**self.kwargs)

    class AsyncLLMEngine:  # never instantiated in these tests
        def __init__(self, *args, **kwargs):
            raise RuntimeError("engine stub must not be instantiated")

    class KVTransferConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class BaseConfig:  # only referenced by main()
        pass

    def _module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod

    class SchedulerOutput:  # only used in isinstance checks
        pass

    class StaticSchedulerOutput:  # only used in isinstance checks
        pass

    _module("infinilm.base_config", BaseConfig=BaseConfig)
    _module("infinilm.config", KVTransferConfig=KVTransferConfig)
    _module(
        "infinilm.llm",
        AsyncLLMEngine=AsyncLLMEngine,
        FinishReason=FinishReason,
        SamplingParams=SamplingParams,
    )
    _module(
        "infinilm.llm.scheduler",
        SchedulerOutput=SchedulerOutput,
    )
    _module(
        "infinilm.llm.static_scheduler",
        StaticSchedulerOutput=StaticSchedulerOutput,
    )
    _module(
        "infinilm.moe_config",
        configure_moe_ep_backend=lambda *args, **kwargs: ("disabled", 1),
    )


_ensure_infinilm_stub()
try:
    from infinilm.llm import FinishReason  # noqa: F401
except Exception:
    _install_engine_stubs()

try:
    from infinilm.server.inference_server import InferenceServer

    SERVER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    InferenceServer = None
    SERVER_IMPORT_ERROR = exc


def _make_server(**kwargs) -> "InferenceServer":
    kwargs.setdefault("model_path", "dummy-model")
    return InferenceServer(**kwargs)


def _openai_chunk(delta=None, finish_reason=None) -> str:
    chunk = {
        "id": "cmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "dummy",
        "choices": [
            {
                "index": 0,
                "delta": delta or {},
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _parse_sse_events(raw_events):
    """Parse 'event: X\\ndata: {...}\\n\\n' strings into (type, data) pairs."""
    parsed = []
    for raw in raw_events:
        lines = raw.splitlines()
        event_type = lines[0].split(": ", 1)[1]
        data = json.loads(lines[1].split(": ", 1)[1]) if len(lines) > 1 else {}
        parsed.append((event_type, data))
    return parsed


@unittest.skipIf(
    InferenceServer is None,
    f"inference_server not importable in this environment: {SERVER_IMPORT_ERROR}",
)
@unittest.skipIf(
    InferenceServer is None,
    f"inference_server not importable in this environment: {SERVER_IMPORT_ERROR}",
)
class TestPerRequestParsers(unittest.TestCase):
    """P1-2: parser state must not be shared between concurrent requests."""

    def test_stream_parsers_are_independent(self):
        from infinilm.agents import AgentStreamParser

        tools = [
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {}},
            }
        ]
        parser_a = AgentStreamParser(tool_call_parser="llama31", tools=tools)
        parser_b = AgentStreamParser(tool_call_parser="llama31", tools=tools)
        parser_a.process_delta('<|python_tag|>{"name":"get_weather", ')
        # Resetting B's detector must not touch A's buffered state.
        parser_b._tool_call_parser.detector.clear()
        delta = parser_a.flush()
        self.assertTrue(
            parser_a._tool_call_parser.detector._buffer or delta.tool_calls,
            "request A state was clobbered by B",
        )

    def test_no_shared_parser_instances_on_server(self):
        server = _make_server(tool_call_parser="glm", reasoning_parser="think")
        self.assertFalse(hasattr(server, "_tool_call_parser_instance"))
        self.assertFalse(hasattr(server, "_reasoning_parser_instance"))

    def test_disabled_parsers_pass_text_through(self):
        from infinilm.agents import AgentStreamParser

        parser = AgentStreamParser()
        delta = parser.process_delta("plain text")
        self.assertEqual(delta.content, "plain text")
        self.assertEqual(delta.reasoning_content, "")
        self.assertEqual(delta.tool_calls, [])
        self.assertFalse(parser.has_tool_calls)


try:
    from infinilm.processors.basic_llm_processor import BasicLLMProcessor

    PROCESSOR_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    BasicLLMProcessor = None
    PROCESSOR_IMPORT_ERROR = exc


@unittest.skipIf(
    BasicLLMProcessor is None,
    f"basic_llm_processor not importable in this environment: {PROCESSOR_IMPORT_ERROR}",
)
class TestConversationNormalization(unittest.TestCase):
    """Engine-level content handling (moved out of the HTTP layer)."""

    def test_text_only_list_joined_into_string(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part1"},
                    {"type": "text", "text": "part2"},
                ],
            }
        ]
        self.assertEqual(
            BasicLLMProcessor.normalize_conversation(conversation),
            [{"role": "user", "content": "part1part2"}],
        )

    def test_string_content_untouched(self):
        conversation = [{"role": "user", "content": "hello"}]
        self.assertEqual(
            BasicLLMProcessor.normalize_conversation(conversation), conversation
        )

    def test_metadata_and_observation_messages_untouched(self):
        conversation = [
            {"role": "assistant", "metadata": "get_weather", "content": "{}"},
            {"role": "observation", "content": '{"aqi": 42}'},
        ]
        self.assertEqual(
            BasicLLMProcessor.normalize_conversation(conversation), conversation
        )

    def test_non_text_parts_rejected_for_text_models(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "http://x/y.jpg"}},
                ],
            }
        ]
        with self.assertRaises(ValueError):
            BasicLLMProcessor.normalize_conversation(conversation)


@unittest.skipIf(
    InferenceServer is None,
    f"inference_server not importable in this environment: {SERVER_IMPORT_ERROR}",
)
class TestAnthropicStream(unittest.IsolatedAsyncioTestCase):
    """P1-5: content blocks need explicit types and monotonic indices."""

    async def _run_stream(self, chunks):
        server = _make_server()

        async def fake_stream_chat(request_id, data, http_request):
            for chunk in chunks:
                yield chunk
            yield "data: [DONE]\n\n"

        server._stream_chat = fake_stream_chat
        raw_events = []
        async for event in server._anthropic_stream("msg_test", {}, None):
            raw_events.append(event)
        return _parse_sse_events(raw_events)

    async def test_thinking_text_tool_sequence(self):
        """Reasoning, then text, then a tool call: three blocks, 0 -> 1 -> 2."""
        chunks = [
            _openai_chunk(delta={"reasoning_content": "Let me think"}),
            _openai_chunk(delta={"content": "The weather is"}),
            _openai_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ]
                }
            ),
            _openai_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"city":"Beijing"}'},
                        }
                    ]
                }
            ),
            _openai_chunk(finish_reason="tool_calls"),
        ]
        events = await self._run_stream(chunks)

        starts = [
            (data["index"], data["content_block"]["type"])
            for event_type, data in events
            if event_type == "content_block_start"
        ]
        deltas = [
            (data["index"], data["delta"]["type"])
            for event_type, data in events
            if event_type == "content_block_delta"
        ]
        stops = [
            data["index"]
            for event_type, data in events
            if event_type == "content_block_stop"
        ]

        self.assertEqual(starts, [(0, "thinking"), (1, "text"), (2, "tool_use")])
        self.assertEqual(
            deltas,
            [(0, "thinking_delta"), (1, "text_delta"), (2, "input_json_delta")],
        )
        self.assertEqual(stops, [0, 1, 2])

        # Every delta must target a block declared with the matching type.
        declared = dict(starts)
        delta_type_for_block = {
            "thinking": "thinking_delta",
            "text": "text_delta",
            "tool_use": "input_json_delta",
        }
        for index, delta_type in deltas:
            self.assertEqual(delta_type, delta_type_for_block[declared[index]])

        message_delta = next(
            data for event_type, data in events if event_type == "message_delta"
        )
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")
        event_types = [event_type for event_type, _ in events]
        self.assertEqual(event_types[0], "message_start")
        self.assertEqual(event_types[-1], "message_stop")

    async def test_tool_block_index_not_reused(self):
        """Regression: tool_use used to reuse index 0 after thinking/text."""
        chunks = [
            _openai_chunk(delta={"reasoning_content": "hmm"}),
            _openai_chunk(delta={"content": "answer"}),
            _openai_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ]
                }
            ),
            _openai_chunk(finish_reason="tool_calls"),
        ]
        events = await self._run_stream(chunks)
        start_indices = [
            data["index"]
            for event_type, data in events
            if event_type == "content_block_start"
        ]
        # Indices must be unique and monotonically increasing.
        self.assertEqual(start_indices, sorted(set(start_indices)))
        self.assertEqual(len(start_indices), 3)

    async def test_two_parallel_tool_calls(self):
        """Two tool calls become two separate tool_use blocks."""
        chunks = [
            _openai_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ]
                }
            ),
            _openai_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_time", "arguments": "{}"},
                        }
                    ]
                }
            ),
            _openai_chunk(finish_reason="tool_calls"),
        ]
        events = await self._run_stream(chunks)
        starts = [
            (
                data["index"],
                data["content_block"]["type"],
                data["content_block"]["name"],
            )
            for event_type, data in events
            if event_type == "content_block_start"
        ]
        self.assertEqual(
            starts,
            [(0, "tool_use", "get_weather"), (1, "tool_use", "get_time")],
        )

    async def test_text_only_stream(self):
        chunks = [
            _openai_chunk(delta={"content": "Hello "}),
            _openai_chunk(delta={"content": "world"}),
            _openai_chunk(finish_reason="stop"),
        ]
        events = await self._run_stream(chunks)
        starts = [
            (data["index"], data["content_block"]["type"])
            for event_type, data in events
            if event_type == "content_block_start"
        ]
        deltas = [
            (data["index"], data["delta"].get("text"))
            for event_type, data in events
            if event_type == "content_block_delta"
        ]
        self.assertEqual(starts, [(0, "text")])
        self.assertEqual(deltas, [(0, "Hello "), (0, "world")])
        message_delta = next(
            data for event_type, data in events if event_type == "message_delta"
        )
        self.assertEqual(message_delta["delta"]["stop_reason"], "end_turn")


@unittest.skipIf(
    InferenceServer is None,
    f"inference_server not importable in this environment: {SERVER_IMPORT_ERROR}",
)
class TestConcurrentStreams(unittest.IsolatedAsyncioTestCase):
    """P1-2 regression: two interleaved streams must not share parser state."""

    async def test_two_interleaved_tool_call_streams(self):
        from infinilm.llm import FinishReason

        call_text_a = (
            '<|python_tag|>{"name":"get_weather", "arguments":{"city":"Beijing"}}'
        )
        call_text_b = (
            '<|python_tag|>{"name":"get_weather", "arguments":{"city":"Shanghai"}}'
        )

        class FakeRequest:
            def __init__(self, request_id, texts):
                self.request_id = request_id
                self._texts = texts
                self._finished = False

            def is_finished(self):
                return self._finished

            def get_prompt_length(self):
                return 10

            def get_num_generated_tokens(self):
                return len(self._texts)

            def get_total_length(self):
                return 10 + len(self._texts)

        class FakeInnerEngine:
            eos_token_ids = []

        class FakeEngine:
            def __init__(self, streams):
                self.engine = FakeInnerEngine()
                self._streams = streams

            def add_chat_request(self, *, request_id, **kwargs):
                return FakeRequest(request_id, self._streams[request_id])

            async def stream_request(self, req, timeout=None, request_timeout=None):
                for i, text in enumerate(req._texts):
                    await asyncio.sleep(0)  # force interleaving with other streams
                    finished = i == len(req._texts) - 1
                    yield types.SimpleNamespace(
                        token_id=0,
                        token_text=text,
                        finished=finished,
                        finish_reason=FinishReason.EOS_TOKEN if finished else None,
                    )
                req._finished = True

            def add_aborted_req(self, req, reason):
                pass

        class FakeHttpRequest:
            async def is_disconnected(self):
                return False

        # Stream the two tool calls in small interleaved chunks.
        def split(text, n):
            step = -(-len(text) // n)
            return [
                text[i * step : (i + 1) * step] for i in range(n) if text[i * step :]
            ]

        streams = {"req-a": split(call_text_a, 4), "req-b": split(call_text_b, 4)}

        server = _make_server(tool_call_parser="llama31")
        server.engine = FakeEngine(streams)

        data = {
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "chat_template_kwargs": {},
        }

        async def collect(request_id):
            events = []
            async for raw in server._stream_chat(
                request_id, dict(data), FakeHttpRequest()
            ):
                events.append(raw)
            return events

        events_a, events_b = await asyncio.gather(collect("req-a"), collect("req-b"))

        def tool_calls_from(events):
            names, arg_parts, content_parts = [], [], []
            for raw in events:
                if not raw.startswith("data: ") or raw.startswith("data: [DONE]"):
                    continue
                chunk = json.loads(raw[6:].strip())
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    content_parts.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        names.append(fn["name"])
                    if fn.get("arguments"):
                        arg_parts.append(fn["arguments"])
            return names, "".join(arg_parts), "".join(content_parts)

        names_a, args_a, content_a = tool_calls_from(events_a)
        names_b, args_b, content_b = tool_calls_from(events_b)

        # Each stream must see exactly its own complete tool call; nothing
        # may leak into plain content.
        self.assertEqual(names_a, ["get_weather"])
        self.assertEqual(json.loads(args_a), {"city": "Beijing"})
        self.assertEqual(content_a, "")
        self.assertEqual(names_b, ["get_weather"])
        self.assertEqual(json.loads(args_b), {"city": "Shanghai"})
        self.assertEqual(content_b, "")


if __name__ == "__main__":
    unittest.main()
