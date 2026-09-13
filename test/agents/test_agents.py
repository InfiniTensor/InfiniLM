"""
Basic unit tests for agent support modules.
These tests do not require GPU and can run in a pure Python environment.
"""

import json
import sys
import unittest
from pathlib import Path

# Ensure infinilm is importable when running tests directly. The Python
# sources live under <repo root>/python.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
for _path in (str(PYTHON_ROOT), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from infinilm.agents import (
        Function,
        FunctionCallParser,
        ReasoningParser,
        StreamingParseResult,
        Tool,
        ToolCallItem,
    )
    from infinilm.agents.utils import parse_arguments
except Exception:
    # ``infinilm/__init__.py`` eagerly loads the native engine stack, which
    # the pure-Python agents package does not need. Register a minimal
    # namespace stub so ``infinilm.agents`` can be imported standalone in
    # environments without the compiled extension (CPU-only test setups).
    import types

    sys.modules.pop("infinilm", None)
    _infinilm_stub = types.ModuleType("infinilm")
    _infinilm_stub.__path__ = [str(PYTHON_ROOT / "infinilm")]
    sys.modules["infinilm"] = _infinilm_stub

    from infinilm.agents import (
        Function,
        FunctionCallParser,
        ReasoningParser,
        StreamingParseResult,
        Tool,
        ToolCallItem,
    )
    from infinilm.agents.utils import parse_arguments

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_weather_tools(backend="llama31"):
    """Return a list of Tool objects for get_weather / get_time tools."""
    weather_tool = Tool(
        type="function",
        function=Function(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        ),
    )
    time_tool = Tool(
        type="function",
        function=Function(
            name="get_time",
            description="Get current time",
            parameters={"type": "object", "properties": {}},
        ),
    )
    return [weather_tool, time_tool]


def _make_weather_dict_tools():
    """Return weather/time tools as plain dicts (like from HTTP JSON)."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get current time",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


# ===========================================================================
# ReasoningParser
# ===========================================================================


class TestReasoningParser(unittest.TestCase):
    """Tests for ReasoningParser and BaseReasoningFormatDetector."""

    # ---- GLM45 (keep existing) ------------------------------------------------

    def test_glm45_detector_non_streaming(self):
        parser = ReasoningParser(reasoning_parser_name="glm45")
        text = "thinkI need to think step by step./thinkThe answer is 42."
        results = parser.extract_reasoning_content(text)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].reasoning_content, "I need to think step by step.")
        self.assertEqual(results[0].complete, True)
        self.assertEqual(results[1].normal_text, "The answer is 42.")

    def test_glm45_detector_streaming(self):
        parser = ReasoningParser(reasoning_parser_name="glm45")
        # Simulate streaming: prefill already started reasoning.
        # The start_token is not in the stream because prefill included it.
        # The base detector treats content as reasoning until end_token appears.
        deltas = ["Thinking...", " done", "/thinkAnswer: 42"]
        current = ""
        normal_parts = []
        reasoning_parts = []
        for d in deltas:
            prev = current
            current += d
            res = parser.extract_reasoning_content_streaming(prev, current, d)
            if res.reasoning_content:
                reasoning_parts.append(res.reasoning_content)
            if res.normal_text:
                normal_parts.append(res.normal_text)

        # Verify that reasoning content is emitted incrementally
        self.assertIn("Thinking... done", "".join(reasoning_parts))
        # After end_token, normal text is emitted
        self.assertIn("Answer: 42", "".join(normal_parts))

    # ---- Think tag detector ---------------------------------------------------

    def test_think_tag_detector_non_streaming(self):
        """ReasoningParser(name='think') extracts content between <thinking> tags."""
        parser = ReasoningParser(reasoning_parser_name="think")
        text = (
            "<thinking>Let me reason about this carefully.</thinking>The answer is 42."
        )
        results = parser.extract_reasoning_content(text)
        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[0].reasoning_content, "Let me reason about this carefully."
        )
        self.assertEqual(results[0].complete, True)
        self.assertEqual(results[1].normal_text, "The answer is 42.")

    def test_think_tag_detector_streaming(self):
        """Stream tokens through ThinkTagDetector and verify accumulation."""
        parser = ReasoningParser(reasoning_parser_name="think")
        text = "<thinking>Step 1: Analyse.</thinking>Final answer: yes."
        current = ""
        reasoning_parts = []
        normal_parts = []
        # Feed character by character to stress-test streaming
        for ch in text:
            prev = current
            current += ch
            res = parser.extract_reasoning_content_streaming(prev, current, ch)
            if res.reasoning_content:
                reasoning_parts.append(res.reasoning_content)
            if res.normal_text:
                normal_parts.append(res.normal_text)

        self.assertEqual("".join(reasoning_parts), "Step 1: Analyse.")
        self.assertEqual("".join(normal_parts), "Final answer: yes.")

    def test_think_tag_detector_streaming_word_chunks(self):
        """Stream larger word-level chunks through ThinkTagDetector."""
        parser = ReasoningParser(reasoning_parser_name="think")
        # Use a single <thinking>...</thinking> pair split across chunks.
        chunks = ["<thin", "king>First ", "thought.</thinking>done"]
        current = ""
        reasoning_parts = []
        normal_parts = []
        for chunk in chunks:
            prev = current
            current += chunk
            res = parser.extract_reasoning_content_streaming(prev, current, chunk)
            if res.reasoning_content:
                reasoning_parts.append(res.reasoning_content)
            if res.normal_text:
                normal_parts.append(res.normal_text)

        full_reasoning = "".join(reasoning_parts)
        self.assertIn("First thought.", full_reasoning)
        self.assertEqual("".join(normal_parts), "done")

    # ---- No-op parser ----------------------------------------------------------

    def test_no_op_parser(self):
        parser = ReasoningParser(reasoning_parser_name=None)
        res = parser.extract_reasoning_content_streaming("", "hello", "hello")
        self.assertEqual(res.normal_text, "hello")
        self.assertEqual(res.reasoning_content, "")

    # ---- Think tag without think tags -------------------------------------------

    def test_think_tag_no_think(self):
        """Text without <thinking> tags should all be returned as normal_text."""
        parser = ReasoningParser(reasoning_parser_name="think")
        text = "Hello, world! No reasoning here, just a plain response."
        results = parser.extract_reasoning_content(text)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].reasoning_content, "")
        self.assertEqual(results[0].normal_text, text)
        self.assertEqual(results[0].complete, True)

    def test_think_tag_partial_no_close(self):
        """Text with opening tag but no closing tag."""
        parser = ReasoningParser(reasoning_parser_name="think")
        text = "<thinking>Partial reasoning here"
        results = parser.extract_reasoning_content(text)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].reasoning_content, "Partial reasoning here")
        self.assertEqual(results[0].normal_text, "")
        self.assertEqual(results[0].complete, False)

    # ---- DeepSeek-R1 / QwQ (<think> tag) ---------------------------------------

    def test_deepseek_r1_detector_non_streaming(self):
        """deepseek-r1 uses the short <think> tag, not <thinking>."""
        parser = ReasoningParser(reasoning_parser_name="deepseek-r1")
        text = "<think>hidden reasoning</think>answer"
        results = parser.extract_reasoning_content(text)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].reasoning_content, "hidden reasoning")
        self.assertEqual(results[0].complete, True)
        self.assertEqual(results[1].normal_text, "answer")

    def test_qwq_detector_non_streaming(self):
        """qwq is an alias for the same <think> tag format."""
        parser = ReasoningParser(reasoning_parser_name="qwq")
        text = "<think>Let me think.</think>42"
        results = parser.extract_reasoning_content(text)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].reasoning_content, "Let me think.")
        self.assertEqual(results[1].normal_text, "42")

    def test_deepseek_r1_detector_streaming(self):
        """Stream a <think> block character by character."""
        parser = ReasoningParser(reasoning_parser_name="deepseek-r1")
        text = "<think>Step 1: analyse the problem.</think>The answer is 42."
        current = ""
        reasoning_parts = []
        normal_parts = []
        for ch in text:
            prev = current
            current += ch
            res = parser.extract_reasoning_content_streaming(prev, current, ch)
            if res.reasoning_content:
                reasoning_parts.append(res.reasoning_content)
            if res.normal_text:
                normal_parts.append(res.normal_text)

        self.assertEqual("".join(reasoning_parts), "Step 1: analyse the problem.")
        self.assertEqual("".join(normal_parts), "The answer is 42.")

    def test_deepseek_r1_detector_chunked_streaming(self):
        """Stream a <think> block in chunks split across tag boundaries."""
        parser = ReasoningParser(reasoning_parser_name="deepseek-r1")
        chunks = ["<thi", "nk>deep ", "thoughts</thi", "nk>final answer"]
        current = ""
        reasoning_parts = []
        normal_parts = []
        for chunk in chunks:
            prev = current
            current += chunk
            res = parser.extract_reasoning_content_streaming(prev, current, chunk)
            if res.reasoning_content:
                reasoning_parts.append(res.reasoning_content)
            if res.normal_text:
                normal_parts.append(res.normal_text)

        self.assertEqual("".join(reasoning_parts), "deep thoughts")
        self.assertEqual("".join(normal_parts), "final answer")

    def test_deepseek_alias_does_not_match_thinking_tag(self):
        """deepseek-r1 must not silently fall back to <thinking> parsing."""
        parser = ReasoningParser(reasoning_parser_name="deepseek-r1")
        text = "<thinking>not the r1 format</thinking>answer"
        results = parser.extract_reasoning_content(text)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].normal_text, text)
        self.assertEqual(results[0].reasoning_content, "")

    # ---- Edge cases ------------------------------------------------------------

    def test_unknown_reasoning_parser_name(self):
        """Passing an unknown parser name produces empty detectors (same as no-op)."""
        parser = ReasoningParser(reasoning_parser_name="nonexistent")
        text = "<thinking>test</thinking>answer"
        results = parser.extract_reasoning_content(text)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].normal_text, text)
        self.assertEqual(results[0].reasoning_content, "")


# ===========================================================================
# ToolCallParser
# ===========================================================================


class TestToolCallParser(unittest.TestCase):
    """Tests for FunctionCallParser with various backends."""

    # ---- Registration (keep existing) ------------------------------------------

    def test_llama_registration(self):
        parser = FunctionCallParser(tool_call_parser="llama31")
        self.assertIsNotNone(parser.detector)

    def test_glm_registration(self):
        parser = FunctionCallParser(tool_call_parser="glm")
        self.assertIsNotNone(parser.detector)

    def test_non_stream_simple(self):
        parser = FunctionCallParser(tool_call_parser="llama31", tools=[])
        text = "Hello world"
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(normal, text)
        self.assertEqual(calls, [])

    # ---- Llama 3.1 non-stream with tools ---------------------------------------

    def test_llama31_non_stream_with_tools(self):
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="llama31", tools=tools)
        text = (
            'Hello<|python_tag|>{"name":"get_weather", "arguments":{"city":"Beijing"}}'
        )
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(normal.strip(), "Hello")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        params = json.loads(calls[0].parameters)
        self.assertEqual(params, {"city": "Beijing"})

    def test_llama31_non_stream_with_tools_no_prefix(self):
        """Tool call starts immediately without leading text."""
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="llama31", tools=tools)
        text = '<|python_tag|>{"name":"get_time", "arguments":{}}'
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(normal.strip(), "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_time")
        params = json.loads(calls[0].parameters)
        self.assertEqual(params, {})

    def test_llama31_non_stream_no_tool_match(self):
        """Model calls a tool not in the provided list."""
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="llama31", tools=tools)
        text = '<|python_tag|>{"name":"nonexistent_fn", "arguments":{"x":1}}'
        normal, calls = parser.parse_non_stream(text)
        # Undefined tools are filtered out by the detector.
        self.assertEqual(len(calls), 0)

    # ---- Llama 3.1 streaming with tools ----------------------------------------

    def test_llama31_stream_with_tools(self):
        """Stream a complete tool call in chunks; verify name then params."""
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="llama31", tools=tools)
        call_text = (
            '<|python_tag|>{"name":"get_weather", "arguments":{"city":"Beijing"}}'
        )

        # Feed in two chunks to simulate streaming.
        # Split right after the bot token so the first chunk never contains
        # partial JSON keys / values that could confuse the incremental parser.
        bot_len = len("<|python_tag|>")
        result1 = parser.parse_streaming_increment("", call_text[:bot_len], tools)
        result2 = parser.parse_streaming_increment("", call_text[bot_len:], tools)
        _, end_calls = parser.parse_stream_end()

        all_calls = result1.calls + result2.calls + end_calls
        all_names = [c.name for c in all_calls if c.name]
        self.assertIn("get_weather", all_names)
        # All streamed argument fragments must reassemble into the full JSON.
        streamed_args = "".join(c.parameters for c in all_calls)
        self.assertEqual(json.loads(streamed_args), {"city": "Beijing"})

    def test_llama31_stream_with_tools_single_chunk(self):
        """Stream entire JSON block at once; arguments must not be lost."""
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="llama31", tools=tools)
        call_text = '<|python_tag|>{"name":"get_time", "arguments":{}}'
        result = parser.parse_streaming_increment("", call_text, tools)
        _, end_calls = parser.parse_stream_end()

        all_calls = result.calls + end_calls
        names = [c.name for c in all_calls if c.name]
        self.assertIn("get_time", names)
        streamed_args = "".join(c.parameters for c in all_calls)
        self.assertEqual(json.loads(streamed_args), {})

    def test_llama31_stream_token_by_token(self):
        """Stream a tool call one character at a time; params must survive."""
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="llama31", tools=tools)
        call_text = (
            '<|python_tag|>{"name":"get_weather", "arguments":{"city":"Beijing"}}'
        )

        names = []
        arg_parts = []
        for ch in call_text:
            res = parser.parse_streaming_increment("", ch, tools)
            for call in res.calls:
                if call.name:
                    names.append(call.name)
                if call.parameters:
                    arg_parts.append(call.parameters)
        _, end_calls = parser.parse_stream_end()
        for call in end_calls:
            if call.name:
                names.append(call.name)
            if call.parameters:
                arg_parts.append(call.parameters)

        self.assertIn("get_weather", names)
        self.assertEqual(json.loads("".join(arg_parts)), {"city": "Beijing"})

    def test_llama31_stream_end_flushes_buffered_args(self):
        """parse_stream_end() must flush buffered arguments.

        Regression test for the reviewer repro: a parser constructed without
        bound tools used to return an empty result from parse_stream_end(),
        leaving the streamed arguments stuck in the detector buffer.
        """
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="llama31")
        call_text = (
            '<|python_tag|>{"name":"get_weather", "arguments":{"city":"Beijing"}}'
        )
        # Split inside the arguments object so part of it stays buffered.
        split = len('<|python_tag|>{"name":"get_weather", "arguments":{"ci')
        increment_results = [
            parser.parse_streaming_increment("", call_text[:split], tools),
            parser.parse_streaming_increment("", call_text[split:], tools),
        ]
        normal, calls = parser.parse_stream_end(tools)

        all_calls = increment_results[0].calls + increment_results[1].calls + calls
        self.assertIn("get_weather", [c.name for c in all_calls if c.name])
        streamed_args = "".join(c.parameters for c in all_calls)
        self.assertEqual(json.loads(streamed_args), {"city": "Beijing"})
        self.assertEqual(normal, "")

    def test_llama31_plain_json_response_not_swallowed_streaming(self):
        """A JSON answer without a tool name must not be held back forever."""
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="llama31", tools=tools)
        text = '{"answer": 42}'
        parts = []
        for ch in text:
            res = parser.parse_streaming_increment("", ch, tools)
            if res.normal_text:
                parts.append(res.normal_text)
        normal, calls = parser.parse_stream_end()
        if normal:
            parts.append(normal)
        self.assertEqual("".join(parts), text)
        self.assertEqual(calls, [])

    def test_llama31_plain_json_response_not_swallowed_non_stream(self):
        """Non-streaming parse also passes plain JSON answers through."""
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="llama31", tools=tools)
        text = '{"answer": 42}'
        normal, calls = parser.parse_non_stream(text, tools=tools)
        self.assertEqual(normal, text)
        self.assertEqual(calls, [])

    def test_llama31_stream_multiple_tools(self):
        """Parse multiple tool calls in separate stream chunks."""
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="llama31", tools=tools)
        chunk1 = '<|python_tag|>{"name":"get_weather", "arguments":{"city":"Beijing"}};'
        chunk2 = '<|python_tag|>{"name":"get_time", "arguments":{}}'

        result1 = parser.parse_streaming_increment("", chunk1, tools)
        # Reset before the second chunk so state from the first call does not leak.
        parser.detector.clear()
        result2 = parser.parse_streaming_increment("", chunk2, tools)
        all_names = []
        all_params = []
        for r in (result1, result2):
            for c in r.calls:
                if c.name:
                    all_names.append(c.name)
                if c.parameters:
                    all_params.append(c.parameters)

        self.assertIn("get_weather", all_names)
        self.assertIn("get_time", all_names)

    # ---- GLM non-stream with tools ---------------------------------------------

    def test_glm_non_stream_with_tools(self):
        """Parse GLM-style <tool_call> format."""
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="glm", tools=tools)
        text = (
            "Let me check.<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n"
            "</tool_call>"
        )
        normal, calls = parser.parse_non_stream(text)
        self.assertIn("Let me check.", normal)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        params = json.loads(calls[0].parameters)
        self.assertEqual(params, {"city": "Beijing"})

    def test_glm_non_stream_multiple_tools(self):
        """Parse multiple GLM tool calls in one text."""
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="glm", tools=tools)
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n"
            "</tool_call>\n"
            "<tool_call>get_time\n"
            "</tool_call>"
        )
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[1].name, "get_time")
        params0 = json.loads(calls[0].parameters)
        self.assertEqual(params0, {"city": "Beijing"})

    # ---- GLM streaming with tools -----------------------------------------------

    def test_glm_stream_with_tools(self):
        """Stream GLM tool call token by token."""
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="glm", tools=tools)
        call_text = (
            "some_prefix<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n"
            "</tool_call>tail"
        )
        result = parser.parse_streaming_increment("", call_text, tools)
        # Should produce at least one ToolCallItem with name get_weather
        names = [c.name for c in result.calls if c.name]
        self.assertIn("get_weather", names)

    def test_glm_detector_clear_resets_all_state(self):
        """clear() must reset GLM streaming state so the parser is reusable."""
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="glm", tools=tools)
        # Leave the detector mid-tool-call with dirty streaming state.
        parser.parse_streaming_increment(
            "", "<tool_call>get_weather\n<arg_key>city</arg_key>"
        )
        parser.detector.clear()
        # A fresh tool call must parse cleanly after the reset.
        result = parser.parse_streaming_increment(
            "", "<tool_call>get_time\n</tool_call>"
        )
        _, end_calls = parser.parse_stream_end()
        all_calls = result.calls + end_calls
        self.assertIn("get_time", [c.name for c in all_calls if c.name])
        streamed_args = "".join(c.parameters for c in all_calls)
        self.assertEqual(json.loads(streamed_args), {})

    # ---- has_tool_call ---------------------------------------------------------

    def test_has_tool_call(self):
        """has_tool_call detects bot_token in text."""
        # Llama
        parser = FunctionCallParser(
            tool_call_parser="llama31", tools=_make_weather_tools()
        )
        self.assertFalse(parser.has_tool_call("plain text"))
        self.assertTrue(parser.has_tool_call("prefix<|python_tag|>json"))
        self.assertTrue(parser.has_tool_call('{"name":"f"}'))

        # GLM
        parser2 = FunctionCallParser(
            tool_call_parser="glm", tools=_make_weather_tools()
        )
        self.assertFalse(parser2.has_tool_call("plain text"))
        self.assertTrue(
            parser2.has_tool_call("before\n<tool_call>fn\n</tool_call>after")
        )

    def test_has_tool_call_no_tools(self):
        """has_tool_call returns False when no tools are configured."""
        parser = FunctionCallParser(tool_call_parser="llama31", tools=[])
        self.assertFalse(parser.has_tool_call("<|python_tag|>json"))
        self.assertFalse(parser.has_tool_call("plain"))

        parser2 = FunctionCallParser(tool_call_parser="glm", tools=None)
        self.assertFalse(parser2.has_tool_call("<tool_call>fn</tool_call>"))

    # ---- parse_stream_end -------------------------------------------------------

    def test_parse_stream_end(self):
        """parse_stream_end flushes buffered state."""
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="llama31", tools=tools)
        normal, calls = parser.parse_stream_end()
        self.assertEqual(normal, "")
        self.assertEqual(calls, [])

    def test_parse_stream_end_no_tools(self):
        """parse_stream_end returns empty when no tools."""
        parser = FunctionCallParser(tool_call_parser="llama31")
        normal, calls = parser.parse_stream_end()
        self.assertEqual(normal, "")
        self.assertEqual(calls, [])

    # ---- tools dict conversion --------------------------------------------------

    def test_tools_dict_conversion(self):
        """Passing plain dict tools works the same as Tool objects."""
        dict_tools = _make_weather_dict_tools()
        parser = FunctionCallParser(tool_call_parser="llama31", tools=dict_tools)
        text = '<|python_tag|>{"name":"get_weather", "arguments":{"city":"Shanghai"}}'
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        params = json.loads(calls[0].parameters)
        self.assertEqual(params, {"city": "Shanghai"})

    def test_tools_dict_conversion_glm(self):
        """GLM parser also accepts plain dict tools."""
        dict_tools = _make_weather_dict_tools()
        parser = FunctionCallParser(tool_call_parser="glm", tools=dict_tools)
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>Shanghai</arg_value>\n"
            "</tool_call>"
        )
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        params = json.loads(calls[0].parameters)
        self.assertEqual(params, {"city": "Shanghai"})

    def test_tools_override_in_parse_call(self):
        """tools passed at parse time override constructor tools."""
        default_tools = _make_weather_tools()[0:1]  # only get_weather
        full_tools = _make_weather_tools()  # get_weather + get_time
        parser = FunctionCallParser(tool_call_parser="llama31", tools=default_tools)
        text = '<|python_tag|>{"name":"get_time", "arguments":{}}'
        normal, calls = parser.parse_non_stream(text, tools=full_tools)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_time")


# ===========================================================================
# Glm4Chat0414Detector (GLM-4-9B-0414 metadata format)
# ===========================================================================


class TestGlm4Chat0414Parser(unittest.TestCase):
    """Tests for the GLM-4-9B-0414 metadata-style tool call format:

    function_name
    {"arg": "value"}
    """

    def test_registration(self):
        for name in ("glm4", "glm49b", "glm4-9b-0414", "glm-4-9b-0414"):
            parser = FunctionCallParser(tool_call_parser=name)
            self.assertIsNotNone(parser.detector)

    def test_non_stream_simple_call(self):
        """Exact shape observed from GLM-4-9B-0414 on the server."""
        parser = FunctionCallParser(
            tool_call_parser="glm4-9b-0414", tools=_make_weather_dict_tools()
        )
        text = 'get_weather\n{"city": "北京"}'
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(normal, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters), {"city": "北京"})

    def test_non_stream_prefix_text(self):
        parser = FunctionCallParser(
            tool_call_parser="glm4-9b-0414", tools=_make_weather_dict_tools()
        )
        text = 'I will check the weather.\nget_weather\n{"city": "Shanghai"}'
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(normal, "I will check the weather.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Shanghai"})

    def test_non_stream_plain_text_untouched(self):
        parser = FunctionCallParser(
            tool_call_parser="glm4-9b-0414", tools=_make_weather_dict_tools()
        )
        text = "Just a normal answer with no tool call."
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(normal, text)
        self.assertEqual(calls, [])

    def test_non_stream_plain_json_answer_not_parsed_as_call(self):
        parser = FunctionCallParser(
            tool_call_parser="glm4-9b-0414", tools=_make_weather_dict_tools()
        )
        text = 'The result is:\n{"answer": 42}'
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(calls, [])
        self.assertIn('{"answer": 42}', normal)

    def test_non_stream_unknown_tool_name_kept_as_text(self):
        parser = FunctionCallParser(
            tool_call_parser="glm4-9b-0414", tools=_make_weather_dict_tools()
        )
        text = 'nonexistent_fn\n{"x": 1}'
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(calls, [])
        self.assertIn("nonexistent_fn", normal)

    def test_non_stream_multiple_calls_with_assistant_marker(self):
        """Parallel calls are separated by <|assistant|> markers."""
        parser = FunctionCallParser(
            tool_call_parser="glm4-9b-0414", tools=_make_weather_dict_tools()
        )
        marker = "<|" + "assistant|>"
        text = 'get_weather\n{"city": "Beijing"}' + marker + "get_time\n{}"
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(normal, "")
        self.assertEqual([c.name for c in calls], ["get_weather", "get_time"])
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Beijing"})
        self.assertEqual(json.loads(calls[1].parameters), {})

    def test_stream_char_by_char(self):
        parser = FunctionCallParser(
            tool_call_parser="glm4-9b-0414", tools=_make_weather_dict_tools()
        )
        text = 'get_weather\n{"city": "北京"}'
        names = []
        arg_parts = []
        for ch in text:
            res = parser.parse_streaming_increment("", ch)
            for call in res.calls:
                if call.name:
                    names.append(call.name)
                if call.parameters:
                    arg_parts.append(call.parameters)
        normal, end_calls = parser.parse_stream_end()
        for call in end_calls:
            if call.name:
                names.append(call.name)
            if call.parameters:
                arg_parts.append(call.parameters)

        self.assertEqual(names, ["get_weather"])
        self.assertEqual(json.loads("".join(arg_parts)), {"city": "北京"})
        self.assertEqual(normal, "")

    def test_stream_plain_text_not_held_back(self):
        """Text that cannot become a tool call must stream out immediately."""
        parser = FunctionCallParser(
            tool_call_parser="glm4-9b-0414", tools=_make_weather_dict_tools()
        )
        parts = []
        for chunk in ["Hello ", "world, ", "no tools ", "here."]:
            res = parser.parse_streaming_increment("", chunk)
            if res.normal_text:
                parts.append(res.normal_text)
        self.assertEqual("".join(parts), "Hello world, no tools here.")

    def test_stream_name_held_until_decidable(self):
        """A line matching a tool-name prefix is held back, then resolves."""
        parser = FunctionCallParser(
            tool_call_parser="glm4-9b-0414", tools=_make_weather_dict_tools()
        )
        res1 = parser.parse_streaming_increment("", "get_wea")
        self.assertEqual(res1.normal_text, "")
        self.assertEqual(res1.calls, [])
        res2 = parser.parse_streaming_increment("", 'ther\n{"city": "Beijing"}')
        self.assertEqual([c.name for c in res2.calls if c.name], ["get_weather"])
        args = "".join(c.parameters for c in res2.calls)
        self.assertEqual(json.loads(args), {"city": "Beijing"})

    def test_stream_held_line_released_when_not_a_tool(self):
        """A held line that turns out not to be a tool call is released."""
        parser = FunctionCallParser(
            tool_call_parser="glm4-9b-0414", tools=_make_weather_dict_tools()
        )
        res1 = parser.parse_streaming_increment("", "get_we")
        self.assertEqual(res1.normal_text, "")
        # The line completes to something that is not a tool name.
        res2 = parser.parse_streaming_increment("", "lcome to Beijing!")
        self.assertIn("get_welcome to Beijing!", res2.normal_text)

    def test_stream_multiple_calls_with_split_marker(self):
        """Parallel calls whose <|assistant|> separator arrives split."""
        marker = "<|" + "assistant|>"
        parser = FunctionCallParser(
            tool_call_parser="glm4-9b-0414", tools=_make_weather_dict_tools()
        )
        text = 'get_weather\n{"city": "北京"}' + marker + "get_time\n{}"
        names = []
        arg_parts = []
        # Feed in fixed awkward chunks that split inside the marker.
        chunks = [text[:15], text[15:27], text[27:33], text[33:41], text[41:]]
        fed = ""
        for chunk in chunks:
            self.assertTrue(text.startswith(fed + chunk))
            fed += chunk
            res = parser.parse_streaming_increment("", chunk)
            for call in res.calls:
                if call.name:
                    names.append(call.name)
                if call.parameters:
                    arg_parts.append(call.parameters)
        self.assertEqual(fed, text)
        normal, end_calls = parser.parse_stream_end()
        for call in end_calls:
            if call.name:
                names.append(call.name)
            if call.parameters:
                arg_parts.append(call.parameters)

        self.assertEqual(names, ["get_weather", "get_time"])
        self.assertNotIn(marker, normal)
        self.assertNotIn("<|", normal)

    def test_stream_marker_fragment_in_plain_text_not_duplicated(self):
        """A partial marker inside plain text must not leak or duplicate."""
        parser = FunctionCallParser(
            tool_call_parser="glm4-9b-0414", tools=_make_weather_dict_tools()
        )
        marker = "<|" + "assistant|>"
        text = "第一段" + marker + "第二段"
        parts = []
        for i in range(len(text)):
            res = parser.parse_streaming_increment("", text[i])
            if res.normal_text:
                parts.append(res.normal_text)
        normal, _ = parser.parse_stream_end()
        if normal:
            parts.append(normal)
        combined = "".join(parts)
        # Both text parts survive exactly once; the separator is dropped.
        self.assertEqual(combined.count("第一段"), 1)
        self.assertEqual(combined.count("第二段"), 1)
        self.assertNotIn("<|", combined)


# ===========================================================================
# AgentStreamParser / parse_full_response (protocol-level helpers)
# ===========================================================================


class TestAgentStreamParser(unittest.TestCase):
    """Per-request stream parsing into protocol-ready deltas."""

    def test_tool_call_stream_into_openai_deltas(self):
        from infinilm.agents import AgentStreamParser

        parser = AgentStreamParser(
            tool_call_parser="llama31", tools=_make_weather_dict_tools()
        )
        text = '<|python_tag|>{"name":"get_weather", "arguments":{"city":"Beijing"}}'

        emitted = []
        for ch in text:
            delta = parser.process_delta(ch)
            if delta.tool_calls:
                emitted.extend(delta.tool_calls)
        end = parser.flush()
        emitted.extend(end.tool_calls)

        names = [tc["function"]["name"] for tc in emitted if tc["function"]["name"]]
        args = "".join(tc["function"]["arguments"] for tc in emitted)
        self.assertEqual(names, ["get_weather"])
        self.assertEqual(json.loads(args), {"city": "Beijing"})
        self.assertTrue(parser.has_tool_calls)
        # All deltas must be OpenAI-protocol shaped.
        for tc in emitted:
            self.assertEqual(tc["type"], "function")
            self.assertIn("index", tc)
            self.assertIn("id", tc)

    def test_reasoning_and_content_split(self):
        from infinilm.agents import AgentStreamParser

        parser = AgentStreamParser(reasoning_parser="deepseek-r1")
        text = "<think>hidden</think>answer"
        reasoning_parts, content_parts = [], []
        for ch in text:
            delta = parser.process_delta(ch)
            if delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)
            if delta.content:
                content_parts.append(delta.content)
        self.assertEqual("".join(reasoning_parts), "hidden")
        self.assertEqual("".join(content_parts), "answer")
        self.assertFalse(parser.has_tool_calls)

    def test_no_parsers_is_passthrough(self):
        from infinilm.agents import AgentStreamParser

        parser = AgentStreamParser()
        delta = parser.process_delta("hello")
        self.assertEqual(delta.content, "hello")
        self.assertEqual(delta.reasoning_content, "")
        self.assertEqual(delta.tool_calls, [])
        self.assertEqual(parser.flush().content, "")

    def test_instances_do_not_share_state(self):
        from infinilm.agents import AgentStreamParser

        tools = _make_weather_dict_tools()
        parser_a = AgentStreamParser(tool_call_parser="llama31", tools=tools)
        parser_b = AgentStreamParser(tool_call_parser="llama31", tools=tools)
        parser_a.process_delta('<|python_tag|>{"name":"get_weather", ')
        delta_b = parser_b.process_delta("plain text")
        self.assertEqual(delta_b.content, "plain text")
        self.assertEqual(delta_b.tool_calls, [])


class TestParseFullResponse(unittest.TestCase):
    """One-shot parsing for non-streaming responses."""

    def test_tool_call_response(self):
        from infinilm.agents import parse_full_response

        text = '<|python_tag|>{"name":"get_weather", "arguments":{"city":"Beijing"}}'
        reasoning, content, tool_calls = parse_full_response(
            text,
            tool_call_parser="llama31",
            tools=_make_weather_dict_tools(),
        )
        self.assertIsNone(reasoning)
        self.assertEqual(content, "")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["id"], "call_0")
        self.assertEqual(tool_calls[0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]),
            {"city": "Beijing"},
        )

    def test_reasoning_response_glm4_metadata_format(self):
        from infinilm.agents import parse_full_response

        text = 'get_weather\n{"city": "北京"}'
        reasoning, content, tool_calls = parse_full_response(
            text,
            tool_call_parser="glm4-9b-0414",
            tools=_make_weather_dict_tools(),
        )
        self.assertIsNone(reasoning)
        self.assertEqual(content, "")
        self.assertEqual(tool_calls[0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]),
            {"city": "北京"},
        )

    def test_plain_text_response(self):
        from infinilm.agents import parse_full_response

        reasoning, content, tool_calls = parse_full_response(
            "just an answer",
            tool_call_parser="llama31",
            tools=_make_weather_dict_tools(),
        )
        self.assertIsNone(reasoning)
        self.assertEqual(content, "just an answer")
        self.assertEqual(tool_calls, [])


# ===========================================================================
# Qwen3XmlDetector (Qwen3 xml tool-call format)
# ===========================================================================


def _qwen3_call(name: str, args: dict) -> str:
    """Build a Qwen3-format tool_call block."""
    open_tag = "<" + "tool_call" + ">"
    close_tag = "</" + "tool_call" + ">"
    return f'{open_tag}\n{{"name": "{name}", "arguments": {json.dumps(args, ensure_ascii=False)}}}\n{close_tag}'


class TestQwen3XmlParser(unittest.TestCase):
    def test_registration(self):
        for alias in ("qwen3", "qwen3-30b-a3b"):
            parser = FunctionCallParser(tool_call_parser=alias)
            self.assertIsNotNone(parser.detector)

    def test_non_stream_single_call(self):
        parser = FunctionCallParser(
            tool_call_parser="qwen3", tools=_make_weather_dict_tools()
        )
        text = "I will check the weather.\n" + _qwen3_call(
            "get_weather", {"city": "北京"}
        )
        normal, calls = parser.parse_non_stream(text)
        self.assertIn("I will check the weather.", normal)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters), {"city": "北京"})

    def test_non_stream_two_calls(self):
        parser = FunctionCallParser(
            tool_call_parser="qwen3", tools=_make_weather_dict_tools()
        )
        text = (
            _qwen3_call("get_weather", {"city": "北京"})
            + "\n"
            + _qwen3_call("get_time", {})
        )
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(normal, "")
        self.assertEqual([c.name for c in calls], ["get_weather", "get_time"])
        self.assertEqual(json.loads(calls[0].parameters), {"city": "北京"})
        self.assertEqual(json.loads(calls[1].parameters), {})

    def test_non_stream_unknown_tool_dropped(self):
        parser = FunctionCallParser(
            tool_call_parser="qwen3", tools=_make_weather_dict_tools()
        )
        text = _qwen3_call("no_such_tool", {"x": 1})
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(calls, [])

    def test_stream_char_by_char(self):
        parser = FunctionCallParser(
            tool_call_parser="qwen3", tools=_make_weather_dict_tools()
        )
        text = "Let me see.\n" + _qwen3_call("get_weather", {"city": "Beijing"})
        names, arg_parts, normals = [], [], []
        for ch in text:
            res = parser.parse_streaming_increment("", ch)
            for c in res.calls:
                if c.name:
                    names.append(c.name)
                if c.parameters:
                    arg_parts.append(c.parameters)
            if res.normal_text:
                normals.append(res.normal_text)
        normal_end, end_calls = parser.parse_stream_end()
        for c in end_calls:
            if c.name:
                names.append(c.name)
            if c.parameters:
                arg_parts.append(c.parameters)

        self.assertEqual(names, ["get_weather"])
        self.assertEqual(json.loads("".join(arg_parts)), {"city": "Beijing"})
        self.assertIn("Let me see.", "".join(normals) + normal_end)

    def test_stream_two_consecutive_calls(self):
        parser = FunctionCallParser(
            tool_call_parser="qwen3", tools=_make_weather_dict_tools()
        )
        text = _qwen3_call("get_weather", {"city": "北京"}) + _qwen3_call(
            "get_time", {}
        )
        names, arg_parts = [], []
        for ch in text:
            res = parser.parse_streaming_increment("", ch)
            for c in res.calls:
                if c.name:
                    names.append(c.name)
                if c.parameters:
                    arg_parts.append(c.parameters)
        normal_end, end_calls = parser.parse_stream_end()
        for c in end_calls:
            if c.name:
                names.append(c.name)
            if c.parameters:
                arg_parts.append(c.parameters)

        self.assertEqual(names, ["get_weather", "get_time"])
        # Arguments arrive per call; concatenate-and-split by object boundaries.
        combined = "".join(arg_parts)
        self.assertIn("北京", combined)

    def test_stream_plain_text_not_held_forever(self):
        parser = FunctionCallParser(
            tool_call_parser="qwen3", tools=_make_weather_dict_tools()
        )
        text = "Just a plain answer without any tool call."
        parts = []
        for ch in text:
            res = parser.parse_streaming_increment("", ch)
            if res.normal_text:
                parts.append(res.normal_text)
        normal_end, end_calls = parser.parse_stream_end()
        self.assertEqual(end_calls, [])
        # Only a trailing partial-token fragment may be held back.
        emitted = "".join(parts) + normal_end
        self.assertTrue(text.startswith(emitted) or emitted.startswith(text[:-9]))

    def test_truncated_call_released_at_finish(self):
        parser = FunctionCallParser(
            tool_call_parser="qwen3", tools=_make_weather_dict_tools()
        )
        block = _qwen3_call("get_weather", {"city": "北京"})
        truncated = block[: len(block) // 2]  # cut inside the block
        for ch in truncated:
            parser.parse_streaming_increment("", ch)
        normal_end, end_calls = parser.parse_stream_end()
        # Nothing parsed as a call; the truncated block must not vanish.
        self.assertEqual(end_calls, [])
        self.assertTrue(normal_end)


# ===========================================================================
# adapt_messages (GLM-4 metadata tool-history convention)
# ===========================================================================


class TestAdaptMessages(unittest.TestCase):
    """Adaptation of OpenAI tool history for GLM-4 metadata models."""

    def test_no_parser_passes_messages_through(self):
        from infinilm.agents.message_adapter import adapt_messages

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        self.assertEqual(adapt_messages(messages, None), messages)

    def test_other_parsers_keep_openai_roles(self):
        from infinilm.agents.message_adapter import adapt_messages

        messages = [
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        self.assertEqual(adapt_messages(messages, "llama31"), messages)

    def test_multimodal_message_kept_untouched(self):
        from infinilm.agents.message_adapter import adapt_messages

        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "http://x/cat.jpg"}},
            ],
        }
        self.assertEqual(adapt_messages([message], "glm4-9b-0414"), [message])

    def test_tool_messages_become_observation(self):
        from infinilm.agents.message_adapter import adapt_messages

        messages = [
            {"role": "tool", "tool_call_id": "c1", "content": '{"aqi": 42}'},
        ]
        self.assertEqual(
            adapt_messages(messages, "glm4-9b-0414"),
            [
                {
                    "role": "observation",
                    "tool_call_id": "c1",
                    "content": '{"aqi": 42}',
                }
            ],
        )

    def test_assistant_tool_calls_become_metadata_messages(self):
        from infinilm.agents.message_adapter import adapt_messages

        messages = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Beijing"}',
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            },
        ]
        self.assertEqual(
            adapt_messages(messages, "glm4-9b-0414"),
            [
                {"role": "assistant", "content": "Let me check."},
                {
                    "role": "assistant",
                    "metadata": "get_weather",
                    "content": '{"city": "Beijing"}',
                },
                {
                    "role": "assistant",
                    "metadata": "get_time",
                    "content": "{}",
                },
            ],
        )


# ===========================================================================
# AnthropicStreamConverter (OpenAI stream -> Anthropic SSE)
# ===========================================================================


class TestAnthropicStreamConverter(unittest.TestCase):
    """Pure converter tests (no server needed)."""

    @staticmethod
    def _chunk(delta=None, finish_reason=None):
        return {
            "choices": [
                {"index": 0, "delta": delta or {}, "finish_reason": finish_reason}
            ]
        }

    @staticmethod
    def _parse(events):
        parsed = []
        for raw in events:
            lines = raw.splitlines()
            event_type = lines[0].split(": ", 1)[1]
            data = json.loads(lines[1].split(": ", 1)[1]) if len(lines) > 1 else {}
            parsed.append((event_type, data))
        return parsed

    def test_thinking_text_tool_sequence(self):
        from infinilm.agents.anthropic import AnthropicStreamConverter

        converter = AnthropicStreamConverter(message_id="msg_x", model="m")
        events = list(converter.begin())
        events += converter.feed(
            self._chunk(delta={"reasoning_content": "Let me think"})
        )
        events += converter.feed(self._chunk(delta={"content": "The answer"}))
        events += converter.feed(
            self._chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_0",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Beijing"}',
                            },
                        }
                    ]
                }
            )
        )
        events += converter.feed(self._chunk(finish_reason="tool_calls"))
        events += list(converter.end())

        parsed = self._parse(events)
        starts = [
            (d["index"], d["content_block"]["type"])
            for t, d in parsed
            if t == "content_block_start"
        ]
        deltas = [
            (d["index"], d["delta"]["type"])
            for t, d in parsed
            if t == "content_block_delta"
        ]
        self.assertEqual(starts, [(0, "thinking"), (1, "text"), (2, "tool_use")])
        self.assertEqual(
            deltas,
            [(0, "thinking_delta"), (1, "text_delta"), (2, "input_json_delta")],
        )
        message_delta = next(d for t, d in parsed if t == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

    def test_usage_propagated_to_message_delta(self):
        from infinilm.agents.anthropic import AnthropicStreamConverter

        converter = AnthropicStreamConverter(message_id="msg_x", model="m")
        events = list(converter.begin())
        events += converter.feed(self._chunk(delta={"content": "hi"}))
        events += converter.feed(
            self._chunk(
                finish_reason="stop",
                delta={},
            )
        )
        # The finish chunk of the OpenAI stream carries usage.
        events += converter.feed(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 7},
            }
        )
        events += list(converter.end())

        parsed = self._parse(events)
        message_delta = next(d for t, d in parsed if t == "message_delta")
        self.assertEqual(
            message_delta["usage"], {"input_tokens": 100, "output_tokens": 7}
        )


# ===========================================================================
# Request preparation and SSE rendering helpers
# ===========================================================================


class TestPrepareChatTemplateKwargs(unittest.TestCase):
    def test_tools_packed_into_chat_template_kwargs(self):
        from infinilm.agents.message_adapter import prepare_chat_template_kwargs

        data = {"tools": _make_weather_dict_tools(), "tool_choice": "auto"}
        prepare_chat_template_kwargs(data)
        self.assertEqual(data["chat_template_kwargs"]["tools"], data["tools"])
        self.assertEqual(data["chat_template_kwargs"]["tool_choice"], "auto")

    def test_existing_kwargs_kept(self):
        from infinilm.agents.message_adapter import prepare_chat_template_kwargs

        data = {
            "tools": _make_weather_dict_tools(),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        prepare_chat_template_kwargs(data)
        self.assertIs(data["chat_template_kwargs"]["enable_thinking"], False)
        self.assertIn("tools", data["chat_template_kwargs"])

    def test_no_tools_no_tools_key(self):
        from infinilm.agents.message_adapter import prepare_chat_template_kwargs

        data = {}
        prepare_chat_template_kwargs(data)
        self.assertNotIn("tools", data["chat_template_kwargs"])


class TestDeltaSseRendering(unittest.TestCase):
    def test_delta_events_render_openai_sse_lines(self):
        from infinilm.agents import AgentStreamParser

        parser = AgentStreamParser(reasoning_parser="deepseek-r1")
        # Long enough that the detector can rule out a pending start tag.
        events = parser.delta_events("hello world, no tags here", "cmpl-1", "m")
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].startswith("data: "))
        chunk = json.loads(events[0][6:].strip())
        self.assertEqual(
            chunk["choices"][0]["delta"]["content"], "hello world, no tags here"
        )

    def test_delta_events_empty_when_nothing_to_emit(self):
        from infinilm.agents import AgentStreamParser

        parser = AgentStreamParser()
        self.assertEqual(parser.delta_events("", "cmpl-1", "m"), [])

    def test_tool_call_flush_events(self):
        from infinilm.agents import AgentStreamParser

        parser = AgentStreamParser(
            tool_call_parser="llama31", tools=_make_weather_dict_tools()
        )
        text = '<|python_tag|>{"name":"get_weather", "arguments":{"city":"北京"}}'
        events = []
        for ch in text:
            events.extend(parser.delta_events(ch, "cmpl-1", "m"))
        events.extend(parser.flush_events("cmpl-1", "m"))

        chunks = [json.loads(e[6:].strip()) for e in events]
        all_calls = [
            tc
            for c in chunks
            for tc in (c["choices"][0]["delta"].get("tool_calls") or [])
        ]
        names = [tc["function"]["name"] for tc in all_calls if tc["function"]["name"]]
        args = "".join(tc["function"]["arguments"] for tc in all_calls)
        self.assertEqual(names, ["get_weather"])
        self.assertEqual(json.loads(args), {"city": "北京"})
        self.assertTrue(parser.has_tool_calls)


class TestOpenaiSseLineParsing(unittest.TestCase):
    def test_parse_data_line(self):
        from infinilm.agents.anthropic import parse_openai_sse_line

        chunk = parse_openai_sse_line('data: {"choices": []}\n\n')
        self.assertEqual(chunk, {"choices": []})

    def test_non_data_lines_return_none(self):
        from infinilm.agents.anthropic import parse_openai_sse_line

        self.assertIsNone(parse_openai_sse_line("data: [DONE]\n\n"))
        self.assertIsNone(parse_openai_sse_line(": keepalive\n\n"))
        self.assertIsNone(parse_openai_sse_line("data: {invalid json}\n\n"))

    def test_anthropic_error_body(self):
        from infinilm.agents.anthropic import anthropic_error_body

        body = anthropic_error_body("boom")
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertEqual(body["error"]["message"], "boom")


class TestConvertOpenaiSseStream(unittest.TestCase):
    def test_full_stream_conversion(self):
        import asyncio

        from infinilm.agents.anthropic import convert_openai_sse_stream

        def sse(delta=None, finish_reason=None):
            chunk = {
                "choices": [
                    {"index": 0, "delta": delta or {}, "finish_reason": finish_reason}
                ]
            }
            return f"data: {json.dumps(chunk)}\n\n"

        async def fake_stream():
            yield sse(delta={"reasoning_content": "think"})
            yield sse(delta={"content": "answer"})
            yield "data: [DONE]\n\n"

        async def run():
            events = []
            async for event in convert_openai_sse_stream(
                fake_stream(), message_id="msg_x", model="m"
            ):
                events.append(event)
            return events

        events = asyncio.run(run())
        joined = "".join(events)
        for fragment in (
            "message_start",
            '"type": "thinking"',
            "thinking_delta",
            '"type": "text"',
            "text_delta",
            "content_block_stop",
            '"stop_reason": "end_turn"',
            "message_stop",
        ):
            self.assertIn(fragment, joined)


# ===========================================================================
# parse_arguments utility
# ===========================================================================


class TestParseArguments(unittest.TestCase):
    """Tests for the parse_arguments utility function."""

    def test_string_json(self):
        args, ok = parse_arguments('{"a": 1}')
        self.assertTrue(ok)
        self.assertEqual(args, {"a": 1})

    def test_dict_passthrough(self):
        args, ok = parse_arguments({"b": 2})
        self.assertTrue(ok)
        self.assertEqual(args, {"b": 2})

    def test_literal_eval(self):
        args, ok = parse_arguments("{'c': 3}")
        self.assertTrue(ok)
        self.assertEqual(args, {"c": 3})

    def test_list_input(self):
        args, ok = parse_arguments("[1, 2, 3]")
        self.assertTrue(ok)
        self.assertEqual(args, [1, 2, 3])

    def test_number_string(self):
        args, ok = parse_arguments("42")
        self.assertTrue(ok)
        self.assertEqual(args, 42)

    def test_boolean_string(self):
        args, ok = parse_arguments("true")
        self.assertTrue(ok)
        self.assertEqual(args, True)

    def test_escaped_string_value(self):
        args, ok = parse_arguments('{"key": "value with \\"quotes\\""}')
        self.assertTrue(ok)
        parsed = json.loads('{"key": "value with \\"quotes\\""}')
        self.assertEqual(args, parsed)


# ===========================================================================
# StreamingParseResult
# ===========================================================================


class TestStreamingParseResult(unittest.TestCase):
    """Tests for StreamingParseResult dataclass."""

    def test_defaults(self):
        r = StreamingParseResult()
        self.assertEqual(r.normal_text, "")
        self.assertEqual(r.calls, [])

    def test_with_calls(self):
        r = StreamingParseResult(
            normal_text="hi",
            calls=[ToolCallItem(tool_index=0, name="test", parameters="{}")],
        )
        self.assertEqual(len(r.calls), 1)
        self.assertEqual(r.calls[0].name, "test")

    def test_multiple_calls(self):
        r = StreamingParseResult(
            normal_text="",
            calls=[
                ToolCallItem(tool_index=0, name="f1", parameters='{"a":1}'),
                ToolCallItem(tool_index=1, name="f2", parameters='{"b":2}'),
            ],
        )
        self.assertEqual(len(r.calls), 2)
        self.assertEqual(r.calls[1].name, "f2")

    def test_no_name_call(self):
        """ToolCallItem can have name=None (parameter-only streaming events)."""
        r = StreamingParseResult(
            normal_text="",
            calls=[
                ToolCallItem(tool_index=0, name=None, parameters='{"city":"NYC"}'),
            ],
        )
        self.assertIsNone(r.calls[0].name)
        self.assertEqual(r.calls[0].parameters, '{"city":"NYC"}')


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestIntegration(unittest.TestCase):
    """Tests that simulate the inference_server pipeline."""

    def test_reasoning_then_tool_call_non_stream_glm45(self):
        """Extract reasoning via glm45, then parse tool calls from remaining text."""
        # Step 1: Extract reasoning
        reasoning_parser = ReasoningParser(reasoning_parser_name="glm45")
        full_text = (
            "thinkI should call get_weather for Beijing."
            "/thinkChecking weather...\n"
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n"
            "</tool_call>"
        )
        reasoning_results = reasoning_parser.extract_reasoning_content(full_text)

        self.assertEqual(len(reasoning_results), 2)
        rc = reasoning_results[0]
        nt = reasoning_results[1]
        self.assertEqual(rc.reasoning_content, "I should call get_weather for Beijing.")
        self.assertTrue(rc.complete)
        self.assertIn("Checking weather...", nt.normal_text)

        # Step 2: Parse tool calls from the normal_text part
        tools = _make_weather_tools()
        tool_parser = FunctionCallParser(tool_call_parser="glm", tools=tools)
        normal, calls = tool_parser.parse_non_stream(nt.normal_text)
        self.assertIn("Checking weather...", normal)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        params = json.loads(calls[0].parameters)
        self.assertEqual(params, {"city": "Beijing"})

    def test_reasoning_then_tool_call_non_stream_think(self):
        """Extract reasoning via think tags, then parse tool calls via Llama."""
        reasoning_parser = ReasoningParser(reasoning_parser_name="think")
        full_text = (
            "<thinking>I should call get_time.</thinking>The time is now.\n"
            '<|python_tag|>{"name":"get_time", "arguments":{}}'
        )
        results = reasoning_parser.extract_reasoning_content(full_text)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].reasoning_content, "I should call get_time.")
        self.assertTrue(results[0].complete)

        remaining = results[1].normal_text
        tools = _make_weather_tools()
        tool_parser = FunctionCallParser(tool_call_parser="llama31", tools=tools)
        normal, calls = tool_parser.parse_non_stream(remaining)
        self.assertIn("The time is now.", normal)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_time")

    def test_llama31_with_raw_dict_tools(self):
        """Full pipeline: raw dict tools from HTTP JSON -> parse tool calls."""
        # Tools from HTTP request body (plain dicts, not Tool objects)
        raw_tools = _make_weather_dict_tools()

        parser = FunctionCallParser(tool_call_parser="llama31", tools=raw_tools)
        text = '<|python_tag|>{"name":"get_weather", "arguments":{"city":"London"}}'
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        params = json.loads(calls[0].parameters)
        self.assertEqual(params, {"city": "London"})

    def test_llama31_streaming_with_raw_dict_tools(self):
        """Streaming with raw dict tools."""
        raw_tools = _make_weather_dict_tools()
        parser = FunctionCallParser(tool_call_parser="llama31", tools=raw_tools)
        call_text = '<|python_tag|>{"name":"get_time", "arguments":{}}'
        result = parser.parse_streaming_increment("", call_text, raw_tools)
        names = [c.name for c in result.calls if c.name]
        self.assertIn("get_time", names)


class TestIntegrationEdgeCases(unittest.TestCase):
    """Edge-case integration tests."""

    def test_empty_reasoning_then_tool_call(self):
        """Empty reasoning (no output) followed by tool call still parsed."""
        parser = FunctionCallParser(
            tool_call_parser="llama31", tools=_make_weather_tools()
        )
        text = 'Hello<|python_tag|>{"name":"get_time", "arguments":{}}'
        normal, calls = parser.parse_non_stream(text)
        self.assertIn("Hello", normal)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_time")

    def test_only_tool_call_no_text(self):
        """Pure tool call with no surrounding text."""
        tools = _make_weather_tools()
        parser = FunctionCallParser(tool_call_parser="llama31", tools=tools)
        text = '<|python_tag|>{"name":"get_weather", "arguments":{"city":"Tokyo"}}'
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        params = json.loads(calls[0].parameters)
        self.assertEqual(params, {"city": "Tokyo"})


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    unittest.main()
