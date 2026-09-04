"""
Regression test: ``AsyncLLMEngine.add_chat_request()`` must forward
``chat_template_kwargs`` (in particular tool definitions) through to
``apply_chat_template()`` so the tools actually enter the prompt.

The test runs against the real ``infinilm/llm/llm.py`` code; engine internals
that require the native stack are stubbed only when the real modules cannot
be imported (CPU-only environments).
"""

import enum
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


def _namespace_stub(name: str, path: str):
    mod = sys.modules.get(name)
    if mod is None or not hasattr(mod, "__path__"):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    mod.__path__ = [path]
    return mod


def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


try:
    from infinilm.llm.llm import AsyncLLMEngine, LLMEngine
    from infinilm.llm.sampling_params import SamplingParams

    _STUBBED = False
except Exception:
    # CPU-only environment without the compiled engine: stand in for the
    # native-dependent modules so the real llm.py logic can still run.

    class FinishReason(enum.Enum):
        EOS_TOKEN = "eos_token"
        STOP_STRING = "stop_string"
        STOP = "stop"
        LENGTH = "length"
        CANCELED = "canceled"
        TIMEOUT = "timeout"
        ERROR = "error"

    class SamplingParams:  # minimal mirror of the real dataclass
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            for key, value in kwargs.items():
                setattr(self, key, value)

        def clone(self):
            return SamplingParams(**self.kwargs)

    class InferenceRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @property
        def output_queue(self):
            return None

    def _empty_mm_inputs(messages):
        return {
            "images": [],
            "image_urls": [],
            "videos": [],
            "video_urls": [],
            "audios": [],
            "audio_urls": [],
        }

    if "janus" not in sys.modules:
        try:
            import janus  # noqa: F401
        except Exception:
            _stub("janus", Queue=object)

    _namespace_stub("infinilm", str(PYTHON_ROOT / "infinilm"))
    _namespace_stub("infinilm.llm", str(PYTHON_ROOT / "infinilm" / "llm"))
    _namespace_stub(
        "infinilm.llm.model_runner",
        str(PYTHON_ROOT / "infinilm" / "llm" / "model_runner"),
    )
    _namespace_stub("infinilm.multimodal", str(PYTHON_ROOT / "infinilm" / "multimodal"))
    _namespace_stub("infinilm.config", str(PYTHON_ROOT / "infinilm" / "config"))

    _stub("infinilm.config.engine_config", EngineConfig=object)
    _stub("infinilm.config.kv_transfer", KVTransferConfig=object)
    _stub(
        "infinilm.infer_engine",
        model_uses_mamba_cache=lambda config: False,
        read_hf_config=lambda path: {},
    )
    _stub(
        "infinilm.kv_connector",
        KVConnectorFactory=object,
        KVConnectorRole=object,
    )
    _stub("infinilm.llm.model_runner.model_runner", ModelRunner=object)
    _stub(
        "infinilm.llm.request",
        FinishReason=FinishReason,
        InferenceRequest=InferenceRequest,
        RequestOutput=object,
        TokenOutput=object,
    )
    _stub("infinilm.llm.sampling_params", SamplingParams=SamplingParams)
    _stub("infinilm.llm.scheduler", Scheduler=object, SchedulerOutput=object)
    _stub(
        "infinilm.llm.static_scheduler",
        StaticScheduler=object,
        StaticSchedulerOutput=object,
    )
    _stub(
        "infinilm.multimodal.multimodal",
        resolve_multimodal_inputs=_empty_mm_inputs,
    )

    from infinilm.llm.llm import AsyncLLMEngine, LLMEngine

    _STUBBED = True


def _make_weather_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


class _FakeInputIds:
    """Minimal stand-in for a token-id tensor."""

    def flatten(self):
        return self

    def tolist(self):
        return [1, 2, 3]


class TestChatTemplateKwargsForwarding(unittest.TestCase):
    """P1-1: tools sent via chat_template_kwargs must reach the template."""

    def _make_engine(self, captured: dict):
        class FakeProcessor:
            def apply_chat_template(
                self, conversation, add_generation_prompt, tokenize, **kwargs
            ):
                captured.update(kwargs)
                return "PROMPT_WITH_TOOLS" if "tools" in kwargs else "PROMPT"

            def __call__(self, prompt, images=None, videos=None, audios=None, **kw):
                return {"input_ids": _FakeInputIds()}

            def get_mm_token_index_list(self, *args, **kwargs):
                return {}

        class FakeScheduler:
            def __init__(self):
                self.added = []

            def add_request(self, req):
                self.added.append(req)

        llm_engine = object.__new__(LLMEngine)
        llm_engine.processor = FakeProcessor()
        llm_engine.scheduler = FakeScheduler()
        llm_engine.eos_token_ids = [2]

        engine = object.__new__(AsyncLLMEngine)
        engine.engine = llm_engine
        engine.config = types.SimpleNamespace(max_tokens=64)
        return engine

    def test_tools_reach_chat_template(self):
        captured = {}
        engine = self._make_engine(captured)
        tools = _make_weather_tools()

        request = engine.add_chat_request(
            messages=[{"role": "user", "content": "weather in Beijing?"}],
            sampling_params=SamplingParams(max_tokens=32),
            request_id="cmpl-tools",
            chat_template_kwargs={"tools": tools, "tool_choice": "auto"},
        )

        self.assertEqual(request.prompt, "PROMPT_WITH_TOOLS")
        self.assertIn("tools", captured)
        self.assertEqual(captured["tools"], tools)
        self.assertEqual(captured["tool_choice"], "auto")

    def test_no_tools_no_template_kwargs(self):
        captured = {}
        engine = self._make_engine(captured)

        request = engine.add_chat_request(
            messages=[{"role": "user", "content": "hello"}],
            sampling_params=SamplingParams(max_tokens=32),
            request_id="cmpl-plain",
        )

        self.assertEqual(request.prompt, "PROMPT")
        self.assertNotIn("tools", captured)

    def test_add_request_accepts_chat_template_kwargs(self):
        """The lower-level add_request() also forwards chat_template_kwargs."""
        captured = {}
        engine = self._make_engine(captured)
        tools = _make_weather_tools()

        engine.add_request(
            messages=[{"role": "user", "content": "hi"}],
            sampling_params=SamplingParams(max_tokens=32),
            request_id="cmpl-lowlevel",
            chat_template_kwargs={"tools": tools},
        )

        self.assertEqual(captured.get("tools"), tools)


if __name__ == "__main__":
    unittest.main()
