"""CPU contracts for shared processor input transformations."""

import ast
import copy
import json
import types
from unittest.mock import Mock, patch

import pytest
from test_agent_protocol import SERVER, load_module


def processor_class(filename, methods, base=object, **namespace):
    tree = ast.parse(
        (SERVER.parent / "processors" / filename).read_text(encoding="utf-8")
    )
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    namespace.update(Base=base, override=lambda function: function)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return namespace[cls.name]


@pytest.mark.parametrize(
    "filename",
    ["qwen3_5_processor.py", "qwen3_next_processor.py", "kimi_k3_processor.py"],
)
@pytest.mark.parametrize("prefill", [False, True])
def test_shared_recurrent_indices_through_model_processors(filename, prefill):
    core = types.ModuleType("infinilm._core")
    core.int32 = "int32"
    core.from_list = lambda values, dtype: (tuple(values), dtype)
    scheduler = type("Scheduler", (), {})
    output = scheduler()
    output.is_prefill = prefill
    output.scheduled_requests = [
        types.SimpleNamespace(request_id=str(index), mamba_cache_index=index)
        for index in [3, 7]
    ]

    class Base:
        def build_model_inputs(self, *args, **kwargs):
            return {"preserved": True}

        _build_model_input_from_batch_scheduler_output = build_model_inputs
        _build_model_input_from_static_scheduler_output = build_model_inputs

        def _append_qwen35_mm_inputs(self, model_inputs, output):
            pass

        _append_qwen35_position_ids = _append_qwen35_mm_inputs
        _append_multimodal_inputs = _append_qwen35_mm_inputs

    helper = load_module(
        "infinilm.processors.recurrent_inputs",
        SERVER.parent / "processors/recurrent_inputs.py",
    )
    cls = processor_class(
        filename,
        {"build_model_inputs"},
        Base,
        SchedulerOutput=scheduler,
        StaticSchedulerOutput=type("Static", (), {}),
        append_recurrent_state_inputs=helper.append_recurrent_state_inputs,
    )
    with patch.dict(
        "sys.modules",
        {"infinilm": types.ModuleType("infinilm"), "infinilm._core": core},
    ):
        actual = cls().build_model_inputs(output)
        assert actual == {
            "preserved": True,
            "mamba_init_state_indices": ((0, 0) if prefill else (3, 7), "int32"),
            "mamba_final_state_indices": ((3, 7), "int32"),
        }
        output.scheduled_requests[-1].mamba_cache_index = None
        with pytest.raises(RuntimeError, match="mamba cache index"):
            cls().build_model_inputs(output)


def test_text_normalization_keeps_history_and_mamba_fallback():
    basic = processor_class(
        "basic_llm_processor.py", {"_normalize_text_messages", "apply_chat_template"}
    )
    mamba = processor_class("mamba_processor.py", {"apply_chat_template"}, basic)
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "answer"}],
        "tool_calls": [{"id": "opaque"}],
        "reasoning_content": "saved",
    }
    original = copy.deepcopy(message)
    for cls in [basic, mamba]:
        instance = cls()
        instance.tokenizer = types.SimpleNamespace(
            chat_template=True, apply_chat_template=lambda **kwargs: kwargs
        )
        result = instance.apply_chat_template([message], tokenize=False)
        assert result["conversation"] == [{**message, "content": "answer"}]
        assert message == original
    instance.tokenizer.chat_template = None
    assert instance.apply_chat_template([message], tokenize=False) == "answer"


@pytest.mark.parametrize("has_tokenizer", [False, True])
def test_kimi_tokenizer_fallback_is_lazy(tmp_path, has_tokenizer):
    tokenizer = types.SimpleNamespace(apply_chat_template=object())
    processor = types.SimpleNamespace(tokenizer=tokenizer if has_tokenizer else None)
    fallback = Mock(return_value=tokenizer)
    cls = processor_class(
        "kimi_k3_processor.py",
        {"__init__"},
        json=json,
        os=__import__("os"),
        torch=types.SimpleNamespace(bfloat16="bf16"),
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **kw: processor),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=fallback),
        kimi_history_template=lambda template: template,
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"media_placeholder_token_id": 42})
    )
    instance = cls(str(tmp_path))
    assert instance.tokenizer is tokenizer
    assert fallback.call_count == (0 if has_tokenizer else 1)


def test_sentencepiece_shared_conversion_keeps_tokenizer_hooks():
    basic = types.ModuleType("infinilm.processors.basic_llm_processor")
    basic.BasicLLMProcessor = object
    stubs = {
        "infinilm": types.ModuleType("infinilm"),
        "infinilm.processors": types.ModuleType("infinilm.processors"),
        "infinilm.processors.basic_llm_processor": basic,
    }
    sentencepiece = load_module(
        "infinilm.processors.sentencepiece_processor",
        SERVER.parent / "processors/sentencepiece_processor.py",
        stubs,
    )
    chatglm = load_module(
        "infinilm.processors.chatglm_processor",
        SERVER.parent / "processors/chatglm_processor.py",
        {**stubs, "infinilm.processors.sentencepiece_processor": sentencepiece},
    )
    tokens = ["▁hi", "<0xE4>", "<0xB8>", "<0xAD>"]
    tokenizer = types.SimpleNamespace(
        all_special_tokens=["[EOS]"],
        convert_ids_to_tokens=lambda *a, **kw: tokens + ["[EOS]"],
    )
    sentencepiece.SentencePieceProcessor._fix_tokenizer_decode(tokenizer)
    chatglm.ChatGLMProcessor._fix_tokenizer_decode(tokenizer)
    assert tokenizer.decode([1], skip_special_tokens=True) == " hi中"
    assert tokenizer.convert_tokens_to_string(tokens) == " hi中"
