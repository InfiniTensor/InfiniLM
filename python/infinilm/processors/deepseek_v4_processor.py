"""Use the checkpoint's official DeepSeek V4 prompt encoder."""

import copy
import importlib.util
import json
from pathlib import Path

from .basic_llm_processor import BasicLLMProcessor


class DeepSeekV4Processor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        super().__init__(model_dir_path)
        source = Path(model_dir_path) / "encoding/encoding_dsv4.py"
        spec = importlib.util.spec_from_file_location("deepseek_v4_encoding", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.encode_messages = module.encode_messages

    def apply_chat_template(
        self, conversation, add_generation_prompt=False, tokenize=True, **kwargs
    ):
        messages = copy.deepcopy(conversation)
        tools = kwargs.get("tools")
        if tools:
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": ""})
            messages[0]["tools"] = tools
        for message in messages:
            if isinstance(message.get("content"), list):
                message["content"] = "".join(
                    part["text"] for part in message["content"]
                )
            for call in message.get("tool_calls") or []:
                fn = call["function"]
                if isinstance(fn.get("arguments"), dict):
                    fn["arguments"] = json.dumps(
                        fn["arguments"], ensure_ascii=False, allow_nan=False
                    )
        mode = kwargs.get(
            "thinking_mode",
            "thinking" if kwargs.get("enable_thinking", True) else "chat",
        )
        prompt = self.encode_messages(
            messages,
            thinking_mode=mode,
            reasoning_effort=kwargs.get("reasoning_effort"),
        )
        if not add_generation_prompt:
            prompt = prompt.removesuffix(
                "<｜Assistant｜>" + ("<think>" if mode == "thinking" else "</think>")
            )
        return (
            self.tokenizer.encode(prompt, add_special_tokens=False)
            if tokenize
            else prompt
        )
