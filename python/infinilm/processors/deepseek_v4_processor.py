"""Use the checkpoint's official DeepSeek V4 prompt encoder."""

import copy
import importlib.util
import json
from pathlib import Path

from .basic_llm_processor import BasicLLMProcessor
from .history_encoding import deepseek_history_encoder


class DeepSeekV4Processor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        from transformers import PreTrainedTokenizerFast

        # The checkpoint supplies a generic fast tokenizer. AutoTokenizer can
        # try to instantiate the unregistered DeepSeek V4 Transformers config.
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir_path)
        source = Path(model_dir_path) / "encoding/encoding_dsv4.py"
        self.encode_messages = None
        if source.is_file():
            spec = importlib.util.spec_from_file_location("deepseek_v4_encoding", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.encode_messages = deepseek_history_encoder(module.encode_messages)

    def apply_chat_template(
        self, conversation, add_generation_prompt=False, tokenize=True, **kwargs
    ):
        """Render official text, or return an unbatched token-ID list when tokenized."""
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
        if self.encode_messages is not None:
            prompt = self.encode_messages(
                messages,
                thinking_mode=mode,
                reasoning_effort=kwargs.get("reasoning_effort"),
                drop_thinking=False,
            )
        elif (
            len(messages) == 1
            and messages[0].get("role") == "user"
            and isinstance(messages[0].get("content"), str)
            and not messages[0].get("tool_calls")
            and kwargs.get("reasoning_effort") is None
            and mode in {"thinking", "chat"}
        ):
            # The W8A8 export omits the official encoder. Preserve the
            # acceptance version's plain single-turn token format.
            user, assistant = self.tokenizer.convert_ids_to_tokens([128803, 128804])
            thinking = "<think>" if mode == "thinking" else "</think>"
            prompt = (
                self.tokenizer.bos_token + user + messages[0]["content"] + assistant + thinking
            )
        else:
            raise ValueError(
                "DeepSeek V4 multi-turn, tools and reasoning effort require "
                "encoding/encoding_dsv4.py in the checkpoint directory"
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
