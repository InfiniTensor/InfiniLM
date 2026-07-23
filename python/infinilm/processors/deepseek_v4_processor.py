import os

from typing_extensions import override
from transformers import PreTrainedTokenizerFast

from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor

_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
_EOS_TOKEN = "<｜end▁of▁sentence｜>"
_USER_TOKEN = "<｜User｜>"
_ASSISTANT_TOKEN = "<｜Assistant｜>"
_THINKING_END_TOKEN = "</think>"


@register_processor("deepseek_v4")
class DeepseekV4Processor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str, tokenizer=None):
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                model_dir_path,
                trust_remote_code=True,
                local_files_only=os.environ.get("TRANSFORMERS_LOCAL_FILES_ONLY", "1")
                != "0",
            )

    @override
    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        normalized_conversation = []
        for message in conversation:
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = "".join(text_parts)

            normalized_conversation.append(
                {"role": message.get("role", "user"), "content": str(content)}
            )

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                conversation=normalized_conversation,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                **kwargs,
            )

        prompt = self._apply_dsv4_chat_template(
            normalized_conversation, add_generation_prompt=add_generation_prompt
        )
        if tokenize:
            return self.tokenizer.encode(prompt, add_special_tokens=False, **kwargs)
        return prompt

    def _apply_dsv4_chat_template(
        self, conversation: list[dict], add_generation_prompt: bool
    ) -> str:
        prompt = _BOS_TOKEN

        for index, message in enumerate(conversation):
            role = message["role"]
            content = message["content"]
            next_role = (
                conversation[index + 1]["role"]
                if index + 1 < len(conversation)
                else None
            )

            if role in ("user", "developer"):
                prompt += _USER_TOKEN + content
                if next_role in ("assistant", "latest_reminder") or (
                    next_role is None and add_generation_prompt
                ):
                    prompt += _ASSISTANT_TOKEN + _THINKING_END_TOKEN
            elif role == "assistant":
                prompt += content + _EOS_TOKEN
            elif role == "system":
                prompt += content
            elif role == "latest_reminder":
                prompt += "<｜latest_reminder｜>" + content
            else:
                raise NotImplementedError(f"Unknown role: {role}")

        return prompt
