from typing_extensions import override
from transformers import PreTrainedTokenizerFast

from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("deepseek_v4")
class DeepseekV4Processor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_dir_path, trust_remote_code=True
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

        prompt = "\n".join(
            message["content"] for message in normalized_conversation if message["content"]
        )
        if add_generation_prompt:
            prompt += "\n"
        if tokenize:
            return self.tokenizer(prompt, add_special_tokens=False, **kwargs)
        return prompt
