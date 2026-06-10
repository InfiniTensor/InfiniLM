from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("gpt2")
class GPT2Processor(BasicLLMProcessor):
    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        prompt = self._messages_to_prompt(conversation)
        if tokenize:
            return self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return prompt

    @staticmethod
    def _messages_to_prompt(conversation) -> str:
        parts = []
        for message in conversation:
            if not isinstance(message, dict):
                parts.append(str(message))
                continue

            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = "".join(text_parts)

            if content:
                parts.append(str(content))

        return "\n".join(parts)
