from typing_extensions import override

from .basic_llm_processor import BasicLLMProcessor


class MambaProcessor(BasicLLMProcessor):
    @override
    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        normalized_conversation = self._normalize_text_messages(conversation)

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                conversation=normalized_conversation,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                **kwargs,
            )

        text = "\n".join(
            str(message.get("content", "")) for message in normalized_conversation
        )
        if tokenize:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return text
