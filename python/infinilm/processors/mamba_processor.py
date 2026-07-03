from typing_extensions import override

from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("mamba")
class MambaProcessor(BasicLLMProcessor):
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
            if isinstance(message["content"], list):
                assert (
                    len(message["content"]) == 1
                ), "Only one content item supported in list"
                content_item = message["content"][0]
                assert (
                    "type" in content_item and "text" in content_item
                ), "Content dict must have 'type' and 'text' keys"
                normalized_conversation.append(
                    {"role": message["role"], "content": content_item["text"]}
                )
            else:
                normalized_conversation.append(message)

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
