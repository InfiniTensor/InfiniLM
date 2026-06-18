import logging

from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor

logger = logging.getLogger(__name__)


@register_processor("rwkv5")
class Rwkv5Processor(BasicLLMProcessor):
    @classmethod
    def resolve_cache_type(cls, cache_type: str) -> str:
        if cache_type == "paged":
            logger.info("RWKV5 uses recurrent state; using static cache scheduling")
            return "static"
        return cache_type

    def prepare_model_forward(self, scheduler_output, model_engine, engine_config):
        if engine_config.cache_type == "static" and scheduler_output.is_prefill:
            scheduler_output.prefix_hit_len = 0
            model_engine.reset_cache(model_engine.get_cache_config())

    def default_stop_strings(self) -> list[str]:
        return ["\n\nUser:", "\nUser:"]

    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        normalized_conversation = self._normalize_conversation(conversation)
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                conversation=normalized_conversation,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                **kwargs,
            )

        prompt_parts = [
            "User: hi\n"
            "Assistant: Hi. I am your assistant and I will provide expert full response in full details. "
            "Please feel free to ask any question and I will always answer it."
        ]
        for message in normalized_conversation:
            role = "Assistant" if message.get("role") == "assistant" else "User"
            prompt_parts.append(f"{role}: {message.get('content', '')}")
        prompt = "\n\n".join(prompt_parts)
        if add_generation_prompt:
            prompt = prompt.rstrip() + "\n\nAssistant:"
        if not tokenize:
            return prompt
        return self.tokenizer(prompt, add_special_tokens=False, **kwargs)["input_ids"]
