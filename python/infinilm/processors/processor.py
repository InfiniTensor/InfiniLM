import json


class InfinilmProcessor:
    def __init__(self, model_dir_path: str):
        """Initialize the processor with the model directory path."""
        raise NotImplementedError("ModelInputProcessor is not implemented yet")

    def __call__(
        self,
        prompt,
        images=None,
        videos=None,
        audios=None,
        return_tensors: str = None,
        **kwargs,
    ) -> dict:
        """Process the input prompt and media into final inputs."""
        raise NotImplementedError("__call__ is not implemented yet")

    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        """Apply chant template given input messages"""
        raise NotImplementedError("apply_chat_template is not implemented yet")

    def build_model_inputs(self, scheduler_output, **kwargs) -> dict:
        """Build batched infinilm model inputs from the scheduler output."""
        raise NotImplementedError("build_model_inputs is not implemented yet")

    def get_tokenizer(self):
        """Return the text tokenizer associated with this processor."""
        raise NotImplementedError("get_tokenizer is not implemented yet")

    def get_mm_token_index_list(
        self, prompt_token_ids, image_ids=None, video_ids=None, audio_ids=None, **kwargs
    ):
        """
        Get the list of starting token index and identifier mapping for multimodal inputs, sorted by index.
        Return: [{"start_index": <token_id>, "identifier": <id>}, ...]
        """
        raise NotImplementedError("get_mm_token_index_list is not implemented yet")


def normalize_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI JSON-string tool arguments to template mappings."""
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue

        normalized_message = message.copy()
        if "content" not in normalized_message and normalized_message.get("tool_calls"):
            normalized_message["content"] = None

        tool_calls = normalized_message.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    normalized_calls.append(tool_call)
                    continue
                normalized_call = tool_call.copy()
                function = normalized_call.get("function")
                if isinstance(function, dict):
                    normalized_function = function.copy()
                    arguments = normalized_function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                "tool call function.arguments must be valid JSON"
                            ) from exc
                        if not isinstance(arguments, dict):
                            raise ValueError(
                                "tool call function.arguments must decode to an object"
                            )
                        normalized_function["arguments"] = arguments
                    normalized_call["function"] = normalized_function
                normalized_calls.append(normalized_call)
            normalized_message["tool_calls"] = normalized_calls

        normalized.append(normalized_message)
    return normalized


# Global registry mapping model_type strings to their Processor classes
_PROCESSOR_REGISTRY = {}


def register_processor(model_type: str):
    """Decorator to register a Processor class for a specific model type."""

    def decorator(cls):
        if model_type in _PROCESSOR_REGISTRY:
            raise ValueError(
                f"Duplicate processor registration: model_type '{model_type}' "
                f"is already registered by {_PROCESSOR_REGISTRY[model_type].__name__}"
            )
        _PROCESSOR_REGISTRY[model_type] = cls
        return cls

    return decorator


def get_processor_class(model_type: str):
    """Retrieve the processor class for a given model type.

    Falls back to the "default" registered processor if the model type
    is not explicitly recognized.
    """
    return _PROCESSOR_REGISTRY.get(model_type, _PROCESSOR_REGISTRY["default"])
