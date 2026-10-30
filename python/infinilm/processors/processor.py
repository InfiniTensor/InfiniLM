import importlib


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


# Global registry mapping model_type strings to their Processor classes
_PROCESSOR_REGISTRY = {}

# Built-in processors have no registration side effects: several model types
# share a module, while an explicit user registration may override just one.
_BUILTIN_PROCESSORS = {
    "default": ("basic_llm_processor", "BasicLLMProcessor"),
    "baichuan": ("baichuan_processor", "BaichuanProcessor"),
    "chatglm": ("chatglm_processor", "ChatGLMProcessor"),
    "deepseek": ("deepseek_processor", "DeepSeekProcessor"),
    "deepseek_v2": ("sentencepiece_processor", "DeepSeekV2Processor"),
    "deepseek_v4": ("deepseek_v4_processor", "DeepSeekV4Processor"),
    "ernie4_5_moe_vl": ("ernie4_5_vl_processor", "Ernie45VLProcessor"),
    "fm9g": ("sentencepiece_processor", "FM9GProcessor"),
    "gpt2": ("gpt2_processor", "GPT2Processor"),
    "internlm3": ("sentencepiece_processor", "InternLMProcessor"),
    "kimi_k3": ("kimi_k3_processor", "KimiK3Processor"),
    "llama": ("llama_processor", "LlamaProcessor"),
    "mamba": ("mamba_processor", "MambaProcessor"),
    "minicpm": ("sentencepiece_processor", "MiniCPMProcessor"),
    "minicpm4": ("sentencepiece_processor", "MiniCPMProcessor"),
    "minicpmv": ("minicpmv_processor", "MiniCPMVProcessor"),
    "mistral": ("sentencepiece_processor", "MistralProcessor"),
    "qwen3_5": ("qwen3_5_processor", "Qwen35Processor"),
    "qwen3_5_moe": ("qwen3_5_processor", "Qwen35Processor"),
    "qwen3_next": ("qwen3_next_processor", "Qwen3NextProcessor"),
    "videonsa": ("videonsa_processor", "VideoNSAProcessor"),
}


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
    registered = _PROCESSOR_REGISTRY.get(model_type)
    if registered is not None:
        return registered
    if model_type not in _BUILTIN_PROCESSORS:
        registered_default = _PROCESSOR_REGISTRY.get("default")
        if registered_default is not None:
            return registered_default
    module_name, class_name = _BUILTIN_PROCESSORS.get(
        model_type, _BUILTIN_PROCESSORS["default"]
    )
    module = importlib.import_module(f".{module_name}", __package__)
    return getattr(module, class_name)
