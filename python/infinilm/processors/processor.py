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

    def configure_paged_cache(
        self, scheduler_block_size: int, kernel_block_size: int
    ) -> None:
        """Configure worker-side virtual block splitting."""
        if scheduler_block_size % kernel_block_size != 0:
            raise ValueError(
                "scheduler_block_size must be divisible by kernel_block_size"
            )
        self.cache_block_size_factor = scheduler_block_size // kernel_block_size

    def expand_block_table_for_kernel(self, block_table: list[int]) -> list[int]:
        """Map scheduler block IDs to consecutive physical kernel block IDs."""
        factor = getattr(self, "cache_block_size_factor", 1)
        if factor == 1:
            return list(block_table)
        expanded = []
        for block_id in block_table:
            if block_id < 0:
                expanded.extend([-1] * factor)
            else:
                first_kernel_block = block_id * factor
                expanded.extend(range(first_kernel_block, first_kernel_block + factor))
        return expanded

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
