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
        raise NotImplementedError("InfinilmProcessor is not implemented yet")

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        """Apply chant template given input messages"""
        raise NotImplementedError("InfinilmProcessor is not implemented yet")

    def build_model_inputs(self, scheduler_output, **kwargs) -> dict:
        """Build batched infinilm model inputs from the scheduler output."""
        raise NotImplementedError("InfinilmProcessor is not implemented yet")

    def get_tokenizer(self):
        """Return the text tokenizer associated with this processor."""
        raise NotImplementedError("InfinilmProcessor is not implemented yet")
