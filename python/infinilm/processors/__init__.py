from .processor import InfinilmProcessor
from .basic_llm_processor import BasicLLMProcessor
from .llama_processor import LlamaProcessor
from .baichuan_processor import BaichuanProcessor

from transformers import AutoConfig


class AutoInfinilmProcessor:
    @classmethod
    def from_pretrained(cls, model_dir_path: str, **kwargs) -> InfinilmProcessor:
        """Factory method to get the appropriate processor based on model config."""
        config = AutoConfig.from_pretrained(model_dir_path, trust_remote_code=True)
        model_type = config.model_type.lower()

        if model_type in ["llama"]:
            return LlamaProcessor(model_dir_path)
        elif model_type in ["baichuan"]:
            return BaichuanProcessor(model_dir_path)
        else:
            return BasicLLMProcessor(model_dir_path)
