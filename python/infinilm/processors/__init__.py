from .processor import InfinilmProcessor
from .basic_llm_processor import BasicLLMProcessor
from .llama_processor import LlamaProcessor
from .minicpmv_processor import MiniCPMVProcessor

from transformers import AutoConfig


class AutoInfinilmProcessor:
    @classmethod
    def from_pretrained(cls, model_dir_path: str, **kwargs) -> InfinilmProcessor:
        """Factory method to get the appropriate processor based on model config."""
        config = AutoConfig.from_pretrained(model_dir_path, trust_remote_code=True)
        model_type = config.model_type.lower()

        if model_type in ["llama"]:
            return LlamaProcessor(model_dir_path)
        elif model_type in ["minicpmv"]:
            return MiniCPMVProcessor(model_dir_path)
        else:
            return BasicLLMProcessor(model_dir_path)
