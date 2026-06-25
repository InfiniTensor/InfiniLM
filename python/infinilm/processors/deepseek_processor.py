from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor

from typing import List, Optional, Union

from transformers.models.llama import LlamaTokenizerFast


class DeepseekTokenizerFast(LlamaTokenizerFast):
    """Fast tokenizer for DeepSeek models (MoE V1, V2, etc.).

    HuggingFace AutoTokenizer may load these checkpoints as generic LlamaTokenizer
    when tokenization_deepseek_fast.py is missing from the model directory, which
    breaks Chinese tokenization. Use this class directly instead.
    """

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            token = self._tokenizer.id_to_token(index)
            tokens.append(token if token is not None else "")
        return tokens

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        token = self._tokenizer.id_to_token(int(index))
        return token if token is not None else ""


@register_processor("deepseek")
class DeepSeekProcessor(BasicLLMProcessor):
    """Processor for DeepSeek MoE V1 models (model_type=deepseek)."""

    def __init__(self, model_dir_path: str):
        self.tokenizer = DeepseekTokenizerFast.from_pretrained(
            model_dir_path, trust_remote_code=True
        )
