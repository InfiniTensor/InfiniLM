import re
import types
from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


class SentencePieceProcessor(BasicLLMProcessor):
    """Generic processor for models using SentencePiece tokenizer
    (InternLM, Mistral, FM9G, etc.).

    Fixes leading space loss during incremental decoding with Fast tokenizers.
    """

    def __init__(self, model_dir_path: str):
        super().__init__(model_dir_path)
        self._fix_tokenizer_decode(self.tokenizer)

    @staticmethod
    def _fix_tokenizer_decode(tokenizer):
        """Fix leading space loss when tokenizer decodes incrementally.

        Problem: Fast tokenizer's Rust backend trims leading spaces derived
        from ▁ (U+2581) during single-token decoding, causing words to concatenate.

        Fix: patch tokenizer.decode() to manually replace ▁ → space
        and handle byte fallback.
        """

        def patched_decode(self_tok, token_ids, skip_special_tokens=False, **kwargs):
            # 1. Normalize input to list of token IDs
            if isinstance(token_ids, int):
                token_ids = [token_ids]

            # 2. Convert token IDs to raw token strings (preserving ▁)
            tokens = self_tok.convert_ids_to_tokens(
                token_ids, skip_special_tokens=skip_special_tokens
            )
            if isinstance(tokens, str):
                tokens = [tokens]

            # 3. Remove special tokens if requested
            if skip_special_tokens:
                special = set(self_tok.all_special_tokens)
                tokens = [t for t in tokens if t not in special]

            # 4. Join and replace ▁ (U+2581) with space
            text = "".join(tokens).replace("\u2581", " ")

            # 5. Handle SentencePiece byte fallback: consecutive <0xHH> → UTF-8
            def byte_fallback_replace(match):
                hex_strs = re.findall(r"<0x([0-9A-Fa-f]{2})>", match.group(0))
                byte_values = bytes([int(h, 16) for h in hex_strs])
                return byte_values.decode("utf-8", errors="replace")

            text = re.sub(r"(<0x[0-9A-Fa-f]{2}>)+", byte_fallback_replace, text)

            return text

        tokenizer.decode = types.MethodType(patched_decode, tokenizer)


@register_processor("fm9g")
class FM9GProcessor(SentencePieceProcessor):
    pass


# Register model-specific aliases
@register_processor("internlm3")
class InternLMProcessor(SentencePieceProcessor):
    pass


@register_processor("mistral")
class MistralProcessor(SentencePieceProcessor):
    pass


@register_processor("deepseek_v2")
class DeepSeekV2Processor(BasicLLMProcessor):
    pass
