import re
import types
from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("mistral")
class MistralProcessor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        super().__init__(model_dir_path)
        self._fix_tokenizer_decode(self.tokenizer)

    @staticmethod
    def _fix_tokenizer_decode(tokenizer):
        """Fix Mistral tokenizer incremental decoding space loss.

        LlamaTokenizerFast.decode() calls Rust backend directly, which
        trims leading spaces derived from ▁ (U+2581) during single-token
        decoding, causing English words to concatenate.

        Fix: patch tokenizer.decode() to:
        1. Convert token IDs to raw token strings (preserving ▁)
        2. Manually replace ▁ → space and handle byte fallback
        """
        original_decode = tokenizer.decode

        def patched_decode(self_tok, token_ids, skip_special_tokens=False, **kwargs):
            # 1. Get raw token strings (preserving ▁)
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            tokens = self_tok.convert_ids_to_tokens(
                token_ids, skip_special_tokens=skip_special_tokens
            )
            if isinstance(tokens, str):
                tokens = [tokens]

            # 2. Remove special tokens if requested
            if skip_special_tokens:
                special = set(self_tok.all_special_tokens)
                tokens = [t for t in tokens if t not in special]

            # 3. Join + replace ▁ (U+2581) with space
            text = "".join(tokens).replace("\u2581", " ")

            # 4. Handle SentencePiece byte fallback: consecutive <0xHH> → UTF-8
            def byte_fallback_replace(match):
                hex_strs = re.findall(r"<0x([0-9A-Fa-f]{2})>", match.group(0))
                byte_values = bytes([int(h, 16) for h in hex_strs])
                return byte_values.decode("utf-8", errors="replace")

            text = re.sub(r"(<0x[0-9A-Fa-f]{2}>)+", byte_fallback_replace, text)

            # 5. Strip leading/trailing whitespace only if ALL tokens were special
            #    (preserve inter-word spaces from ▁)
            return text

        tokenizer.decode = types.MethodType(patched_decode, tokenizer)

