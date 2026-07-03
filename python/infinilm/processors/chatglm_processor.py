# python/infinilm/processors/chatglm_processor.py

import re
import types
from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("chatglm")
class ChatGLMProcessor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        super().__init__(model_dir_path)
        self._fix_tokenizer_decode(self.tokenizer)

    @staticmethod
    def _fix_tokenizer_decode(tokenizer):
        """Fix ChatGLM tokenizer: patch convert_tokens_to_string.
        
        ChatGLM uses SentencePiece which encodes spaces as ▁ (U+2581).
        Its convert_tokens_to_string calls self.tokenizer.decode_tokens(tokens),
        which strips ▁ when decoding tokens incrementally, losing inter-word spaces.
        
        Fix: replace convert_tokens_to_string to:
        1. Join tokens + replace ▁ → space (the only thing decode_tokens gets wrong)
        2. Handle byte fallback: consecutive <0xHH> sequences → UTF-8 chars
        
        This keeps decode()'s other logic (skip_special_tokens, 
        clean_up_tokenization_spaces, etc.) intact.
        
        ▁ (U+2581) and _ (U+005F) are different characters in SentencePiece,
        so this replacement will NOT affect real underscores.
        """
        def patched_convert_tokens_to_string(self_tok, tokens):
            # 1. Join tokens + replace ▁ (U+2581) with space
            text = "".join(tokens).replace("\u2581", " ")
            
            # 2. Handle SentencePiece byte fallback: consecutive <0xHH> → UTF-8
            def byte_fallback_replace(match):
                hex_strs = re.findall(r"<0x([0-9A-Fa-f]{2})>", match.group(0))
                byte_values = bytes([int(h, 16) for h in hex_strs])
                return byte_values.decode("utf-8", errors="replace")
            
            text = re.sub(r"(<0x[0-9A-Fa-f]{2}>)+", byte_fallback_replace, text)
            return text

        tokenizer.convert_tokens_to_string = types.MethodType(
            patched_convert_tokens_to_string, tokenizer
        )
