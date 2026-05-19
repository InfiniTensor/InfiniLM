from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor
from tokenizers import decoders as _dec


@register_processor("llama")
class LlamaProcessor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        super().__init__(model_dir_path)
        self._fix_tokenizer_decoder(self.tokenizer)

    @staticmethod
    def _fix_tokenizer_decoder(tokenizer):
        """Fix tokenizer decoder for llama models."""
        backend = getattr(tokenizer, "backend_tokenizer", None)
        target = getattr(backend, "_tokenizer", backend)
        norm = getattr(target, "normalizer", None)
        dec = getattr(target, "decoder", None)
        sn = repr(norm)[:800] if norm is not None else ""
        sd = repr(dec)[:800] if dec is not None else ""
        has_prepend = "Prepend" in sn
        has_strip = "Strip" in sd
        if has_prepend and has_strip:
            target.decoder = _dec.Sequence(
                [
                    _dec.Replace("▁", " "),
                    _dec.ByteFallback(),
                    _dec.Fuse(),
                ]
            )
