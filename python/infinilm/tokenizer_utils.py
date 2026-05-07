"""
Tokenizer utilities for InfiniLM.

This module provides InfiniLMTokenizer class that encapsulates all tokenizer
operations including initialization, encoding/decoding, chat template handling,
and model-specific fixes.
"""

import os
import json
from typing import List, Optional, Union, Any

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from tokenizers import decoders as _dec


class InfiniLMTokenizer:
    """Unified tokenizer wrapper for InfiniLM.

    This class encapsulates all tokenizer-related operations including:
    - Model-specific initialization and fixes
    - Encoding/decoding
    - Chat template application
    - Pad token configuration

    Attributes:
        tokenizer: The underlying HuggingFace tokenizer instance.
        model_type: The model type string (e.g., 'llama', 'qwen2', 'minicpm').
        eos_token_id: End-of-sequence token ID(s).
    """

    def __init__(
        self,
        model_path: str,
        trust_remote_code: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize the tokenizer for a given model.

        Args:
            model_path: Path to the model directory containing config.json and tokenizer files.
            trust_remote_code: Whether to trust remote code.
                If None, will be determined based on model type.
            **kwargs: Additional keyword arguments passed to AutoTokenizer.from_pretrained().

        Raises:
            FileNotFoundError: If config.json is not found in model_path.
            ValueError: If tokenizer initialization fails.
        """
        self._model_path = os.path.expanduser(model_path)

        # Load model config to determine model type
        config_path = os.path.join(self._model_path, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config.json not found in {self._model_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        self.model_type = config_dict.get("model_type", "").lower()

        # Determine trust_remote_code based on model type if not explicitly specified
        if trust_remote_code is None:
            trust_remote_code = self._should_trust_remote_code(self.model_type)

        # Initialize the underlying tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize tokenizer from {self._model_path}: {e}"
            )

        # Apply model-specific fixes
        self._apply_model_specific_fixes()

        # Configure pad token
        self._configure_pad_token()

        # Extract EOS token ID from config
        eos_token_id = config_dict.get("eos_token_id")
        if eos_token_id is not None:
            self.eos_token_id = (
                [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
            )
        else:
            self.eos_token_id = []

        # Ensure EOS token ID is always a list
        if isinstance(self.eos_token_id, int):
            self.eos_token_id = [self.eos_token_id]

    @staticmethod
    def _should_trust_remote_code(model_type: str) -> bool:
        """Determine if trust_remote_code should be True based on model type.

        Some model types require custom tokenizer code (trust_remote_code=True),
        while others work with standard tokenizers.

        Args:
            model_type: The model type string.

        Returns:
            True if trust_remote_code should be enabled, False otherwise.
        """
        # Models that typically require custom code
        require_trust = {
            "fm9g",
            "minicpm",
            "fm9g7b",
            "minicpmv",
            "qwen2",
            "qwen3",
        }

        # Models that work with standard tokenizers (no trust_remote_code needed)
        standard_models = {
            "llama",
            "mistral",
            "gemma",
        }

        if model_type in require_trust:
            return True
        elif model_type in standard_models:
            return False
        else:
            # Default to True for unknown model types
            return True

    def _apply_model_specific_fixes(self) -> None:
        """Apply model-specific tokenizer fixes.

        Currently handles:
        - Llama models: Fix decoder to handle space replacement properly.
        """
        if self.model_type == "llama":
            self._fix_llama_decoder()

    def _fix_llama_decoder(self) -> None:
        """Fix Llama tokenizer decoder for proper space handling.

        Llama tokenizers often have a decoder that prepends spaces, causing
        double spaces or incorrect spacing in decoded text. This fix replaces
        the decoder with one that handles spaces correctly.
        """
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        target = getattr(backend, "_tokenizer", backend) if backend else None

        if target is None:
            return

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

    def _configure_pad_token(self) -> None:
        """Configure pad token and pad_token_id for the tokenizer.

        If no pad_token is set, this method tries to use eos_token as pad_token.
        If that fails, it adds a new [PAD] token.
        """
        if self.tokenizer.pad_token is not None:
            return

        if self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            # Add a new pad token
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def encode(self, text: Union[str, List[str]]) -> List[int]:
        """Encode text(s) into token IDs.

        Args:
            text: A single text string or a list of text strings.

        Returns:
            If input is a single string, returns a list of token IDs.
            If input is a list of strings, returns a list of lists of token IDs.

        Raises:
            TypeError: If text is not a string or list of strings.
        """
        if isinstance(text, str):
            return self.tokenizer.encode(text)
        elif isinstance(text, list):
            return [self.tokenizer.encode(t) for t in text]
        else:
            raise TypeError(f"Expected str or List[str], got {type(text).__name__}")

    def decode(
        self,
        token_ids: Union[List[int], Any],
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs to decode, or any iterable of integers.
            skip_special_tokens: Whether to skip special tokens in decoding.
            **kwargs: Additional keyword arguments passed to tokenizer.decode().

        Returns:
            Decoded text string.
        """
        # Convert to list if necessary (e.g., from numpy array or tensor)
        if not isinstance(token_ids, list):
            try:
                token_ids = list(token_ids)
            except TypeError:
                token_ids = [token_ids]

        return self.tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens, **kwargs
        )

    def apply_chat_template(
        self,
        conversation: List[dict],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        **kwargs: Any,
    ) -> Union[str, List[int]]:
        """Apply chat template to a conversation.

        Args:
            conversation: List of message dicts with 'role' and 'content' keys.
                Example: [{"role": "user", "content": "Hello"}]
            add_generation_prompt: Whether to add generation prompt at the end.
            tokenize: Whether to tokenize the output.
            **kwargs: Additional keyword arguments passed to tokenizer.apply_chat_template().

        Returns:
            Formatted conversation string, or list of token IDs if tokenize=True.

        Raises:
            ValueError: If the tokenizer does not have a chat template and tokenize=False.
        """
        if (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        ):
            return self.tokenizer.apply_chat_template(
                conversation=conversation,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                **kwargs,
            )
        else:
            # Fallback: construct a simple prompt from the conversation
            text_parts = []
            for msg in conversation:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    text_parts.append(f"System: {content}")
                elif role == "user":
                    text_parts.append(f"User: {content}")
                elif role == "assistant":
                    text_parts.append(f"Assistant: {content}")
                else:
                    text_parts.append(f"{role}: {content}")

            text = "\n".join(text_parts)
            if add_generation_prompt:
                text += "\nAssistant: "

            if tokenize:
                return self.encode(text)
            return text

    @property
    def model_max_length(self) -> int:
        """Get the maximum sequence length supported by the model."""
        return getattr(self.tokenizer, "model_max_length", 2048)

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> Optional[int]:
        """Get the pad token ID."""
        return self.tokenizer.pad_token_id

    @property
    def eos_token(self) -> Optional[str]:
        """Get the EOS token string."""
        return self.tokenizer.eos_token

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the underlying HuggingFace tokenizer instance.

        Returns:
            The PreTrainedTokenizer or PreTrainedTokenizerFast instance.
        """
        return self.tokenizer

    def __repr__(self) -> str:
        return (
            f"InfiniLMTokenizer(model_type='{self.model_type}', "
            f"vocab_size={self.vocab_size}, "
            f"model_max_length={self.model_max_length})"
        )


# Legacy function for backward compatibility
def infinilm_encode(tokenizer, text):
    """Encode text into token ids using the provided tokenizer.

    Deprecated: Use InfiniLMTokenizer.encode() instead.
    This function is kept for backward compatibility.
    """
    return tokenizer.encode(text)
