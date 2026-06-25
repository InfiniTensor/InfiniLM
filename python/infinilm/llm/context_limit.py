"""Context-length limits aligned with vLLM ``max_model_len`` semantics."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


_HF_MAX_LEN_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_seq_len",
    "model_max_length",
    "max_sequence_length",
)


def hf_max_position_embeddings(hf_config: Mapping[str, Any]) -> int:
    """Smallest known context cap from HF config (mirrors vLLM key scan)."""
    values: list[int] = []
    configs = [hf_config]
    text_config = hf_config.get("text_config")
    if isinstance(text_config, Mapping):
        configs.append(text_config)

    for cfg in configs:
        for key in _HF_MAX_LEN_KEYS:
            raw = cfg.get(key)
            if raw is not None:
                values.append(int(raw))

    if not values:
        return 2048
    return min(values)


def effective_max_model_len(
    hf_config: Mapping[str, Any],
    *,
    compile_max_seq: Optional[int] = None,
) -> int:
    """Runtime context cap: min(HF max position, ``INFINI_COMPILE_MAX_SEQ``)."""
    config_cap = hf_max_position_embeddings(hf_config)
    if compile_max_seq is None:
        try:
            from infinilm.compile.env import compile_max_seq_len

            compile_max_seq = compile_max_seq_len(config_cap)
        except ImportError:
            compile_max_seq = config_cap
    return min(config_cap, int(compile_max_seq))


def resolve_truncate_limit(
    truncate_prompt_tokens: Optional[int],
    max_model_len: int,
) -> Optional[int]:
    if truncate_prompt_tokens is None:
        return None
    if truncate_prompt_tokens <= -1:
        return max_model_len
    if truncate_prompt_tokens > max_model_len:
        raise ValueError(
            f"truncate_prompt_tokens ({truncate_prompt_tokens}) is greater than "
            f"max_model_len ({max_model_len})."
        )
    return int(truncate_prompt_tokens)


def truncate_prompt_token_ids(
    prompt_token_ids: Sequence[int],
    *,
    truncate_prompt_tokens: Optional[int],
    max_model_len: int,
) -> list[int]:
    """Keep the last *truncate_prompt_tokens* prompt ids (vLLM OpenAI compat)."""
    limit = resolve_truncate_limit(truncate_prompt_tokens, max_model_len)
    if limit is None:
        return list(prompt_token_ids)
    if len(prompt_token_ids) <= limit:
        return list(prompt_token_ids)
    return list(prompt_token_ids[-limit:])


def validate_prompt_length(prompt_len: int, max_model_len: int) -> None:
    if prompt_len <= 0:
        raise ValueError("The decoder prompt cannot be empty")
    if prompt_len > max_model_len:
        raise ValueError(
            f"The decoder prompt (length {prompt_len}) is longer than the "
            f"maximum model length of {max_model_len}. Make sure that "
            f"max_model_len is no smaller than the number of text tokens."
        )


def cap_max_tokens(
    max_tokens: Optional[int],
    prompt_len: int,
    max_model_len: int,
    *,
    default_max_tokens: Optional[int] = None,
) -> int:
    """Cap generation budget to ``max_model_len - prompt_len`` (vLLM ``get_max_tokens``)."""
    allowed = max_model_len - prompt_len
    if allowed <= 0:
        raise ValueError(
            f"prompt length {prompt_len} leaves no room for generation "
            f"(max_model_len={max_model_len})"
        )

    candidates = [allowed]
    if max_tokens is not None:
        candidates.append(int(max_tokens))
    if default_max_tokens is not None:
        candidates.append(int(default_max_tokens))
    return min(candidates)


def context_length_exceeded(total_length: int, max_model_len: int) -> bool:
    return total_length >= max_model_len
