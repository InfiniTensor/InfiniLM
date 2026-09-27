def resolve_attention_backend(backend: str, cache_type: str) -> str:
    """Resolve the default attention implementation for the selected KV cache."""
    if backend == "paged-attn":
        raise ValueError(
            "`paged-attn` was removed. Use `flash-attn` with a paged KV cache."
        )
    if backend == "default":
        return "flash-attn" if cache_type == "paged" else "static-attn"
    return backend
