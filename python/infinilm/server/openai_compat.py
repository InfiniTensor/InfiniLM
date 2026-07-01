"""OpenAI-compatible request field resolution."""


def resolve_chat_template_kwargs(data: dict) -> dict:
    """Merge chat_template_kwargs from extra_body and top-level request fields.

    Precedence matches vLLM: extra_body.chat_template_kwargs first, then
    top-level chat_template_kwargs overrides on key conflicts.
    """
    extra = data.get("extra_body")
    from_extra = extra.get("chat_template_kwargs") if isinstance(extra, dict) else {}
    from_top = data.get("chat_template_kwargs")
    merged = {}
    if isinstance(from_extra, dict):
        merged.update(from_extra)
    if isinstance(from_top, dict):
        merged.update(from_top)
    return merged
