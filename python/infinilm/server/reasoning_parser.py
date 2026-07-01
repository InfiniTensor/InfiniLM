"""Split Qwen3-style thinking tags from assistant output."""

from __future__ import annotations

import re

_THINK_OPEN = re.compile(r"<\s*(?:think|redacted_thinking)\s*>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"<\s*/\s*(?:think|redacted_thinking)\s*>", re.IGNORECASE)
_REDACTED_OPEN = "<" + "redacted_thinking" + ">"
_OPEN_TAG_PREFIXES = (
    _REDACTED_OPEN,
    "<" + "think" + ">",
)


def _might_be_partial_open_tag(text: str) -> bool:
    """True if text may still be an incomplete thinking open tag."""
    lower = text.lstrip().lower()
    if not lower.startswith("<"):
        return False
    if _THINK_OPEN.search(text):
        return False
    for tag in _OPEN_TAG_PREFIXES:
        if len(lower) < len(tag) and tag.startswith(lower):
            return True
    return False


def split_thinking_content(text: str) -> tuple[str, str]:
    """Return (reasoning, visible_content) from raw model output."""
    if not text:
        return "", ""

    match_open = _THINK_OPEN.search(text)
    if not match_open:
        if _might_be_partial_open_tag(text):
            return "", ""
        return "", text

    match_close = _THINK_CLOSE.search(text, match_open.end())
    if not match_close:
        return text[match_open.end() :].strip(), ""

    reasoning = text[match_open.end() : match_close.start()].strip()
    content = text[match_close.end() :].lstrip("\n")
    return reasoning, content


class ReasoningStreamSplitter:
    """Incrementally hide thinking tags from streamed token text."""

    def __init__(self) -> None:
        self._raw = ""
        self._emitted_content_len = 0

    def feed(self, piece: str) -> str:
        if not piece:
            return ""
        self._raw += piece
        _, content = split_thinking_content(self._raw)
        if len(content) < self._emitted_content_len:
            self._emitted_content_len = 0
        new_content = content[self._emitted_content_len :]
        self._emitted_content_len = len(content)
        return new_content
