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
# Explicit / late answer cues (avoid scraping the first A-D in mid-think).
_EXPLICIT_ANSWER_RE = re.compile(
    r"(?:答案|answer)\s*[:：是为]?\s*([A-D])\b",
    re.IGNORECASE,
)
_LATE_ANSWER_RES = (
    re.compile(r"(?:应选|选择|最终选)\s*([A-D])\b"),
    re.compile(r"选项\s*([A-D])\s*(?:正确|是正确的|不恰当|不正确)"),
    re.compile(r"([A-D])\s*(?:选项)?\s*(?:正确|不恰当|不正确)"),
)
_OPTION_LINE_RE = re.compile(
    r"([A-D])[\.、．]\s*(.+?)(?=\n[A-D][\.、．]|\n答案|$)",
    re.S,
)
_HAS_LETTER_RE = re.compile(r"\b([A-D])\b")


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


def _extract_answer_letter(text: str) -> str:
    """Best-effort letter from reasoning; prefer late/explicit cues."""
    if not text:
        return ""
    explicit = _EXPLICIT_ANSWER_RE.search(text)
    if explicit:
        return explicit.group(1).upper()
    best_pos = -1
    best_letter = ""
    for cre in _LATE_ANSWER_RES:
        for m in cre.finditer(text):
            if m.start() >= best_pos:
                best_pos = m.start()
                best_letter = m.group(1).upper()
    if best_letter:
        return best_letter
    suffix = text[-240:]
    m = re.search(r"([A-D])", suffix)
    return m.group(1).upper() if m else ""


def has_unclosed_thinking(text: str) -> bool:
    """True if an open thinking tag has no matching close tag yet."""
    if not text:
        return False
    match_open = _THINK_OPEN.search(text)
    if not match_open:
        return False
    return _THINK_CLOSE.search(text, match_open.end()) is None


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


def parse_mc_options_from_text(text: str) -> dict[str, str]:
    """Parse ``A. … B. …`` options from a CEval-style user prompt."""
    if not text:
        return {}
    opts: dict[str, str] = {}
    for m in _OPTION_LINE_RE.finditer(text):
        opts[m.group(1)] = m.group(2).strip()
    return opts


def match_option_letter(content: str, options: dict[str, str]) -> str:
    """If ``content`` restates exactly one option's text (no A-D letter), return it."""
    if not content or not options:
        return ""
    if _HAS_LETTER_RE.search(content):
        return ""
    hits = [L for L, t in options.items() if t and t in content]
    return hits[0] if len(hits) == 1 else ""


def options_from_chat_messages(messages: list | None) -> dict[str, str]:
    """Options from the last user message in an OpenAI-style chat."""
    if not messages:
        return {}
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            # multimodal: concatenate text parts
            parts = [
                str(p.get("text", ""))
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = "\n".join(parts)
        if isinstance(content, str) and content.strip():
            return parse_mc_options_from_text(content)
    return {}


def resolve_chat_visible_content(
    output_text: str,
    *,
    messages: list | None = None,
) -> tuple[str, str]:
    """Split thinking and choose API ``content`` for unfinished / empty cases.

    CEval-style filters take the first ``[A-D]`` in ``content``. Dumping an
    unfinished ``<think>`` body scrapes the first option letter in reasoning.
    Prefer an extracted answer letter; raw fallback only when think is closed.
    When the model answers with option prose and no letter, map unique option
    text from the last user message to A-D.
    """
    reasoning, visible_content = split_thinking_content(output_text)
    raw = (output_text or "").strip()
    options = options_from_chat_messages(messages)

    def _maybe_option_letter(text: str) -> str:
        return match_option_letter(text, options) if options else ""

    if (visible_content or "").strip():
        cleaned = _THINK_CLOSE.sub("", visible_content).strip()
        # Close-tag spam or long post-think prose: expose letter only.
        if (not cleaned) or _THINK_CLOSE.search(visible_content) or len(cleaned) > 80:
            letter = (
                _extract_answer_letter(visible_content)
                or _extract_answer_letter(reasoning)
                or _maybe_option_letter(cleaned or visible_content)
            )
            if letter:
                return reasoning, letter
        letter = _maybe_option_letter(cleaned or visible_content)
        if letter and not _HAS_LETTER_RE.search(cleaned or visible_content):
            return reasoning, letter
        return reasoning, cleaned or visible_content

    if not raw:
        return reasoning, visible_content

    letter = (
        _extract_answer_letter(raw)
        or _extract_answer_letter(reasoning)
        or _maybe_option_letter(raw)
        or _maybe_option_letter(reasoning)
    )
    if letter:
        return reasoning, letter
    if has_unclosed_thinking(raw):
        return reasoning, ""
    return reasoning, raw


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
