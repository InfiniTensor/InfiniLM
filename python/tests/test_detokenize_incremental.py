"""Regression tests for full-sequence decode (fixes slice-decode missing spaces)."""

from __future__ import annotations

import os
import unittest


def _update_generated_text_from_tokens(tokenizer, req) -> bool:
    """Mirror of LLMEngine._update_generated_text_from_tokens (llm.py)."""
    full_text = tokenizer.decode(req.generated_token_ids)
    holds_back = bool(full_text) and full_text.endswith("\ufffd")
    if not holds_back:
        req.generated_text = full_text
    return holds_back


def _slice_decode_append(tokenizer, token_ids: list[int]) -> str:
    """Legacy buggy path: decode only new tokens since last offset and append."""
    text = ""
    offset = 0
    for i in range(len(token_ids)):
        pending = token_ids[offset : i + 1]
        text += tokenizer.decode(pending)
        offset = i + 1
    return text


class _Req:
    def __init__(self):
        self.generated_token_ids: list[int] = []
        self.generated_text: str = ""


class DetokenizeIncrementalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise unittest.SkipTest("transformers not installed") from None

        model = os.environ.get("INFINILM_TEST_TOKENIZER", "gpt2")
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(model)
        except OSError as exc:
            raise unittest.SkipTest(f"tokenizer unavailable: {model}") from exc

    def test_slice_decode_differs_from_full_on_english(self):
        sample = "Okay, the user is asking for a detailed explanation"
        ids = self.tokenizer.encode(sample, add_special_tokens=False)
        self.assertGreater(len(ids), 3)
        buggy = _slice_decode_append(self.tokenizer, ids)
        reference = self.tokenizer.decode(ids)
        if buggy == reference:
            self.skipTest("sample does not trigger slice-decode bug on this tokenizer")
        self.assertNotEqual(buggy, reference)
        self.assertIn(" ", reference)

    def test_full_decode_helper_matches_reference(self):
        sample = "Hello, the user is asking for help with chunked prefill."
        ids = self.tokenizer.encode(sample, add_special_tokens=False)
        req = _Req()
        for tid in ids:
            req.generated_token_ids.append(tid)
            _update_generated_text_from_tokens(self.tokenizer, req)
        self.assertEqual(req.generated_text, self.tokenizer.decode(ids))

    def test_utf8_holdback_does_not_commit_partial_text(self):
        req = _Req()
        req.generated_text = "committed"
        req.generated_token_ids = [1, 2, 3]

        class _HoldbackTokenizer:
            def decode(self, _token_ids):
                return "partial\ufffd"

        holds_back = _update_generated_text_from_tokens(_HoldbackTokenizer(), req)
        self.assertTrue(holds_back)
        self.assertEqual(req.generated_text, "committed")


if __name__ == "__main__":
    unittest.main()
