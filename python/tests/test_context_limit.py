"""Tests for context-length / max_tokens admission helpers."""

from __future__ import annotations

import unittest

from infinilm.llm.context_limit import cap_max_tokens


class CapMaxTokensTest(unittest.TestCase):
    def test_request_max_not_clamped_by_server_default(self):
        # Serve --max-new-tokens 256 must not override client max_tokens=1024.
        self.assertEqual(
            cap_max_tokens(1024, prompt_len=100, max_model_len=8192, default_max_tokens=256),
            1024,
        )

    def test_default_used_when_request_omits(self):
        self.assertEqual(
            cap_max_tokens(None, prompt_len=100, max_model_len=8192, default_max_tokens=256),
            256,
        )

    def test_context_room_always_wins(self):
        self.assertEqual(
            cap_max_tokens(1024, prompt_len=8000, max_model_len=8192, default_max_tokens=256),
            192,
        )


if __name__ == "__main__":
    unittest.main()
