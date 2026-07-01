"""Tests for OpenAI-compatible request field resolution."""

from __future__ import annotations

import unittest

from infinilm.server.openai_compat import resolve_chat_template_kwargs


class ResolveChatTemplateKwargsTest(unittest.TestCase):
    def test_ultrarag_extra_body_path(self):
        data = {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        self.assertEqual(
            resolve_chat_template_kwargs(data),
            {"enable_thinking": False},
        )

    def test_top_level_only(self):
        data = {"chat_template_kwargs": {"enable_thinking": True}}
        self.assertEqual(
            resolve_chat_template_kwargs(data),
            {"enable_thinking": True},
        )

    def test_top_level_overrides_extra_body(self):
        data = {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "foo": "from_extra",
                }
            },
            "chat_template_kwargs": {"enable_thinking": True},
        }
        self.assertEqual(
            resolve_chat_template_kwargs(data),
            {"enable_thinking": True, "foo": "from_extra"},
        )

    def test_missing_kwargs(self):
        self.assertEqual(resolve_chat_template_kwargs({}), {})

    def test_extra_body_not_dict(self):
        data = {"extra_body": "not-a-dict"}
        self.assertEqual(resolve_chat_template_kwargs(data), {})

    def test_kwargs_not_dict(self):
        data = {
            "extra_body": {"chat_template_kwargs": "not-a-dict"},
            "chat_template_kwargs": 42,
        }
        self.assertEqual(resolve_chat_template_kwargs(data), {})


if __name__ == "__main__":
    unittest.main()
