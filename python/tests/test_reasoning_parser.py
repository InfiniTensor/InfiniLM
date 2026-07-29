"""Tests for Qwen3 thinking tag splitting."""

from __future__ import annotations

import unittest

from infinilm.server.reasoning_parser import (
    ReasoningStreamSplitter,
    has_unclosed_thinking,
    resolve_chat_visible_content,
    split_thinking_content,
)


class SplitThinkingContentTest(unittest.TestCase):
    def test_no_thinking_passthrough(self):
        reasoning, content = split_thinking_content("Hello world")
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "Hello world")

    def test_redacted_thinking_tags(self):
        open_tag = "<" + "redacted_thinking" + ">"
        close_tag = "</" + "redacted_thinking" + ">"
        raw = f"{open_tag}\nchain\n{close_tag}\n\n1943"
        reasoning, content = split_thinking_content(raw)
        self.assertEqual(reasoning, "chain")
        self.assertEqual(content, "1943")

    def test_think_tags(self):
        open_tag = "<" + "think" + ">"
        close_tag = "</" + "think" + ">"
        raw = f"{open_tag}\nchain\n{close_tag}\n\nanswer"
        reasoning, content = split_thinking_content(raw)
        self.assertEqual(reasoning, "chain")
        self.assertEqual(content, "answer")

    def test_partial_open_tag_holds_back(self):
        open_tag = "<" + "redacted_thinking" + ">"
        reasoning, content = split_thinking_content(open_tag[:-1])
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "")

    def test_unclosed_thinking(self):
        open_tag = "<" + "redacted_thinking" + ">"
        reasoning, content = split_thinking_content(f"{open_tag}\nstill thinking")
        self.assertEqual(reasoning, "still thinking")
        self.assertEqual(content, "")


class HasUnclosedThinkingTest(unittest.TestCase):
    def test_open_without_close(self):
        open_tag = "<" + "think" + ">"
        self.assertTrue(has_unclosed_thinking(f"{open_tag}\noption A then B"))

    def test_closed(self):
        open_tag = "<" + "think" + ">"
        close_tag = "</" + "think" + ">"
        self.assertFalse(has_unclosed_thinking(f"{open_tag}\nx{close_tag}\nD"))

    def test_no_tags(self):
        self.assertFalse(has_unclosed_thinking("D"))


class ResolveChatVisibleContentTest(unittest.TestCase):
    def test_unfinished_think_does_not_dump_raw(self):
        open_tag = "<" + "think" + ">"
        raw = f"{open_tag}\n先看选项A正确，再看B。"
        reasoning, content = resolve_chat_visible_content(raw)
        self.assertIn("选项A", reasoning)
        # Late "选项A正确" cue extracts A (better than dumping full think).
        self.assertEqual(content, "A")

    def test_unfinished_think_explicit_answer(self):
        open_tag = "<" + "think" + ">"
        raw = f"{open_tag}\n讨论A和B。答案：C"
        _, content = resolve_chat_visible_content(raw)
        self.assertEqual(content, "C")

    def test_closed_empty_body_falls_back_to_raw(self):
        open_tag = "<" + "think" + ">"
        close_tag = "</" + "think" + ">"
        raw = f"{open_tag}\n好的\n{close_tag}"
        _, content = resolve_chat_visible_content(raw)
        self.assertEqual(content, raw)

    def test_normal_answer_after_think(self):
        open_tag = "<" + "think" + ">"
        close_tag = "</" + "think" + ">"
        raw = f"{open_tag}\nchain\n{close_tag}\nD"
        _, content = resolve_chat_visible_content(raw)
        self.assertEqual(content, "D")

    def test_close_tag_spam_extracts_letter(self):
        close_tag = "</" + "think" + ">"
        raw = "选项C正确。\n" + "\n".join([close_tag] * 5)
        _, content = resolve_chat_visible_content(raw)
        self.assertEqual(content, "C")

    def test_option_prose_maps_to_letter(self):
        messages = [
            {
                "role": "user",
                "content": (
                    "下列属于印花税纳税人的是____。\n"
                    "A. 合同的证人\nB. 合同的担保人\n"
                    "C. 合同的鉴定人\nD. 合同的当事人\n答案："
                ),
            }
        ]
        raw = "根据我国《印花税暂行条例》的规定，下列各项中，属于印花税的纳税人的是合同的当事人。"
        _, content = resolve_chat_visible_content(raw, messages=messages)
        self.assertEqual(content, "D")


class ReasoningStreamSplitterTest(unittest.TestCase):
    def test_streams_only_visible_content(self):
        open_tag = "<" + "redacted_thinking" + ">"
        close_tag = "</" + "redacted_thinking" + ">"
        splitter = ReasoningStreamSplitter()
        out = []
        for piece in [f"{open_tag}\n", "hidden\n", f"{close_tag}\n\n", "Hi"]:
            out.append(splitter.feed(piece))
        self.assertEqual("".join(out), "Hi")


if __name__ == "__main__":
    unittest.main()
