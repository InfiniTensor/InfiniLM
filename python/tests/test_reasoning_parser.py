"""Tests for Qwen3 thinking tag splitting."""

from __future__ import annotations

import unittest

from infinilm.server.reasoning_parser import ReasoningStreamSplitter, split_thinking_content


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
