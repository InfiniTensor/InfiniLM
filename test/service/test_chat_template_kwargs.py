"""Regression tests for chat template options in request preparation."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from infinilm.llm.llm import AsyncLLMEngine


class ChatTemplateKwargsTest(unittest.TestCase):
    def setUp(self):
        self.engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
        self.engine.engine = SimpleNamespace(
            apply_chat_template=Mock(return_value="formatted prompt"),
            process=Mock(return_value={"input_ids": torch.tensor([[1, 2]])}),
            processor=SimpleNamespace(get_mm_token_index_list=Mock(return_value=None)),
            tokenize=Mock(return_value=[7, 8]),
        )
        self.engine._add_prepared_request = lambda prepared, sampling, data: prepared
        self.messages = [{"role": "user", "content": "7*8?"}]

    def check_options(self, options):
        self.engine.engine.apply_chat_template.assert_called_once_with(
            self.messages, add_generation_prompt=True, chat_template_kwargs=options
        )

    def test_sync_chat_forwards_false_and_custom_options(self):
        options = {"enable_thinking": False, "custom_option": "value"}
        result = self.engine.add_chat_request(self.messages, chat_template_kwargs=options)
        self.check_options(options)
        self.assertEqual(result["prompt_token_ids"], [1, 2])
        self.assertEqual(options, {"enable_thinking": False, "custom_option": "value"})

    def test_async_chat_forwards_false(self):
        options = {"enable_thinking": False}
        result = asyncio.run(
            self.engine.add_chat_request_async(
                messages=self.messages, chat_template_kwargs=options
            )
        )
        self.check_options(options)
        self.assertEqual(result["prompt"], "formatted prompt")

    def test_omitted_options_keep_default(self):
        self.engine.add_chat_request(self.messages)
        self.check_options(None)

    def test_true_is_preserved(self):
        self.engine.add_chat_request(
            self.messages, chat_template_kwargs={"enable_thinking": True}
        )
        self.check_options({"enable_thinking": True})

    def test_raw_prompt_bypasses_template(self):
        result = self.engine.add_request(
            None, prompt="raw", chat_template_kwargs={"enable_thinking": False}
        )
        self.engine.engine.apply_chat_template.assert_not_called()
        self.assertEqual(result["prompt_token_ids"], [7, 8])

    def test_token_ids_bypass_template(self):
        result = asyncio.run(
            self.engine.add_request_async(
                None, prompt_token_ids=[3, 4], chat_template_kwargs={"enable_thinking": False}
            )
        )
        self.engine.engine.apply_chat_template.assert_not_called()
        self.assertEqual(result["prompt_token_ids"], [3, 4])


if __name__ == "__main__":
    unittest.main()
