"""
Detector for Llama 3.2 / 3.1 models with json tool call format.
"""

import json
import logging
import re
from typing import List

from infinilm.agents.base_detector import BaseFormatDetector
from infinilm.agents.protocol import Tool
from infinilm.agents.types import StreamingParseResult, StructureInfo, _GetInfoFunc
from infinilm.agents.utils import safe_literal_eval

logger = logging.getLogger(__name__)


class Llama32Detector(BaseFormatDetector):
    """
    Detector for Llama 3.2 models with json tool call format.
    Format Structure:
      <|python_tag>{"name":"xxx", "arguments":{...}}
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<|python_tag|>"
        self.tool_call_separator = ";"

    def _convert_python_dict_to_json(self, text: str) -> str:
        """Convert Python dict strings to JSON format."""
        try:
            parsed = safe_literal_eval(text.strip())
            if isinstance(parsed, dict):
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass
        return text

    def has_tool_call(self, text: str) -> bool:
        return "<|python_tag|>" in text or text.startswith("{")

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if "<|python_tag|>" not in text and not text.startswith("{"):
            return StreamingParseResult(normal_text=text, calls=[])

        if "<|python_tag|>" in text:
            normal_text, action_text = text.split("<|python_tag|>", maxsplit=1)
        else:
            normal_text, action_text = "", text

        decoder = json.JSONDecoder()
        idx = 0
        safe_idx = idx
        all_actions = []
        action_text_len = len(action_text)
        while idx < action_text_len:
            try:
                obj, end = decoder.raw_decode(action_text[idx:])
                all_actions.append(obj)
                idx += end + len(self.tool_call_separator)
                safe_idx = idx
            except json.JSONDecodeError:
                try:
                    dict_end = idx
                    brace_count = 0
                    for i in range(idx, action_text_len):
                        if action_text[i] == "{":
                            brace_count += 1
                        elif action_text[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                dict_end = i + 1
                                break
                    if dict_end > idx:
                        potential_dict = action_text[idx:dict_end]
                        json_version = self._convert_python_dict_to_json(potential_dict)
                        if json_version != potential_dict:
                            obj, _ = decoder.raw_decode(json_version)
                            all_actions.append(obj)
                            idx = dict_end + len(self.tool_call_separator)
                            safe_idx = idx
                            continue
                except Exception:
                    pass
                next_obj_start = action_text.find('{"name":', idx + 1)
                if next_obj_start == -1:
                    break
                idx = next_obj_start

        # Objects without a "name" key are plain JSON answers, not tool
        # calls; pass the original text through untouched.
        if all_actions and not any(
            isinstance(action, dict) and "name" in action for action in all_actions
        ):
            return StreamingParseResult(normal_text=text, calls=[])

        calls = self.parse_base_json(all_actions, tools) if all_actions else []
        trailing_text = (
            action_text[safe_idx:].strip() if safe_idx < action_text_len else ""
        )
        return StreamingParseResult(
            normal_text=normal_text + trailing_text, calls=calls
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        converted_buffer = self._buffer
        converted_buffer = re.sub(r"'([^']*)':", r'"\1":', converted_buffer)
        converted_buffer = re.sub(r":\s*'([^']*)'", r': "\1"', converted_buffer)
        original_buffer = self._buffer
        self._buffer = converted_buffer
        try:
            result = super().parse_streaming_increment("", tools)
            return result
        except Exception:
            self._buffer = original_buffer
            # original_buffer already contains new_text, so pass empty string
            # to avoid adding new_text twice.
            return super().parse_streaming_increment("", tools)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<|python_tag|>{"name":"' + name + '", "arguments":',
            end="}",
            trigger="<|python_tag|>",
        )
