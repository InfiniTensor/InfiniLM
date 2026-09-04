"""
Detector for Qwen3 models' xml tool call format.
"""

import json
import logging
import re
from typing import List

from infinilm.agents.base_detector import Allow, BaseFormatDetector
from infinilm.agents.protocol import Tool
from infinilm.agents.types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from infinilm.agents.utils import _find_common_prefix, _partial_json_loads

logger = logging.getLogger(__name__)


class Qwen3XmlDetector(BaseFormatDetector):
    """
    Detector for Qwen3 models with xml tool call format.
    Format Structure:
       {"name":"xxx", "arguments":{...}}
    wrapped in opening/closing "tool_call" xml tags.

    Streaming model: outside a block, text is emitted as normal content until
    the opening tag is seen. Inside a block, content accumulates in a private
    buffer; partial-JSON parsing streams the tool name and argument deltas as
    they arrive, and the closing tag finalizes the call (guaranteeing the full
    arguments are delivered exactly once).
    """

    def __init__(self):
        super().__init__()
        # Built via concatenation so the literal tag sequences stay intact
        # in this source file.
        self.bot_token = "<" + "tool_call" + ">"
        self.eot_token = "</" + "tool_call" + ">"
        self.tool_call_separator = "\n"
        self._in_tool_block = False
        self._block_buffer = ""

    def clear(self):
        super().clear()
        self._in_tool_block = False
        self._block_buffer = ""

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    @staticmethod
    def _hold_len(value: str, token: str) -> int:
        """How many trailing chars of ``value`` could be the start of ``token``."""
        for i in range(min(len(value), len(token) - 1), 0, -1):
            if value.endswith(token[:i]):
                return i
        return 0

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        tool_indices = self._get_tool_indices(tools)
        calls = []
        normal_parts: List[str] = []

        while self._buffer or self._in_tool_block:
            if not self._in_tool_block:
                bot_idx = self._buffer.find(self.bot_token)
                if bot_idx == -1:
                    # Hold back a trailing fragment that could start the tag.
                    hold = self._hold_len(self._buffer, self.bot_token)
                    if hold:
                        normal_parts.append(self._buffer[:-hold])
                        self._buffer = self._buffer[-hold:]
                    else:
                        normal_parts.append(self._buffer)
                        self._buffer = ""
                    # A stray closing tag outside a block is stripped.
                    emitted = "".join(normal_parts)
                    normal_parts = [emitted.replace(self.eot_token, "")]
                    break
                # Emit text before the opening tag (strip stray closing tags,
                # hold a trailing fragment that could start one).
                before = self._buffer[:bot_idx]
                eot_hold = self._hold_len(before, self.eot_token)
                if eot_hold:
                    normal_parts.append(before[:-eot_hold])
                    self._buffer = before[-eot_hold:] + self._buffer[bot_idx:]
                    break
                normal_parts.append(before.replace(self.eot_token, ""))
                self._buffer = self._buffer[bot_idx + len(self.bot_token) :]
                self._in_tool_block = True
                self._block_buffer = ""
                continue

            # --- inside a tool block ---
            self._block_buffer += self._buffer
            self._buffer = ""

            eot_idx = self._block_buffer.find(self.eot_token)
            if eot_idx != -1:
                calls.extend(self._finalize_block(eot_idx, tool_indices))
                self._in_tool_block = False
                continue

            # Block not closed yet: stream name / argument deltas.
            calls.extend(self._stream_partial_block(tool_indices))
            break

        normal_text = "".join(normal_parts)
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _ensure_call_slot(self):
        if self.current_tool_id == -1:
            self.current_tool_id = len(self.prev_tool_call_arr)
            self.streamed_args_for_tool.append("")
            self.prev_tool_call_arr.append({})

    def _stream_partial_block(self, tool_indices) -> List[ToolCallItem]:
        """Emit name / argument deltas for a still-open block."""
        calls: List[ToolCallItem] = []
        flags = (
            Allow.ALL
            if (Allow is not None and self.current_tool_name_sent)
            else (Allow.ALL & ~Allow.STR if Allow is not None else None)
        )
        try:
            obj, _ = _partial_json_loads(self._block_buffer.strip(), flags)
        except Exception:
            return calls
        if not isinstance(obj, dict):
            return calls

        name = obj.get("name")
        if name and name in tool_indices and not self.current_tool_name_sent:
            self._ensure_call_slot()
            calls.append(self._call_item(tool_indices[name], name, ""))
            self.current_tool_name_sent = True

        args = obj.get("arguments")
        if args is not None and self.current_tool_name_sent:
            self._ensure_call_slot()
            cur_args_json = json.dumps(args, ensure_ascii=False)
            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                "arguments"
            )
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = args
            if prev_arguments is not None:
                prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                if cur_args_json != prev_args_json:
                    sent = self.streamed_args_for_tool[self.current_tool_id]
                    prefix = _find_common_prefix(prev_args_json, cur_args_json)
                    diff = prefix[len(sent) :]
                    if diff:
                        self.streamed_args_for_tool[self.current_tool_id] += diff
                        calls.append(
                            self._call_item(
                                tool_indices.get(name, self.current_tool_id),
                                None,
                                diff,
                            )
                        )
        return calls

    def _finalize_block(self, eot_idx: int, tool_indices) -> List[ToolCallItem]:
        """Close the block at ``eot_idx`` and deliver any unsent remainder."""
        calls: List[ToolCallItem] = []
        content = self._block_buffer[:eot_idx].strip()
        remainder = self._block_buffer[eot_idx + len(self.eot_token) :]
        self._block_buffer = ""

        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            obj = None

        if isinstance(obj, dict):
            name = obj.get("name")
            if name and name in tool_indices:
                if not self.current_tool_name_sent:
                    self._ensure_call_slot()
                    calls.append(self._call_item(tool_indices[name], name, ""))
                    self.current_tool_name_sent = True
                args = obj.get("arguments") or obj.get("parameters") or {}
                final_args_json = json.dumps(args, ensure_ascii=False)
                sent = self.streamed_args_for_tool[self.current_tool_id]
                if final_args_json.startswith(sent):
                    diff = final_args_json[len(sent) :]
                else:
                    # Snapshots diverged from what was streamed; re-emit the
                    # full arguments to guarantee a correct, complete value.
                    diff = final_args_json
                    sent = ""
                if diff:
                    self.streamed_args_for_tool[self.current_tool_id] = sent + diff
                    calls.append(self._call_item(tool_indices[name], None, diff))

        # Reset per-call state for the next block; reprocess any remainder.
        self.current_tool_name_sent = False
        self.current_tool_id = -1
        self._buffer = remainder + self._buffer
        return calls

    @staticmethod
    def _call_item(tool_index: int, name, parameters: str) -> ToolCallItem:
        return ToolCallItem(tool_index=tool_index, name=name, parameters=parameters)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Flush buffered state; release a truncated block as normal text."""
        result = StreamingParseResult()
        sp_result = self.parse_streaming_increment("", tools)
        result.normal_text += sp_result.normal_text
        result.calls.extend(sp_result.calls)

        leftover = ""
        if self._in_tool_block and self._block_buffer:
            # Truncated block at end of stream: emit it verbatim.
            leftover = self.bot_token + self._block_buffer
            self._block_buffer = ""
            self._in_tool_block = False
        if self._buffer:
            leftover += self._buffer
            self._buffer = ""
        if leftover:
            result.normal_text += leftover
        return result

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse all complete tool_call blocks in one shot."""
        pattern = re.compile(
            re.escape(self.bot_token) + r"(.*?)" + re.escape(self.eot_token),
            re.DOTALL,
        )
        matches = pattern.findall(text)
        if not matches:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text = pattern.sub("", text).strip()
        calls = []
        for content in matches:
            try:
                action = json.loads(content.strip())
            except json.JSONDecodeError:
                continue
            calls.extend(self.parse_base_json(action, tools))
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        """Return a builder for Qwen3 XML tool-call structural tags.

        The tag bounds are ``<tool_call>...</tool_call>``.  This metadata
        is ready for XGrammar / constrained-decoding integration; the tag
        is not yet enforced because MVP disables ``get_structural_tag()``.
        """
        return lambda name: StructureInfo(
            begin="<tool_call>",
            end="</tool_call>",
            trigger="",
        )
