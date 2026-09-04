"""
Base format detector for tool-call parsing.
Removes structural_tag / XGrammar-dependent methods.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Union

from .protocol import Tool, ToolChoice
from .types import StreamingParseResult, ToolCallItem
from .utils import _find_common_prefix, _is_complete_json, _partial_json_loads

logger = logging.getLogger(__name__)


try:
    from partial_json_parser.core.options import Allow

    PARTIAL_JSON_AVAILABLE = True
except ImportError:
    Allow = None
    PARTIAL_JSON_AVAILABLE = False


class BaseFormatDetector(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(self):
        # Streaming state management
        self._buffer = ""
        # Stores complete tool call info for each tool being parsed.
        self.prev_tool_call_arr: List[Dict] = []
        # Index of currently streaming tool call.
        self.current_tool_id: int = -1
        # Flag for whether current tool's name has been sent to client.
        self.current_tool_name_sent: bool = False
        # Tracks raw JSON string content streamed to client for each tool's arguments.
        self.streamed_args_for_tool: List[str] = []

        # Token configuration (override in subclasses)
        self.bot_token = ""
        self.eot_token = ""
        self.tool_call_separator = ", "

    def clear(self):
        self._buffer = ""
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []
        if hasattr(self, "_tool_indices"):
            delattr(self, "_tool_indices")

    def _get_tool_indices(self, tools: List[Tool]) -> Dict[str, int]:
        """Get a mapping of tool names to their indices in the tools list."""
        return {
            tool.function.name: i for i, tool in enumerate(tools) if tool.function.name
        }

    def parse_base_json(self, action: Any, tools: List[Tool]) -> List[ToolCallItem]:
        tool_indices = self._get_tool_indices(tools)
        if not isinstance(action, list):
            action = [action]

        results = []
        for act in action:
            name = act.get("name")
            if not (name and name in tool_indices):
                logger.warning(f"Model attempted to call undefined function: {name}")
                continue

            results.append(
                ToolCallItem(
                    tool_index=tool_indices.get(name, -1),
                    name=name,
                    parameters=json.dumps(
                        act.get("parameters") or act.get("arguments", {}),
                        ensure_ascii=False,
                    ),
                )
            )

        return results

    @abstractmethod
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parses the text in one shot."""
        action = json.loads(text)
        return StreamingParseResult(calls=self.parse_base_json(action, tools))

    def _ends_with_partial_token(self, buffer: str, bot_token: str) -> int:
        """Check if buffer ends with a partial bot_token."""
        for i in range(1, min(len(buffer) + 1, len(bot_token))):
            if bot_token.startswith(buffer[-i:]):
                return i
        return 0

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.
        Base implementation works best with formats where:
        1. bot_token is followed immediately by JSON
        2. JSON can be parsed incrementally using partial_json_loads
        3. Multiple tool calls are separated by "; " or ", "
        """
        self._buffer += new_text
        current_text = self._buffer

        if not (
            self.has_tool_call(current_text)
            or (
                self.current_tool_id > 0
                and current_text.startswith(self.tool_call_separator)
            )
        ):
            if not self._ends_with_partial_token(self._buffer, self.bot_token):
                normal_text = self._buffer
                self._buffer = ""
                if self.eot_token in normal_text:
                    normal_text = normal_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=normal_text)
            else:
                return StreamingParseResult()

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        flags = (
            Allow.ALL
            if (Allow is not None and self.current_tool_name_sent)
            else (Allow.ALL & ~Allow.STR if Allow is not None else None)
        )
        if flags is None:
            flags = 0  # fallback when partial_json_parser unavailable

        try:
            try:
                used_separator_branch = False
                if self.current_tool_id > 0 and current_text.startswith(
                    self.tool_call_separator
                ):
                    start_idx = len(self.tool_call_separator)
                    used_separator_branch = True
                else:
                    tool_call_pos = current_text.find(self.bot_token)
                    if tool_call_pos != -1:
                        start_idx = tool_call_pos + len(self.bot_token)
                    else:
                        start_idx = 0

                if start_idx >= len(current_text):
                    return StreamingParseResult()

                try:
                    obj, end_idx = _partial_json_loads(current_text[start_idx:], flags)
                except (Exception, json.JSONDecodeError):
                    if used_separator_branch and self.bot_token in current_text:
                        start_idx = current_text.find(self.bot_token) + len(
                            self.bot_token
                        )
                        if start_idx >= len(current_text):
                            return StreamingParseResult()
                        obj, end_idx = _partial_json_loads(
                            current_text[start_idx:], flags
                        )
                    else:
                        raise

                is_current_complete = _is_complete_json(
                    current_text[start_idx : start_idx + end_idx]
                )

                # A complete JSON object without a "name" field can never
                # become a tool call; release the buffer as normal text
                # instead of holding it back forever.
                if is_current_complete and "name" not in obj:
                    normal_text = self._buffer
                    self._buffer = ""
                    if self.eot_token in normal_text:
                        normal_text = normal_text.replace(self.eot_token, "")
                    return StreamingParseResult(normal_text=normal_text)

                if "name" in obj and obj["name"] not in self._tool_indices:
                    self._buffer = ""
                    self.current_tool_id = -1
                    self.current_tool_name_sent = False
                    if self.streamed_args_for_tool:
                        self.streamed_args_for_tool.pop()
                    return StreamingParseResult()

                if "parameters" in obj:
                    obj["arguments"] = obj["parameters"]

                current_tool_call = obj

            except (Exception, json.JSONDecodeError):
                return StreamingParseResult()

            if not current_tool_call:
                return StreamingParseResult()

            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")

                if function_name and function_name in self._tool_indices:
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.streamed_args_for_tool.append("")
                    elif self.current_tool_id >= len(self.streamed_args_for_tool):
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                    res = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        ],
                    )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()

                if cur_arguments is not None:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_arguments = None
                    if self.current_tool_id < len(self.prev_tool_call_arr):
                        prev_arguments = self.prev_tool_call_arr[
                            self.current_tool_id
                        ].get("arguments")

                    argument_diff = None
                    completing_tool_id = self.current_tool_id

                    if is_current_complete:
                        argument_diff = cur_args_json[sent:]
                        completing_tool_id = self.current_tool_id
                        self._buffer = current_text[start_idx + end_idx :]

                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                        if cur_args_json != prev_args_json:
                            prefix = _find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    if self.current_tool_id >= 0:
                        while len(self.prev_tool_call_arr) <= self.current_tool_id:
                            self.prev_tool_call_arr.append({})
                        self.prev_tool_call_arr[self.current_tool_id] = (
                            current_tool_call
                        )

                    if is_current_complete:
                        self.current_tool_name_sent = False
                        self.current_tool_id += 1

                    if argument_diff is not None:
                        tool_index_to_use = (
                            completing_tool_id
                            if is_current_complete
                            else self.current_tool_id
                        )
                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=tool_index_to_use,
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        self.streamed_args_for_tool[tool_index_to_use] += argument_diff

            return res

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult()

    @abstractmethod
    def has_tool_call(self, text: str) -> bool:
        """Check if the given text contains function call markers."""
        raise NotImplementedError()

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Called once when the stream ends; flush any buffered state."""
        if self._buffer:
            return self.parse_streaming_increment("", tools)
        return StreamingParseResult()

    def supports_structural_tag(self) -> bool:
        return False

    def parses_required_natively(self) -> bool:
        return False

    @abstractmethod
    def structure_info(self):
        """Return a function that creates StructureInfo for constrained generation."""
        raise NotImplementedError()

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ):
        """Return a model-native structural tag when supported. (Disabled in InfiniLM MVP)"""
        return None

    def get_auto_tool_call_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ):
        return None
