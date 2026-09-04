"""
Detector for GLM-4-9B-Chat-0414 style models ("metadata" tool call format).

The official protocol (see the THUDM/glm-4-9b-chat-0414 model card):

- tools are listed in a ``# 可用工具`` system section of the prompt;
- the model calls a tool by emitting the function name on one line and a
  JSON object with the arguments on the next line::

      get_weather
      {"city": "北京"}

- the official decoder detects calls with the regex
  ``([^\\n`]*?)\\n({.*?})(?=\\w*\\n|$)`` and parses the arguments with
  ``json.loads`` (falling back to ``ast.literal_eval``);
- tool results are fed back with the ``observation`` role, and parallel
  calls are separated by ``<|assistant|>`` markers inside the completion.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from infinilm.agents.base_detector import BaseFormatDetector
from infinilm.agents.protocol import Tool
from infinilm.agents.types import StreamingParseResult, _GetInfoFunc
from infinilm.agents.utils import safe_literal_eval

logger = logging.getLogger(__name__)


# Official detection pattern from the model card (used by has_tool_call()).
_OFFICIAL_FC_PATTERN = re.compile(r"([^\n`]*?)\n({.*?})(?=\w*\n|$)", re.DOTALL)


class Glm4Chat0414Detector(BaseFormatDetector):
    """Detector for the GLM-4-9B-0414 metadata-style tool call format.

    Format structure::

        function_name
        {"arg": "value"}

    with an optional ``<|assistant|>`` marker between parallel calls.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = ""
        self.eot_token = "<|assistant|>"
        self.tool_call_separator = "\n"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_json_object(text: str, start: int) -> int:
        """Return the end offset of the JSON object starting at ``start``.

        Returns -1 while the object is still incomplete.
        """
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        return -1

    @staticmethod
    def _parse_arguments(args_str: str):
        """Parse tool arguments like the official example does."""
        try:
            return json.loads(args_str)
        except json.JSONDecodeError:
            pass
        try:
            return safe_literal_eval(args_str)
        except (ValueError, SyntaxError):
            return None

    @staticmethod
    def _could_be_tool_name(line: str, tool_indices: Dict[str, int]) -> bool:
        """Whether ``line`` can still (become) a known tool name."""
        line = line.strip()
        if not line or "`" in line:
            return False
        return any(name == line or name.startswith(line) for name in tool_indices)

    def _find_candidate(
        self, text: str, pos: int, tool_indices: Dict[str, int]
    ) -> Tuple[int, int, Optional[str], bool]:
        """Locate the next ``name\\n{`` candidate at or after ``pos``.

        Returns ``(brace_pos, name_line_start, name, True)`` when a candidate
        exists, otherwise ``(hold_from, -1, None, False)`` where ``hold_from``
        marks how much of ``text[pos:]`` is safe to emit as normal text (the
        rest is held back because it could still grow into a tool call or a
        ``<|assistant|>`` call separator).
        """
        search_from = pos
        while True:
            brace_idx = text.find("{", search_from)
            if brace_idx == -1 or brace_idx == 0:
                break
            if text[brace_idx - 1] != "\n":
                search_from = brace_idx + 1
                continue
            name_line_start = max(text.rfind("\n", pos, brace_idx - 1), pos - 1) + 1
            name = text[name_line_start : brace_idx - 1]
            if "`" in name or not name.strip():
                search_from = brace_idx + 1
                continue
            return brace_idx, name_line_start, name.strip(), True

        # No candidate. Hold back a tail that could still grow into
        # "name\n{" with future increments: the last line, but only if it
        # can still become a known tool name.
        hold_from = len(text)
        if text.endswith("\n"):
            prev_nl = text.rfind("\n", pos, len(text) - 1)
            line_before = text[prev_nl + 1 : len(text) - 1]
            if self._could_be_tool_name(line_before, tool_indices):
                hold_from = prev_nl + 1
        else:
            last_nl = text.rfind("\n", pos)
            tail = text[last_nl + 1 :]
            if self._could_be_tool_name(tail, tool_indices):
                hold_from = last_nl + 1

        # The trailing text could also be a partial "<|assistant|>"
        # separator; never release characters that are a suffix of it.
        for i in range(min(len(self.eot_token), len(text) - pos), 0, -1):
            if text.endswith(self.eot_token[:i]):
                hold_from = min(hold_from, len(text) - i)
                break
        return hold_from, -1, None, False

    # ------------------------------------------------------------------
    # one-shot parsing
    # ------------------------------------------------------------------

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        tool_indices = self._get_tool_indices(tools)
        text = text.replace(self.eot_token, "\n")

        normal_parts: List[str] = []
        calls = []
        pos = 0
        while pos < len(text):
            nl_brace = text.find("\n{", pos)
            if nl_brace == -1:
                normal_parts.append(text[pos:])
                break
            name_line_start = max(text.rfind("\n", pos, nl_brace), pos - 1) + 1
            name = text[name_line_start:nl_brace].strip()
            json_end = self._scan_json_object(text, nl_brace + 1)
            if json_end == -1:
                # Truncated JSON at end of text: keep everything verbatim.
                normal_parts.append(text[pos:])
                break

            if name in tool_indices:
                arguments = self._parse_arguments(text[nl_brace + 1 : json_end])
                if arguments is not None:
                    normal_parts.append(text[pos:name_line_start])
                    calls.extend(
                        self.parse_base_json(
                            {"name": name, "parameters": arguments}, tools
                        )
                    )
                else:
                    # Unparseable arguments: keep the block as normal text.
                    normal_parts.append(text[pos:json_end])
            else:
                # Not a known tool: keep the text verbatim.
                normal_parts.append(text[pos:json_end])
            pos = json_end

        normal_text = "".join(normal_parts).strip()
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    # ------------------------------------------------------------------
    # streaming parsing
    # ------------------------------------------------------------------

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        tool_indices = self._get_tool_indices(tools)

        # Call separators never belong to user-visible text. A separator may
        # arrive split across increments: complete ones are replaced right
        # away, while a trailing partial separator is held back (safe_end)
        # until the next increment decides what it is.
        if self.eot_token in self._buffer:
            self._buffer = self._buffer.replace(self.eot_token, "\n")
        safe_end = len(self._buffer)
        for k in range(min(len(self.eot_token) - 1, len(self._buffer)), 0, -1):
            if self._buffer.endswith(self.eot_token[:k]):
                safe_end = len(self._buffer) - k
                break

        calls = []
        normal_chunks: List[str] = []
        pos = 0
        while True:
            brace_pos, name_line_start, name, found = self._find_candidate(
                self._buffer, pos, tool_indices
            )
            if not found:
                emit_end = min(brace_pos, safe_end)
                if emit_end > pos:
                    normal_chunks.append(self._buffer[pos:emit_end])
                self._buffer = self._buffer[emit_end:]
                return StreamingParseResult(
                    normal_text="".join(normal_chunks), calls=calls
                )

            json_end = self._scan_json_object(self._buffer, brace_pos)

            if name in tool_indices:
                if json_end == -1:
                    # Incomplete JSON of a known tool: hold it back until the
                    # object is complete (or the stream ends).
                    if name_line_start > pos:
                        normal_chunks.append(self._buffer[pos:name_line_start])
                    self._buffer = self._buffer[name_line_start:]
                    return StreamingParseResult(
                        normal_text="".join(normal_chunks), calls=calls
                    )
                arguments = self._parse_arguments(self._buffer[brace_pos:json_end])
                if arguments is not None:
                    if name_line_start > pos:
                        normal_chunks.append(self._buffer[pos:name_line_start])
                    calls.extend(
                        self.parse_base_json(
                            {"name": name, "parameters": arguments}, tools
                        )
                    )
                else:
                    normal_chunks.append(self._buffer[pos:json_end])
                pos = json_end
            else:
                # Unknown name: not a tool call; release it as normal text.
                if json_end == -1:
                    emit_end = max(pos, min(safe_end, len(self._buffer)))
                    if emit_end > pos:
                        normal_chunks.append(self._buffer[pos:emit_end])
                    self._buffer = self._buffer[emit_end:]
                    return StreamingParseResult(
                        normal_text="".join(normal_chunks), calls=calls
                    )
                normal_chunks.append(self._buffer[pos:json_end])
                pos = json_end

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Flush remaining buffer at stream end."""
        result = StreamingParseResult()
        if self._buffer:
            sp_result = self.parse_streaming_increment("", tools)
            result.normal_text += sp_result.normal_text
            result.calls.extend(sp_result.calls)
            # Anything still buffered now is a truncated tool call; release
            # it as text so no generated content is silently lost.
            if self._buffer:
                result.normal_text += self._buffer
                self._buffer = ""
        return result

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------

    def has_tool_call(self, text: str) -> bool:
        return bool(_OFFICIAL_FC_PATTERN.search(text))

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()
