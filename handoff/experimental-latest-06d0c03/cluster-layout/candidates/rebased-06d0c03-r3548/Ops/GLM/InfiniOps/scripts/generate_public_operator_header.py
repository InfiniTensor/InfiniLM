#!/usr/bin/env python3
"""Generate the public `operator.h` installed for downstream consumers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_CALL_SIGNATURE_RE = re.compile(
    r"(?P<indent>^[ \t]*)template\s*<typename\.\.\.\s+Args>\s*\n"
    r"(?P=indent)static\s+void\s+Call\s*\(\s*"
    r"const\s+Handle\s*&\s*handle\s*,\s*"
    r"const\s+Config\s*&\s*config\s*,\s*"
    r"const\s+Args\s*&\s*\.\.\.\s*args\s*\)\s*",
    re.MULTILINE,
)

# Provider implementations are visible only while building InfiniOps. Their
# lookup and construction bodies must not be optimized in a downstream TU,
# even when that TU sees an `extern template` declaration.
_PROVIDER_SIGNATURES = (
    (
        "Operator::CanMake(vector<Tensor>)",
        re.compile(
            r"(?P<indent>^[ \t]*)template\s*<typename\.\.\.\s+Args>\s*\n"
            r"(?P=indent)static\s+bool\s+CanMake\s*\(\s*"
            r"const\s+Config\s*&\s*config\s*,\s*"
            r"const\s+std::vector<Tensor>\s+tensors\s*,\s*"
            r"const\s+Args\s*&\s*\.\.\.\s*args\s*\)\s*",
            re.MULTILINE,
        ),
    ),
    (
        "Operator::CanMake(Tensor)",
        re.compile(
            r"(?P<indent>^[ \t]*)template\s*<typename\.\.\.\s+Args>\s*\n"
            r"(?P=indent)static\s+bool\s+CanMake\s*\(\s*"
            r"const\s+Config\s*&\s*config\s*,\s*const\s+Tensor\s+tensor\s*,\s*"
            r"const\s+Args\s*&\s*\.\.\.\s*args\s*\)\s*",
            re.MULTILINE,
        ),
    ),
    (
        "Operator::Make(Tensor)",
        re.compile(
            r"(?P<indent>^[ \t]*)template\s*<typename\.\.\.\s+Args>\s*\n"
            r"(?P=indent)static\s+std::unique_ptr<Operator>\s+Make\s*\(\s*"
            r"const\s+Config\s*&\s*config\s*,\s*const\s+Tensor\s+tensor\s*,\s*"
            r"Args\s*&&\s*\.\.\.\s*args\s*\)\s*",
            re.MULTILINE,
        ),
    ),
    (
        "Operator::Make(vector<Tensor>)",
        re.compile(
            r"(?P<indent>^[ \t]*)template\s*<typename\.\.\.\s+Args>\s*\n"
            r"(?P=indent)static\s+std::unique_ptr<Operator>\s+Make\s*\(\s*"
            r"const\s+Config\s*&\s*config\s*,\s*"
            r"const\s+std::vector<Tensor>\s+tensors\s*,\s*"
            r"Args\s*&&\s*\.\.\.\s*args\s*\)\s*",
            re.MULTILINE,
        ),
    ),
    (
        "Operator::active_implementation_indices",
        re.compile(
            r"(?P<indent>^[ \t]*)static\s+std::vector<std::size_t>\s+"
            r"active_implementation_indices\s*\(\s*Device::Type\s+dev_type\s*\)\s*",
            re.MULTILINE,
        ),
    ),
    (
        "Operator::DefaultConfig",
        re.compile(
            r"(?P<indent>^[ \t]*)static\s+Config\s+"
            r"DefaultConfig\s*\(\s*Device::Type\s+dev_type\s*\)\s*",
            re.MULTILINE,
        ),
    ),
    (
        "Operator::DefaultImplementationIndex",
        re.compile(
            r"(?P<indent>^[ \t]*)static\s+std::size_t\s+"
            r"DefaultImplementationIndex\s*\(\s*Device::Type\s+dev_type\s*\)\s*",
            re.MULTILINE,
        ),
    ),
    (
        "Operator::MakeWithDevice",
        re.compile(
            r"(?P<indent>^[ \t]*)template\s*<typename\.\.\.\s+Args>\s*\n"
            r"(?P=indent)static\s+std::unique_ptr<Operator>\s+"
            r"MakeWithDevice\s*\(\s*const\s+Config\s*&\s*config\s*,\s*"
            r"Device::Type\s+dispatch_device_type\s*,\s*"
            r"Args\s*&&\s*\.\.\.\s*args\s*\)\s*",
            re.MULTILINE,
        ),
    ),
    (
        "Operator::SelectWithDevice",
        re.compile(
            r"(?P<indent>^[ \t]*)template\s*<typename\s+Selected\s*,\s*typename\.\.\.\s+Args>\s*\n"
            r"(?P=indent)static\s+bool\s+SelectWithDevice\s*\(\s*"
            r"const\s+Config\s*&\s*config\s*,\s*"
            r"Device::Type\s+dispatch_device_type\s*,\s*"
            r"bool\s+strict\s*,\s*Selected\s*&&\s*selected\s*,\s*"
            r"Args\s*&&\s*\.\.\.\s*args\s*\)\s*",
            re.MULTILINE,
        ),
    ),
)


def _find_matching_brace(text: str, open_brace: int) -> int:
    depth = 0
    index = open_brace
    state = "normal"
    raw_string_end = ""

    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if state == "line_comment":
            if char == "\n":
                state = "normal"
            index += 1
            continue

        if state == "block_comment":
            if char == "*" and next_char == "/":
                state = "normal"
                index += 2
                continue
            index += 1
            continue

        if state == "string":
            if char == "\\":
                index += 2
                continue
            if char == '"':
                state = "normal"
            index += 1
            continue

        if state == "char":
            if char == "\\":
                index += 2
                continue
            if char == "'":
                state = "normal"
            index += 1
            continue

        if state == "raw_string":
            if text.startswith(raw_string_end, index):
                state = "normal"
                index += len(raw_string_end)
                continue
            index += 1
            continue

        if char == "/" and next_char == "/":
            state = "line_comment"
            index += 2
            continue

        if char == "/" and next_char == "*":
            state = "block_comment"
            index += 2
            continue

        if char == "R" and next_char == '"':
            delimiter_start = index + 2
            delimiter_end = text.find("(", delimiter_start)

            if delimiter_end != -1:
                delimiter = text[delimiter_start:delimiter_end]
                raw_string_end = f'){delimiter}"'
                state = "raw_string"
                index = delimiter_end + 1
                continue

        if char == '"':
            state = "string"
            index += 1
            continue

        if char == "'":
            state = "char"
            index += 1
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                return index

        index += 1

    raise RuntimeError("Could not find the end of `Operator::Call`.")


def _hide_definition(
    text: str, signature: re.Pattern[str], name: str, replacement: str = ";"
) -> str:
    match = signature.search(text)

    if match is None:
        raise RuntimeError(f"Could not find the `{name}` signature.")

    body_start = text.find("{", match.end())

    if body_start == -1:
        raise RuntimeError(f"Could not find the `{name}` body.")

    if text[match.end() : body_start].strip():
        raise RuntimeError(f"Unexpected tokens before the `{name}` body.")

    body_end = _find_matching_brace(text, body_start)
    signature = text[match.start() : match.end()].rstrip()

    return f"{text[: match.start()]}{signature}{replacement}{text[body_end + 1 :]}"


def generate_public_operator_header(source: Path, output: Path) -> None:
    public_text = _hide_definition(
        source.read_text(),
        _CALL_SIGNATURE_RE,
        "Operator::Call",
        " {\n    return detail::OperatorFactory<Key>::Call(handle, config, args...);\n  }",
    )
    for name, signature in _PROVIDER_SIGNATURES:
        replacement = ";"
        if name.startswith("Operator::Make(") or name.startswith("Operator::CanMake("):
            method = "CanMake" if "CanMake" in name else "Make"
            first = "tensors" if "vector<Tensor>" in name else "tensor"
            replacement = (
                " {\n    return detail::OperatorFactory<Key>::"
                f"{method}(config, {first}, args...);\n  }}"
            )
        public_text = _hide_definition(public_text, signature, name, replacement)
    public_text = public_text.replace('#include "host_range_profiler.h"\n', "")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(public_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("src/operator.h"))
    parser.add_argument(
        "--output", type=Path, default=Path("generated/include/operator.h")
    )
    args = parser.parse_args()
    generate_public_operator_header(args.source, args.output)


if __name__ == "__main__":
    main()
