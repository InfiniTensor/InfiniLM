"""
Utility functions for agent parsing.
"""

import ast
import json
import logging
import threading
import warnings
from json import JSONDecodeError, JSONDecoder
from json.decoder import WHITESPACE
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


try:
    import partial_json_parser
    from partial_json_parser.core.options import Allow

    PARTIAL_JSON_AVAILABLE = True
except ImportError:
    partial_json_parser = None
    Allow = None
    PARTIAL_JSON_AVAILABLE = False


def _find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def _partial_json_loads(input_str: str, flags: Any) -> Tuple[Any, int]:
    """
    Parse incomplete or partial JSON strings commonly encountered during streaming.
    Falls back to standard JSONDecoder if partial_json_parser is unavailable.
    """
    if PARTIAL_JSON_AVAILABLE and partial_json_parser is not None:
        try:
            return (partial_json_parser.loads(input_str, flags), len(input_str))
        except (JSONDecodeError, IndexError) as e:
            msg = getattr(e, "msg", str(e))
            if "Extra data" in msg or "pop from empty list" in msg:
                start = WHITESPACE.match(input_str, 0).end()
                obj, end = JSONDecoder().raw_decode(input_str, start)
                return obj, end
            raise
        except AssertionError as e:
            raise JSONDecodeError(
                "partial_json_parser assertion (treat as incomplete)", input_str, 0
            ) from e
    else:
        # Fallback: use standard JSONDecoder which handles complete JSON only
        # In streaming context, this will raise on incomplete JSON, which is expected
        start = WHITESPACE.match(input_str, 0).end()
        obj, end = JSONDecoder().raw_decode(input_str, start)
        return obj, end


def _is_complete_json(input_str: str) -> bool:
    try:
        json.loads(input_str)
        return True
    except JSONDecodeError:
        return False


_safe_ast_lock = threading.Lock()


def _run_ast_quiet(fn, *args):
    with _safe_ast_lock, warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return fn(*args)


def safe_literal_eval(value: str) -> Any:
    return _run_ast_quiet(ast.literal_eval, value)


def get_schema_properties(schema: Any) -> Dict[str, Any]:
    """Top-level ``properties`` of a tool ``parameters`` schema."""
    if not isinstance(schema, dict):
        return {}
    properties = schema.get("properties")
    if isinstance(properties, dict):
        return properties
    merged: Dict[str, Any] = {}
    for keyword in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            for branch in branches:
                for key, value in get_schema_properties(branch).items():
                    merged.setdefault(key, value)
    return merged


def infer_type_from_json_schema(schema: Dict[str, Any]) -> Optional[str]:
    """Infer the primary type of a parameter from JSON Schema."""
    if not isinstance(schema, dict):
        return None

    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            return type_value
        elif isinstance(type_value, list) and type_value:
            non_null_types = [t for t in type_value if t != "null"]
            if non_null_types:
                return non_null_types[0]
            return "string"

    if "anyOf" in schema or "oneOf" in schema:
        schemas = schema.get("anyOf") or schema.get("oneOf")
        types = []
        if isinstance(schemas, list):
            for sub_schema in schemas:
                inferred_type = infer_type_from_json_schema(sub_schema)
                if inferred_type:
                    types.append(inferred_type)
        if types:
            if len(set(types)) == 1:
                return types[0]
            if len(set(types)) == 2 and "null" in types:
                return [t for t in types if t != "null"][0]
            if "string" in types:
                return "string"
            return types[0]

    if "enum" in schema and isinstance(schema["enum"], list):
        if not schema["enum"]:
            return "string"
        enum_types = set()
        for value in schema["enum"]:
            if value is None:
                enum_types.add("null")
            elif isinstance(value, bool):
                enum_types.add("boolean")
            elif isinstance(value, int):
                enum_types.add("integer")
            elif isinstance(value, float):
                enum_types.add("number")
            elif isinstance(value, str):
                enum_types.add("string")
            elif isinstance(value, list):
                enum_types.add("array")
            elif isinstance(value, dict):
                enum_types.add("object")
        if len(enum_types) == 1:
            return enum_types.pop()
        return "string"

    if "allOf" in schema and isinstance(schema["allOf"], list):
        schemas = schema["allOf"]
        for sub_schema in schemas:
            inferred_type = infer_type_from_json_schema(sub_schema)
            if inferred_type and inferred_type != "string":
                return inferred_type
        return "string"

    if "properties" in schema:
        return "object"
    if "items" in schema:
        return "array"

    return None


def _convert_to_number(value: str) -> Any:
    """Convert string to appropriate number type (int or float)."""
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        else:
            return int(value)
    except (ValueError, AttributeError):
        return value


def parse_arguments(
    json_value: str, arg_type: Optional[str] = None
) -> Tuple[Any, bool]:
    """Parse argument value with multiple fallback strategies."""
    if not isinstance(json_value, str):
        return json_value, True
    try:
        parsed_value = json.loads(json_value)
        if arg_type == "number" and isinstance(parsed_value, str):
            parsed_value = _convert_to_number(parsed_value)
        return parsed_value, True
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        wrapped = json.loads('{"tmp": "' + json_value + '"}')
        parsed_value = json.loads(wrapped["tmp"])
        if arg_type == "number" and isinstance(parsed_value, str):
            parsed_value = _convert_to_number(parsed_value)
        return parsed_value, True
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    if arg_type == "string":
        if (
            len(json_value) >= 2
            and json_value[0] == json_value[-1]
            and json_value[0] in {'"', "'"}
        ):
            return json_value[1:-1], True
        return json_value, True

    try:
        parsed_value = safe_literal_eval(json_value)
        return parsed_value, True
    except (ValueError, SyntaxError):
        pass

    try:
        quoted_value = json.dumps(str(json_value))
        return json.loads(quoted_value), True
    except (json.JSONDecodeError, ValueError):
        return json_value, False
