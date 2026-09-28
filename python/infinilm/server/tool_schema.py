"""Bound verbose tool documentation while retaining the complete parameter schema."""
import copy
import os


def compact_tool_descriptions(tools: list) -> list:
    if os.getenv("INFINILM_COMPACT_TOOL_DESCRIPTIONS", "1") == "0":
        return tools
    function_limit = int(os.getenv("INFINILM_TOOL_DESCRIPTION_CHARS", "512"))
    parameter_limit = int(os.getenv("INFINILM_PARAMETER_DESCRIPTION_CHARS", "160"))
    if function_limit <= 0 or parameter_limit <= 0:
        raise ValueError("Tool description limits must be positive.")
    result = copy.deepcopy(tools)

    def shorten(value, limit):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "description" and isinstance(item, str):
                    if len(item) > limit:
                        value[key] = item[:limit] + "..."
                else:
                    shorten(item, parameter_limit)
        elif isinstance(value, list):
            for item in value:
                shorten(item, parameter_limit)

    for tool in result:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        description = function.get("description")
        if isinstance(description, str) and len(description) > function_limit:
            function["description"] = description[:function_limit] + "..."
        shorten(function.get("parameters"), parameter_limit)
    return result
