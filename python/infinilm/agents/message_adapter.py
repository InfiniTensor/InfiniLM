"""
Adapt incoming chat requests to model-family tool-call conventions.
"""

from typing import Optional

# Tool-call parsers whose models use the GLM-4 "metadata" convention:
# assistant tool calls are rendered via the message ``metadata`` field and
# tool results use the ``observation`` role (see the GLM-4-9B-Chat-0414
# chat template). OpenAI-style histories must be rewritten accordingly,
# otherwise the template silently drops them and the tool loop cannot close.
GLM4_METADATA_PARSER_ALIASES = {
    "glm4",
    "glm49b",
    "glm4-9b-0414",
    "glm-4-9b-0414",
}

# Qwen3 models natively understand OpenAI-format tool messages in their
# chat templates, so no rewriting is required.
QWEN3_PARSER_ALIASES = {
    "qwen3",
    "qwen3-30b-a3b",
}


def adapt_messages(messages: list, tool_call_parser: Optional[str]) -> list:
    """Rewrite OpenAI-format tool history for parsers that need it.

    Applies only to the GLM-4 metadata-style convention
    (``GLM4_METADATA_PARSER_ALIASES``): ``tool`` messages become
    ``observation`` messages, and assistant messages carrying
    ``tool_calls`` become one ``metadata`` message per call. All other
    requests pass through unchanged.
    """
    if tool_call_parser not in GLM4_METADATA_PARSER_ALIASES:
        # Qwen3, Llama and other parsers pass OpenAI-format messages straight
        # through; their chat templates understand the standard roles.
        return messages

    adapted = []
    for msg in messages:
        if not isinstance(msg, dict):
            adapted.append(msg)
            continue
        if msg.get("role") == "tool":
            adapted.append({**msg, "role": "observation"})
        elif msg.get("role") == "assistant" and msg.get("tool_calls"):
            if msg.get("content"):
                adapted.append({"role": "assistant", "content": msg["content"]})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                adapted.append(
                    {
                        "role": "assistant",
                        "metadata": fn.get("name", ""),
                        "content": fn.get("arguments", "") or "{}",
                    }
                )
        else:
            adapted.append(msg)
    return adapted


def prepare_chat_template_kwargs(data: dict) -> None:
    """Pack tool definitions from the request into ``chat_template_kwargs``.

    Pops nothing: ``tools``/``tool_choice`` stay in ``data`` for the output
    post-processing, and are additionally forwarded to ``apply_chat_template``
    so the model prompt actually sees the tool definitions.
    """
    tools = data.get("tools") or []
    tool_choice = data.get("tool_choice", "auto")
    chat_template_kwargs = data.get("chat_template_kwargs") or {}
    if tools:
        chat_template_kwargs["tools"] = tools
    if tool_choice:
        chat_template_kwargs["tool_choice"] = tool_choice
    data["chat_template_kwargs"] = chat_template_kwargs
