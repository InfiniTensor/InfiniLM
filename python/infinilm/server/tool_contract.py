import copy


def _append_contract(message: dict, contract: str) -> None:
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = content.rstrip() + chr(10) + chr(10) + contract
    elif isinstance(content, list):
        content.append({"type": "text", "text": contract})
    else:
        message["content"] = contract


def apply_tool_contract(
    messages: list, original_tools: list, exposed_tools: list
) -> list:
    """Make a server-side tool filter explicit in the system context."""
    original_names = {
        tool.get("function", {}).get("name")
        for tool in original_tools
        if isinstance(tool, dict)
    }
    exposed_names = {
        tool.get("function", {}).get("name")
        for tool in exposed_tools
        if isinstance(tool, dict)
    }
    if not exposed_names or original_names == exposed_names:
        return messages

    quoted_names = ", ".join(f"`{name}`" for name in sorted(exposed_names))
    single_tool_instruction = (
        f" Create the requested deliverable content and call `{next(iter(exposed_names))}` "
        "directly."
        if len(exposed_names) == 1
        else " When an action is required, choose one tool from this exact set."
    )
    contract = (
        "<InfiniLM tool contract>\n"
        f"The complete set of tools exposed by the runtime is exactly: {quoted_names}. "
        "Do not call, mention, simulate, or substitute any other tool."
        f"{single_tool_instruction}\n"
        "</InfiniLM tool contract>"
    )

    normalized_messages = copy.deepcopy(messages)
    system_message = next(
        (
            message
            for message in normalized_messages
            if isinstance(message, dict) and message.get("role") == "system"
        ),
        None,
    )
    if system_message is None:
        normalized_messages.insert(0, {"role": "system", "content": contract})
    else:
        _append_contract(system_message, contract)

    user_message = next(
        (
            message
            for message in reversed(normalized_messages)
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        None,
    )
    if user_message is not None:
        _append_contract(user_message, contract)
    return normalized_messages
