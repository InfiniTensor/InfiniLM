"""Keep historical reasoning independent of the next response's mode."""

from types import FunctionType, MethodType


def _with_globals(function, **overrides):
    # Isolate hooks per processor: never mutate checkpoint module globals shared
    # by other processors or concurrent chat/thinking requests.
    clone = FunctionType(
        function.__code__,
        dict(function.__globals__, **overrides),
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


def deepseek_history_encoder(encode):
    render = encode.__globals__["render_message"]

    def render_history(index, messages, thinking_mode, **kwargs):
        message = messages[index]
        next_message = messages[index + 1] if index + 1 < len(messages) else {}
        historical_reasoning = (
            message.get("role") == "assistant" and message.get("reasoning_content")
        ) or (
            message.get("role") in {"user", "developer"}
            and message.get("task") is None
            and next_message.get("role") == "assistant"
            and next_message.get("reasoning_content")
        )
        if historical_reasoning and thinking_mode != "thinking":
            thinking_mode = "thinking"
            # Effort belongs to the pending response, not a historical prefix.
            kwargs["reasoning_effort"] = None
        return render(index, messages, thinking_mode=thinking_mode, **kwargs)

    return _with_globals(encode, render_message=render_history)


def kimi_history_template(template):
    build = template.__func__.__globals__["build_chat_segments"]
    render = build.__globals__["_render_assistant_segments"]

    def render_history(message, image_state, thinking=True):
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        return render(
            message,
            image_state,
            thinking=thinking or (reasoning is not None and str(reasoning) != ""),
        )

    build = _with_globals(build, _render_assistant_segments=render_history)
    function = _with_globals(template.__func__, build_chat_segments=build)
    return MethodType(function, template.__self__)
