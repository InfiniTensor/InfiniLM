"""
Minimal agent client for the InfiniLM inference server.

Demonstrates the complete tool-call loop over the OpenAI-compatible
``/v1/chat/completions`` endpoint:

    request (with tools) -> model returns tool_calls
    -> client executes the tools locally
    -> results are appended as assistant/tool messages
    -> repeat until the model answers in plain text

This is the same loop an agent framework (e.g. Claude Code) drives; it is
kept dependency-free and with a short system prompt on purpose, so the loop
can be validated on small models where huge agent prompts degrade tool use.

Usage (server started with ``--tool-call-parser llama31`` or
``--tool-call-parser glm4-9b-0414`` etc.):

    python examples/agent_client.py --url http://127.0.0.1:8000 \
        --model GLM-4-9B-0414 "北京天气怎么样？顺便看看当前目录有什么文件"
"""

import argparse
import json
import os
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories at a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"}
                },
                "required": ["path"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool locally and return its result as a string."""
    if name == "get_weather":
        city = arguments.get("city", "")
        return json.dumps(
            {"city": city, "weather": "晴", "temperature": "26度"}, ensure_ascii=False
        )
    if name == "list_dir":
        path = arguments.get("path", ".")
        try:
            entries = sorted(os.listdir(path))
        except OSError as e:
            return f"error: {e}"
        return "\n".join(entries) if entries else "(empty directory)"
    return f"error: unknown tool {name}"


def chat(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def run(url: str, model: str, question: str, max_turns: int = 6):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the available tools "
            "to answer the user's question.",
        },
        {"role": "user", "content": question},
    ]

    for turn in range(max_turns):
        response = chat(
            url,
            {
                "model": model,
                "messages": messages,
                "tools": TOOLS,
                "max_tokens": 1024,
                "stream": False,
            },
        )
        choice = response["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            print(f"\n[turn {turn}] assistant: {message.get('content', '')}")
            return

        # Record the assistant tool-call turn, execute, feed results back.
        messages.append(message)
        for call in tool_calls:
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result = execute_tool(name, arguments)
            print(f"[turn {turn}] tool_call: {name}({arguments}) -> {result[:80]}")
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )

    print("\n[max turns reached without a final answer]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="GLM-4-9B-0414")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("question", nargs="+")
    args = parser.parse_args()
    run(args.url, args.model, " ".join(args.question), args.max_turns)


if __name__ == "__main__":
    main()
