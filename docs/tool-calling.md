# Tool calls through Chat Completions

Install the server dependencies with `pip install '.[server]'`. Enable the
appropriate parser with `--tool-call-parser NAME`, or set `tool_call_parser`
in the server configuration. The default is
`none`, which retains plain chat output and rejects requests declaring tools.
Use the model's own tokenizer/template. Recurrent Qwen3.5 configurations also
require prefix caching to be disabled as documented by the model backend.

| Parser | Native format | Model/template examples |
| --- | --- | --- |
| `qwen3_coder` | `<tool_call><function=...><parameter=...>` | Qwen3-Coder, Qwen3.5 |
| `hermes` | `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` | Qwen2.5-Instruct, Qwen3, Hermes tool-use templates |
| `llama3_json` | JSON with `name` and `parameters`, optionally preceded by `<\|python_tag\|>` | Llama 3.1/3.3 JSON tool mode |
| `mistral` | `[TOOL_CALLS]` followed by a JSON array | Mistral-7B-Instruct-v0.3 |
| `minicpm5` | `<function name="..."><param name="...">...</param></function>` | MiniCPM5-1B/2B |

Choose by the checkpoint's actual chat template, rather than its architecture
alone: Hermes and MiniCPM5 can both use `model_type=llama`, with different tool
formats. The parser does not add missing model/backend support or tool training.
Llama JSON mode excludes its Python-call and built-in code interpreter formats.
Hermes tokenizers with named templates must provide a `tool_use` template.

Template references: [Qwen](https://qwen.readthedocs.io/en/stable/framework/function_call.html),
[Hermes](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B),
[Llama](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/prompt_format.md),
[Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), and
[MiniCPM](https://github.com/OpenBMB/MiniCPM).

Send function definitions in the request's top-level `tools` array. They are
forwarded to the chat template, rather than appended to a user message.
Responses contain OpenAI-style `tool_calls`: stable IDs, a function name and a
JSON-encoded `arguments` string. The serving process never executes the tool.

To continue a conversation, append the returned assistant message and one
`role: "tool"` message for each call. Each result must include the matching
`tool_call_id` and result `content`. Missing, duplicate or unrelated call IDs
are rejected before generation. Tool history fields survive message content
normalization and reach the model template. Mistral uses generated nine-character
alphanumeric IDs; replayed IDs must have the same format.

Both ordinary and SSE responses separate `reasoning_content` from answer text.
SSE calls are emitted only after a complete call passes argument schema
validation. Tagged calls can complete before the turn ends; Llama/Mistral JSON
without a closing marker is buffered until the turn ends. Chunk boundaries may
split any marker or JSON value. Normal tool
completion uses `finish_reason: "tool_calls"`; truncation and errors do not
become successful tool completion. `stream_options.include_usage` requests a
final usage chunk before `[DONE]`.

`tool_choice` supports `auto`, `none`, `required` and a named function. Required
and named choices are checked against the generated result; they do not enable
constrained decoding. `parallel_tool_calls: false` rejects multiple calls.
`strict: true` and non-local JSON Schema references are rejected. Malformed
generated calls return HTTP 502 for ordinary responses, or an error event for
an already-open SSE response. Clients should execute tools only after a
successful completed turn, and handle tool permissions and side effects in the
agent framework.

An open `<think>` at the end of the rendered generation prompt starts reasoning
mode; explicit `<think>`/`</think>` output is also recognized. Thinking controls
and `reasoning_effort` are passed to the native template, whose accepted values
can differ between models. MiniCPM5 CDATA string parameters preserve literal XML
tags and newlines. Ordinary Llama JSON objects without a call shape stay text.

## Scope and layout

The first step is model-side tool protocol support, kept in the InfiniLM
repository. `server/agent_protocol.py` owns request/history validation and shared
call objects; `server/tool_parsers/` owns the native output scanners. New formats
reuse the HTTP/SSE service and argument validation.

Agent execution, browser sessions, memory stores and orchestration belong in an
agent framework or a separate integration package. They do not belong in model
processors or the native inference engine.

DeepSeek-V3, GLM-4.5, InternLM, Kimi-K2, MiniCPM4-MCP and Gemma native tool formats
are deferred: their framing, history, reasoning or model-backend requirements
need separate adapters and validation. MiniCPM4-MCP's Python-style calls are not
the MiniCPM5 XML format; vision/audio MiniCPM support is also a separate claim.

This layer provides text/tool protocol adaptation. Admission quotas, gateway
authentication, embeddings and durable memory are separate components.
