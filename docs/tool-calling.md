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
| `glm5` | `<tool_call>name<arg_key>...</arg_key><arg_value>...</arg_value>` | GLM-5.3 key/value XML |
| `deepseek_v4` | DSML `tool_calls` / `invoke` / typed `parameter` blocks | DeepSeek-V4 Flash/Pro |
| `kimi_k3` | XTML `tools` / indexed `call` / typed `argument` or JSON blocks | Kimi K3 |

Choose by the checkpoint's actual chat template, rather than its architecture
alone: Hermes and MiniCPM5 can both use `model_type=llama`, with different tool
formats. The parser does not add missing model/backend support or tool training.
Llama JSON mode excludes its Python-call and built-in code interpreter formats.
Hermes tokenizers with named templates must provide a `tool_use` template.
These adapters assume that the selected checkpoint already runs correctly on
the installed InfiniLM native backend. Adding a parser does not add C++ model
implementations or certify untested weights.

Template references: [Qwen](https://qwen.readthedocs.io/en/stable/framework/function_call.html),
[Hermes](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B),
[Llama](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/prompt_format.md),
[Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), and
[MiniCPM](https://github.com/OpenBMB/MiniCPM).
Current-family references: [DeepSeek V4 encoding](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/README.md),
[Kimi K3 encoding](https://huggingface.co/moonshotai/Kimi-K3/blob/main/encoding_k3.py),
and [GLM 5.3 template](https://huggingface.co/zai-org/GLM-5.3/blob/main/chat_template.jinja).

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

For GLM, a named choice retains all declared tool definitions so the model can
understand the full task; output validation still permits only the named tool.
Thinking controls follow the checkpoint: GLM-5.3 may ignore `enable_thinking`,
so allow enough output tokens for reasoning and enough context for the complete
tool history. A short output budget can truncate calls in either HTTP or SSE.
An engine timeout returns HTTP 504 or an SSE `timeout_error` followed by a
`timeout` finish. It never becomes a successful tool completion.

An open `<think>` at the end of the rendered generation prompt starts reasoning
mode; explicit `<think>`/`</think>` output is also recognized. Thinking controls
and `reasoning_effort` are passed to the native template, whose accepted values
can differ between models. MiniCPM5 CDATA string parameters preserve literal XML
tags and newlines. Ordinary Llama JSON objects without a call shape stay text.

Kimi K3 uses XTML think/response channels, rather than K2's tool-section tokens.
`enable_thinking` and `reasoning_effort` are translated to its native `thinking`
and `thinking_effort` keywords. Call indices must follow call order; external
OpenAI call IDs remain opaque and are matched by the native encoder on replay.

DeepSeek V4 checkpoints must retain their official
`encoding/encoding_dsv4.py`. The registered `DeepSeekV4Processor` delegates prompt
encoding to that file, places tool definitions on a system message and serializes
historical arguments as JSON strings. It uses the checkpoint tokenizer for token
IDs and does not replace the official encoder with a Jinja approximation.
`thinking_mode` can select `chat` or `thinking`; accepted effort values follow
the installed encoder. DSML's explicit string flag and K3's argument type are
honored before shared schema validation.

Historical reasoning is retained independently of the next response's thinking
mode. DeepSeek disables history pruning; its original message renderer and K3's
original segment renderer retain the historical thinking channel while the new
generation prefix follows the requested mode. Hooks use per-processor function
globals, without modifying checkpoint files or shared module state. These paths
require the checkpoint's Python encoders (`render_message` for DeepSeek and
`build_chat_segments` for K3); custom tokenizer replacements need separate tests.
Text-only K3 chat inputs use that tokenizer's segment-aware encoding directly,
so literal special-token-looking user/tool text is not reinterpreted as a
structural marker. Raw text prompts and multimodal encoding retain their
existing paths and are separate validation scopes.

Qwen3.5, DeepSeek V4, GLM5 and K3 receive per-turn prompt instructions for required/named
tools, disabled tools and `parallel_tool_calls=false`. Output validation remains
mandatory: these instructions are not constrained decoding guarantees.

## Scope and layout

The first step is model-side tool protocol support, kept in the InfiniLM
repository. `server/agent_protocol.py` owns request/history validation and shared
call objects; `server/tool_parsers/` owns the native output scanners. New formats
reuse the HTTP/SSE service and argument validation.

Agent execution, browser sessions, memory stores and orchestration belong in an
agent framework or a separate integration package. They do not belong in model
processors or the native inference engine.

DeepSeek-V3, InternLM, Kimi-K2, MiniCPM4-MCP and Gemma native tool formats
are deferred: their framing, history, reasoning or model-backend requirements
need separate adapters and validation. MiniCPM4-MCP's Python-style calls are not
the MiniCPM5 XML format; vision/audio MiniCPM support is also a separate claim.

This layer provides text/tool protocol adaptation. Admission quotas, gateway
authentication, embeddings and durable memory are separate components.
