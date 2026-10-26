# Qwen tool calls through Chat Completions

Install the server dependencies with `pip install '.[server]'`. Enable the
Qwen3.5 XML parser with `--tool-call-parser qwen3_coder`, or set
`"tool_call_parser": "qwen3_coder"` in the server configuration. The default is
`none`, which retains plain chat output and rejects requests declaring tools.
Use the model's own tokenizer/template. Recurrent Qwen3.5 configurations also
require prefix caching to be disabled as documented by the model backend.

Send function definitions in the request's top-level `tools` array. They are
forwarded to the chat template, rather than appended to a user message.
Responses contain OpenAI-style `tool_calls`: stable IDs, a function name and a
JSON-encoded `arguments` string. The serving process never executes the tool.

To continue a conversation, append the returned assistant message and one
`role: "tool"` message for each call. Each result must include the matching
`tool_call_id` and result `content`. Missing, duplicate or unrelated call IDs
are rejected before generation. Tool history fields survive message content
normalization and reach the Qwen template.

Both ordinary and SSE responses separate `reasoning_content` from answer text.
SSE calls are emitted only after a complete XML call passes argument schema
validation. Chunk boundaries may split any marker or JSON value. Normal tool
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

This layer provides text/tool protocol adaptation. Admission quotas, gateway
authentication, embeddings and durable memory are separate components.
