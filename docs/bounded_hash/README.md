# Opt-in bounded-hash routing for complete replicas

This draft adds a separate HTTP gateway for independent full-model replicas.
Each replica runs its own Prefill and Decode. The existing server and PD paths
keep their defaults. Enable the new backend contract and launch the gateway
explicitly to use it. This is an unexecuted upstream adaptation of the historical
V3 experiment, with no claim that its performance transfers to current main.

## Supported surface

The initial backend accepts dense Qwen3 or Llama text models, TP1, a paged cache,
prefix caching, whole prefills, and one complete model per backend process.
It rejects PP, PD/KV transfer, MoE EP, MLA/NSA, Mamba, speculative decoding,
chunked prefills, tools and multimodal requests. The gateway accepts
`POST /v1/chat/completions`, in streaming or non-streaming mode, with text
`messages` (system/user/assistant), optional `model`, a positive explicit
`max_tokens`, `temperature`, `top_p`, `top_k`, `add_generation_prompt`, and
`chat_template_kwargs.enable_thinking`. Streaming callers may request
`stream_options.include_usage`. Unsupported fields are rejected before generation.
The backend retains the server's EOS configuration. Normal `stop` and `length`
completions are valid; online requests do not require a fixed-length benchmark
finish. Errors, timeouts, incomplete SSE and disconnects are failures.

The gateway does not load another model or tokenizer. It uses an installed
InfiniLM package and the `bounded-hash` optional dependencies (`fastapi`,
`uvicorn`, `httpx`, `jsonschema`), along with the existing package dependencies.
No dependencies were installed during this adaptation. Native imports follow
the current `_infinilm._core` layout; no old binary or import-path injection is
included.

## Invocation and configuration

These commands are documentation examples and were **not executed**. Model paths
and environment-variable values are deployment placeholders.

Start each isolated backend on its assigned device using the normal server:

```sh
export INFINILM_REPLICA_KEY='<deployment-secret>'
python -m infinilm.server.inference_server \
  --model /path/to/model --device moore --dtype bfloat16 --tp 1 \
  --enable-paged-attn --attn paged-attn \
  --num-blocks 128 --block-size 256 --max-batch-size 3 \
  --host 127.0.0.1 --port 8000 \
  --bounded-hash-backend-key-env INFINILM_REPLICA_KEY \
  --bounded-hash-cache-policy prefix_lru \
  --bounded-hash-max-inflight 3 \
  --bounded-hash-hardware-id '<vendor/model/memory-and-runtime-label>'
```

Create a deployment copy of [gateway.example.json](gateway.example.json), set
`enabled` to `true`, configure all replica origins and secret environment-variable
names, and provide a compatible cost file. The example has `enabled: false`.
Then run a single gateway process:

```sh
python -m infinilm.server.bounded_hash_gateway --config /path/to/gateway.json
```

All traffic to these replicas must pass through this one state owner. The CLI
uses one HTTP worker and holds an exclusive `state_lock` for its lifetime.
Every process targeting a pool must use the same lock file. Multiple hosts or
different lock paths do not share capacity state and are unsupported. Do not
run this app with multiple ASGI workers or multiple gateways for the same pool.
Backend authorization is private to the gateway; generation requires its bearer
secret, current instance ID and verified token digest. Exposing the gateway
outside a trusted network requires deployment authentication at its ingress.

The historical settings in the example are explicit choices: 128 virtual nodes
per replica, at most three in flight, skew 1.25, five-second cache-hint TTL,
32768 KV tokens in 256-token pages, `prefix_lru`, and up to 1000 ms of active
reuse waiting. These are not universal hardware recommendations. `queue_timeout_s`
is a separate capacity-queue timeout; the V3 active-wait limit is not a limit on
all queueing.

`prefix_tokens` selects a page-aligned proper prefix to be hashed from the
backend's actual input token IDs. A shorter prompt has zero reusable prefix.
Set this field to zero (or null) for conservative routing without affinity.
Clients cannot supply cache-hit claims or `X-Fourway-Item`. No character-count
estimate or synthetic cache hit is used.

## Cost compatibility

A cost file is mandatory; absence, invalid coefficients or incompatible
metadata fails explicitly. [cost.template.json](cost.template.json) is an
intentionally incomplete schema illustration, **not calibration**. Replace its
nulls with separately obtained measurements and the running backends' exact
`identity`, page size, KV capacity, policy and concurrency. Do not copy the old
MUSA coefficients and call them current-main calibration. No calibration command
is run automatically and none was run in this adaptation.

The authenticated `/internal/bounded-hash/contract` endpoint accepts
`{"request": <supported chat request>, "prefix_tokens": <integer or null>}`.
It returns token counts/digests, capacity, a boot instance ID, and identity
including model configuration/tokenizer/processor/source hashes, dtype, device,
TP and an explicit hardware label. The gateway compares every compatibility
field on every request and requires identical token contracts across the pool.
The hardware label is an operator assertion, not a device probe. Model config
hashes do not prove equality of weight bytes; identical weights and deployed
native binaries remain deployment prerequisites. No model weight files are
read by the gateway or added to this PR.

The phase model keeps the original explicit three-coefficient Prefill curve and
positive Decode samples for every batch size through the configured cap. It
charges preceding pending Prefills, the incoming Prefill and one co-resident
Decode round. It is a ranking estimate, not a latency guarantee. All source
hashes and identity values must match the deployed adaptation; the base revision
alone does not identify an unbuilt native runtime or establish calibrated speed.

## Actual call chain and lifecycle

1. The gateway validates supported input and asks each backend for a token
   contract. The backend calls the same existing processor, tokenizer, chat
   template and sampling builder used by generation, without another model.
2. `Gateway.prepare` checks model/tokenization/cost compatibility and updates
   `Router.observe_backend` with authoritative instance IDs. Restarted instances
   invalidate prior observations and active-Decode predictions.
3. `Router.acquire` uses the V3/V2 decision chain and reserves the **complete**
   page-rounded input plus maximum output, independent of estimated reuse.
   Per-replica in-flight capacity is hard. The projected request-count bound is
   soft and retains its least-loaded fallback. Consistent hashing keeps one
   primary and at most one persistent hot secondary.
4. `Gateway.generate` sends exactly one generation POST with bound instance,
   token digest and prefix length headers. The backend verifies them against
   its current tokenizer before calling `AsyncLLMEngine.add_chat_request`.
   Generation POSTs are not retried or redirected.
5. Existing `stream_options.include_usage` is reused. Opt-in internal progress
   events add actual generated-token counts, initial cache-hit counts and CPU
   residency snapshots with boot ID, epoch and observation age. Ordinary token
   usage alone is not a cache snapshot. The owner scheduler publishes metadata
   after its existing completion/publication path, without pin/unpin or GPU
   synchronization. Repeated epochs do not renew the TTL, and pending Prefills
   invalidate observed hints. Forecast reuse remains distinct from observed hits.
6. Streaming is parsed as SSE; non-streaming responses aggregate the same
   verified backend stream. Normal EOS stops are accepted. Child async
   generators are explicitly closed and a single `finally` releases the lease
   after success, error, cancellation or downstream disconnect. A successful
   completion waits for current-main CPU resource retirement. On an uncertain
   generation POST failure, the gateway quarantines that replica until its boot
   instance changes. This avoids admitting more work onto a possibly still
   executing request and does not replay the request.

V3 considers waiting only for another healthy, reusable replica at a hard
capacity limit with no pending Prefill. It checks that the earliest release
would fit the **full** reservation and that waiting plus warm Prefill has a
predicted benefit. The deadline is fixed once per request, with a budget of
`min(warm_wait_ms, 25% of cold/warm Prefill savings)`, capped at 1000 ms in this
entry. It cannot be refreshed by later Decode progress.

## Allocator compatibility and scope of changes

- `base_config.py` and `server/inference_server.py`: explicit backend flags,
  authenticated internal contract, bound generation and opt-in feedback.
- `server/bounded_hash_backend.py`: real token contracts and bounded CPU prefix
  observations, instance-scoped metadata, supported-topology guards.
- `llm/cache_manager.py`: startup-only `prefix_lru` selection, pin removal from
  the idle list, tail-before-prefix idle ordering on final unpin, and eviction of
  zero-reference idle blocks. Existing allocation, rollback, hash publication,
  NSA owner checks and reference counts keep their current-main paths. No old
  cache-manager fingerprint is replaced, disabled or bypassed; no live
  `manager.__init__` reset exists.
- `llm/scheduler.py`: capture the initial cache hit and publish read-only
  observations on the existing resource-owner thread.
- `routing/bounded_hash/*`: package migration of the actual V3 dependency chain,
  instance/TTL handling, bounded routing event history and cancellation cleanup.
- `server/bounded_hash_{config,gateway}.py`: explicit compatible cost loading,
  single-owner admission, one-attempt transport and stream/non-stream handling.
- `pyproject.toml`: a separate opt-in dependency extra; default service extras
  and all community workflows are unchanged.

Source modules, hashes and licensing are attributed in
[routing NOTICE](../../python/infinilm/routing/bounded_hash/NOTICE) and
[historical provenance](history/SOURCE.json). The vendored hash-ring module and
Apache license are preserved. Historical `cache_policy.py`, `nh_stream_usage.py`,
`gateway.py` and `streaming.py` were reviewed as dependencies; their obsolete
monkey patches and fixed-length benchmark validator were not installed.

## Draft status and validation

The code provides an actual opt-in request path, but runtime behavior has **not
been validated**. Outstanding review work includes current-main native/MUSA
compatibility, allocator failure/rollback stress, token-contract equality on real
models, stream cancellation/resource-retirement timing, and per-deployment cost
calibration. Multi-owner capacity, dynamic membership, multimodal/tool requests
and unsupported topologies are outside this initial scope. An interrupted
backend currently requires a restart to leave quarantine; automated recovery is
intentionally absent. Replica contract retrieval currently requires all
configured backends to respond, so a missing member fails admission conservatively.

See [ADAPTATION_VALIDATION.json](ADAPTATION_VALIDATION.json) for this adaptation's
static-only checks and NOT_RUN record. New test source is included for future
review and was not executed. The separate unmodified historical
[PACKAGING_VALIDATION.json](history/PACKAGING_VALIDATION.json) records 262 CPU
checks from archive packaging, 98 unchanged originals and zero new GPU runs at
that earlier step. It does not cover this code.

No local tests, benchmarks, model runs, builds, or calibration runs were executed during this adaptation, at the author's request. The reported MUSA results belong to the historical V3 experiment and do not validate the current upstream adaptation. Any automatically triggered upstream CI results are reported separately.

The current `ci_test.yml` and `ruff.yml` include push/PR triggers despite the
older PR template's manual-only CI description. The native device CI job is
gated on `workflow_dispatch`. This task does not manually dispatch or rerun CI.
Automatically triggered checks are separate from local execution and historical
results. [Historical MUSA results](history/README.md) include adverse p95 and
128-token results and their single-seed, high-reuse synthetic-workload limits.
