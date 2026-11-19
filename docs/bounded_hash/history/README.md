# Historical V3 MUSA results

These are preserved results for the frozen V3 experiment, not measurements of the current upstream adaptation. [summary.csv](summary.csv) is an unmodified copy of the [public historical CSV](https://github.com/nhcf/InfiniLM/tree/c6f1005f3e81d0799229d66bb6db6c37f1d257ac/examples/bounded_hash/results/summary.csv). [SOURCE.json](SOURCE.json) records the fixed source commit and file digests. The raw `optimized_run_c16_20261104T020635_636924Z_report.zip` was unavailable during adaptation; the available success records are the CSV and [historical RESULTS.md](https://github.com/nhcf/InfiniLM/tree/c6f1005f3e81d0799229d66bb6db6c37f1d257ac/examples/bounded_hash/RESULTS.md).

## Experiment conditions

- Eight MTT S5000 cards, approximately 80 GB per card; Qwen3-32B BF16; eight TP1 complete-model replicas, each performing its own Prefill and Decode.
- Global concurrency 16, at most three in-flight requests per replica; prompt 8192 tokens, shared prefix 7936 tokens (96.875%); eight documents, 25% hot fraction; outputs 32, 64 and 128 tokens.
- One round, seed 33962. Each of nine method/length cells has 64 learning requests and 128 measured requests; all 1728 requests succeeded. Learning establishes cache and routing state and is not model training.
- InfiniLM KV budget: 128 blocks x 256 tokens. vLLM: 2048 blocks x 16 tokens. Both reserve 32768 tokens per replica.
- Each cell starts a new model pool. Its measured stage inherits the learning-stage cache. Method order rotates between output lengths; this is not repeated paired measurement within each scenario.
- vLLM uses the historical MUSA-adapted environment, not an official unmodified release. The exact successful-run package versions could not be independently verified because the raw report archive was unavailable; versions from the older reference environment are not attributed to this result panel.

The historical source uses 128 virtual nodes per replica, in-flight cap 3, skew 1.25, cache-hint TTL 5 seconds and `prefix_lru`. The active-wait budget is `min(1000 ms, 25% of the estimated cold/warm Prefill saving)` with a fixed deadline for each request. It bounds the deliberate reuse wait, not all queueing time. Historical cost parameters and settings are specific to that environment and are not calibration for current main or other hardware.

The [original reference environment](https://github.com/nhcf/InfiniLM/tree/c6f1005f3e81d0799229d66bb6db6c37f1d257ac/examples/bounded_hash/experiment/reference_environment.json) records the historical runtime baseline plus local changes and older `cache_confirm_run_20261028T081428_439145Z` provenance. It contains machine paths and is linked instead of copied. Its original SHA256 is in SOURCE.json. This older environment record must not be interpreted as a current-main V3 execution record.

## Measured stage

| Output tokens | Method | Output token/s | Mean TTFT ms | p95 TTFT ms |
| --- | --- | ---: | ---: | ---: |
| 32 | Bounded hash V3 | 103.666 | 696.12 | 2990.36 |
| 32 | V2 | 93.037 | 913.41 | 4811.38 |
| 32 | vLLM least-active | 85.117 | 1545.53 | 3846.80 |
| 64 | Bounded hash V3 | 109.824 | 758.49 | 4794.39 |
| 64 | V2 | 105.615 | 979.30 | 4835.07 |
| 64 | vLLM least-active | 101.539 | 1496.09 | 2978.82 |
| 128 | Bounded hash V3 | 114.589 | 1026.75 | 4867.01 |
| 128 | V2 | 115.613 | 853.38 | 4845.87 |
| 128 | vLLM least-active | 114.841 | 1408.70 | 2976.20 |

Relative to vLLM, V3 throughput is +21.79% with 32 output tokens, +8.16% with 64 and -0.22% with 128. Mean TTFT decreases by 54.96% and 49.30% at 32 and 64 output tokens. The p95 TTFT is worse than vLLM at 64 and 128 output tokens. At 128 output tokens, V3 throughput is 0.89% below V2 and mean TTFT is 20.32% higher than V2.

## Learning plus measured stages

| Output tokens | V3 output token/s | V2 output token/s | vLLM output token/s |
| --- | ---: | ---: | ---: |
| 32 | 89.023 | 84.516 | 80.393 |
| 64 | 102.966 | 100.579 | 98.716 |
| 128 | 113.046 | 113.990 | 112.945 |

Combined throughput divides output tokens from both stages by the sum of their timed durations. Model startup, probes and calibration are excluded, so this is not full deployment cold-start performance. The 128-token V3/vLLM difference is approximately a tie (+0.09%). Measured rows are subsets of `learn_plus_measured` rows; adding both phases double counts requests.

## Limits and validation status

This is one round and one seed on synthetic traffic with high prefix reuse. It does not demonstrate broad superiority over vLLM, statistical significance or improved answer quality. No native round-robin or old mean comparison was made in this panel. October C16/C32/C64 results are not included.

[PACKAGING_VALIDATION.json](PACKAGING_VALIDATION.json) preserves the historical archive check unchanged: 262 CPU tests passed, 98 original experiment files were unchanged, and that packaging step ran zero new GPU experiments. These are historical checks and do not cover the new adaptation. The original candidate README and `experiment/VALIDATION.json` contained GPU `NOT_RUN` before the later successful experiment documented in RESULTS.md; the original records have not been rewritten.

No local tests, benchmarks, model runs, builds, or calibration runs were executed during this adaptation, at the author's request. The reported MUSA results belong to the historical V3 experiment and do not validate the current upstream adaptation. Any automatically triggered upstream CI results are reported separately.

Current adaptation checks and NOT_RUN status are recorded separately from the unmodified historical validation file. Static parsing and data-table comparison do not establish runtime correctness.

## Source and license

The source snapshot is [nhcf/InfiniLM at `c6f1005f3e81d0799229d66bb6db6c37f1d257ac`](https://github.com/nhcf/InfiniLM/tree/c6f1005f3e81d0799229d66bb6db6c37f1d257ac/examples/bounded_hash). Its SOURCE_MANIFEST.json attributes 98 original experiment files to `infinilm_cache_v3_20261104.zip`, recorded SHA256 `25407351f3589f65e992b0781cff18000a194c16623f867730d6b7c3f1cf5b45`. All 98 local source files, fixed-commit blobs and upload-package members matched the recorded file digests during read-only discovery. The original source ZIP itself was unavailable, so its recorded archive digest was not recomputed.

The archive's Apache-2.0 routing license is preserved in [LICENSE](LICENSE); original authorship and license notices remain unchanged. This history directory records provenance and measured data, and does not replace the current cache manager or native runtime with historical code.
