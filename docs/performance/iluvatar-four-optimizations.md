# Iluvatar inference optimizations and Prefill integration

This opt-in change submits the fused causal Softmax through the normal runtime
execution path. Enable `INFINILM_ILUVATAR_FUSED_CAUSAL_PREFILL=1` only with the
companion InfiniOps FP32 causal provider. Unset it to retain the existing
Fill/Triu/Tril/Add + Softmax path. The environment flag is exact (`1`), with
eligibility restricted to contiguous 3D Iluvatar BF16, 2 <= Q <= K <= 4096.
The default remains unchanged, including Decode Attention.

The operator submission preserves owner checks, stream/recording policy,
normal dispatch and runtime profiling. It removes the experimental direct
kernel entry point and counter. The compatibility operator already exists;
the companion PR specializes its FP32 accumulation order without adding APIs.
An older InfiniOps build exposes the same operator name but does not have this
FP32 provider: install the companion change before enabling the flag.

## Four retained optimizations

| Item | Bottleneck addressed | Narrow implementation |
| --- | --- | --- |
| Gate/Up GEMV | BLAS overhead and weight traffic at B1/B2 | BF16 K4096/N32768, shared weight loads for two rows |
| Down GEMV | Small-row Down projection | BF16 K16384/N4096, same four-accumulator scheme |
| Prefill causal Softmax | Mask allocations and separate masking launches | Register causal mask, FP32 max/exp/sum, BF16 exponent store/divide |
| Decode QK reduction | Repeated CTA barriers across eight token scores | Same FP32 trees, 48→6 QK barriers, CTA64/tile8/split4 |

Gate/Up and Down share one InfiniOps PR. Prefill has an InfiniOps provider PR
and this InfiniLM integration PR. Decode has a separate InfiniOps PR. These
changes are independent of the earlier R5 modernization submissions.

## Historical same-card application evidence

The four-item combination was compared with official InfiniLM
`6e7f9537173cd33bb30539ebf8d92ab6d869660d` on TG-V200/CoreX using 100 new
fixed GSM8K targets, eight demonstrations from a disjoint test-corpus subset,
greedy decoding, max_tokens=512, EOS=[2,73440], and strict last-`####` numeric
scoring. The model, prompts, weights, answer references and scoring were held
fixed. This is an internal paired evaluation, not a standard benchmark split.

Each target's two versions ran on the same physical card. The first 20 targets
ran serially on one card; the remaining 80 were sharded over two cards. Each
model had two fixed warmup requests before four formal requests. Fifty model
processes completed normally (scope return, maps, atexit and OS exit0), with
200 formal and 100 warmup requests. Repeated warmups add no independent quality
samples. Two physical cache rows and two active requests were used throughout.

| Metric | Official | Four-item combination |
| --- | ---: | ---: |
| Correct / 100 | 82 | 84 |
| Completed / parseable | 100 / 100 | 100 / 100 |
| Errors / truncated | 0 / 0 | 0 / 0 |
| Sum of formal paired-wave makespans, s | 749.203818 | 505.326616 |
| Requests/s over that sum | 0.133475 | 0.197892 |
| Published tokens | 13894 | 14213 |
| Published tokens/s | 18.545020 | 28.126363 |
| Median first TokenOutput, ms | 2521.843 | 1802.338 |
| Median TPOT, ms | 69.509 | 44.958 |
| Median request completion latency, ms | 11942.692 | 8004.228 |

Time fell **32.5515%**, request throughput rose **48.2613%**, and token
throughput rose **51.6653%**. The time denominator is the sum of each version's
same-card formal wave makespans, including queue and text processing, excluding
model load, warmup and final shutdown. It is not the parallel run's host wall
time. Output lengths changed; token throughput is not substituted for completed
request throughput. These are combined gains, not sums of component gains.

### Token differences and task regressions

34/100 targets had different token sequences: 26 remained correct, four
remained wrong, one regressed, and three improved. Accuracy increased by two
percentage points while preserving this individual regression:

| Target ID | Reference | Official prediction | Combined prediction | Change |
| --- | ---: | ---: | ---: | --- |
| 1097 | 480 | 480 | 448 | Correct → wrong |
| 672 | 2 | 9 | 2 | Wrong → correct |
| 1259 | 7200 | 720 | 7200 | Wrong → correct |
| 1203 | 240 | 420 | 240 | Wrong → correct |

99 request latencies decreased. Target 672 took 73.5054% longer as its output
grew from 57 to 238 tokens and its answer became correct. Token differences and
correctness are recorded separately. Finite sample accuracy is not a claim of
universal per-task non-regression.

## Current submission and remaining validation

This PR is rebased on main `cc6d52ae04f4c9ff17229bc7fc13a9ad105aca6d`,
which adds automatic operator-tuning initialization after the measured baseline.
The companion InfiniOps PRs are based on master
`1a68b261d19f0ed6f76bbabeab7a25f479f05e07`. The original GPU results apply
to the recorded private builds; this revised normal-dispatch integration still
needs a current-base Iluvatar smoke build and the same two-task application
regression. It is submitted as a draft with the flag off by default.

Focused host check:

```sh
c++ -std=c++17 -I. test/runtime/causal_prefill_policy_test.cpp -o /tmp/causal-prefill-policy
/tmp/causal-prefill-policy
```

The companion provider includes focused device tests for masked values,
rectangular masks and in-place/out-of-place output. No dataset expansion is
needed to validate this integration.
