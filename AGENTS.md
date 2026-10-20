# Hygon migration handoff

For work on this migration, read `handoff/README.zh-CN.md` and
`handoff/AGENT_HANDOFF.zh-CN.md` before changing build or numerical behavior.
The `legacy` and `current` snapshots are separate dependency and numerical
profiles. Preserve exact pins, negative evidence, owner/lifecycle checks, and
EOS-aware useful-token accounting. Experimental component results are not
whole-model performance qualifications. Do not cancel allocations, reset GPUs,
or hot-replace libraries used by running jobs. Keep keys, credentials, weights,
machine-specific access configuration, and large raw artifacts out of Git.

Latest explicit human numerical contract (2026-10-20): require exact complete generated token IDs to the frozen reference, including EOS and repeats. Logit bitwise equality across rows/ranks/repeats/precision is diagnostic only. Keep finite-logit/CPU argmax, owner/lifecycle and Slurm checks. Prefer native BF16/FP32; F64 is reference/control only. Historical raw-only rejection is superseded, actual token mismatch still fails. Read handoff/evidence/token-acceptance-policy.json.
