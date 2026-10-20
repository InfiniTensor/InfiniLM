# Hygon migration handoff

For work on this migration, read `handoff/README.zh-CN.md` and
`handoff/AGENT_HANDOFF.zh-CN.md` before changing build or numerical behavior.
The `legacy` and `current` snapshots are separate dependency and numerical
profiles. Preserve exact pins, negative evidence, owner/lifecycle checks, and
EOS-aware useful-token accounting. Experimental component results are not
whole-model performance qualifications. Do not cancel allocations, reset GPUs,
or hot-replace libraries used by running jobs. Keep keys, credentials, weights,
machine-specific access configuration, and large raw artifacts out of Git.
