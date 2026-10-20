<!--
Thanks for contributing to InfiniOps! Please read `CONTRIBUTING.md` before
opening a pull request and fill out every section below. Delete any section
that is genuinely not applicable (and note why).

The PR title MUST follow Conventional Commits, e.g.
  feat: add fused RMSNorm kernel
  fix: correct stride handling for batched matmul
See: https://www.conventionalcommits.org/
-->

## Summary

<!--
A concise description of **what** this PR changes. Prefer bullet points over
prose. Reference files with backtick-fenced paths (e.g. `src/cuda/gemm/blas.h`).
-->

-
-

## Motivation

<!--
Explain **why** this change is needed. Link to any related issue, bug, or
discussion. If this is a performance change, include before/after numbers
(hardware, shape, dtype, and the measurement methodology).
-->

Closes #

## Type of Change

<!-- Tick one or more. The type MUST match the Conventional Commits prefix
in the PR title and the branch name (see `CONTRIBUTING.md` §Branches). -->

- [ ] `feat` — new feature / new operator / new platform
- [ ] `fix` — bug fix
- [ ] `perf` — performance improvement (no behavioral change)
- [ ] `refactor` — code restructuring without behavior change
- [ ] `test` — adding or fixing tests only
- [ ] `docs` — documentation only
- [ ] `build` / `ci` — build system or CI configuration
- [ ] `chore` — tooling, formatting, or other non-code changes
- [ ] Breaking change (requires a `!` in the Conventional Commits prefix or a `BREAKING CHANGE:` footer)

## Platforms Affected

<!-- Tick every backend whose code is touched **or** whose behavior may change. -->

- [ ] CPU (`WITH_CPU`)
- [ ] NVIDIA (`WITH_NVIDIA`)
- [ ] Iluvatar (`WITH_ILUVATAR`)
- [ ] MetaX (`WITH_METAX`)
- [ ] Cambricon (`WITH_CAMBRICON`)
- [ ] Moore (`WITH_MOORE`)
- [ ] Ascend (`WITH_ASCEND`)
- [ ] PyTorch C++ bindings (`WITH_TORCH`)
- [ ] Build system / CMake / CI
- [ ] Python bindings / user-facing API

## Smoke Test Result

<!--
Paste the smoke test command and trimmed output for every affected platform.
Default PR validation is an affected-platform smoke build plus smoke test, e.g.
`python -m pip install .[dev] --no-build-isolation --no-deps --config-settings=cmake.define.INFINI_OPS_SMOKE_BUILD=ON`
and `python -m pytest tests -m smoke -q`.
-->

```text
paste smoke test output here
```

## Test Results on Supported Platforms

<!--
Per `CONTRIBUTING.md` §Pull Requests, build and run the smoke test set on every
affected platform. Use `smoke passed`, `full passed`, or `N/A - not affected` in
the result columns. Run the full suite for high-risk changes, release prep,
maintainer spot checks, or changes affecting shared build, dispatch, wrapper
generation, or cross-platform behavior.

If an affected platform was not tested, state the reason and tag a reviewer or
owner with access. Reviewers may request full-suite or full-platform validation
when the risk profile justifies it.
-->

| Platform   | Affected | Build / Smoke Result | Full Result / Notes |
| ---------- | :------: | ------------------- | ------------------- |
| NVIDIA     |          |                     |                     |
| Iluvatar   |          |                     |                     |
| MetaX      |          |                     |                     |
| Cambricon  |          |                     |                     |
| Moore      |          |                     |                     |
| Ascend     |          |                     |                     |

<details>
<summary>Full `pytest` output (optional)</summary>

```text
paste here
```

</details>

## Benchmark / Performance Impact

<!--
Required for `perf` PRs; optional otherwise. Describe the benchmark harness,
shapes, dtypes, hardware, and include baseline vs. new numbers. If the PR is
not performance-sensitive, write "N/A".
-->

## Notes for Reviewers

<!--
Anything reviewers should focus on: subtle invariants, known trade-offs,
follow-up work intentionally left out of scope, etc.
-->
