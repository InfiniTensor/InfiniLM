<!--
Thanks for contributing to InfiniRT! Please read `CONTRIBUTING.md` before
opening a pull request and fill out every section below. Delete any section
that is genuinely not applicable, and note why.

The PR title MUST follow Conventional Commits, e.g.
  docs: add backend compatibility notes
  fix: correct runtime dispatch for CPU builds
  feat: add Hygon graph capture support
See: https://www.conventionalcommits.org/
-->

## Summary

<!--
A concise description of what this PR changes. Prefer bullet points over prose.
Reference files with backtick-fenced paths, such as `src/runtime.h`.
-->

-
-

## Motivation

<!--
Explain why this change is needed. Link to any related issue, bug, or
discussion. If this is a performance change, include before/after numbers with
hardware, shape, dtype, and measurement methodology.
-->

Closes #

## Type of Change

<!-- Tick one or more. The type MUST match the Conventional Commits prefix in
the PR title and branch name. See `CONTRIBUTING.md` Branches. -->

- [ ] `feat` - new feature / new backend capability / new public API
- [ ] `fix` - bug fix
- [ ] `perf` - performance improvement without behavior change
- [ ] `refactor` - code restructuring without behavior change
- [ ] `test` - adding or fixing tests only
- [ ] `docs` - documentation only
- [ ] `build` / `ci` - build system or CI configuration
- [ ] `chore` - tooling, formatting, or other non-code changes
- [ ] Breaking change (requires a `!` in the Conventional Commits prefix or a `BREAKING CHANGE:` footer)

## Platforms Affected

<!-- Tick every backend whose code is touched or whose behavior may change. -->

- [ ] CPU (`WITH_CPU`)
- [ ] NVIDIA (`WITH_NVIDIA`)
- [ ] Iluvatar (`WITH_ILUVATAR`)
- [ ] Hygon (`WITH_HYGON`)
- [ ] MetaX (`WITH_METAX`)
- [ ] Moore (`WITH_MOORE`)
- [ ] Cambricon (`WITH_CAMBRICON`)
- [ ] Ascend (`WITH_ASCEND`)
- [ ] Build system / CMake / generated headers
- [ ] Public headers / installed consumer API
- [ ] Documentation only

## Smoke Build and Test Result

<!--
Paste the smoke configure, build, and test command plus trimmed output for every
affected platform.

Example:
`cmake -S . -B build -DWITH_CPU=ON -DINFINI_RT_BUILD_TESTING=ON`
`cmake --build build -j`
`ctest --test-dir build --output-on-failure`
-->

```text
paste smoke build and test output here
```

## Test Results on Supported Platforms

<!--
Per `CONTRIBUTING.md` Pull Requests, build and run smoke tests on every affected
platform. Use `smoke passed`, `full passed`, or `N/A - not affected` in result
columns. Run broader validation for high-risk changes, release prep, maintainer
spot checks, or changes affecting shared build, dispatch, generated headers,
public headers, installation, or cross-platform behavior.

If an affected platform was not tested, state the reason and tag a reviewer or
owner with access.
-->

| Platform | Affected | Build / Smoke Result | Full Result / Notes |
| --- | :---: | --- | --- |
| CPU | | | |
| NVIDIA | | | |
| Iluvatar | | | |
| Hygon | | | |
| MetaX | | | |
| Moore | | | |
| Cambricon | | | |
| Ascend | | | |

<details>
<summary>Full `ctest` output (optional)</summary>

```text
paste here
```

</details>

## Benchmark / Performance Impact

<!--
Required for `perf` PRs; optional otherwise. Describe the benchmark harness,
hardware, backend, and include baseline vs. new numbers. If the PR is not
performance-sensitive, write "N/A".
-->

## Notes for Reviewers

<!--
Anything reviewers should focus on: subtle invariants, known trade-offs, or
follow-up work intentionally left out of scope.
-->
