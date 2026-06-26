<!--
Thanks for contributing to InfiniLM! Please read `CONTRIBUTING.md` before
opening a pull request and fill out every section below. Delete any section
that is genuinely not applicable (and note why), but do not delete the
"Checklist" section — it must be filled in for every PR.

The PR title MUST follow Conventional Commits, e.g.
  feat: support Llama 3 models
  fix: correct linear attention calculations
See: https://www.conventionalcommits.org/
-->

## Summary

<!--
A concise description of **what** this PR changes. Prefer bullet points over
prose. Reference files with backtick-fenced paths (e.g. `csrc/models/llama/*`).
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

<!-- Tick one or more. -->

- [ ] `feat` — new feature / new model
- [ ] `fix` — bug fix
- [ ] `perf` — performance improvement (no behavioral change)
- [ ] `refactor` — code restructuring without behavior change
- [ ] `test` — adding or fixing tests only
- [ ] `docs` — documentation only
- [ ] `build` / `ci` — build system or CI configuration
- [ ] `chore` — tooling, formatting, or other non-code changes
- [ ] Breaking change

## Test Results of Involved Models on Supported Platforms (Please attach screenshots)

<!--
For now, provide the screenshots for involved tests performed on platforms supposed to support the changes.

Tests that might be involved:
Single request test: examples/test_infer.py
Offline performance: examples/bench.py
Sanity test: test/bench/test_benchmark.py
Service: python/infinilm/server/inference_server.py + scripts/test_perf.py

Model adaptations may involve all four tests, unless specifying partial support for vastly new structures and confirmed with admin.

Python-level changes may involve less tests depending on which files/features are modified. 

Framework-level and Python-level changes may affect different models. Test a few models that might be affected.
-->

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

## CI / ChatOps

<!--
CI does not run automatically on pull requests. Trigger it manually from the
Actions tab (workflow: **CI**, branch: this PR's head branch), or ask a
maintainer to comment `/retest` or `/test` on this PR.
-->

---

## Checklist

> Every contributor **must** verify every item below before requesting
> review. Tick each box only after the check has actually been performed —
> do not tick speculatively. If an item truly does not apply, replace the
> checkbox with `N/A` and briefly explain why in an inline comment.

### Title, Branch, and Commits

- [ ] PR **title** follows [Conventional Commits](https://www.conventionalcommits.org/) (e.g. `feat(nvidia): …`, `fix(cuda/gemm): …`).
- [ ] Branch name follows `<type>/xxx-yyyy-zzzz` where `<type>` matches the PR title's Conventional Commits type and words are joined with hyphens (see `CONTRIBUTING.md` §Branches).
- [ ] Each **commit** message follows Conventional Commits.
- [ ] Small PR is a **single squashable commit**; or, for a large PR, every commit is meaningful, well-formed, and independently reviewable (see `CONTRIBUTING.md` §Pull Requests).
- [ ] No stray merge commits from `main` — the branch is rebased cleanly on top of the current `main`.
- [ ] No `fixup!` / `squash!` / `wip` commits remain.
- [ ] Existing PR/branch/commit that followed the legacy issue format.

### Scope and Design

- [ ] Changes are **minimal** — nothing unrelated to the stated motivation was added (`CONTRIBUTING.md` §Code/General).
- [ ] No dead code, commented-out blocks, debug prints, `printf`/`std::cout`/`print(...)` left behind, or `TODO` without an owner and issue link.
- [ ] No unrelated formatting churn that would obscure the diff.
- [ ] Public API changes (if any) are intentional, documented, and reflected in affected callers/tests.

### General Code Hygiene (applies to all languages)

- [ ] The code is self-explanatory; comments were added **only** where the *why* is non-obvious (`CONTRIBUTING.md` §Code/General).
- [ ] Every modified or added file **ends with a single trailing newline** (`CONTRIBUTING.md` §Code/General).
- [ ] No trailing whitespace, tab/space mixing, or stray BOMs.
- [ ] Identifiers in comments and error messages are wrapped in backticks (e.g. ``the `seqlens_k` tensor``) (`CONTRIBUTING.md` §Code/General).
- [ ] All comments and error messages are in **English** (`CONTRIBUTING.md` §Code/General).
- [ ] Comments and error messages are complete sentences — capitalized first letter, terminal punctuation — **unless** the language/framework convention says otherwise (`CONTRIBUTING.md` §Code/General; §Python).

### C++ Specific (if C++ files changed)

- [ ] Code follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) strictly.
- [ ] Error and warning message wording follows the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html#error-and-warning-messages) (`CONTRIBUTING.md` §C++).
- [ ] Constructor **initializer list order matches member declaration order** (`CONTRIBUTING.md` §C++).
- [ ] No raw `new`/`delete`; RAII / smart pointers / existing allocators are used.
- [ ] Changed files are formatted by `scripts/format.py`.
- [ ] No changes/reference to `csrc/models/llama_legacy/`.

### Python Specific (if Python files changed)

- [ ] Code is [PEP 8](https://peps.python.org/pep-0008/) compliant.
- [ ] Comments are complete English sentences, starting with a capital letter and ending with punctuation; Markdown backticks are used for code references (`CONTRIBUTING.md` §Python).
- [ ] Docstrings (if any) follow [PEP 257](https://peps.python.org/pep-0257/) (`CONTRIBUTING.md` §Python).
- [ ] Changed files are formatted by `scripts/format.py`.
- [ ] No changes/reference to `python/infinilm/auto_config.py`.

### Testing

- [ ] For any platform that could not be tested, an explicit reason is given in the table and a reviewer with access has been tagged.
- [ ] Passed single request test (`examples/test_infer.py`), or specify the reason for skipping.
- [ ] Passed offline performance test (`examples/bench.py`), or specify the reason for skipping.
- [ ] Passed sanity test (`test/bench/test_benchmark.py`), or specify the reason for skipping.
- [ ] Passed service test (`python/infinilm/server/inference_server.py` + `scripts/test_perf.py`), or specify the reason for skipping.

### Build, CI, and Tooling

- [ ] The project builds cleanly from a fresh directory on at least one affected platform.
- [ ] CI has been triggered manually (Actions → **CI** on this branch), or `/retest` was requested.

### Documentation

- [ ] `README.md`, `CONTRIBUTING.md`, or inline docs updated when behavior, build flags, or developer workflow changed.
- [ ] Any user-visible breaking change is called out explicitly under "Motivation" **and** in the commit/PR title with a `!` or `BREAKING CHANGE:` footer.

### Security and Safety

- [ ] No secrets, access tokens, internal URLs, customer data, or personal hardware identifiers have been committed.
- [ ] Third-party code is license-compatible and attributed.
- [ ] No unsafe pointer arithmetic, uninitialized reads, or missing bounds checks were introduced.
