<!--
Thanks for contributing to InfiniCCL! Please read `CONTRIBUTING.md` before
opening a pull request and fill out every section below. Delete any section
that is genuinely not applicable (and note why), but do not delete the
"Checklist" section — it must be filled in for every PR.

The PR title MUST follow Conventional Commits, e.g.
  feat: support NCCL backend
  fix: correct the averaging logic in `AllReduce`
See: https://www.conventionalcommits.org/
-->

## Summary

<!--
A concise description of **what** this PR changes and **why** this change is needed. Reference files with backtick-fenced paths (e.g. `src/device.h`).
-->


## Changes

<!--
Provide a categorized summary of the changes introduced in this PR.

Guidelines:
- Use first-level bullet points with bolded category titles.
- Under each category, describe the specific changes in concise bullet points.
- Keep descriptions concise and focused on what changed and why it matters.
- Multiple nesting levels are allowed when necessary, but should generally not exceed three levels.
-->

- **Category 1**
  - Brief explanation of change 1
  - Brief explanation of change 2
  - ...

- **Category 2**
  - Brief explanation of change 1
  - Brief explanation of change 2
  - ...

- **Category 3**
  - Brief explanation of change 1
  - Brief explanation of change 2
  - ...

## Platform and Backend Affected

<!--
Please check all the platforms and/or backends this PR affects (i.e., code is touched, behavior may change, etc.). 
-->

### Platform

- [ ] CPU
- [ ] NVIDIA GPU
- [ ] Iluvatar GPU
- [ ] MetaX GPU
- [ ] Moore Threads GPU
- [ ] Cambricon MLU
- [ ] HYGON DCU

### Backend

- [ ] OpenMPI
- [ ] MPICH
- [ ] NCCL
- [ ] MCCL

## Performance Impact

- [ ] No performance impact
- [ ] Performance improved
- [ ] Performance regression possible

<!--
Benchmark is required for `perf` PRs; optional otherwise. Describe the benchmark harness,
data size, dtypes, hardware, and include baseline vs. new numbers. If the PR is not 
performance-sensitive, write "N/A".
-->
If applicable, provide benchmark results.

## Known Issues & Future Work

<!--
Describe any known limitations, caveats, or unresolved issues introduced by this PR in bullet points.

If applicable, also outline potential follow-up improvements, extensions, or future work related to these changes.
-->

## Test Results

<!--
Describe the test environment and list the test results. 

Check the corresponding boxes for all applicable test setups. 

At minimum, attach the log files generated from running `scripts/run_examples.py` with all applicable example programs and configurations related to this PR.

See `CONTRIBUTING.md` § Pull Requests for the official testing requirements and expectations for submitted PRs.
-->

### Test Involved Platform

- [ ] CPU
- [ ] NVIDIA GPU
- [ ] Iluvatar GPU
- [ ] MetaX GPU
- [ ] Moore Threads GPU
- [ ] Cambricon MLU
- [ ] HYGON DCU

### Test Involved Backend

- [ ] OpenMPI
- [ ] MPICH
- [ ] NCCL
- [ ] MCCL

---

## Checklist

> Every contributor **must** verify every item below before requesting
> review. Tick each box only after the check has actually been performed —
> do not tick speculatively. If an item truly does not apply, replace the
> checkbox with `N/A` and briefly explain why in an inline comment.

### Title, Branch, and Commits

- [ ] PR **title** follows [Conventional Commits](https://www.conventionalcommits.org/) (e.g. `feat: …`, `fix(nccl): …`).
- [ ] Branch name follows `<type>/xxx-yyyy-zzzz` where `<type>` matches the PR title's Conventional Commits type and words are joined with hyphens (see `CONTRIBUTING.md` §Branches).
- [ ] Each **commit** message follows Conventional Commits.
- [ ] Small PR is a **single squashable commit**; or, for a large PR, every commit is meaningful, well-formed, and independently reviewable (see `CONTRIBUTING.md` §Pull Requests).
- [ ] No stray merge commits from `master` — the branch is rebased cleanly on top of the current `master`.
- [ ] No `fixup!` / `squash!` / `wip` commits remain.

### Scope and Design

- [ ] Changes are **minimal** — no unrelated modifications were introduced (`CONTRIBUTING.md` §Code/General).
- [ ] No dead code, commented-out blocks, debug prints, `printf`/`std::cout`/`print(...)` left behind, or `TODO` without an owner and issue link.
- [ ] No unrelated formatting churn that would obscure the diff.
- [ ] Public API changes (if any) are intentional, documented, and reflected in affected callers/tests.

### General Code Hygiene

- [ ] The code is self-explanatory; comments were added **only** where the intent or rationale is non-obvious (`CONTRIBUTING.md` §Code/General).
- [ ] Every modified or added file **ends with a single trailing newline** (`CONTRIBUTING.md` §Code/General).
- [ ] No trailing whitespace, inconsistent indentation, or mixed formatting styles remain.
- [ ] Identifiers referenced in comments or error messages are wrapped in Markdown backticks (e.g. ``the `AllReduce` implementation``) (`CONTRIBUTING.md` §Code/General).
- [ ] All comments and error messages are in **English** (`CONTRIBUTING.md` §Code/General).
- [ ] Comments and error messages are complete sentences — capitalized first letter, terminal punctuation — **unless** the language/framework convention says otherwise (`CONTRIBUTING.md` §Code/General; §Python).

### C++ Specific (if C++ files changed)

- [ ] Code follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) strictly.
- [ ] `clang-format` (version **16**, per `.github/workflows/clang-format.yml`) has been run against all modified applicable files; the diff is clean.
- [ ] **No exceptions** are thrown. Error paths use `assert` with messages that include at least `__FILE__`, `__LINE__`, and `__func__` (`CONTRIBUTING.md` §C++).
- [ ] Error and warning message wording follows the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html#error-and-warning-messages) (`CONTRIBUTING.md` §C++).
- [ ] Constructor **initializer list order matches member declaration order** (`CONTRIBUTING.md` §C++).
- [ ] Exactly **one blank line** between classes, between classes and functions, and between functions (`CONTRIBUTING.md` §C++).
- [ ] Exactly **one blank line** between members (functions *and* variables) within a class (`CONTRIBUTING.md` §C++).
- [ ] Exactly **one blank line** before and after the contents of a namespace (`CONTRIBUTING.md` §C++).

### Python Specific (if Python files changed)

- [ ] Code is [PEP 8](https://peps.python.org/pep-0008/) compliant; `ruff check` passes cleanly on CI (see `.github/workflows/ruff.yml`).
- [ ] `ruff format --check` passes cleanly — if not, run `ruff format` and commit the result.
- [ ] Comments are complete English sentences, starting with a capital letter and ending with punctuation; Markdown backticks are used for code references (`CONTRIBUTING.md` §Python).
- [ ] Framework-specific conventions (e.g. lowercase `pytest.skip` messages without terminal period) are honored where applicable (`CONTRIBUTING.md` §Python).
- [ ] **No blank line** between the function signature and the body when there is no docstring or comment (`CONTRIBUTING.md` §Python).
- [ ] A blank line is present **before and after** `if`, `for`, and similar control-flow statements (`CONTRIBUTING.md` §Python).
- [ ] A blank line appears **before** each `return`, except when it directly follows a control-flow statement (`CONTRIBUTING.md` §Python).
- [ ] Docstrings (if any) follow [PEP 257](https://peps.python.org/pep-0257/) (`CONTRIBUTING.md` §Python).
- [ ] Type hints are added / kept consistent with the surrounding code.

### Testing

- [ ] All applicable example programs have been built and tested successfully on at least one supported heterogeneous cluster setup.

### Build, CI, and Tooling

- [ ] New backends or devices have been added to auto-detection in `CMakeLists.txt` under `if(AUTO_DETECT_DEVICES)` or to `if(AUTO_DETECT_BACKENDS)` if applicable.
- [ ] Both CI workflows (`clang-format.yml`, `ruff.yml`) are green locally (or expected to be green on CI).

### Documentation

- [ ] `README.md`, `CONTRIBUTING.md`, or inline docs updated when behavior, build flags, or developer workflow changed.
- [ ] Any user-visible breaking change is called out explicitly under "Summary" **and** in the commit/PR title with a `!` or `BREAKING CHANGE:` footer.

### Security and Safety

- [ ] No secrets, access tokens, internal URLs, customer data, or personal hardware identifiers have been committed.
- [ ] Third-party code is license-compatible and attributed.
- [ ] No unsafe pointer arithmetic, uninitialized reads, or missing bounds checks were introduced.
