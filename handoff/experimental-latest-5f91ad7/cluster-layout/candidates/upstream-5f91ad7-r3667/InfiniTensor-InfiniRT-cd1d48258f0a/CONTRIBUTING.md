# Contributing Guide

For build and test commands, see [Development Guide](#development-guide) below.

## Code

Please review these details before committing, especially for AI-generated code.

### General

1. Keep changes minimal. Do not add what is not necessary.
2. Prefer self-explanatory code over abundant comments.
3. Files must end with a newline.
4. Use Markdown syntax, such as backticks, when referencing identifiers in
   comments and error messages.
5. Comments and error messages must be in English.
6. Comments and error messages should follow the language's conventions first.
   If the language does not specify, use complete sentences: capitalize the
   first letter and end with punctuation.

### C++

1. Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
   and the repository `.clang-format`.
2. Do not use exceptions in runtime code. Return status values where an API
   surface requires runtime error propagation, and use assertions only for
   internal invariants.
3. Error and warning messages follow the
   [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html#error-and-warning-messages).
4. Initializer list order must match member declaration order.
5. Put one blank line between classes, between classes and functions, and
   between functions.
6. Put one blank line between each member, including both functions and
   variables, within a class.
7. Put one blank line before and after the contents of a namespace.

### Python

Follow [PEP 8](https://peps.python.org/pep-0008/) as the primary style guide.
For anything PEP 8 does not cover in detail, refer to the
[GDScript style guide](https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_styleguide.html)
for non-syntax conventions.

#### Additional Rules

1. Comments should be complete English sentences, starting with a capital
   letter and ending with punctuation. Use Markdown syntax when referencing
   code within comments.
2. When a framework has an established error-message convention, follow that
   convention. Otherwise, use the same rules as comments.
3. If a function has no docstring or comment, do not add a blank line between
   the function signature and the function body.
4. Add a blank line before and after `if`, `for`, and similar control-flow
   statements when it improves readability.
5. Add a blank line before a `return` statement, unless it directly follows a
   control-flow statement.
6. Follow [PEP 257](https://peps.python.org/pep-0257/) for docstrings.

## Commits

Commit messages must follow
[Conventional Commits](https://www.conventionalcommits.org/).

## Pull Requests

1. Small PRs should be squashed. Large PRs may keep multiple commits, but each
   commit must be meaningful and well-formed.
2. PR titles follow the same Conventional Commits format as commit messages.
3. Pull requests should include smoke build and smoke test evidence for each
   affected platform. Full-suite or full-platform validation is expected for
   high-risk changes, release preparation, maintainer spot checks, and changes
   that affect shared build, dispatch, generated headers, public headers,
   installation, or cross-platform behavior.
4. Use `.github/PULL_REQUEST_TEMPLATE.md` and fill every section. Delete a
   section only when it is genuinely not applicable, and state why.

## Branches

Branch names use the format `<type>/xxx-yyyy-zzzz`, where `<type>` matches the
PR title's Conventional Commits type and words are joined with hyphens.

---

# Development Guide

## Prerequisites

- C++17-compatible compiler
- CMake 3.18+
- Python 3 for public-header generation
- Backend SDKs for the accelerator being built, such as CUDA Toolkit for
  NVIDIA or DTK for Hygon
- Optional tools for local checks: `clang-format`, `ruff`, Doxygen, and
  Graphviz

## Build

Configure and build InfiniRT with CMake:

```bash
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=ON
cmake --build build -j
```

If no backend option is enabled, CPU is enabled by default. CPU can be enabled
together with one accelerator backend. Only one accelerator backend can be
enabled in a single build.

Common backend options are:

```bash
-DWITH_CPU=ON
-DWITH_NVIDIA=ON
-DWITH_ILUVATAR=ON
-DWITH_HYGON=ON
-DWITH_METAX=ON
-DWITH_MOORE=ON
-DWITH_CAMBRICON=ON
-DWITH_ASCEND=ON
-DAUTO_DETECT_DEVICES=ON
```

Use separate build directories when switching platforms, for example
`build-cpu` and `build-nvidia`.

## Testing

Enable tests with:

```bash
-DINFINI_RT_BUILD_TESTING=ON
```

Run the configured test suite with:

```bash
ctest --test-dir build --output-on-failure
```

For pull requests, run a smoke build plus `ctest` on every affected platform.
At minimum, include the exact configure, build, and test commands in the PR
description. If an affected platform cannot be tested locally, state why and
request review from a maintainer with access to that platform.

The `test_install_consumer` test installs InfiniRT to a temporary prefix,
compiles a small external consumer against the installed prefix, and verifies
that downstream projects can consume the installed headers and library.

## Installing

Install the configured build with:

```bash
cmake --install build
```

The install prefix should contain public headers under `include/` and the
InfiniRT library under `lib/`.

## Documentation

Build the Doxygen reference with:

```bash
cmake -S . -B build -DWITH_CPU=ON -DINFINI_RT_BUILD_DOCS=ON
cmake --build build --target infinirt_docs
```

The generated HTML is written under `build/docs/reference/html`.

## Formatting

C++ code should pass the repository `clang-format` workflow. The workflow uses
`clang-format` 21, so CI is the source of truth when local formatter versions
disagree.

Python scripts should pass Ruff:

```bash
ruff format --check .
ruff check .
```

## Adding or Updating a Backend

1. Add or update the backend option and detection logic in `CMakeLists.txt`.
2. Add public-header generation support in `scripts/generate_public_headers.py`
   when a new public backend wrapper is needed.
3. Keep backend-specific native headers under `src/native/<backend>/` or the
   existing backend family layout.
4. Add or update tests under `tests/`, including runtime dispatch and installed
   consumer coverage when public headers or libraries change.
5. Update `docs/backends.md` and compatibility documentation when public
   support or backend requirements change.

## Troubleshooting

1. **CMake cannot find a backend SDK**: Check the backend-specific environment
   variables documented in `docs/backends.md`, such as `DTK_ROOT`,
   `MACA_PATH`, `MUSA_ROOT`, `NEUWARE_HOME`, or `ASCEND_HOME_PATH`.
2. **Switching between backends gives stale build errors**: Use a separate
   build directory per platform or delete the stale build directory.
3. **Generated headers are stale**: Re-run CMake configure. Public headers are
   generated during configuration.
4. **`test_install_consumer` fails to configure, link, or run**: Verify that the
   install prefix contains the CMake package, headers, and libraries. Check that
   `CMAKE_PREFIX_PATH` and the backend SDK root variables documented above are
   available to the consumer build and runtime linker.
5. **Documentation build cannot find Doxygen**: Install Doxygen and Graphviz, or
   rely on the Documentation Pages workflow for validation.
